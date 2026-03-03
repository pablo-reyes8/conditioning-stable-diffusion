import os
import shutil
import sys
import time

from src.training.autocast import make_grad_scaler
from src.training.ema import ema_health, ema_reinit_from_model, ema_set_decay
from src.training.schedule import WarmupCosineLR
from src.training.train_one_epoch import train_one_epoch


def _fmt_hms(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _rule(w=92, ch="─"):  # línea separadora
    return ch * w


def _is_colab():
    return "google.colab" in sys.modules


def _ensure_drive_mounted():
    if _is_colab():
        drive_root = "/content/drive"
        if not os.path.isdir(drive_root):
            try:
                from google.colab import drive
                drive.mount(drive_root, force_remount=False)
            except Exception as e:
                print(f"[DRIVE] No se pudo montar automáticamente: {e}")


def _copy_ckpt_to_drive_fixed(src_path: str, drive_dir: str, fixed_name: str = "latest_ddpm.pt"):
    """Copia el checkpoint a Drive con nombre fijo; si existe, lo reemplaza."""
    try:
        if not drive_dir:
            return
        if drive_dir.startswith("/content/drive"):
            _ensure_drive_mounted()
        os.makedirs(drive_dir, exist_ok=True)
        dst_path = os.path.join(drive_dir, fixed_name)
        if os.path.exists(dst_path):
            os.remove(dst_path)
            print(f"└─ [DRIVE]  eliminado previo → {dst_path}")
        shutil.copy2(src_path, dst_path)
        print(f"└─ [DRIVE]  copiado (fixed) → {dst_path}")
    except Exception as e:
        print(f"└─ [DRIVE]  ERROR al copiar a Drive: {e}")


import os, time, torch

def train_ldm(
    *,
    model, diffusion, train_loader, optimizer,
    ema,
    vae, label_encoder,
    ema_target=None,
    device: str = "cuda",
    epochs: int = 50,
    base_lr: float = 2e-4,
    min_lr: float = 2e-5,
    warmup_steps: int = 1000,
    grad_clip: float = 1.0,
    use_autocast: bool = True,
    amp_dtype: str = "bf16",
    scaler=None,
    cfg_drop_prob: float = 0.12,
    latent_scaling: float = 0.18215,
    latent_hw: int = 32,

    # sampling
    sample_every: int = 1,
    sample_n: int = 36,
    sample_steps: int = 50,
    sample_eta: float = 0.0,
    guidance_scale: float = 7.5,
    sample_seed: int | None = 1234,
    sample_labels: torch.Tensor | None = None,
    sample_fn=None,

    # NEW: sampling EMA control
    sample_with_ema: bool = True,                 # default: usa EMA
    sample_no_ema_first_n_epochs: int = 0,        # ej: 5 -> primeras 5 épocas (del run actual) sample sin EMA

    # ckpt
    ckpt_dir: str = "checkpoints",
    run_name: str = "ldm",
    save_every: int = 1,
    save_last: bool = True,
    resume_path: str | None = None,
    ckpt_utils=None,  # (save_ckpt, load_ckpt)

    # train loop options
    grad_accum_steps: int = 1,
    use_channels_last: bool = False,
    on_oom: str = "skip",

    # diag
    log_every: int = 0,
    probe_timesteps: list[int] | None = None,
    log_mem: bool = False,
    log_grad_norm: bool = False,

    # resume overrides / EMA control
    reset_optimizer_state: bool = False,
    override_lr: float | None = None,
    override_weight_decay: float | None = None,
    override_ema_decay: float | None = None,

    # NEW: EMA reset options
    reset_ema_on_resume: bool = False,            # hard reset EMA inmediatamente tras cargar ckpt
    reset_ema_at_epoch: int | None = None,        # hard reset EMA en una época específica (epoch global)
    ema_decay_after_reset: float = 0.9995,

    # existing EMA repair (sanity)
    repair_ema_on_resume: bool = False,
    ema_decay_after_repair: float = 0.9995,

    # Drive copy
    drive_ckpt_dir: str | None = None,
    copy_fixed_to_drive: bool = True,
    fixed_drive_name: str = "latest_ldm.pt",
):
    import os, time
    os.makedirs(ckpt_dir, exist_ok=True)

    save_ckpt_fn, load_ckpt_fn = (None, None)
    if ckpt_utils is not None:
        save_ckpt_fn, load_ckpt_fn = ckpt_utils

    # scaler (bf16 => None; fp16 => GradScaler)
    if scaler is None and use_autocast:
        scaler = make_grad_scaler(device=device, enabled=True, amp_dtype=amp_dtype)

    # total steps del run actual
    steps_per_epoch = max(1, len(train_loader) // max(1, grad_accum_steps))
    total_steps = epochs * steps_per_epoch

    # Scheduler SIEMPRE existe (aunque el ckpt viejo no lo tenga)
    scheduler = WarmupCosineLR(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=min_lr
    )

    # ----------------------------
    # Resume
    # ----------------------------
    global_step, start_epoch = 0, 0
    resumed = False

    if resume_path and load_ckpt_fn is not None and os.path.exists(resume_path):
        opt_for_load = None if reset_optimizer_state else optimizer
        ema_for_load = ema

        step_loaded, extra = load_ckpt_fn(
            resume_path, model,
            optimizer=opt_for_load,
            scaler=scaler,
            ema=ema_for_load,
            scheduler=scheduler,
            map_location=device
        )

        if isinstance(extra, dict):
            global_step = int(extra.get("global_step", step_loaded or 0))
            start_epoch = int(extra.get("epoch", 0)) + 1
        else:
            global_step = int(step_loaded or 0)
            start_epoch = 0

        print(f"[RESUME] {resume_path} | global_step={global_step} | start_epoch={start_epoch}")
        resumed = True

        # overrides
        if reset_optimizer_state:
            print("[RESUME] Optimizer: estado NO cargado (reset).")

        if override_lr is not None:
            for g in optimizer.param_groups:
                g["lr"] = float(override_lr)
            print(f"[RESUME] override_lr → {override_lr:.3e}")

        if override_weight_decay is not None:
            for g in optimizer.param_groups:
                g["weight_decay"] = float(override_weight_decay)
            print(f"[RESUME] override_weight_decay → {override_weight_decay:.3e}")

        if override_ema_decay is not None and (ema is not None) and hasattr(ema, "decay"):
            ema.decay = float(override_ema_decay)
            print(f"[RESUME] override_ema_decay → {ema.decay:.6f}")

        scheduler.base_lrs = [float(base_lr) for _ in optimizer.param_groups]
        scheduler.min_lr = float(min_lr)
        #scheduler.total_steps = int(total_steps)
        #scheduler.warmup_steps = int(warmup_steps)

        # Important: align step_num with global_step (your logs use global_step)
        scheduler.step_num = int(global_step)
        scheduler._set_lr(scheduler.step_num)

        print(f"[RESUME][SCHED] forced override | step_num={scheduler.step_num} "
              f"base_lr={base_lr:.2e} min_lr={min_lr:.2e} total_steps={scheduler.total_steps} "
              f"warmup_steps={scheduler.warmup_steps} lr_now={optimizer.param_groups[0]['lr']:.2e}")

        # --- NEW: hard reset EMA on resume (deliberado) ---
        if (ema is not None) and reset_ema_on_resume:
            target_module = ema_target if ema_target is not None else model
            ema_reinit_from_model(ema, target_module)
            ema_set_decay(ema, float(ema_decay_after_reset))
            print(f"[RESUME][EMA] reset_ema_on_resume=True -> reinit | decay={ema.decay:.6f}")

        # --- existing: EMA repair (solo si está “rota”) ---
        if (ema is not None) and repair_ema_on_resume:
            ok, reason, rel = ema_health(ema, model, rel_tol=5.0)
            if not ok:
                ema_reinit_from_model(ema, model)
                ema_set_decay(ema, float(ema_decay_after_repair))
                print(f"[RESUME][EMA] inválida ({reason}, rel={rel:.3f}) -> reinit | decay={ema.decay:.6f}")
            else:
                print(f"[RESUME][EMA] saludable (rel={rel:.3f})")

    else:
        # from scratch
        scheduler.base_lrs = [float(base_lr) for _ in optimizer.param_groups]
        scheduler.step_num = 0
        scheduler._set_lr(0)

    # ----------------------------
    # Header
    # ----------------------------
    ema_decay_val = getattr(ema, "decay", None)
    ema_str = f"{ema_decay_val:.6f}" if isinstance(ema_decay_val, (float, int)) else "on"
    print(_rule())
    print(f"LDM run: {run_name}")
    print(f"Device: {device} | autocast: {use_autocast}({amp_dtype}) | EMA: {ema_str} | "
          f"epochs: {epochs} | base_lr: {base_lr:.2e} | warmup_steps: {warmup_steps} | cfg_drop={cfg_drop_prob}")
    if resumed:
        print(f"[RESUME] start_epoch={start_epoch} | global_step={global_step} | lr_now={optimizer.param_groups[0]['lr']:.2e}")
    print(_rule())
    print(f"{'ep':>3} | {'step':>8} | {'loss':>10} | {'lr':>9} | "
          f"{'batches':>8} | {'images':>8} | {'imgs/s':>7} | {'time':>8} | {'warmup':>6}")
    print(_rule())

    # ----------------------------
    # Train loop
    # ----------------------------
    total_time = 0.0

    for epoch in range(start_epoch, epochs):
        # --- NEW: hard reset EMA at a specific epoch (deliberado) ---
        if (ema is not None) and (reset_ema_at_epoch is not None) and (epoch == reset_ema_at_epoch):
            target_module = ema_target if ema_target is not None else model
            ema_reinit_from_model(ema, target_module)
            ema_set_decay(ema, float(ema_decay_after_reset))
            print(f"[EMA] reset at epoch={epoch} | decay={ema.decay:.6f}")

        t0 = time.time()

        avg_loss, n_batches, n_images, global_step = train_one_epoch(
            model=model,
            diffusion=diffusion,
            dataloader=train_loader,
            optimizer=optimizer,
            vae=vae,
            label_encoder=label_encoder,
            latent_scaling=latent_scaling,
            cfg_drop_prob=cfg_drop_prob,
            amp_dtype=amp_dtype,
            scaler=scaler,
            ema=ema,
            ema_target=ema_target,
            scheduler=scheduler,
            device=device,
            grad_clip=grad_clip,
            use_autocast=use_autocast,
            grad_accum_steps=grad_accum_steps,
            use_channels_last=use_channels_last,
            on_oom=on_oom,
            global_step=global_step,
            log_every=log_every,
            probe_timesteps=probe_timesteps,
            log_mem=log_mem,
            log_grad_norm=log_grad_norm,
            latent_hw=latent_hw
        )

        sec = time.time() - t0
        total_time += sec
        ips = (n_images / sec) if sec > 0 else 0.0
        lr_now = optimizer.param_groups[0]["lr"]
        warm_prog = 0.0 if not warmup_steps else min(1.0, global_step / float(warmup_steps))

        print(f"{epoch:3d} | {global_step:8d} | {avg_loss:10.5f} | {lr_now:9.2e} | "
              f"{n_batches:8d} | {n_images:8d} | {ips:7.1f} | {_fmt_hms(sec):>8} | {int(100*warm_prog):3d}%")

        # ----------------------------
        # Samples
        # ----------------------------
        if (sample_fn is not None) and ((epoch % sample_every == 0) or (epoch == epochs - 1)):
            out_path = os.path.join(ckpt_dir, f"{run_name}_samples_e{epoch:03d}.png")

            # --- NEW: primeras N épocas del run actual samplea sin EMA ---
            epochs_since_start = epoch - start_epoch
            force_no_ema_now = (sample_no_ema_first_n_epochs > 0) and (epochs_since_start < sample_no_ema_first_n_epochs)

            use_ema_for_sample = (
                (ema is not None)
                and sample_with_ema
                and (not force_no_ema_now)
            )

            rel = 0.0
            backup = None

            if use_ema_for_sample:
                target_module = ema_target if ema_target is not None else model
                ok, _, rel = ema_health(ema, target_module)
                backup = {k: v.detach().cpu().clone() for k, v in target_module.state_dict().items()}
                ema.copy_to(target_module)

            if sample_seed is not None:
                torch.manual_seed(sample_seed)

            _ = sample_fn(
                model=model,
                diffusion=diffusion,
                vae=vae,
                label_encoder=label_encoder,
                n=sample_n,
                latent_hw=latent_hw,
                device=device,
                steps=sample_steps,
                eta=sample_eta,
                guidance_scale=guidance_scale,
                c=sample_labels,
                latent_scaling=latent_scaling,
                save_path=out_path
            )

            if use_ema_for_sample and backup is not None:
                target_module = ema_target if ema_target is not None else model
                target_module.load_state_dict({k: v.to(device) for k, v in backup.items()})

            print(f"└─ [SAMPLE] DDIM grid → {out_path} | EMA_used={use_ema_for_sample} | rel={rel:.3f} | no_ema_firstN={force_no_ema_now}")

        # ----------------------------
        # Save ckpt (GUARDA scheduler)
        # ----------------------------
        if (save_ckpt_fn is not None) and ((epoch % save_every == 0) or (epoch == epochs - 1)):
            ckpt_path = os.path.join(ckpt_dir, f"{run_name}_e{epoch:03d}.pt")
            save_ckpt_fn(
                ckpt_path, model, optimizer, scaler, ema,
                step=global_step,
                extra={"epoch": epoch, "global_step": global_step},
                scheduler=scheduler
            )
            print(f"└─ [CKPT] saved → {ckpt_path}")

            if copy_fixed_to_drive and drive_ckpt_dir:
                _copy_ckpt_to_drive_fixed(ckpt_path, drive_ckpt_dir, fixed_name=fixed_drive_name)

    # save_last
    if save_last and (save_ckpt_fn is not None):
        ckpt_path = os.path.join(ckpt_dir, f"{run_name}_last.pt")
        save_ckpt_fn(
            ckpt_path, model, optimizer, scaler, ema,
            step=global_step,
            extra={"epoch": epochs - 1, "global_step": global_step},
            scheduler=scheduler
        )
        print(f"└─ [CKPT] saved → {ckpt_path}")
        if copy_fixed_to_drive and drive_ckpt_dir:
            _copy_ckpt_to_drive_fixed(ckpt_path, drive_ckpt_dir, fixed_name=fixed_drive_name)

    print(_rule())
    print(f"Entrenamiento finalizado en {_fmt_hms(total_time)}")
    print(_rule())

