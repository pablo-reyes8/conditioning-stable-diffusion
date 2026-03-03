import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
from src.training.autocast import *

def compute_grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return total ** 0.5

def gpu_mem_mb(device="cuda"):
    if torch.cuda.is_available() and device == "cuda":
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserv = torch.cuda.memory_reserved() / (1024**2)
        return alloc, reserv
    return 0.0, 0.0

def train_one_epoch(
    model,                 # UNet: forward(z_t, t, cond_tokens) -> eps_pred
    diffusion,
    dataloader,
    optimizer,
    *,
    vae,                   # AutoencoderKL frozen (eval)
    label_encoder,         # LabelTokenEncoder trainable
    latent_scaling: float = 0.18215,
    cfg_drop_prob: float = 0.12,     # 10-15%
    amp_dtype: str = "bf16",         # "bf16" recomendado en A100
    scaler=None,                     # None para bf16; GradScaler para fp16
    ema=None,                        # EMA object (opcional)
    ema_target=None,                 # módulo al que aplicas EMA.update (por defecto model)
    device: str = "cuda",
    max_batches: int | None = None,
    grad_clip: float | None = 1.0,
    use_autocast: bool = True,
    grad_accum_steps: int = 1,
    use_channels_last: bool = False,
    on_oom: str = "skip",
    # warmup por paso
    base_lr: float | None = None,
    warmup_steps: int | None = None,
    global_step: int = 0,
    # DIAGNÓSTICOS
    log_every: int = 0,
    probe_timesteps: list[int] | None = None,
    log_mem: bool = False,
    log_grad_norm: bool = False,
    # latente size (para baseline/diagnósticos)
    latent_hw: int = 32,             # 32 para 256px, 64 para 512px

    scheduler=None
):
    """
    Entrena 1 epoch en espacio latente (Stable Diffusion style):
      x -> z0 (VAE frozen) -> zt (diffusion) -> eps_pred = UNet(zt,t,cond_tokens) -> MSE(eps_pred, eps)

    CFG-dropout:
      con prob cfg_drop_prob, c_used = 0-vector => tokens nulos => aprende rama unconditional.
    """

    current_label_encoder = getattr(model, "label_encoder", label_encoder)
    model.train()
    current_label_encoder.train()
    vae.eval()  # congelado

    if ema_target is None:
        ema_target = model

    if use_channels_last:
        # channels_last ayuda en convs; en latentes también puede ayudar.
        model.to(memory_format=torch.channels_last)

    grad_accum_steps = max(1, int(grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)

    total_loss, n_seen_batches, n_seen_images = 0.0, 0, 0

    # Header diagnóstico
    did_print_in_epoch_header = False
    if log_every and global_step == 0:
        with torch.no_grad():
            zb = torch.randn(32, 4, latent_hw, latent_hw, device=device)
            base = float((zb**2).mean().item())  # ~1
        print("┆ In-epoch statistics (Latent Diffusion)")
        print(f"┆   (baseline)  ε-MSE ≈ {base:.3f}  (esperado ~1.0)")
        print("┆   {:>8} | {:>9} | {:>8} | {:>8} | {:>10}{}".format(
            "step", "lr", "loss", "dt(ms)", "grad_norm",
            (" | probes[t]" if probe_timesteps else "")
        ))
        print("┆   " + "─"*72)
        did_print_in_epoch_header = True

    # ---- Training loop ----
    for i, (x, c) in enumerate(dataloader):
        if (max_batches is not None) and (i >= max_batches):
            break

        try:
            t_start = time.perf_counter()

            # x: (B,3,256,256) en [-1,1]
            x = x.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)

            if use_channels_last:
                x = x.to(memory_format=torch.channels_last)

            B = x.size(0)
            t = diffusion.sample_timesteps(B, device=device)

            # ---- VAE encode (no grad) -> z0 ----
            with torch.no_grad():
                x_in = x.to(dtype=vae.dtype)
                posterior = vae.encode(x_in).latent_dist
                z0 = (posterior.sample() * latent_scaling).to(torch.float32)

            # ---- forward diffusion in latent space ----
            eps = torch.randn_like(z0)
            zt = diffusion.q_sample(z0, t, eps=eps)

            # ---- CFG dropout on labels ----
            if cfg_drop_prob and cfg_drop_prob > 0.0:
                drop_mask = (torch.rand(B, device=device) < cfg_drop_prob)
                c_used = c.clone()
                c_used[drop_mask] = 0
            else:
                c_used = c

            # ---- label -> tokens for cross-attn ----
            cond_tokens = current_label_encoder(c_used)  # (B,11,D)

            # ---- forward + loss ----
            with autocast_ctx(device=device, enabled=bool(use_autocast), dtype=amp_dtype):
                eps_pred = model(zt, t, cond=cond_tokens)

                raw_loss = F.mse_loss(eps_pred, eps, reduction='none')
                raw_loss = raw_loss.mean(dim=list(range(1, raw_loss.ndim)))

                # Min-SNR Weighting clásico y riguroso
                snr = diffusion.get_snr(t).squeeze().clamp_min(1e-8)
                gamma = 5.0  # El estándar empírico demostrado por Hang et al. (2023)

                # w será 1.0 para t grandes (mucho ruido, snr < 5)
                # w decaerá suavemente para t pequeños (poco ruido, snr > 5)
                w = torch.clamp(snr, max=gamma) / snr

                loss = (raw_loss * w).mean() / grad_accum_steps

            # ---- backward ----
            if use_autocast and (scaler is not None):
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_now = ((i + 1) % grad_accum_steps) == 0
            gnorm = None

            if step_now:
              # actualizar LR UNA VEZ (para este update)
              if scheduler is not None:
                  scheduler.step()

              did_unscale = False

              if log_grad_norm:
                  if use_autocast and (scaler is not None) and (not did_unscale):
                      scaler.unscale_(optimizer)
                      did_unscale = True
                  gnorm = compute_grad_norm(model)

              if grad_clip is not None:
                  if use_autocast and (scaler is not None) and (not did_unscale):
                      scaler.unscale_(optimizer)
                      did_unscale = True

                  params_to_clip = (
                      model.parameters()
                      if hasattr(model, "label_encoder")
                      else list(model.parameters()) + list(label_encoder.parameters()))

                  torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip)

              # optimizer step (usa el LR que acabas de setear)
              if use_autocast and (scaler is not None):
                  scaler.step(optimizer)
                  scaler.update()
              else:
                  optimizer.step()

              optimizer.zero_grad(set_to_none=True)

              if ema is not None:
                  ema.update(ema_target)

              global_step += 1

            total_loss += float(loss.detach()) * grad_accum_steps
            n_seen_batches += 1
            n_seen_images += B

            # ---- Logs ----
            if log_every and step_now and (global_step % log_every == 0):
                if not did_print_in_epoch_header:
                    print("┆ In-epoch statistics (Latent Diffusion)")
                    print("┆   {:>8} | {:>9} | {:>8} | {:>8} | {:>10}{}".format(
                        "step", "lr", "loss", "dt(ms)", "grad_norm",
                        (" | probes[t]" if probe_timesteps else "")
                    ))
                    print("┆   " + "─"*72)
                    did_print_in_epoch_header = True

                probe_msg = ""
                if probe_timesteps:
                    vals = []
                    with torch.no_grad():
                        # Para probes, usamos condición real (sin dropout)
                        cond_probe = label_encoder(c)

                        for tau in probe_timesteps:
                            t_fix = torch.full((B,), int(tau), device=device, dtype=torch.long)
                            epsp = torch.randn_like(z0)
                            ztp = diffusion.q_sample(z0, t_fix, eps=epsp)
                            with autocast_ctx(device=device, enabled=True, dtype=amp_dtype):
                                eps_pred_p = model(ztp, t_fix, cond=cond_probe)
                                v = F.mse_loss(eps_pred_p, epsp).item()
                            vals.append(f"t={tau}:{v:.3f}")
                    probe_msg = " | " + " ".join(vals)

                mem_msg = ""
                if log_mem:
                    alloc, reserv = gpu_mem_mb(device)
                    mem_msg = f" | mem={alloc:.0f}/{reserv:.0f}MB"

                lr_now = optimizer.param_groups[0]["lr"]
                dt = (time.perf_counter() - t_start) * 1000.0
                gn_str = (f"{gnorm:.2e}" if (gnorm is not None) else "—")
                loss_val = (loss.detach() * grad_accum_steps).item()

                print("┆   {:8d} | {:9.2e} | {:8.4f} | {:8.1f} | {:>10}{}{}".format(
                    global_step, lr_now, loss_val, dt, gn_str, mem_msg, probe_msg))

        except RuntimeError as e:
            if ("CUDA out of memory" in str(e)) and (on_oom == "skip"):
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                print(f"[WARN][OOM] Batch {i} omitido. Limpié cache y sigo…")
                optimizer.zero_grad(set_to_none=True)
                continue
            else:
                raise

    avg_loss = total_loss / max(1, n_seen_batches)
    return avg_loss, n_seen_batches, n_seen_images, global_step
