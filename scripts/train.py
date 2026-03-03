#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.factory import build_training_runtime
from src.training.train_model import train_ldm
from src.utils.config import load_yaml


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training CLI for the Stable Diffusion project.")
    parser.add_argument("--config", default="config/training/maad_256.yaml")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    runtime = build_training_runtime(config, config_path=config_path)

    train_cfg = config["training"]
    checkpoint_cfg = config.get("checkpoint", {})
    sample_cfg = config.get("sampling", {})
    sample_kind = str(sample_cfg.get("kind", "ddim")).lower()
    if sample_kind != "ddim":
        raise ValueError("Training sampling currently supports only DDIM grids.")
    from src.training.ddim_for_training import sample_latent_ddim_cfg

    train_ldm(
        model=runtime["model"],
        diffusion=runtime["diffusion"],
        train_loader=runtime["train_loader"],
        optimizer=runtime["optimizer"],
        ema=runtime["ema"],
        vae=runtime["vae"],
        label_encoder=runtime["label_encoder"],
        device=runtime["device"],
        epochs=int(train_cfg.get("epochs", 50)),
        base_lr=float(train_cfg.get("base_lr", 2e-4)),
        min_lr=float(train_cfg.get("min_lr", 2e-5)),
        warmup_steps=int(train_cfg.get("warmup_steps", 1000)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        use_autocast=bool(train_cfg.get("use_autocast", True)),
        amp_dtype=str(train_cfg.get("amp_dtype", "bf16")),
        scaler=runtime["scaler"],
        cfg_drop_prob=float(train_cfg.get("cfg_drop_prob", 0.12)),
        latent_scaling=float(config.get("vae", {}).get("latent_scaling", 0.18215)),
        latent_hw=int(train_cfg.get("latent_hw", 32)),
        sample_every=int(sample_cfg.get("every", 1)),
        sample_n=int(sample_cfg.get("sample_n", 16)),
        sample_steps=int(sample_cfg.get("steps", 50)),
        sample_eta=float(sample_cfg.get("eta", 0.0)),
        guidance_scale=float(sample_cfg.get("guidance_scale", 7.5)),
        sample_seed=sample_cfg.get("seed", 1234),
        sample_labels=runtime["sample_labels"],
        sample_fn=sample_latent_ddim_cfg,
        sample_with_ema=bool(sample_cfg.get("sample_with_ema", True)),
        sample_no_ema_first_n_epochs=int(sample_cfg.get("sample_no_ema_first_n_epochs", 0)),
        ckpt_dir=runtime["checkpoint_dir"],
        run_name=str(checkpoint_cfg.get("run_name", "ldm")),
        save_every=int(checkpoint_cfg.get("save_every", 1)),
        save_last=bool(checkpoint_cfg.get("save_last", True)),
        resume_path=checkpoint_cfg.get("resume_path"),
        ckpt_utils=runtime["ckpt_utils"],
        grad_accum_steps=int(train_cfg.get("grad_accum_steps", 1)),
        use_channels_last=bool(train_cfg.get("use_channels_last", False)),
        on_oom=str(train_cfg.get("on_oom", "skip")),
        log_every=int(train_cfg.get("log_every", 0)),
        probe_timesteps=train_cfg.get("probe_timesteps"),
        log_mem=bool(train_cfg.get("log_mem", False)),
        log_grad_norm=bool(train_cfg.get("log_grad_norm", False)),
        reset_optimizer_state=bool(checkpoint_cfg.get("reset_optimizer_state", False)),
        override_lr=checkpoint_cfg.get("override_lr"),
        override_weight_decay=checkpoint_cfg.get("override_weight_decay"),
        override_ema_decay=checkpoint_cfg.get("override_ema_decay"),
        reset_ema_on_resume=bool(checkpoint_cfg.get("reset_ema_on_resume", False)),
        reset_ema_at_epoch=checkpoint_cfg.get("reset_ema_at_epoch"),
        ema_decay_after_reset=float(checkpoint_cfg.get("ema_decay_after_reset", 0.9995)),
        repair_ema_on_resume=bool(checkpoint_cfg.get("repair_ema_on_resume", False)),
        ema_decay_after_repair=float(checkpoint_cfg.get("ema_decay_after_repair", 0.9995)),
        drive_ckpt_dir=checkpoint_cfg.get("drive_ckpt_dir"),
        copy_fixed_to_drive=bool(checkpoint_cfg.get("copy_fixed_to_drive", False)),
        fixed_drive_name=str(checkpoint_cfg.get("fixed_drive_name", "latest_ldm.pt")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
