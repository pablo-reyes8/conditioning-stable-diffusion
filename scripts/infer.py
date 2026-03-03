#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.factory import build_inference_runtime
from src.utils.config import load_yaml


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference CLI for the Stable Diffusion project.")
    parser.add_argument("--config", default="config/inference/ddim_256.yaml")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    runtime = build_inference_runtime(config, config_path=config_path)

    inference_cfg = config["inference"]
    sampler = str(inference_cfg.get("sampler", "ddim")).lower()
    attributes = inference_cfg.get("attributes")
    out_path = inference_cfg.get("out_path", "outputs/inference.png")

    if sampler == "ddpm":
        from src.inference.ddpm import ddpm_infer_sample

        ddpm_infer_sample(
            model=runtime["model"],
            diffusion=runtime["diffusion"],
            vae=runtime["vae"],
            label_encoder=runtime["label_encoder"],
            attr_dict=attributes,
            n=int(inference_cfg.get("n", 16)),
            latent_hw=int(inference_cfg.get("latent_hw", 32)),
            device=runtime["device"],
            guidance_scale=float(inference_cfg.get("guidance_scale", 7.5)),
            latent_scaling=float(config.get("vae", {}).get("latent_scaling", 0.18215)),
            ema=runtime["ema"],
            out_path=str(out_path),
            save_individual=bool(inference_cfg.get("save_individual", False)),
            out_dir=str(inference_cfg.get("out_dir", "outputs/individual")),
            seed=inference_cfg.get("seed", 1234),
        )
        return 0

    if sampler == "ddim":
        from src.inference.ddim import ddim_latent_infer_sample

        ddim_latent_infer_sample(
            model=runtime["model"],
            diffusion=runtime["diffusion"],
            vae=runtime["vae"],
            label_encoder=runtime["label_encoder"],
            attr_dict=attributes,
            n=int(inference_cfg.get("n", 16)),
            latent_hw=int(inference_cfg.get("latent_hw", 32)),
            device=runtime["device"],
            guidance_scale=float(inference_cfg.get("guidance_scale", 7.5)),
            latent_scaling=float(config.get("vae", {}).get("latent_scaling", 0.18215)),
            ema=runtime["ema"],
            out_path=str(out_path),
            save_individual=bool(inference_cfg.get("save_individual", False)),
            out_dir=str(inference_cfg.get("out_dir", "outputs/individual")),
            seed=inference_cfg.get("seed", 1234),
            steps=int(inference_cfg.get("steps", 50)),
            eta=float(inference_cfg.get("eta", 0.0)),
            schedule_kind=str(inference_cfg.get("schedule_kind", "t_linear")),
        )
        return 0

    raise ValueError(f"Unsupported sampler: {sampler}")


if __name__ == "__main__":
    raise SystemExit(main())
