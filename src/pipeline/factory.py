from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.constants import DEFAULT_ATTRIBUTES
from src.model.diffusion import Diffusion
from src.model.label_encoder import LabelTokenEncoder
from src.model.unet import UNetDenoiser
from src.training.autocast import make_grad_scaler
from src.training.checkpoints import load_ckpt, save_ckpt
from src.training.ema import EMA
from src.training.schedule import build_diffusion_param_groups
from src.utils.config import attrs_to_tensor, default_device, load_component_config, parse_dtype, resolve_path


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_model_section(config: dict[str, Any], *, base_dir: Path | None = None) -> dict[str, Any]:
    if "model" in config and isinstance(config["model"], dict):
        return config["model"]
    if "model_config" in config:
        return load_component_config(config["model_config"], base_dir=base_dir)
    raise ValueError("Config must define either 'model' or 'model_config'.")


def build_unet_from_config(model_config: dict[str, Any]) -> UNetDenoiser:
    return UNetDenoiser(
        in_channels=int(model_config.get("in_channels", 4)),
        base_channels=int(model_config.get("base_channels", 192)),
        channel_mults=tuple(model_config.get("channel_mults", [1, 2, 3, 4])),
        num_res_blocks=int(model_config.get("num_res_blocks", 2)),
        attn_resolutions=set(model_config.get("attn_resolutions", [8, 4])),
        time_embed_dim=int(model_config.get("time_embed_dim", 512)),
        dropout=float(model_config.get("dropout", 0.1)),
        num_heads=int(model_config.get("num_heads", 4)),
        head_dim=int(model_config.get("head_dim", 64)),
        img_resolution=int(model_config.get("img_resolution", 32)),
        use_cross_attn=bool(model_config.get("use_cross_attn", True)),
        context_dim=int(model_config.get("context_dim", 256)),
        attn_drop=float(model_config.get("attn_drop", 0.0)),
    )


def build_diffusion_from_config(diffusion_config: dict[str, Any]) -> Diffusion:
    return Diffusion(
        T=int(diffusion_config.get("T", 1000)),
        schedule=str(diffusion_config.get("schedule", "linear")),
        beta_min=float(diffusion_config.get("beta_min", 1e-4)),
        beta_max=float(diffusion_config.get("beta_max", 2e-2)),
        cosine_s=float(diffusion_config.get("cosine_s", 0.008)),
        clamp_x0=bool(diffusion_config.get("clamp_x0", True)),
        dynamic_threshold=diffusion_config.get("dynamic_threshold"),
    )


def _build_vae(vae_name: str, *, device: str, dtype_name: str):
    try:
        from diffusers.models import AutoencoderKL
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "diffusers is required to build the VAE. Install the training/inference requirements first."
        ) from exc

    vae = AutoencoderKL.from_pretrained(vae_name)
    vae.to(device=device, dtype=parse_dtype(dtype_name))
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae


def _attribute_names(data_config: dict[str, Any]) -> list[str]:
    return list(data_config.get("attributes", DEFAULT_ATTRIBUTES))


def build_training_runtime(config: dict[str, Any], *, config_path: str | Path | None = None) -> dict[str, Any]:
    config_base_dir = Path(config_path).resolve().parent if config_path else None
    data_config = config["data"]
    model_config = _load_model_section(config, base_dir=config_base_dir)
    diffusion_config = config["diffusion"]
    train_config = config["training"]
    optimizer_config = config.get("optimizer", {})
    checkpoint_config = config.get("checkpoint", {})
    sample_config = config.get("sampling", {})

    device = default_device(config.get("device"))
    seed_everything(config.get("seed"))

    attribute_names = _attribute_names(data_config)
    from src.model.data_loaders import ZipFaceDataset

    dataset = ZipFaceDataset(
        zip_path=str(resolve_path(data_config["archive_path"])),
        labels_path=str(resolve_path(data_config["manifest_path"])),
        attrs=attribute_names,
        img_size=int(data_config.get("image_size", 256)),
        zip_prefix=str(data_config.get("zip_prefix", "train/")),
    )

    train_loader = DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 4)),
        shuffle=bool(data_config.get("shuffle", True)),
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=bool(data_config.get("pin_memory", device == "cuda")),
        drop_last=bool(data_config.get("drop_last", True)),
    )

    model = build_unet_from_config(model_config).to(device)
    label_encoder = LabelTokenEncoder(
        num_labels=len(attribute_names),
        context_dim=int(model_config.get("context_dim", 256)),
    ).to(device)
    diffusion = build_diffusion_from_config(diffusion_config).to(device)

    vae_config = config.get("vae", {})
    vae = _build_vae(
        vae_name=str(vae_config.get("name", "stabilityai/sd-vae-ft-mse")),
        device=device,
        dtype_name=str(vae_config.get("dtype", "float16")),
    )

    param_groups = build_diffusion_param_groups(
        model=model,
        label_encoder=label_encoder,
        weight_decay=float(optimizer_config.get("weight_decay", 1e-4)),
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=float(train_config.get("base_lr", 2e-4)),
        betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
        eps=float(optimizer_config.get("eps", 1e-8)),
    )

    use_autocast = bool(train_config.get("use_autocast", True))
    amp_dtype = str(train_config.get("amp_dtype", "bf16"))
    scaler = make_grad_scaler(device=device, enabled=use_autocast, amp_dtype=amp_dtype)
    ema = EMA(model, decay=float(train_config.get("ema_decay", 0.9995)))

    sample_labels = attrs_to_tensor(
        sample_config.get("attributes"),
        attribute_names,
        batch_size=int(sample_config.get("sample_n", 16)),
        device=device,
    )

    return {
        "device": device,
        "attribute_names": attribute_names,
        "train_loader": train_loader,
        "model": model,
        "label_encoder": label_encoder,
        "diffusion": diffusion,
        "vae": vae,
        "optimizer": optimizer,
        "scaler": scaler,
        "ema": ema,
        "sample_labels": sample_labels,
        "ckpt_utils": (save_ckpt, load_ckpt),
        "checkpoint_dir": str(resolve_path(checkpoint_config.get("dir", "checkpoints"))),
    }


def build_inference_runtime(config: dict[str, Any], *, config_path: str | Path | None = None) -> dict[str, Any]:
    config_base_dir = Path(config_path).resolve().parent if config_path else None
    data_config = config["data"]
    model_config = _load_model_section(config, base_dir=config_base_dir)
    diffusion_config = config["diffusion"]
    inference_config = config["inference"]
    vae_config = config.get("vae", {})

    device = default_device(config.get("device"))
    seed_everything(config.get("seed"))

    attribute_names = _attribute_names(data_config)
    model = build_unet_from_config(model_config).to(device)
    label_encoder = LabelTokenEncoder(
        num_labels=len(attribute_names),
        context_dim=int(model_config.get("context_dim", 256)),
    ).to(device)
    diffusion = build_diffusion_from_config(diffusion_config).to(device)
    vae = _build_vae(
        vae_name=str(vae_config.get("name", "stabilityai/sd-vae-ft-mse")),
        device=device,
        dtype_name=str(vae_config.get("dtype", "float16")),
    )

    ema = None
    if inference_config.get("use_ema", True):
        ema = EMA(model, decay=float(inference_config.get("ema_decay", 0.9995)))

    checkpoint_path = resolve_path(inference_config["checkpoint_path"])
    load_ckpt(
        checkpoint_path,
        model=model,
        optimizer=None,
        scaler=None,
        ema=ema,
        scheduler=None,
        map_location=device,
        strict=bool(inference_config.get("strict_checkpoint", True)),
    )

    return {
        "device": device,
        "attribute_names": attribute_names,
        "model": model,
        "label_encoder": label_encoder,
        "diffusion": diffusion,
        "vae": vae,
        "ema": ema,
    }
