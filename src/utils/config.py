from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_value: str | Path, *, base_dir: str | Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    base = Path(base_dir) if base_dir is not None else PROJECT_ROOT
    return (base / path).resolve()


def load_yaml(path_value: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path_value)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_component_config(config_path: str | Path, *, base_dir: str | Path | None = None) -> dict[str, Any]:
    resolved = resolve_path(config_path, base_dir=base_dir)
    config = load_yaml(resolved)
    config["_config_path"] = str(resolved)
    return config


def parse_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[normalized]


def default_device(device_name: str | None = None) -> str:
    if device_name:
        return device_name
    return "cuda" if torch.cuda.is_available() else "cpu"


def attrs_to_tensor(
    attrs: dict[str, int | float] | None,
    attribute_names: list[str],
    *,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    tensor = torch.zeros(batch_size, len(attribute_names), device=device, dtype=torch.float32)
    if not attrs:
        return tensor

    for index, name in enumerate(attribute_names):
        value = attrs.get(name)
        if value is not None:
            tensor[:, index] = float(value)
    return tensor
