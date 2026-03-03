from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_image_paths(
    root: str | Path,
    *,
    recursive: bool = True,
    max_images: int | None = None,
) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Image root does not exist: {root_path}")

    if root_path.is_file():
        paths = [root_path] if root_path.suffix.lower() in IMAGE_EXTENSIONS else []
    else:
        iterator = root_path.rglob("*") if recursive else root_path.glob("*")
        paths = sorted(
            path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    if max_images is not None:
        return paths[: max(0, int(max_images))]
    return paths


def load_pil_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_image_tensor(
    path: str | Path,
    *,
    image_size: int,
    device: str = "cpu",
) -> torch.Tensor:
    image = load_pil_image(path)
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.to(device)


def iter_image_batches(
    image_paths: Iterable[str | Path],
    *,
    batch_size: int,
    image_size: int,
    device: str = "cpu",
):
    batch: list[torch.Tensor] = []
    for image_path in image_paths:
        batch.append(load_image_tensor(image_path, image_size=image_size, device=device))
        if len(batch) == batch_size:
            yield torch.stack(batch, dim=0)
            batch.clear()

    if batch:
        yield torch.stack(batch, dim=0)
