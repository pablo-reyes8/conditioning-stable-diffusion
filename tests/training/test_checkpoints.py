from pathlib import Path

import torch

from src.training.checkpoints import load_ckpt, save_ckpt
from src.training.ema import EMA


def test_checkpoint_roundtrip(tmp_path: Path):
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = EMA(model, decay=0.9)
    checkpoint_path = tmp_path / "model.pt"

    save_ckpt(checkpoint_path, model, optimizer, scaler=None, ema=ema, step=12, extra={"epoch": 2})

    restored_model = torch.nn.Linear(4, 4)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_ema = EMA(restored_model, decay=0.9)

    step, extra = load_ckpt(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scaler=None,
        ema=restored_ema,
        map_location="cpu",
    )

    assert step == 12
    assert extra["epoch"] == 2
    for left, right in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.allclose(left, right)
