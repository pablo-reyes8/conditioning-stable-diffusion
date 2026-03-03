import torch

from src.training.schedule import WarmupCosineLR


def test_warmup_cosine_lr_changes_learning_rate():
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineLR(optimizer, total_steps=10, warmup_steps=2, min_lr=1e-5)

    scheduler.step()
    lr_after_first_step = optimizer.param_groups[0]["lr"]
    scheduler.step()
    scheduler.step()
    lr_after_third_step = optimizer.param_groups[0]["lr"]

    assert lr_after_first_step > 0.0
    assert lr_after_third_step <= 1e-3
