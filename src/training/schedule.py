import math 
import torch 
import torch.nn as nn

def build_diffusion_param_groups(model: nn.Module, label_encoder: nn.Module, weight_decay: float = 1e-4):
    decay, no_decay = [], []

    # Unimos los iteradores de ambos modelos
    import itertools
    all_named_params = itertools.chain(model.named_parameters(), label_encoder.named_parameters())

    for name, p in all_named_params:
        if not p.requires_grad:
            continue
        # No decay para biases ni normalizaciones (GroupNorm, LayerNorm)
        if name.endswith(".bias") or ("norm" in name.lower()) or ("bn" in name.lower()) or ("ln" in name.lower()):
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0}]

class WarmupCosineLR:
    """Warmup linear for warmup_steps, then cosine to min_lr. Step-based."""
    def __init__(self, optimizer, total_steps: int, warmup_steps: int, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_num = 0

    def _set_lr(self, t: int):
        for i, group in enumerate(self.optimizer.param_groups):
            base = self.base_lrs[i] if i < len(self.base_lrs) else group.get("lr", self.base_lrs[-1])

            if self.warmup_steps > 0 and t <= self.warmup_steps:
                lr = base * (t / self.warmup_steps)
            else:
                tt = min(t, self.total_steps)
                denom = max(1, self.total_steps - self.warmup_steps)
                progress = (tt - self.warmup_steps) / denom
                progress = min(1.0, max(0.0, progress))  # clamp
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = self.min_lr + (base - self.min_lr) * cosine

            group["lr"] = lr

    def step(self):
        self.step_num += 1
        self._set_lr(self.step_num)

    def state_dict(self):
        return {
            "step_num": int(self.step_num),
            "base_lrs": list(self.base_lrs),
            "min_lr": float(self.min_lr),
            "total_steps": int(self.total_steps),
            "warmup_steps": int(self.warmup_steps),
        }

    def load_state_dict(self, d):
        if not isinstance(d, dict):
            return

        self.step_num = int(d.get("step_num", 0))

        blrs = d.get("base_lrs", None)
        if isinstance(blrs, (list, tuple)) and len(blrs) > 0:
            self.base_lrs = list(blrs)

        self.min_lr = float(d.get("min_lr", self.min_lr))
        self.total_steps = int(d.get("total_steps", self.total_steps))
        self.warmup_steps = int(d.get("warmup_steps", self.warmup_steps))

        self._set_lr(self.step_num)