from typing import Literal, Tuple, Optional
import torch
import math
import torch.nn as nn
ScheduleKind = Literal["linear", "cosine"]

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    Toma un vector 1D (longitud T) y lo indexa por t (B,), devolviendo (B, 1, 1, 1)
    broadcastable al tamaño de x.
    """
    t = t.long().clamp(0, a.shape[0] - 1)
    out = a.gather(0, t)
    return out.view((t.shape[0],) + (1,) * (len(x_shape) - 1))

def beta_schedule_linear(T: int, beta_min: float = 1e-4, beta_max: float = 2e-2) -> torch.Tensor:
    """
    Schedule lineal clásico: beta_t en [beta_min, beta_max].
    """
    return torch.linspace(beta_min, beta_max, T, dtype=torch.float32)

def _alpha_bar_cosine(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """
    Alphas-bar continuas según Nichol & Dhariwal (cosine schedule):
        alpha_bar(t) = cos^2( ((t/T + s)/(1+s)) * (pi/2) )
    Asume t ∈ [0, 1]. Retorna alpha_bar(t) ∈ (0,1].
    """
    x = (t + s) / (1.0 + s)
    return torch.cos((math.pi / 2.0) * x).clamp(min=1e-7) ** 2


def beta_schedule_cosine(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Discretiza alpha_bar(t) por diferencias para obtener beta_t.
    """
    steps = torch.arange(T + 1, dtype=torch.float32) / T  # 0..1
    alphas_bar = _alpha_bar_cosine(steps, s=s)
    alphas_bar = alphas_bar / alphas_bar[0]  # normaliza para que alpha_bar(0)=1
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return betas.clamp(min=1e-8, max=0.999)
