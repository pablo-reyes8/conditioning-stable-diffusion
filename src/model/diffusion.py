from typing import Literal, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.diffusion_utils import *

ScheduleKind = Literal["linear", "cosine"]

class Diffusion(nn.Module):
    """
    DDPM/DDIM utilities (works for pixel-space or latent-space):
      - Precomputes schedules and posterior coefficients
      - q_sample, predict_x0
      - loss (eps prediction)
      - sampling steps (DDPM and DDIM)

    For conditional diffusion (StableDiffusion-like), model_eps_pred_fn should accept:
        eps_pred = model_eps_pred_fn(x_t, t, cond=cond)
    where cond could be:
        - context tokens [B, T, D] for cross-attention, or
        - label vector [B, 11] if your UNet embeds internally
    """
    def __init__(
        self,
        T: int = 1000,
        schedule: ScheduleKind = "linear",
        beta_min: float = 1e-4,
        beta_max: float = 2e-2,
        cosine_s: float = 0.008,
        clamp_x0: bool = True,
        dynamic_threshold: Optional[float] = None,
        img_size=None,
    ):
        super().__init__()
        self.T = int(T)
        self.clamp_x0 = clamp_x0
        self.dynamic_threshold = dynamic_threshold
        self.img_size = img_size

        if schedule == "linear":
            betas = beta_schedule_linear(T, beta_min, beta_max)
        elif schedule == "cosine":
            betas = beta_schedule_cosine(T, s=cosine_s)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod), persistent=False)
        self.register_buffer("alphas_cumprod_prev", F.pad(alphas_cumprod[:-1], (1, 0), value=1.0), persistent=False)

        posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20), persistent=False)
        self.register_buffer("posterior_log_variance", torch.log(posterior_variance.clamp(min=1e-20)), persistent=False)

        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod),
            persistent=False
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod),
            persistent=False
        )

    def sample_timesteps(self, batch_size: int, device=None) -> torch.Tensor:
        if device is None:
            device = self.betas.device
        return torch.randint(1, self.T, (batch_size,), device=device, dtype=torch.long)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        if eps is None:
            eps = torch.randn_like(x0)
        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omb = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ab * x0 + sqrt_omb * eps

    @torch.no_grad()
    def predict_x0(self, x_t: torch.Tensor, eps_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_omb = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        x0_hat = (x_t - sqrt_omb * eps_pred) / (sqrt_ab + 1e-12)

        if self.dynamic_threshold is not None:
            s = self.dynamic_threshold
            amax = x0_hat.detach().abs().flatten(1).max(dim=1).values
            amax = torch.maximum(amax, torch.tensor(1.0, device=x0_hat.device, dtype=x0_hat.dtype))
            x0_hat = (x0_hat.transpose(0, 1) / amax.clamp(min=s).unsqueeze(-1).unsqueeze(-1)).transpose(0, 1)
            x0_hat = x0_hat.clamp(-1, 1)
        elif self.clamp_x0:
            x0_hat = x0_hat.clamp(-1, 1)

        return x0_hat

    def posterior_mean_variance(self, x_t: torch.Tensor, x0_hat: torch.Tensor, t: torch.Tensor):
        coef1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x0_hat + coef2 * x_t
        var = extract(self.posterior_variance, t, x_t.shape)
        logvar = extract(self.posterior_log_variance, t, x_t.shape)
        return mean, var, logvar


    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Calcula la relación señal-ruido (SNR) para un timestep t.
        SNR(t) = alpha_cumprod(t) / (1 - alpha_cumprod(t))
        """
        a_t = extract(self.alphas_cumprod, t, (t.shape[0],))
        return a_t / (1.0 - a_t)

    # -------------------------
    # Training loss (conditional-ready)
    # -------------------------
    def loss_simple(
        self,
        model_eps_pred_fn: Callable,    # (x_t, t, cond) -> eps_pred
        x0: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        cond_drop_prob: float = 0.0,
        noise: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        null_cond: Optional[torch.Tensor] = None,
        use_min_snr: bool = True,       # Activado por defecto para mejor convergencia
        min_snr_gamma: float = 5.0      # Parámetro estándar de la literatura
    ) -> torch.Tensor:

        if noise is None:
            noise = torch.randn_like(x0)

        # CFG dropout
        if cond is not None and cond_drop_prob > 0.0:
            B = x0.shape[0]
            drop_mask = (torch.rand(B, device=x0.device) < cond_drop_prob)

            if null_cond is None:
                null_cond = torch.zeros_like(cond)

            view_shape = (B,) + (1,) * (cond.ndim - 1)
            drop_mask = drop_mask.view(view_shape)
            cond_used = torch.where(drop_mask, null_cond, cond)
        else:
            cond_used = cond

        x_t = self.q_sample(x0, t, eps=noise)
        eps_pred = model_eps_pred_fn(x_t, t, cond_used)

        mse = (noise - eps_pred).pow(2).mean(dim=tuple(range(1, eps_pred.ndim)))

        # Ponderación Min-SNR
        if use_min_snr:
            snr = self.get_snr(t).squeeze()
            # Para epsilon-prediction, el peso escalar es min(gamma, SNR) / SNR
            snr_weight = torch.clamp(snr, max=min_snr_gamma)
            min_snr_weight = snr_weight / snr

            # Combinar con cualquier otro peso externo si lo hubiera
            if weight is None:
                weight = min_snr_weight
            else:
                weight = weight * min_snr_weight

        if weight is not None:
            # Asegurar broadcast correcto si weight es 1D (B,)
            if weight.dim() < mse.dim():
                weight = weight.view(-1, *([1] * (mse.dim() - 1)))
            mse = mse * weight

        return mse.mean()

    # -------------------------
    # Sampling steps (conditional-ready)
    # -------------------------
    @torch.no_grad()
    def p_sample_step(
        self,
        model_eps_pred_fn,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        clip_x0: Optional[bool] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if clip_x0 is None:
            clip_x0 = self.clamp_x0

        eps_pred = model_eps_pred_fn(x_t, t, cond)
        x0_hat = self.predict_x0(x_t, eps_pred, t)
        if clip_x0:
            x0_hat = x0_hat.clamp(-1, 1)

        mean, var, logvar = self.posterior_mean_variance(x_t, x0_hat, t)

        nonzero_mask = (t > 0).float().reshape((x_t.shape[0],) + (1,) * (x_t.ndim - 1))
        if noise is None:
            noise = torch.randn_like(x_t)

        return mean + nonzero_mask * torch.exp(0.5 * logvar) * noise

    @torch.no_grad()
    def p_sample_step_ddim(
        self,
        model_eps_pred_fn,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        eta: float = 0.0,
        clip_x0: Optional[bool] = None,
        noise: Optional[torch.Tensor] = None):
        if clip_x0 is None:
            clip_x0 = self.clamp_x0
        if noise is None:
            noise = torch.randn_like(x_t)

        a_t      = extract(self.alphas_cumprod, t,      x_t.shape)
        a_t_prev = extract(self.alphas_cumprod, t_prev, x_t.shape)

        eps_pred = model_eps_pred_fn(x_t, t, cond)
        x0_hat   = self.predict_x0(x_t, eps_pred, t)

        if clip_x0:
            x0_hat = x0_hat.clamp(-1, 1)
            # Si forzamos x0 al hipercubo, recalculamos la dirección del ruido
            # para mantener la coherencia geométrica de la trayectoria
            dir_xt = (x_t - torch.sqrt(a_t) * x0_hat) / torch.sqrt(1.0 - a_t + 1e-12)
        else:
            # Identidad algebraica pura: sin clipping, dir_xt ES eps_pred.
            # Ahorramos cómputo y evitamos inestabilidad numérica.
            dir_xt = eps_pred

        sigma  = eta * torch.sqrt((1.0 - a_t_prev) / (1.0 - a_t + 1e-12)) \
                      * torch.sqrt(1.0 - a_t / (a_t_prev + 1e-12))

        mean   = torch.sqrt(a_t_prev) * x0_hat
        add    = torch.sqrt(torch.clamp(1.0 - a_t_prev - sigma**2, min=0.0)) * dir_xt
        x_prev = mean + add + sigma * noise

        return x_prev

    # -------------------------
    # Optional: CFG-guided eps for sampling
    # -------------------------
    @torch.no_grad()
    def eps_cfg(
        self,
        model_eps_pred_fn,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        guidance_scale: float = 7.5):

        eps_u = model_eps_pred_fn(x_t, t, uncond)
        eps_c = model_eps_pred_fn(x_t, t, cond)
        return eps_u + guidance_scale * (eps_c - eps_u)