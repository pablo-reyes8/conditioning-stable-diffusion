import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from src.model.diffusion import * 
from src.model.unet import *

class StableLatentWrapper(nn.Module):
    """
    baseline: 
        x (B,3,256,256) in [-1,1]
        c (B,11) multi-hot
        -> z (B,4,32,32)
        -> eps_pred (B,4,32,32)
        -> x_rec (B,3,256,256)
    """
    def __init__(
        self,
        unet: nn.Module,
        label_encoder: LabelTokenEncoder,
        vae_name: str = "stabilityai/sd-vae-ft-mse",
        latent_scaling: float = 0.18215,
        vae_dtype: torch.dtype = torch.float16,
        device: str = "cuda",):

        super().__init__()
        self.unet = unet
        self.label_encoder = label_encoder
        self.latent_scaling = latent_scaling

        # Cargar VAE preentrenado
        self.vae = AutoencoderKL.from_pretrained(vae_name)
        self.vae.to(device=device, dtype=vae_dtype)

        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_to_latents(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        x: (B,3,H,W) en [-1,1]
        returns z: (B,4,H/8,W/8)
        """
        x_in = x.to(dtype=self.vae.dtype)

        posterior = self.vae.encode(x_in).latent_dist
        z = posterior.sample() if sample else posterior.mean

        return (z * self.latent_scaling).to(dtype=torch.float32)

    @torch.no_grad()
    def decode_from_latents(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B,4,h,w) scaled
        returns x_hat: (B,3,H,W) en [-1,1]
        """
        z = z / self.latent_scaling

        z_in = z.to(dtype=self.vae.dtype)

        x_hat = self.vae.decode(z_in).sample

        return x_hat.clamp(-1, 1).to(dtype=torch.float32)

    def forward_eps(self, z_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Predict noise in latent space.
        c: (B,11) -> tokens: (B,11,D)
        """
        cond_tokens = self.label_encoder(c)  # (B,11,D)
        return self.unet(z_t, t, cond=cond_tokens)

    @torch.no_grad()
    def sanity_reconstruct(self, x: torch.Tensor, sample_latent: bool = False) -> torch.Tensor:
        """
        Autoencoder reconstruction sanity check (no diffusion).
        """
        z = self.encode_to_latents(x, sample=sample_latent)
        x_hat = self.decode_from_latents(z)
        return x_hat