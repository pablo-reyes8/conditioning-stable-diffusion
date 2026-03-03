import os
import torch
import math
import torchvision.utils as vutils

from src.data.constants import DEFAULT_ATTRIBUTES

ATTRS = DEFAULT_ATTRIBUTES

@torch.no_grad()
def ddpm_infer_sample(
    model,
    diffusion,
    vae,
    label_encoder,
    attr_dict: dict | None = None,       # Ej: {'Male': 1, 'Smiling': 1}
    n: int = 16,                         # Número de imágenes a generar
    latent_hw: int = 64,                 # 32 para 256px
    device: str = "cuda",
    guidance_scale: float = 7.5,
    latent_scaling: float = 0.18215,
    *,
    ema=None,
    out_path: str = "samples_ddpm.png",
    save_individual: bool = False,
    out_dir: str = "samples_individual",
    seed: int | None = 23423):

    """
    Genera n imágenes con DDPM ancestral (T=1000 pasos) en espacio LATENTE
    usando Classifier-Free Guidance (CFG) vectorizado y un diccionario de atributos.
    """
    model_was_training = model.training
    model.eval()
    label_encoder.eval()
    vae.eval()

    backup_state = None
    if ema is not None:
        backup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)

    if seed is not None:
        torch.manual_seed(seed)

    # Parseo del Diccionario de Atributos -> Tensor (n, 11)
    c_tensor = torch.zeros(n, len(ATTRS), device=device)
    if attr_dict:
        for idx, attr_name in enumerate(ATTRS):
            if attr_name in attr_dict and attr_dict[attr_name] == 1:
                c_tensor[:, idx] = 1.0

    # Codificación de Tokens para CFG
    cond_tokens   = label_encoder(c_tensor)
    uncond_tokens = label_encoder(torch.zeros_like(c_tensor))

    z = torch.randn(n, 4, latent_hw, latent_hw, device=device, dtype=torch.float32)

    # Bucle de Muestreo Ancestral DDPM (T pasos)
    for i in reversed(range(diffusion.T)):
        t = torch.full((n,), i, device=device, dtype=torch.long)

        # Envolvemos el modelo en una función que aplique CFG vectorizado
        def batched_cfg_eps(z_t, t_t, cond_unused):
            z_in = torch.cat([z_t, z_t], dim=0)
            t_in = torch.cat([t_t, t_t], dim=0)
            c_in = torch.cat([uncond_tokens, cond_tokens], dim=0)

            with torch.autocast(device_type=("cuda" if "cuda" in device else "cpu"), dtype=torch.bfloat16):
                eps_all = model(z_in, t_in, cond=c_in)

            eps_u, eps_c = eps_all.chunk(2, dim=0)
            eps_u, eps_c = eps_u.float(), eps_c.float()

            return eps_u + guidance_scale * (eps_c - eps_u)

        z = diffusion.p_sample_step(batched_cfg_eps, z, t, cond=None)

    # Decodificación con el VAE
    z_scaled = (z / latent_scaling).to(dtype=vae.dtype)
    x = vae.decode(z_scaled).sample
    x = x.clamp(-1, 1).to(torch.float32)
    x = (x + 1) * 0.5

    nrow = int(math.sqrt(n))
    grid = vutils.make_grid(x, nrow=nrow, padding=2)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    vutils.save_image(grid, out_path)
    print(f"[INFER] Grid DDPM guardado en: {out_path} con atributos: {attr_dict}")

    if save_individual:
        os.makedirs(out_dir, exist_ok=True)
        for idx in range(n):
            vutils.save_image(x[idx], os.path.join(out_dir, f"img_{idx:03d}.png"))
        print(f"[INFER] {n} imágenes individuales guardadas en: {out_dir}")

    if backup_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in backup_state.items()})

    model.train(model_was_training)

    return grid
