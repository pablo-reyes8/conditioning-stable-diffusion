import os, time, torch
from torchvision.utils import make_grid, save_image


@torch.no_grad()
def sample_latent_ddim_cfg(
    model, diffusion, vae, label_encoder,
    *,
    n: int = 36,
    latent_hw: int = 64,                 # 32 para 256px, 64 para 512px
    device: str = "cuda",
    steps: int = 50,                     # DDIM steps
    eta: float = 0.0,                    # 0.0 = DDIM determinista
    guidance_scale: float = 3.5,
    c: torch.Tensor | None = None,       # (n,11) o None => uncond
    latent_scaling: float = 0.18215,
    save_path: str | None = None,
    nrow: int | None = None):

    model.eval()
    label_encoder.eval()
    vae.eval()

    # Init ruido en latentes
    z = torch.randn(n, 4, latent_hw, latent_hw, device=device, dtype=torch.float32)

    T = diffusion.T
    ts = torch.linspace(T - 1, 0, steps, device=device).long()
    ts_prev = torch.cat([ts[1:], ts.new_zeros(1)])

    # Labels condition
    if c is None:
        c = torch.zeros(n, 11, device=device)
    else:
        c = c.to(device)

    cond_tokens   = label_encoder(c)
    uncond_tokens = label_encoder(torch.zeros_like(c))

    for t_i, t_prev_i in zip(ts, ts_prev):
        # Vectores de tiempo para el batch
        t  = torch.full((n,), int(t_i.item()), device=device, dtype=torch.long)
        t_prev = torch.full((n,), int(t_prev_i.item()), device=device, dtype=torch.long)

        # Optimización CFG: Batched Forward Pass
        z_in = torch.cat([z, z], dim=0)
        t_in = torch.cat([t, t], dim=0)
        cond_in = torch.cat([uncond_tokens, cond_tokens], dim=0)

        # Usamos autocast para inferencia si es compatible
        with torch.autocast(device_type=("cuda" if "cuda" in device else "cpu"), dtype=torch.bfloat16):
            eps_all = model(z_in, t_in, cond=cond_in)

        eps_u, eps_c = eps_all.chunk(2, dim=0)

        eps_u, eps_c = eps_u.float(), eps_c.float()
        eps = eps_u + guidance_scale * (eps_c - eps_u)

        # Corrección Boundary Condition (t=0)
        if t_i.item() == 0:
            # En el último paso, extraemos la imagen limpia directamente
            z = diffusion.predict_x0(z, eps, t)
        else:
            # DDIM step normal
            def _eps_fn(_z, _t, _cond_unused):
                return eps
            z = diffusion.p_sample_step_ddim(
                _eps_fn, z, t=t, t_prev=t_prev, eta=eta, cond=None)


    z_scaled = (z / latent_scaling).to(dtype=vae.dtype)
    x = vae.decode(z_scaled).sample

    # Aseguramos que x vuelva a float32 para las métricas e imágenes
    x = x.clamp(-1, 1).to(torch.float32)
    x = (x + 1) * 0.5

    if nrow is None:
        nrow = int(n ** 0.5)

    grid = make_grid(x, nrow=nrow, padding=2)

    if save_path is not None:
        save_image(grid, save_path)

    return grid