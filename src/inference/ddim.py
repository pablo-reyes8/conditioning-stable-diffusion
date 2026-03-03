import os
import torch
import math
import torchvision.utils as vutils

from src.data.constants import DEFAULT_ATTRIBUTES

ATTRS = DEFAULT_ATTRIBUTES

@torch.no_grad()
def ddim_latent_infer_sample(
    model,
    diffusion,
    vae,
    label_encoder,
    attr_dict: dict | None = None,
    n: int = 16,
    latent_hw: int = 64,
    device: str = "cuda",
    guidance_scale: float = 7.5,
    latent_scaling: float = 0.18215,
    *,
    ema=None,
    out_path: str = "samples_ddim.png",
    save_individual: bool = False,
    out_dir: str = "samples_individual",
    seed: int | None = 1234,
    steps: int = 50,
    eta: float = 0.0,
    schedule_kind: str = "t_linear"):

    """
    Inferencia acelerada DDIM en espacio LATENTE con soporte para
    Classifier-Free Guidance (CFG) y diccionarios de atributos.
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

    # Preparación de Atributos (Parser de diccionario)
    c_tensor = torch.zeros(n, len(ATTRS), device=device)
    if attr_dict:
        for idx, attr_name in enumerate(ATTRS):
            if attr_name in attr_dict and attr_dict[attr_name] == 1:
                c_tensor[:, idx] = 1.0

    # Pre-computar Tokens para CFG
    cond_tokens   = label_encoder(c_tensor)
    uncond_tokens = label_encoder(torch.zeros_like(c_tensor))

    # Inicialización Latente (4 canales para Stable Diffusion style)
    z = torch.randn(n, 4, latent_hw, latent_hw, device=device, dtype=torch.float32)

    T = diffusion.T
    if schedule_kind == "t_linear":
        schedule = torch.linspace(T - 1, 0, steps, device=device)
        schedule = torch.unique_consecutive(schedule.round().long(), dim=0)
    else:
        a_bar = diffusion.alphas_cumprod
        u = torch.linspace(0.0, 1.0, steps, device=device)
        targets = (1.0 - u)
        s_list = [(a_bar - z_val).abs().argmin().item() for z_val in targets]
        schedule = torch.tensor(sorted(set(s_list), reverse=True), device=device)

    if schedule[-1].item() != 0:
        schedule = torch.cat([schedule, torch.zeros(1, device=device, dtype=schedule.dtype)])

    # Bucle DDIM con CFG Vectorizado
    for i in range(len(schedule) - 1):
        t_cur  = schedule[i]
        t_prev = schedule[i+1]

        t_cur_t  = torch.full((n,), int(t_cur.item()),  device=device, dtype=torch.long)
        t_prev_t = torch.full((n,), int(t_prev.item()), device=device, dtype=torch.long)

        # Función envolvente para aplicar CFG en un solo batch
        def batched_cfg_eps_ddim(z_t, t_t, cond_unused=None, *args, **kwargs):
            z_in = torch.cat([z_t, z_t], dim=0)
            t_in = torch.cat([t_t, t_t], dim=0)
            c_in = torch.cat([uncond_tokens, cond_tokens], dim=0)

            with torch.autocast(
                device_type=("cuda" if "cuda" in device else "cpu"),
                dtype=torch.bfloat16,
            ):
                eps_all = model(z_in, t_in, cond=c_in)

            eps_u, eps_c = eps_all.chunk(2, dim=0)
            return eps_u.float() + guidance_scale * (eps_c.float() - eps_u.float())

        # Paso de muestreo DDIM (Determinista si eta=0)
        z = diffusion.p_sample_step_ddim(
            batched_cfg_eps_ddim,
            x_t=z,
            t=t_cur_t,
            t_prev=t_prev_t,
            eta=eta,
            clip_x0=True)

    # Decodificación VAE (Latente -> Píxel)
    z_final = (z / latent_scaling).to(dtype=vae.dtype)
    x = vae.decode(z_final).sample
    x = x.clamp(-1, 1).to(torch.float32)
    x = (x + 1) * 0.5


    nrow = int(math.sqrt(n))
    grid = vutils.make_grid(x, nrow=nrow, padding=2)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    vutils.save_image(grid, out_path)
    print(f"[INFER-DDIM] Grid guardado en: {out_path} (steps={len(schedule)-1}, eta={eta})")

    if backup_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in backup_state.items()})
    model.train(model_was_training)

    return grid
