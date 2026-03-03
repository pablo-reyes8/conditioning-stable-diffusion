import torch 
import torch.nn as nn 

class EMA:
    """
    Exponential Moving Average (EMA) sobre parámetros de un modelo.
    Mapea por nombre para garantizar estabilidad al guardar/cargar checkpoints
    o si cambia la configuración de requires_grad.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999, device: str | torch.device | None = None):
        self.decay = float(decay)
        self.device = device

        # Mapeo: {nombre_parametro: tensor_sombra_fp32}
        self.shadow = {}

        for name, p in model.named_parameters():
            if p.requires_grad:
                s = p.detach().to(dtype=torch.float32).clone()
                if self.device is not None:
                    s = s.to(self.device)
                self.shadow[name] = s

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Actualiza la sombra EMA usando los parámetros actuales del modelo."""
        for name, p in model.named_parameters():
            if name in self.shadow and p.requires_grad:
                s = self.shadow[name]
                p32 = p.detach().to(dtype=torch.float32)

                # Mover al dispositivo de la sombra si es necesario
                if s.device != p32.device:
                    p32 = p32.to(s.device)

                s.mul_(self.decay).add_(p32, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        """Copia los pesos EMA de vuelta a un modelo (para inferencia/evaluación)."""
        for name, p in model.named_parameters():
            if name in self.shadow:
                s = self.shadow[name]
                p.data.copy_(s.to(dtype=p.dtype, device=p.device))

    @torch.no_grad()
    def state_dict(self):
        """Genera un diccionario seguro para guardar como checkpoint."""
        return {
            "decay": self.decay,
            # Guardamos los tensores como un dict nominal, igual que nn.Module
            "shadow": {name: s.cpu() for name, s in self.shadow.items()} }

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        """Restaura la sombra desde un checkpoint."""
        self.decay = float(state_dict.get("decay", self.decay))
        loaded_shadow = state_dict.get("shadow", {})

        for name, s in self.shadow.items():
            if name in loaded_shadow:
                # Copiar preservando el dispositivo y dtype actuales de la sombra local
                s.data.copy_(loaded_shadow[name].to(device=s.device, dtype=s.dtype))
            else:
                print(f"Advertencia EMA: Parámetro '{name}' no encontrado en el checkpoint.")


@torch.no_grad()
def ema_health(ema: EMA, model: nn.Module, rel_tol: float = 5.0):
    def _flat(t):
        return t.detach().float().cpu().reshape(-1)

    m_params = []
    e_params = []

    for name, p in model.named_parameters():
        if name in ema.shadow:
            m_params.append(p)
            e_params.append(ema.shadow[name])

    if not m_params:
        return (False, "empty_ema", float("inf"))

    m_flat = torch.cat([_flat(p) for p in m_params], dim=0)
    e_flat = torch.cat([_flat(s) for s in e_params], dim=0)

    if not torch.isfinite(e_flat).all():
        return (False, "nan_or_inf_in_ema", float("inf"))

    m_norm = m_flat.norm().item()
    e_norm = e_flat.norm().item()
    if e_norm < 1e-12:
        return (False, "ema_zero_norm", float("inf"))
    if m_norm < 1e-12:
        return (False, "model_zero_norm", float("inf"))

    rel = (m_flat - e_flat).norm().item() / (m_norm + 1e-8)
    if rel > rel_tol:
        return (False, "large_rel_diff", rel)
    return (True, "ok", rel)


@torch.no_grad()
def ema_reinit_from_model(ema: EMA, model: nn.Module):
    for name, p in model.named_parameters():
        if name in ema.shadow:
            s = ema.shadow[name]
            s.data.copy_(p.detach().to(dtype=torch.float32, device=s.device))

def ema_set_decay(ema: EMA, new_decay: float):
    ema.decay = float(new_decay)