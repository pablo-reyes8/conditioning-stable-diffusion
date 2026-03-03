import inspect
from contextlib import contextmanager, nullcontext
import torch

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16}

def make_grad_scaler(device: str = "cuda", enabled: bool = True, amp_dtype: str = "bf16"):
    if not enabled:
        return None
    if amp_dtype.lower() in ("bf16", "bfloat16"):
        return None

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            sig = inspect.signature(torch.amp.GradScaler)
            if len(sig.parameters) >= 1:
                return torch.amp.GradScaler(device if device in ("cuda", "cpu") else "cuda")
            else:
                return torch.amp.GradScaler()
        except Exception:
            pass

    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler()

    return None


def _cuda_dtype_supported(dtype: torch.dtype) -> bool:
    if not torch.cuda.is_available():
        return False
    # En A100: bf16 y fp16 están soportados.
    return dtype in (torch.bfloat16, torch.float16)

@contextmanager
def autocast_ctx(
    device: str = "cuda",
    enabled: bool = True,
    dtype: str = "bf16",
    cache_enabled: bool = True):
    """
    Autocast robusto:
      - CUDA: usa torch.amp.autocast(device_type="cuda", dtype=...)
      - CPU:  usa torch.amp.autocast(device_type="cpu",  dtype=torch.bfloat16) si enabled
      - Desactiva limpio con nullcontext().
    Notas:
      * En BF16 NO uses GradScaler.
      * En FP16 sí puedes usar GradScaler (p.ej., torch.cuda.amp.GradScaler()).
    """
    if not enabled:
        with nullcontext():
            yield
        return

    if device == "cuda":
        want = _DTYPE_MAP.get(dtype.lower(), torch.bfloat16)
        use = want if _cuda_dtype_supported(want) else torch.float16
        with torch.amp.autocast(device_type="cuda", dtype=use, cache_enabled=cache_enabled):
            yield
        return

    # CPU autocast solo útil con bfloat16 en kernels soportados.
    if device == "cpu":
        try:
            with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, cache_enabled=cache_enabled):
                yield
        except Exception:
            # fallback seguro
            with nullcontext():
                yield
        return

    # Fallback por si aparece otro backend
    with nullcontext():
        yield