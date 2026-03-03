import torch


def save_ckpt(path, model, optimizer, scaler, ema, step: int, extra: dict | None = None, scheduler=None):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "step": int(step),
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if extra:
        state["extra"] = extra
    torch.save(state, path)


def load_ckpt(
    path,
    model,
    optimizer=None,
    scaler=None,
    ema=None,
    scheduler=None,
    map_location="cuda",
    strict: bool = True,
):
    state = torch.load(path, map_location=map_location)
    sd = state["model"]

    has_unet_prefix = any(key.startswith("unet.") for key in sd.keys())

    if hasattr(model, "unet"):
        if has_unet_prefix:
            model.load_state_dict(sd, strict=strict)
        else:
            model.unet.load_state_dict(sd, strict=strict)
    else:
        if has_unet_prefix:
            sd = {key[len("unet.") :]: value for key, value in sd.items() if key.startswith("unet.")}
        model.load_state_dict(sd, strict=strict)

    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    if ema is not None and state.get("ema") is not None:
        ema.load_state_dict(state["ema"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])

    return state.get("step", 0), state.get("extra", {})
