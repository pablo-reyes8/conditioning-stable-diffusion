import importlib
import sys
import types
from pathlib import Path

import torch


def _install_torchvision_stub():
    saved_paths: list[str] = []

    torchvision_module = types.ModuleType("torchvision")
    utils_module = types.ModuleType("torchvision.utils")

    def make_grid(x, nrow=8, padding=2):
        return x[0]

    def save_image(tensor, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"stub-image")
        saved_paths.append(str(path))

    utils_module.make_grid = make_grid
    utils_module.save_image = save_image
    torchvision_module.utils = utils_module

    sys.modules["torchvision"] = torchvision_module
    sys.modules["torchvision.utils"] = utils_module
    return saved_paths


class DummyLabelEncoder(torch.nn.Module):
    def forward(self, c):
        return c.unsqueeze(-1)


class DummyModel(torch.nn.Module):
    def forward(self, z_t, t, cond=None):
        return torch.zeros_like(z_t)


class DummyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float32

    def decode(self, z):
        return types.SimpleNamespace(sample=torch.zeros(z.shape[0], 3, 8, 8, dtype=torch.float32))


class DummyDiffusion:
    def __init__(self, T=4):
        self.T = T
        self.alphas_cumprod = torch.linspace(1.0, 0.1, T)

    def p_sample_step(self, eps_fn, z, t, cond=None):
        return z * 0.5

    def p_sample_step_ddim(self, eps_fn, x_t, t, t_prev, eta=0.0, clip_x0=True):
        return x_t * 0.5


def test_ddpm_sampler_creates_output_file(tmp_path: Path):
    saved_paths = _install_torchvision_stub()
    ddpm = importlib.import_module("src.inference.ddpm")

    out_path = tmp_path / "ddpm.png"
    grid = ddpm.ddpm_infer_sample(
        model=DummyModel(),
        diffusion=DummyDiffusion(T=3),
        vae=DummyVAE(),
        label_encoder=DummyLabelEncoder(),
        attr_dict={"Smiling": 1},
        n=4,
        latent_hw=4,
        device="cpu",
        out_path=str(out_path),
        seed=123,
    )

    assert grid is not None
    assert out_path.exists()
    assert str(out_path) in saved_paths


def test_ddim_sampler_creates_output_file(tmp_path: Path):
    saved_paths = _install_torchvision_stub()
    ddim = importlib.import_module("src.inference.ddim")

    out_path = tmp_path / "ddim.png"
    grid = ddim.ddim_latent_infer_sample(
        model=DummyModel(),
        diffusion=DummyDiffusion(T=5),
        vae=DummyVAE(),
        label_encoder=DummyLabelEncoder(),
        attr_dict={"Young": 1},
        n=4,
        latent_hw=4,
        device="cpu",
        out_path=str(out_path),
        steps=3,
        seed=123,
    )

    assert grid is not None
    assert out_path.exists()
    assert str(out_path) in saved_paths
