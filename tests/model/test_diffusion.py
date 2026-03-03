import torch

from src.model.diffusion import Diffusion


def test_q_sample_preserves_shape():
    diffusion = Diffusion(T=10, schedule="linear")
    x0 = torch.randn(2, 4, 8, 8)
    t = torch.tensor([1, 5], dtype=torch.long)
    xt = diffusion.q_sample(x0, t)

    assert xt.shape == x0.shape


def test_p_sample_step_ddim_preserves_shape():
    diffusion = Diffusion(T=10, schedule="linear")
    x_t = torch.randn(2, 4, 8, 8)
    t = torch.tensor([7, 7], dtype=torch.long)
    t_prev = torch.tensor([6, 6], dtype=torch.long)

    def eps_fn(x, timestep, cond=None):
        return torch.zeros_like(x)

    x_prev = diffusion.p_sample_step_ddim(eps_fn, x_t=x_t, t=t, t_prev=t_prev, eta=0.0)
    assert x_prev.shape == x_t.shape
