import torch

from src.model.unet import UNetDenoiser


def test_unet_forward_preserves_latent_shape():
    model = UNetDenoiser(
        in_channels=4,
        base_channels=32,
        channel_mults=(1, 2),
        num_res_blocks=1,
        attn_resolutions={4},
        time_embed_dim=32,
        num_heads=2,
        head_dim=16,
        img_resolution=8,
        context_dim=16,
    )
    x = torch.randn(2, 4, 8, 8)
    t = torch.tensor([1, 3], dtype=torch.long)
    cond = torch.randn(2, 11, 16)

    out = model(x, t, cond=cond)

    assert out.shape == x.shape
