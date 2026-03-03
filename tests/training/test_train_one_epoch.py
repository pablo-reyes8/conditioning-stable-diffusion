import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model.diffusion import Diffusion
from src.model.label_encoder import LabelTokenEncoder
from src.training.train_one_epoch import train_one_epoch


class DummyPosterior:
    def __init__(self, z):
        self.latent_dist = self
        self._z = z

    def sample(self):
        return self._z


class DummyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float32

    def encode(self, x):
        return DummyPosterior(x[:, :4])


class TinyConditionalModel(torch.nn.Module):
    def __init__(self, context_dim: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, kernel_size=1)
        self.time = torch.nn.Embedding(16, 4)
        self.context = torch.nn.Linear(context_dim, 4)

    def forward(self, z_t, t, cond=None):
        context_bias = self.context(cond.mean(dim=1)).unsqueeze(-1).unsqueeze(-1)
        time_bias = self.time(t).unsqueeze(-1).unsqueeze(-1)
        return self.conv(z_t) + context_bias + time_bias


def test_train_one_epoch_runs_with_small_dummy_components():
    images = torch.randn(4, 4, 8, 8)
    labels = torch.randint(0, 2, (4, 11)).float()
    loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)

    model = TinyConditionalModel(context_dim=8)
    diffusion = Diffusion(T=16, schedule="linear")
    label_encoder = LabelTokenEncoder(num_labels=11, context_dim=8)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(label_encoder.parameters()),
        lr=1e-3,
    )

    avg_loss, n_batches, n_images, global_step = train_one_epoch(
        model=model,
        diffusion=diffusion,
        dataloader=loader,
        optimizer=optimizer,
        vae=DummyVAE(),
        label_encoder=label_encoder,
        latent_scaling=1.0,
        cfg_drop_prob=0.0,
        amp_dtype="bf16",
        scaler=None,
        ema=None,
        device="cpu",
        grad_clip=1.0,
        use_autocast=False,
        grad_accum_steps=1,
        global_step=0,
        latent_hw=8,
    )

    assert avg_loss >= 0.0
    assert n_batches == 2
    assert n_images == 4
    assert global_step == 2
