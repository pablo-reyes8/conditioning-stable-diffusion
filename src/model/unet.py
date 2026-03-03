import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Sequence, Set, Optional

from src.model.attention import *

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.act1  = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))

        self.norm2 = group_norm(out_ch)
        self.act2  = nn.SiLU()
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))

        # Proyectamos el tiempo
        t_bias = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(h)
        h = h + t_bias
        h = self.conv2(self.drop(self.act2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNetDenoiser(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 192,
        channel_mults: Sequence[int] = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Set[int] = frozenset({8, 4}),
        time_embed_dim: int = 512,
        dropout: float = 0.0,
        num_heads: int = 4,
        head_dim: int = 64,
        img_resolution: int = 32,
        use_cross_attn: bool = True,
        context_dim: int = 256,
        attn_drop: float = 0.0,
    ):
        super().__init__()

        self.img_resolution = img_resolution
        self.use_cross_attn = use_cross_attn
        self.context_dim = context_dim

        # Embedding temporal
        self.time_pos_emb = SinusoidalPosEmb(time_embed_dim)
        self.time_mlp = TimeMLP(time_embed_dim, time_embed_dim)

        # Entrada
        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # ---------------------------------------------------------
        # ENCODER
        # ---------------------------------------------------------
        self.downs = nn.ModuleList()
        enc_skip_channels = [base_channels]  # Guardamos el canal de la conv inicial
        resolutions = [img_resolution]
        in_ch = base_channels

        for level_idx, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            is_last_level = (level_idx == len(channel_mults) - 1)
            level_blocks = nn.ModuleList()

            # Bloques residuales + Atención (si aplica)
            for _ in range(num_res_blocks):
                res_block = ResBlock(in_ch, out_ch, time_dim=time_embed_dim, dropout=dropout)
                in_ch = out_ch

                attn_block = nn.Identity()
                if resolutions[-1] in attn_resolutions:
                    if use_cross_attn:
                        attn_block = CrossAttnBlock(in_ch, context_dim=context_dim, num_heads=num_heads, head_dim=head_dim, p_drop=attn_drop)
                    else:
                        attn_block = AttnBlock(in_ch, num_heads=num_heads, head_dim=head_dim, p_drop=attn_drop)

                # Agrupamos como una tupla/lista en el ModuleList para un forward limpio
                level_blocks.append(nn.ModuleList([res_block, attn_block]))
                enc_skip_channels.append(in_ch)  # Skip denso: guardamos después de cada bloque

            # Downsample
            down_block = nn.Identity()
            if not is_last_level:
                down_block = Downsample(in_ch)
                resolutions.append(resolutions[-1] // 2)
                enc_skip_channels.append(in_ch)  # Skip denso: guardamos el downsample

            down_module = nn.Module()
            down_module.blocks = level_blocks
            down_module.down = down_block
            self.downs.append(down_module)

        # ---------------------------------------------------------
        # BOTTLENECK
        # ---------------------------------------------------------
        bottleneck_res = resolutions[-1]
        mid_attn = nn.Identity()
        if bottleneck_res in attn_resolutions:
            mid_attn = CrossAttnBlock(in_ch, context_dim=context_dim, num_heads=num_heads, head_dim=head_dim, p_drop=attn_drop) if use_cross_attn \
                       else AttnBlock(in_ch, num_heads=num_heads, head_dim=head_dim, p_drop=attn_drop)

        self.mid = nn.ModuleList([
            ResBlock(in_ch, in_ch, time_dim=time_embed_dim, dropout=dropout),
            mid_attn,
            ResBlock(in_ch, in_ch, time_dim=time_embed_dim, dropout=dropout),
        ])

        # ---------------------------------------------------------
        # DECODER
        # ---------------------------------------------------------
        self.ups = nn.ModuleList()

        # Recorremos los niveles de forma inversa
        for level_idx, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            is_first_level = (level_idx == 0)
            level_blocks = nn.ModuleList()

            # IMPORTANTE: num_res_blocks + 1 para consumir el skip del downsample (o conv inicial)
            for _ in range(num_res_blocks + 1):
                skip_ch = enc_skip_channels.pop()

                # El in_ch actual + el skip que traemos del encoder
                res_block = ResBlock(in_ch + skip_ch, out_ch, time_dim=time_embed_dim, dropout=dropout)
                in_ch = out_ch

                attn_block = nn.Identity()
                if resolutions[-1] in attn_resolutions:
                    if use_cross_attn:
                        attn_block = CrossAttnBlock(in_ch, context_dim=context_dim, num_heads=num_heads, head_dim=head_dim, p_drop=attn_drop)
                    else:
                        attn_block = AttnBlock(in_ch, num_heads=num_heads, head_dim=head_dim, p_drop=attn_drop)

                level_blocks.append(nn.ModuleList([res_block, attn_block]))

            up_block = nn.Identity()
            if not is_first_level:
                up_block = Upsample(in_ch)
                resolutions.pop()

            up_module = nn.Module()
            up_module.blocks = level_blocks
            up_module.up = up_block
            self.ups.append(up_module)

        # Validamos que todos los skips se consumieron correctamente
        assert len(enc_skip_channels) == 0, "Error de topología: No se consumieron todos los skips."

        # Salida
        self.out_norm = group_norm(in_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(in_ch, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_pos_emb(t))

        # 1. ENCODER
        cur = self.in_conv(x)
        skips = [cur]

        for down in self.downs:
            for res_block, attn_block in down.blocks:
                cur = res_block(cur, t_emb)

                # Procesamos atención si el bloque no es Identity
                if getattr(attn_block, 'context_dim', None) is not None:
                    cur = attn_block(cur, cond)
                elif not isinstance(attn_block, nn.Identity):
                    cur = attn_block(cur)

                skips.append(cur)

            if not isinstance(down.down, nn.Identity):
                cur = down.down(cur)
                skips.append(cur)

        # 2. BOTTLENECK
        cur = self.mid[0](cur, t_emb)
        if getattr(self.mid[1], 'context_dim', None) is not None:
            cur = self.mid[1](cur, cond)
        elif not isinstance(self.mid[1], nn.Identity):
            cur = self.mid[1](cur)
        cur = self.mid[2](cur, t_emb)

        # 3. DECODER
        for up in self.ups:
            for res_block, attn_block in up.blocks:
                skip = skips.pop()
                cur = torch.cat([cur, skip], dim=1)

                cur = res_block(cur, t_emb)

                if getattr(attn_block, 'context_dim', None) is not None:
                    cur = attn_block(cur, cond)
                elif not isinstance(attn_block, nn.Identity):
                    cur = attn_block(cur)

            if not isinstance(up.up, nn.Identity):
                cur = up.up(cur)

        out = self.out_conv(self.out_act(self.out_norm(cur)))
        return out


def build_unet_latent_32(
    in_channels: int = 4,
    base_channels: int = 192,
    channel_mults: Tuple[int, ...] = (1, 2, 3, 4),   # 32 -> 16 -> 8 -> 4
    attn_resolutions: Set[int] = frozenset({8, 4}),
    time_embed_dim: int = 512,
    dropout: float = 0.1,
    num_heads: int = 4,
    head_dim: int = 64,
    use_cross_attn: bool = True,
    context_dim: int = 256,
    attn_drop: float = 0.0):

    return UNetDenoiser(
        in_channels=in_channels,
        base_channels=base_channels,
        channel_mults=channel_mults,
        num_res_blocks=2,
        attn_resolutions=attn_resolutions,
        time_embed_dim=time_embed_dim,
        dropout=dropout,
        num_heads=num_heads,
        head_dim=head_dim,
        img_resolution=32,
        use_cross_attn=use_cross_attn,
        context_dim=context_dim,
        attn_drop=attn_drop,)


def build_unet_latent_64(
    in_channels: int = 4,
    base_channels: int = 192,
    channel_mults: Tuple[int, ...] = (1, 2, 3, 4),
    attn_resolutions: Set[int] = frozenset({16, 8}),
    time_embed_dim: int = 512,
    dropout: float = 0.1,
    num_heads: int = 4,
    head_dim: int = 64,
    use_cross_attn: bool = True,
    context_dim: int = 256,
    attn_drop: float = 0.0):

    return UNetDenoiser(
        in_channels=in_channels,
        base_channels=base_channels,
        channel_mults=channel_mults,
        num_res_blocks=2,
        attn_resolutions=attn_resolutions,
        time_embed_dim=time_embed_dim,
        dropout=dropout,
        num_heads=num_heads,
        head_dim=head_dim,
        img_resolution=64,
        use_cross_attn=use_cross_attn,
        context_dim=context_dim,
        attn_drop=attn_drop)


class TrainableSD(nn.Module):
    def __init__(self, unet, label_encoder):
        super().__init__()
        self.unet = unet
        self.label_encoder = label_encoder

    def forward(self, zt, t, cond):
        # El wrapper simplemente actúa como un proxy hacia la UNet
        return self.unet(zt, t, cond=cond)