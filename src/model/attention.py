import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    """Embedding sinusoidal estándar para timesteps (t) -> R^dim."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb


class TimeMLP(nn.Module):
    """Proyecta el embedding sinusoidal a un espacio (time_dim) con activación."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),)

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)  # (B, out_dim)




def group_norm(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(num_groups, channels), num_channels=channels, eps=1e-6, affine=True)

class AttnBlock(nn.Module):
    """
    Self-attention espacial (tokens = H*W).
    Residual: x + proj(attn(norm(x))).
    """
    def __init__(self, channels: int, num_heads: int = 4, head_dim: int = 64, p_drop: float = 0.0):
        super().__init__()
        assert channels > 0 and num_heads > 0 and head_dim > 0
        self.channels  = channels
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.p_drop    = float(p_drop)

        inner = num_heads * head_dim
        self.norm = group_norm(channels)
        self.qkv  = nn.Conv2d(channels, inner * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(inner, channels, kernel_size=1)

        # Zero-init trick para estabilidad en arquitecturas profundas
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, N)
        q, k, v = qkv.unbind(dim=1)  # (B, heads, d, N)

        # (B, heads, N, d)
        q = q.permute(0, 1, 3, 2).contiguous()
        k = k.permute(0, 1, 3, 2).contiguous()
        v = v.permute(0, 1, 3, 2).contiguous()

        dp = self.p_drop if self.training and self.p_drop > 0.0 else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=False)

        # back to (B, inner, H, W)
        out = out.permute(0, 1, 3, 2).contiguous().reshape(B, self.num_heads * self.head_dim, H, W)
        out = self.proj(out)

        return x + out


class CrossAttnBlock(nn.Module):
    """
    Cross-attention espacial:
      - Queries: de features espaciales (H*W tokens)
      - Keys/Values: de cond/context (T tokens)
    """
    def __init__(
        self,
        channels: int,
        context_dim: int,
        num_heads: int = 4,
        head_dim: int = 64,
        p_drop: float = 0.0):

        super().__init__()
        assert channels > 0 and context_dim > 0
        self.channels = channels
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.p_drop = float(p_drop)

        inner = num_heads * head_dim
        self.norm = group_norm(channels)

        # Añadimos normalización al contexto para evitar que el softmax se sature
        self.norm_context = nn.LayerNorm(context_dim)

        self.to_q = nn.Conv2d(channels, inner, kernel_size=1, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)

        self.proj = nn.Conv2d(inner, channels, kernel_size=1)

        # Zero-init trick
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if cond is None:
            return x

        B, C, H, W = x.shape
        N = H * W
        _, T, D = cond.shape
        assert B == cond.shape[0], "cond batch mismatch"
        assert D == self.context_dim, f"Expected context_dim={self.context_dim}, got {D}"

        h = self.norm(x)

        # Normalizamos el contexto
        cond_norm = self.norm_context(cond)

        # Q: (B, heads, N, d)
        q = self.to_q(h).reshape(B, self.num_heads, self.head_dim, N)
        q = q.permute(0, 1, 3, 2).contiguous()

        # K,V: (B, heads, T, d) - Usando el contexto normalizado
        k = self.to_k(cond_norm).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = self.to_v(cond_norm).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        dp = self.p_drop if self.training and self.p_drop > 0.0 else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=False)

        # (B, inner, H, W)
        out = out.permute(0, 1, 3, 2).contiguous().reshape(B, self.num_heads * self.head_dim, H, W)
        out = self.proj(out)

        return x + out
    



