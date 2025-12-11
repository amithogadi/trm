import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.layers.rotary import PositionEmbedding, RoPE


def _trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    """Initialize tensor with truncated normal distribution."""
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)
    return tensor


class Attention(nn.Module):
    def __init__(
            self, *,
            dim: int,
            seq_len: int = 82,
            num_heads: int = 8,
            pos_emb: PositionEmbedding | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # Initialize with trunc_normal
        std = 1.0 / (dim ** 0.5)
        with torch.no_grad():
            _trunc_normal_init_(self.q_proj.weight, std=std)
            _trunc_normal_init_(self.k_proj.weight, std=std)
            _trunc_normal_init_(self.v_proj.weight, std=std)
            _trunc_normal_init_(self.o_proj.weight, std=std)

        self.pos_emb = pos_emb or RoPE(dim=self.head_dim, seq_len=seq_len)

    """Multi-head attention with pluggable position embeddings (defaults to RoPE)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, seq_len, _ = x.shape

        q = self.q_proj(x).view(B, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, seq_len, self.num_heads, self.head_dim)

        q = self.pos_emb(q)
        k = self.pos_emb(k)

        q = q.transpose(1, 2)  # (B, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).contiguous().view(B, seq_len, -1)
        return self.o_proj(out)
