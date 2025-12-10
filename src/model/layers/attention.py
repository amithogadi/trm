import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims: [a, b] -> [-b, a] for RoPE computation."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Applies rotary position embedding: x = x*cos + rotate_half(x)*sin."""
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim // 2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    cos = cos.repeat(1, 1, 1, 2)  # (1, seq_len, 1, head_dim)
    sin = sin.repeat(1, 1, 1, 2)
    return (x * cos) + (rotate_half(x) * sin)


class Attention(nn.Module):
    """Multi-head attention with RoPE on Q/K (custom because nn.MultiheadAttention lacks RoPE support)."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        B, seq_len, _ = x.shape

        q = self.q_proj(x).view(B, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, seq_len, self.num_heads, self.head_dim)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q = q.transpose(1, 2)  # (B, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).contiguous().view(B, seq_len, -1)
        return self.o_proj(out)
