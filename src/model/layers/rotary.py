from abc import ABC, abstractmethod

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims: [a, b] -> [-b, a] for RoPE computation."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies rotary position embedding: x = x*cos + rotate_half(x)*sin."""
    # x: [B, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim // 2]
    orig_dtype = x.dtype
    x = x.to(cos.dtype)

    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim // 2]
    sin = sin.unsqueeze(0).unsqueeze(2)
    cos = cos.repeat(1, 1, 1, 2)  # [1, seq_len, 1, head_dim]
    sin = sin.repeat(1, 1, 1, 2)

    result = (x * cos) + (_rotate_half(x) * sin)
    return result.to(orig_dtype)


class PositionEmbedding(ABC, nn.Module):
    """Base class for position embeddings. Initialized with dim/seq_len, applies to any vector."""

    def __init__(self, dim: int, seq_len: int):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply position embedding to a single tensor. x: [B, seq_len, num_heads, head_dim]"""
        pass


class RoPE(PositionEmbedding):
    """Rotary Position Embedding."""

    def __init__(self, dim: int, seq_len: int, base: float = 10000.0):
        super().__init__(dim, seq_len)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        # register buffer saves it to model state without making them learnable, so model.cuda() moves them
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]
        return _apply_rotary(x, cos, sin)


class NoPositionEmbedding(PositionEmbedding):
    """No position embedding - returns input unchanged."""

    def __init__(self, dim: int = 0, seq_len: int = 0):
        super().__init__(dim, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x