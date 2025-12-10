import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin rotation matrices for rotary position embeddings."""

    def __init__(self, dim: int, max_seq_len: int = 82, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, seq_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            return self.cos, self.sin
        return self.cos[:seq_len], self.sin[:seq_len]
