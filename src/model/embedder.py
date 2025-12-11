import math

import torch
import torch.nn as nn


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    """Initialize tensor with truncated normal distribution."""
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)
    return tensor


class InputEmbedder(nn.Module):
    """Embeds token inputs to dense vectors."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_scale = math.sqrt(hidden_size)

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        with torch.no_grad():
            trunc_normal_init_(self.embed_tokens.weight, std=1.0 / self.embed_scale)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """(B, seq_len) → (B, seq_len, hidden_size)"""
        return self.embed_tokens(inputs) * self.embed_scale
