import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Normalizes by root-mean-square."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)  # rsqrt = 1 / sqrt
        return x.to(input_dtype)
