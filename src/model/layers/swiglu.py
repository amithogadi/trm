import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """Gated MLP: out = w2(silu(w1(x)) * w3(x)). SiLU(x) = x * sigmoid(x), aka Swish."""

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
