import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_multiple(a: int, b: int) -> int:
    """Round up a to nearest multiple of b."""
    return (-(a // -b)) * b


class SwiGLU(nn.Module):
    """Gated MLP matching original TRM: fused gate/up projection, 2/3 expansion ratio."""

    def __init__(self, dim: int, expansion: float = 4.0):
        super().__init__()
        # Match original: inter = round(expansion * dim * 2/3), rounded to multiple of 256
        inter = _find_multiple(round(expansion * dim * 2 / 3), 256)

        # Fused gate + up projection (single matmul, then chunk)
        self.gate_up_proj = nn.Linear(dim, inter * 2, bias=False)
        self.down_proj = nn.Linear(inter, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)