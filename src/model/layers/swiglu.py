import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_multiple(a: int, b: int) -> int:
    """Round up a to nearest multiple of b."""
    return (-(a // -b)) * b


def _trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    """Initialize tensor with truncated normal distribution."""
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)
    return tensor


class SwiGLU(nn.Module):
    """Gated MLP: fused gate/up projection, 2/3 expansion ratio."""

    def __init__(self, dim: int, expansion: float = 4.0):
        super().__init__()
        inter = _find_multiple(round(expansion * dim * 2 / 3), 256)

        self.gate_up_proj = nn.Linear(dim, inter * 2, bias=False)
        self.down_proj = nn.Linear(inter, dim, bias=False)

        # Initialize with trunc_normal
        with torch.no_grad():
            _trunc_normal_init_(self.gate_up_proj.weight, std=1.0 / (dim ** 0.5))
            _trunc_normal_init_(self.down_proj.weight, std=1.0 / (inter ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)