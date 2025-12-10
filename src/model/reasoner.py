import torch
import torch.nn as nn

from src.model.layers import Attention, RMSNorm, SwiGLU


class Reasoner(nn.Module):
    """General-purpose iterative reasoner - a stack of transformer blocks with Post-LN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int = 2,
        mlp_ratio: int = 4,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict({
                    "attn": Attention(dim, num_heads),
                    "attn_norm": RMSNorm(dim, norm_eps),
                    "mlp": SwiGLU(dim, dim * mlp_ratio),
                    "mlp_norm": RMSNorm(dim, norm_eps),
                })
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = hidden_states + input_injection
        for layer in self.layers:
            x = layer["attn_norm"](x + layer["attn"](x, cos, sin))
            x = layer["mlp_norm"](x + layer["mlp"](x))
        return x
