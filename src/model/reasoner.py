import torch
import torch.nn as nn

from src.model.layers import Attention, RMSNorm, SwiGLU


class Reasoner(nn.Module):
    def __init__(
            self,
            dim: int,
            seq_len: int,
            num_heads: int = 8,
            num_layers: int = 2,
            expansion: float = 4.0,
            norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict({
                    "attn": Attention(dim=dim, seq_len=seq_len, num_heads=num_heads),
                    "attn_norm": RMSNorm(dim, norm_eps),
                    "mlp": SwiGLU(dim, expansion),
                    "mlp_norm": RMSNorm(dim, norm_eps),
                })
            )

    """General-purpose iterative reasoner - a stack of transformer blocks with Post-LN."""

    def forward(
            self,
            hidden_states: torch.Tensor,
            input_injection: torch.Tensor,
    ) -> torch.Tensor:
        x = hidden_states + input_injection
        for layer in self.layers:
            x = layer["attn_norm"](x + layer["attn"](x))
            x = layer["mlp_norm"](x + layer["mlp"](x))
        return x
