from typing import Optional

import torch
import torch.nn as nn

from src.model.embedder import InputEmbedder
from src.model.trm import TRM


class SudokuModel(nn.Module):
    """Sudoku model: embedder + TRM reasoning."""

    def __init__(
            self,
            vocab_size: int = 11,
            hidden_size: int = 512,
            context_len: int = 1,
            seq_len: int = 81,
            num_heads: int = 8,
            num_layers: int = 2,
            expansion: float = 4.0,
            norm_eps: float = 1e-5,
            H_cycles: int = 3,
            L_cycles: int = 6,
            halt_max_steps: int = 16,
    ):
        super().__init__()
        self.embedder = InputEmbedder(vocab_size, hidden_size)
        self.trm = TRM(
            input_dim=hidden_size,
            output_dim=vocab_size,
            context_len=context_len,
            seq_len=seq_len + context_len,
            num_heads=num_heads,
            num_layers=num_layers,
            expansion=expansion,
            norm_eps=norm_eps,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            halt_max_steps=halt_max_steps,
        )

    def forward(
            self,
            inputs: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            inputs: (B, seq_len) token ids
            labels: (B, seq_len) solutions, optional

        Returns:
            logits: (B, seq_len, vocab_size)
            loss: scalar if labels, else None
        """
        input_emb = self.embedder(inputs)
        return self.trm(input_emb, labels)
