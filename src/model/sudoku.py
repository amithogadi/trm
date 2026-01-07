import torch
import torch.nn as nn

from src.model.embedder import InputEmbedder
from src.model.trm import TRM, TRMCarry


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

    def initial_carry(self, inputs: torch.Tensor) -> TRMCarry:
        """Create initial carry state for a batch."""
        input_emb = self.embedder(inputs)
        return self.trm.initial_carry(input_emb)

    def forward(
            self,
            carry: TRMCarry,
            inputs: torch.Tensor,
            halt_exploration_prob: float = 0.1,
    ) -> tuple[TRMCarry, dict[str, torch.Tensor]]:
        """Run ONE ACT step.

        Args:
            carry: Current carry state
            inputs: (B, seq_len) token ids
            halt_exploration_prob: Exploration probability (training only)

        Returns:
            new_carry: Updated carry with halting status
            outputs: Dict with 'logits' and 'q_halt_logits'
        """
        input_emb = self.embedder(inputs)
        return self.trm(carry, input_emb, halt_exploration_prob)
