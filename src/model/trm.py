import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import TRMConfig
from src.model.reasoner import Reasoner


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    """Initialize tensor with truncated normal distribution."""
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)
    return tensor


class TRM(nn.Module):
    """Tiny Recursive Model with Adaptive Computation Time."""

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        # Input embeddings - init with std=1/sqrt(hidden_size) so after scaling variance ≈ 1
        self.embed_scale = math.sqrt(config.hidden_size)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        with torch.no_grad():
            trunc_normal_init_(self.embed_tokens.weight, std=1.0 / self.embed_scale)

        # Puzzle context token (prepended to sequence)
        self.puzzle_context = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        trunc_normal_init_(self.puzzle_context, std=1.0)

        # Learned initial states for z_H and z_L
        # Shape: (1, seq_len + 1, hidden_size) - +1 for puzzle context
        init_shape = (1, config.seq_len + 1, config.hidden_size)
        self.H_init = nn.Parameter(trunc_normal_init_(torch.empty(init_shape), std=1.0))
        self.L_init = nn.Parameter(trunc_normal_init_(torch.empty(init_shape), std=1.0))

        # Reasoner (reused for all cycles)
        self.reasoner = Reasoner(
            dim=config.hidden_size,
            seq_len=config.seq_len + 1,  # +1 for puzzle context
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            expansion=config.expansion,
            norm_eps=config.norm_eps,
        )

        # Output heads
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = nn.Linear(config.hidden_size, 1, bias=True)

        # Special initialization for q_head (encourages exploration early)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)

    def _embed_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Embed inputs and prepend puzzle context token.

        Args:
            inputs: (B, seq_len) token ids

        Returns:
            (B, seq_len + 1, hidden_size) embeddings with puzzle context prepended
        """
        B = inputs.shape[0]
        token_emb = self.embed_tokens(inputs) * self.embed_scale
        puzzle_ctx = self.puzzle_context.expand(B, -1, -1)
        return torch.cat([puzzle_ctx, token_emb], dim=1)

    def _inner_forward(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        input_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run H_cycles x L_cycles reasoning with gradient truncation.

        Only the last H_cycle has gradients.
        """
        H_cycles = self.config.H_cycles
        L_cycles = self.config.L_cycles

        # H_cycles - 1 without gradients
        with torch.no_grad():
            for _ in range(H_cycles - 1):
                for _ in range(L_cycles):
                    z_L = self.reasoner(z_L, z_H + input_emb)
                z_H = self.reasoner(z_H, z_L)

        # Last H_cycle WITH gradients
        for _ in range(L_cycles):
            z_L = self.reasoner(z_L, z_H + input_emb)
        z_H = self.reasoner(z_H, z_L)

        # Output logits (skip puzzle context token at position 0)
        logits = self.lm_head(z_H[:, 1:])  # (B, seq_len, vocab_size)

        # Halting decision from puzzle context token
        q_halt = self.q_head(z_H[:, 0]).squeeze(-1)  # (B,)

        return z_H, z_L, logits, q_halt

    def forward(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run full ACT loop for a batch.

        Args:
            inputs: (B, seq_len) puzzle inputs
            labels: (B, seq_len) solutions, optional

        Returns:
            logits: (B, seq_len, vocab_size) final predictions
            loss: scalar loss if labels provided, else None
        """
        B = inputs.shape[0]
        device = inputs.device

        # Initialize fresh for this batch
        z_H = self.H_init.expand(B, -1, -1).clone()
        z_L = self.L_init.expand(B, -1, -1).clone()
        input_emb = self._embed_inputs(inputs)

        # Track halting state per sequence
        halted = torch.zeros(B, dtype=torch.bool, device=device)
        total_lm_loss = torch.tensor(0.0, device=device)
        total_q_loss = torch.tensor(0.0, device=device)
        num_halted = 0

        # Run ACT steps
        for step in range(self.config.halt_max_steps):
            z_H, z_L, logits, q_halt = self._inner_forward(z_H, z_L, input_emb)

            # Detach z_H/z_L to isolate gradients between ACT steps (matches original)
            z_H = z_H.detach()
            z_L = z_L.detach()

            if labels is not None:
                # Determine which sequences halt this step
                is_last_step = (step == self.config.halt_max_steps - 1)
                with torch.no_grad():
                    seq_correct = (logits.argmax(-1) == labels).all(-1)
                    newly_halted = ~halted & (is_last_step | (q_halt > 0))

                # Compute loss only for newly halted sequences
                if newly_halted.any():
                    # LM loss for halted sequences
                    halted_mask = newly_halted.unsqueeze(1).expand(-1, labels.shape[1])
                    halted_logits = logits[newly_halted]
                    halted_labels = labels[newly_halted]
                    lm_loss = F.cross_entropy(
                        halted_logits.reshape(-1, self.config.vocab_size),
                        halted_labels.reshape(-1),
                        ignore_index=-100,
                    )
                    total_lm_loss = total_lm_loss + lm_loss * newly_halted.sum()

                    # Q-halt loss for halted sequences
                    q_loss = F.binary_cross_entropy_with_logits(
                        q_halt[newly_halted],
                        seq_correct[newly_halted].float(),
                        reduction="sum",
                    )
                    total_q_loss = total_q_loss + q_loss

                    num_halted += newly_halted.sum().item()
                    halted = halted | newly_halted

        # Average loss over halted sequences
        if labels is not None and num_halted > 0:
            loss = (total_lm_loss + 0.5 * total_q_loss) / num_halted
        else:
            loss = None

        return logits, loss
