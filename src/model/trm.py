from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.reasoner import Reasoner


@dataclass
class TRMInnerCarry:
    """Carry state for the inner TRM model (z_H, z_L tensors)."""
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TRMCarry:
    """Carry state for the outer TRM with ACT."""
    inner_carry: TRMInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    """Initialize tensor with truncated normal distribution."""
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)
    return tensor


def _stablemax(x: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    """Stablemax transformation: maps reals to positive values."""
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def stablemax_cross_entropy(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
) -> torch.Tensor:
    """Cross entropy using stablemax instead of softmax."""
    logits = logits.to(torch.float64)
    s_x = _stablemax(logits)
    log_probs = torch.log(s_x / s_x.sum(dim=-1, keepdim=True))

    valid_mask = labels != ignore_index
    safe_labels = torch.where(valid_mask, labels, 0)
    target_log_probs = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

    loss = -torch.where(valid_mask, target_log_probs, torch.zeros_like(target_log_probs))
    return loss.sum() / valid_mask.sum().clamp(min=1)


class TRM(nn.Module):
    """Tiny Recursive Model with Adaptive Computation Time."""

    def __init__(
            self,
            *,
            input_dim: int = 512,
            output_dim: int = 11,
            context_len: int = 1,
            seq_len: int = 128,
            num_heads: int = 8,
            num_layers: int = 2,
            expansion: float = 4.0,
            norm_eps: float = 1e-5,
            H_cycles: int = 3,
            L_cycles: int = 6,
            halt_max_steps: int = 16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_len = context_len
        self.seq_len = seq_len  # total seq_len passed to reasoner (includes context)
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.halt_max_steps = halt_max_steps

        # Context tokens (prepended to input embeddings)
        self.context_tokens = nn.Parameter(torch.zeros(1, context_len, input_dim))

        # Initial states for z_H and z_L (NOT trainable, broadcasts to all positions)
        self.register_buffer("H_init", trunc_normal_init_(torch.empty(input_dim), std=1.0))
        self.register_buffer("L_init", trunc_normal_init_(torch.empty(input_dim), std=1.0))

        # Reasoner (reused for all cycles)
        self.reasoner = Reasoner(
            dim=input_dim,
            seq_len=seq_len,
            num_heads=num_heads,
            num_layers=num_layers,
            expansion=expansion,
            norm_eps=norm_eps,
        )

        # Output heads
        self.lm_head = nn.Linear(input_dim, output_dim, bias=False)
        self.q_head = nn.Linear(input_dim, 1, bias=True)

        # Initialize linear layers with trunc_normal
        with torch.no_grad():
            trunc_normal_init_(self.lm_head.weight, std=1.0 / (input_dim ** 0.5))
            # Special initialization for q_head (encourages exploration early)
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)

    def empty_carry(self, batch_size: int) -> TRMInnerCarry:
        """Create an empty carry state for a batch."""
        device = self.H_init.device
        shape = (batch_size, self.seq_len, self.input_dim)
        return TRMInnerCarry(
            z_H=torch.empty(shape, device=device),
            z_L=torch.empty(shape, device=device),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TRMInnerCarry) -> TRMInnerCarry:
        """Reset carry state for sequences where reset_flag is True."""
        H_init = self.H_init.unsqueeze(0).unsqueeze(0).expand_as(carry.z_H)
        L_init = self.L_init.unsqueeze(0).unsqueeze(0).expand_as(carry.z_L)
        z_H = torch.where(reset_flag.view(-1, 1, 1), H_init, carry.z_H)
        z_L = torch.where(reset_flag.view(-1, 1, 1), L_init, carry.z_L)
        return TRMInnerCarry(z_H=z_H, z_L=z_L)

    def step(
            self,
            carry: TRMInnerCarry,
            input_emb: torch.Tensor,
    ) -> tuple[TRMInnerCarry, torch.Tensor, torch.Tensor]:
        """Run ONE ACT iteration (H_cycles x L_cycles reasoning).

        This matches trm_quest's TRMInner.forward() signature.

        Args:
            carry: Current carry state (z_H, z_L)
            input_emb: (B, seq_len + context_len, input_dim) embeddings WITH context

        Returns:
            new_carry: Updated carry with detached z_H, z_L
            logits: (B, seq_len, output_dim) predictions (context stripped)
            q_halt_logits: (B,) halting scores
        """
        z_H, z_L = carry.z_H, carry.z_L

        # H_cycles - 1 without gradients
        with torch.no_grad():
            for _ in range(self.H_cycles - 1):
                for _ in range(self.L_cycles):
                    z_L = self.reasoner(z_L, z_H + input_emb)
                z_H = self.reasoner(z_H, z_L)

        # Last H_cycle WITH gradients
        for _ in range(self.L_cycles):
            z_L = self.reasoner(z_L, z_H + input_emb)
        z_H = self.reasoner(z_H, z_L)

        # Create new carry with detached tensors (matches trm_quest)
        new_carry = TRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())

        # Output logits (skip context tokens)
        logits = self.lm_head(z_H[:, self.context_len:])

        # Halting decision from first context token
        q_halt_logits = self.q_head(z_H[:, 0]).squeeze(-1)

        return new_carry, logits, q_halt_logits

    def _inner_forward(
            self,
            z_H: torch.Tensor,
            z_L: torch.Tensor,
            input_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run H_cycles x L_cycles reasoning with gradient truncation.

        Only the last H_cycle has gradients.
        """
        # H_cycles - 1 without gradients
        with torch.no_grad():
            for _ in range(self.H_cycles - 1):
                for _ in range(self.L_cycles):
                    z_L = self.reasoner(z_L, z_H + input_emb)
                z_H = self.reasoner(z_H, z_L)

        # Last H_cycle WITH gradients
        for _ in range(self.L_cycles):
            z_L = self.reasoner(z_L, z_H + input_emb)
        z_H = self.reasoner(z_H, z_L)

        # Output logits (skip context tokens)
        logits = self.lm_head(z_H[:, self.context_len:])  # (B, seq_len, output_dim)

        # Halting decision from first context token
        q_halt = self.q_head(z_H[:, 0]).squeeze(-1)  # (B,)

        return z_H, z_L, logits, q_halt

    def forward(
            self,
            input_emb: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run full ACT loop for a batch.

        Args:
            input_emb: (B, seq_len, input_dim) embeddings (NO context)
            labels: (B, seq_len) targets

        Returns:
            logits: (B, seq_len, output_dim) predictions
            loss: scalar loss if labels provided, else None
        """
        B, seq_len, _ = input_emb.shape
        device = input_emb.device

        # Prepend context tokens
        context = self.context_tokens.expand(B, -1, -1)
        input_emb = torch.cat([context, input_emb], dim=1)  # (B, seq_len + context_len, input_dim)
        total_seq_len = seq_len + self.context_len

        # Initialize z_H, z_L for full sequence
        z_H = self.H_init.unsqueeze(0).unsqueeze(0).expand(B, total_seq_len, -1).clone()
        z_L = self.L_init.unsqueeze(0).unsqueeze(0).expand(B, total_seq_len, -1).clone()

        # Track halting state per sequence
        halted = torch.zeros(B, dtype=torch.bool, device=device)
        total_lm_loss = torch.tensor(0.0, device=device)
        total_q_loss = torch.tensor(0.0, device=device)
        num_halted = 0

        # Run ACT steps
        for step in range(self.halt_max_steps):
            z_H, z_L, logits, q_halt = self._inner_forward(z_H, z_L, input_emb)

            # Detach z_H/z_L to isolate gradients between ACT steps (matches original)
            z_H = z_H.detach()
            z_L = z_L.detach()

            if labels is not None:
                # Determine which sequences halt this step
                is_last_step = (step == self.halt_max_steps - 1)
                with torch.no_grad():
                    seq_correct = (logits.argmax(-1) == labels).all(-1)
                    newly_halted = ~halted & (is_last_step | (q_halt > 0))

                # Compute loss only for newly halted sequences
                if newly_halted.any():
                    # LM loss for halted sequences (stablemax)
                    halted_logits = logits[newly_halted]
                    halted_labels = labels[newly_halted]
                    lm_loss = stablemax_cross_entropy(halted_logits, halted_labels, ignore_index=-100)
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
