from dataclasses import dataclass

import torch
import torch.nn as nn

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
    current_inputs: torch.Tensor
    current_labels: torch.Tensor


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

    def _input_embeddings(self, input_emb: torch.Tensor) -> torch.Tensor:
        """Prepend context tokens to input embeddings."""
        B = input_emb.shape[0]
        context = self.context_tokens.expand(B, -1, -1)
        return torch.cat([context, input_emb], dim=1)

    def _inner_step(
            self,
            carry: TRMInnerCarry,
            input_emb: torch.Tensor,
    ) -> tuple[TRMInnerCarry, torch.Tensor, torch.Tensor]:
        """Run ONE ACT iteration (H_cycles x L_cycles reasoning).

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

    def initial_carry(self, inputs: torch.Tensor, labels: torch.Tensor) -> TRMCarry:
        """Create initial carry state for a batch.

        Args:
            inputs: (B, seq_len) or (B, seq_len, dim) - input embeddings
            labels: (B, seq_len) - target labels

        Returns:
            TRMCarry with halted=True (triggers reset on first forward)
        """
        batch_size = inputs.shape[0]
        device = inputs.device
        inner = self.empty_carry(batch_size)
        steps = torch.zeros(batch_size, dtype=torch.int32, device=device)
        halted = torch.ones(batch_size, dtype=torch.bool, device=device)
        current_inputs = torch.empty_like(inputs)
        current_labels = torch.empty_like(labels)
        return TRMCarry(inner, steps, halted, current_inputs, current_labels)

    def forward(
            self,
            carry: TRMCarry,
            input_emb: torch.Tensor,
            labels: torch.Tensor,
            halt_exploration_prob: float = 0.1,
    ) -> tuple[TRMCarry, dict[str, torch.Tensor]]:
        """Run ONE ACT step, matching trm_quest's TRM.forward().

        Args:
            carry: Current carry state (inner_carry, steps, halted, current_inputs, current_labels)
            input_emb: (B, seq_len, input_dim) embeddings (NO context) - new batch data
            labels: (B, seq_len) - new batch labels
            halt_exploration_prob: Probability of exploration (training only)

        Returns:
            new_carry: Updated carry with halting status and persisted inputs/labels
            outputs: Dict with 'logits' and 'q_halt_logits'
        """
        # Reset carry for halted sequences (new puzzles)
        new_inner = self.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)

        # Update current_inputs/labels: halted sequences get new data, non-halted keep old
        halted_view = carry.halted.view(-1, 1, 1) if input_emb.ndim == 3 else carry.halted.view(-1, 1)
        new_current_inputs = torch.where(halted_view, input_emb, carry.current_inputs)
        halted_view_labels = carry.halted.view(-1, 1)
        new_current_labels = torch.where(halted_view_labels, labels, carry.current_labels)

        # Prepend context and run one step using CURRENT inputs (not raw batch)
        input_with_context = self._input_embeddings(new_current_inputs)
        new_inner, logits, q_halt_logits = self._inner_step(new_inner, input_with_context)

        outputs = {"logits": logits, "q_halt_logits": q_halt_logits}

        # Compute halting (no gradients)
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.halt_max_steps
            halted = is_last_step

            if self.training and self.halt_max_steps > 1:
                halted = halted | (q_halt_logits > 0)
                # Exploration: randomly delay halting
                exploration = torch.rand_like(q_halt_logits)
                min_halt_steps = (exploration < halt_exploration_prob) * torch.randint_like(
                    new_steps, 2, self.halt_max_steps + 1
                )
                halted = halted & (new_steps >= min_halt_steps)

        return TRMCarry(new_inner, new_steps, halted, new_current_inputs.detach(), new_current_labels.detach()), outputs
