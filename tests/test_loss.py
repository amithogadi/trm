"""Tests for ACT loss computation."""
import torch
import pytest

from src.model.loss import compute_act_loss


class TestComputeActLoss:
    """Test compute_act_loss function."""

    def test_returns_loss_and_metrics(self):
        """compute_act_loss returns a loss tensor and metrics dict."""
        logits = torch.randn(4, 81, 11, requires_grad=True)
        labels = torch.randint(1, 10, (4, 81))
        q_halt_logits = torch.randn(4, requires_grad=True)
        halted = torch.tensor([True, False, True, False])
        steps = torch.tensor([3, 2, 5, 1])

        loss, metrics = compute_act_loss(logits, labels, q_halt_logits, halted, steps)

        assert loss.shape == ()
        assert loss.requires_grad
        assert "lm_loss" in metrics
        assert "q_halt_loss" in metrics
        assert "count" in metrics
        assert "exact_accuracy" in metrics

    def test_metrics_only_for_halted(self):
        """Metrics are computed only for halted sequences."""
        logits = torch.randn(4, 81, 11)
        labels = torch.randint(1, 10, (4, 81))
        q_halt_logits = torch.randn(4)
        halted = torch.tensor([True, False, False, False])
        steps = torch.tensor([5, 2, 3, 1])

        _, metrics = compute_act_loss(logits, labels, q_halt_logits, halted, steps)

        assert metrics["count"] == 1
        assert metrics["steps"] == 5

    def test_loss_computed_for_all_sequences(self):
        """Loss is computed for all sequences, not just halted."""
        logits = torch.randn(4, 81, 11)
        labels = torch.randint(1, 10, (4, 81))
        q_halt_logits = torch.randn(4)
        steps = torch.tensor([1, 1, 1, 1])

        # All halted
        halted_all = torch.ones(4, dtype=torch.bool)
        loss_all, _ = compute_act_loss(logits, labels, q_halt_logits, halted_all, steps)

        # None halted
        halted_none = torch.zeros(4, dtype=torch.bool)
        loss_none, _ = compute_act_loss(logits, labels, q_halt_logits, halted_none, steps)

        # Loss should be the same (computed for all sequences regardless of halted)
        assert torch.allclose(loss_all, loss_none)

    def test_exact_accuracy_counts_perfect_sequences(self):
        """exact_accuracy counts sequences where all tokens are correct."""
        logits = torch.zeros(2, 81, 11)
        labels = torch.zeros(2, 81, dtype=torch.long)
        logits[:, :, 0] = 10.0  # Make argmax select 0
        labels[:] = 0  # Labels are also 0 -> perfect match

        q_halt_logits = torch.tensor([1.0, 1.0])  # Both would halt
        halted = torch.tensor([True, True])
        steps = torch.tensor([1, 1])

        _, metrics = compute_act_loss(logits, labels, q_halt_logits, halted, steps)

        assert metrics["exact_accuracy"] == 2
