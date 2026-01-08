"""ACT Loss computation for training."""
import torch
import torch.nn.functional as F

from src.model.trm import stablemax_cross_entropy


def compute_act_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        q_halt_logits: torch.Tensor,
        halted: torch.Tensor,
        steps: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute ACT loss and metrics for one step.

    Loss is computed for ALL sequences (not just halted).
    Metrics are computed only for halted sequences.

    Args:
        logits: (B, seq_len, vocab_size) predictions
        labels: (B, seq_len) targets
        q_halt_logits: (B,) halting scores
        halted: (B,) which sequences halted this step
        steps: (B,) step counts

    Returns:
        loss: Scalar loss (lm_loss + 0.5 * q_halt_loss)
        metrics: Dict of metrics
    """
    with torch.no_grad():
        mask = labels != -100
        loss_counts = mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

        preds = logits.argmax(dim=-1)
        is_correct = mask & (preds == labels)
        seq_is_correct = is_correct.sum(-1) == loss_counts

        valid = halted & (loss_counts > 0)
        metrics = {
            "count": valid.sum(),
            "accuracy": torch.where(valid, (is_correct.float() / loss_divisor).sum(-1), 0.0).sum(),
            "exact_accuracy": (valid & seq_is_correct).sum(),
            "q_halt_accuracy": (valid & ((q_halt_logits >= 0) == seq_is_correct)).sum(),
            "steps": torch.where(valid, steps, 0).sum(),
        }

    # Loss for ALL sequences
    # lm_loss: average per-token loss (already normalized)
    # q_halt_loss: average per-sequence loss (need to normalize by batch_size)
    lm_loss = stablemax_cross_entropy(logits, labels, ignore_index=-100)
    q_halt_loss = F.binary_cross_entropy_with_logits(
        q_halt_logits, seq_is_correct.float(), reduction="mean"  # Changed from "sum" to "mean"
    )

    loss = lm_loss + 0.5 * q_halt_loss

    metrics["lm_loss"] = lm_loss.detach()
    metrics["q_halt_loss"] = q_halt_loss.detach()

    return loss, metrics
