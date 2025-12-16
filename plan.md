# Plan: Implement Carry-State Training for TRM Sudoku

## Goal
Reduce training time from ~2.5 years to ~4 weeks by implementing carry-state training (1 ACT iteration per step instead of 16).

## Current vs Target

| Aspect | Current | Target |
|--------|---------|--------|
| ACT iterations per step | 16 | 1 |
| Reasoner calls per step | 336 | 21 |
| State management | Fresh each forward | Carry across batches |
| Expected speedup | 1x | ~16x |

---

## Implementation Steps

### Step 1: Add Carry State Dataclasses to `src/model/trm.py`

```python
from dataclasses import dataclass

@dataclass
class TRMInnerCarry:
    z_H: torch.Tensor  # (B, seq_len + context_len, input_dim)
    z_L: torch.Tensor  # (B, seq_len + context_len, input_dim)

@dataclass
class TRMCarry:
    inner_carry: TRMInnerCarry
    steps: torch.Tensor  # (B,) int32
    halted: torch.Tensor  # (B,) bool
```

### Step 2: Add Methods to TRM Class

Add to `src/model/trm.py`:

1. `initial_carry(batch_size, device)` - Create fresh carry with `halted=True` to trigger reset
2. `_reset_carry(carry, reset_mask, total_seq_len)` - Selectively reset z_H/z_L for halted sequences

### Step 3: Replace TRM.forward() with Carry-State Mode

Remove the full ACT loop. New forward:

```python
def forward(self, input_emb, carry, halt_exploration_prob=0.1):
    """Single ACT step with carry state.

    Returns: (new_carry, outputs_dict)
    """
    # Prepend context tokens
    # Reset carry for halted sequences
    # Run single _inner_forward()
    # Determine halting (with exploration)
    # Return new_carry and outputs
```

### Step 4: Update SudokuModel Wrapper

Modify `src/model/sudoku.py`:
- Add `initial_carry(batch_size, device)` method
- Update `forward(inputs, carry)` to pass carry to TRM
- Return `(new_carry, outputs)` instead of `(logits, loss)`

### Step 5: Modify Training Loop in `train.py`

```python
carry = None

for epoch in range(args.epochs):
    carry = None  # Reset at epoch boundary

    for inputs, labels in train_loader:
        if carry is None:
            carry = model.initial_carry(inputs.shape[0], device)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            carry, outputs = model(inputs, carry=carry)

        # Compute loss only for halted sequences
        if carry.halted.any():
            loss = compute_loss(outputs, labels, carry.halted)
            loss.backward()
            optimizer.step()

        scheduler.step()
```

### Step 6: Update Evaluation

Evaluation needs to run multiple steps until all sequences halt:

```python
def evaluate(model, dataloader, device):
    for inputs, labels in dataloader:
        carry = model.initial_carry(batch_size, device)

        # Run until all sequences halt
        while not carry.halted.all():
            carry, outputs = model(inputs, carry=carry)

        # Use final logits for accuracy computation
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/model/trm.py` | Add dataclasses, `initial_carry()`, `_reset_carry()`, replace `forward()` |
| `src/model/sudoku.py` | Add `initial_carry()`, update `forward()` to use carry |
| `train.py` | Add carry state management, update training loop, update evaluate() |

## Notes

- **No backward compatibility**: Stateless mode removed entirely
- **Distributed training**: Carry state is per-GPU, no sync needed
- **Checkpoints**: Don't save carry state - reinitialize on resume
- **Epoch boundary**: Reset carry at each epoch start

## Expected Outcome

- Training time: ~2.5 years → ~4 weeks (with bfloat16 + carry-state)
- Same results as trm_quest approach
