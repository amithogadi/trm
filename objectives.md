# Objective: Fix Puzzle Persistence in Training Loop

## Problem

The training loop feeds new puzzles to ALL sequences every batch, but trm_quest only feeds new puzzles to HALTED sequences. Non-halted sequences should continue processing their original puzzle.

This causes loss spikes (0.8 -> 3-6) when non-halted sequences receive mismatched puzzle inputs.

## Root Cause

In trm_quest, the carry stores `current_data` (inputs, labels):

```python
# trm_quest/models/recursive_reasoning/trm.py:246-259

# Carry includes current_data
current_data={k: torch.empty_like(v) for k, v in batch.items()}

# On forward: halted sequences get NEW data, others keep OLD
new_current_data = {k: torch.where(carry.halted.view(...), batch[k], v)
                    for k, v in carry.current_data.items()}

# Inner model uses current_data, not batch
self.inner(new_inner_carry, new_current_data)
```

Current trm passes `inputs` directly without persistence.

## Required Changes

1. **TRMCarry** (`src/model/trm.py`): Add `current_inputs` and `current_labels` fields

2. **TRM.initial_carry**: Initialize `current_inputs` and `current_labels` with empty tensors

3. **TRM.forward**:
   - Update current_inputs/labels: halted sequences get new batch data, non-halted keep old
   - Pass `current_inputs` to inner model instead of raw inputs
   - Return updated current_inputs/labels in new carry

4. **train.py**:
   - Pass both inputs AND labels to model forward
   - Use carry's current_labels for loss computation (not batch labels directly)

## Reference

See trm_quest implementation:
- `models/recursive_reasoning/trm.py:237-297` - TinyRecursiveReasoningModel_ACTV1
- `models/losses.py:59` - Loss uses `new_carry.current_data["labels"]`

## Verification

After fix, loss should be stable around 0.8-1.2 without spikes to 3-6.
