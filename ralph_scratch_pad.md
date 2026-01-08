# Ralph Loop Scratch Pad

## Test Baseline (Iteration 0)

**Status**: All tests passing at start

- 30 tests collected at start
- 32 tests now (2 new puzzle persistence tests added)
- All 32 pass

## Objectives Complete

All objectives from objectives.md have been implemented and tested:

1. **TRMCarry**: Added `current_inputs` and `current_labels` fields
2. **TRM.initial_carry**: Accepts labels, initializes with `torch.empty_like()`
3. **TRM.forward**: Persists inputs/labels for non-halted sequences using `torch.where(halted, new, old)`
4. **train.py**: Uses `carry.current_labels` for loss computation (core fix)
5. **SudokuModel**: Updated to pass labels through to TRM

## Key Files Modified

- `src/model/trm.py`: TRMCarry dataclass, initial_carry, forward
- `src/model/sudoku.py`: initial_carry, forward signatures
- `train.py`: Training loop and evaluate function
- `tests/test_trm_carry.py`: API tests + puzzle persistence tests
- `tests/test_sudoku_model.py`: API tests

## Verification

The fix ensures non-halted sequences keep their original puzzle data when new batches arrive.
This should eliminate the loss spikes (0.8 -> 3-6) caused by mismatched puzzle inputs.

## Bug Fix: RuntimeError "backward through graph twice"

**Symptom**: Training crashed with:
```
RuntimeError: Trying to backward through the graph a second time
```

**Root cause**: `new_current_inputs` was stored in carry with gradients attached. On next iteration, using `carry.current_inputs` in `torch.where()` created a dependency to the old computation graph.

**Fix**: Detach tensors before storing in carry (trm.py line 243):
```python
return TRMCarry(..., new_current_inputs.detach(), new_current_labels.detach()), outputs
```

This matches how `z_H` and `z_L` are detached in `_inner_step` (line 164).
