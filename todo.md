# TODO: Match trm_quest Training Architecture

## ✅ COMPLETED - ACT Loop Architecture Refactoring

### Step 1: Split TRM into TRMInner + TRM wrapper
- [x] Add empty_carry() and reset_carry() methods to TRM
- [x] Add _inner_step() method that runs ONE ACT iteration with carry
- [x] TRM.forward() handles carry initialization and halting logic

### Step 2: Create Carry dataclass
- [x] `TRMInnerCarry`: z_H, z_L tensors
- [x] `TRMCarry`: inner_carry, steps, halted

### Step 3: Refactor forward to single-step
- [x] TRM.forward(carry, input_emb) → runs ONE ACT iteration
- [x] Returns (new_carry, outputs_dict)
- [x] initial_carry() creates carry with halted=True

### Step 4: Update training loop
- [x] Carry persists across batches in train.py
- [x] Each batch: forward → compute_act_loss → backward → optimizer step

### Step 5: Fix loss computation
- [x] compute_act_loss() computes loss for ALL sequences at each step
- [x] Metrics computed only for halted sequences

## Verified Working

Training smoke test passed:
- Loss decreasing (2.49 → 1.73 over 420 steps)
- ACT halting mechanism works
- All 25 tests pass
