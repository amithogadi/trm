# TODO: Match trm_quest Training Architecture

## Critical: Refactor ACT Loop Architecture

The current implementation runs all 16 ACT steps in one forward() call with one backward().
trm_quest runs each ACT step as a separate training iteration with its own backward().

### Step 1: Split TRM into TRMInner + TRM wrapper
- [x] Add empty_carry() and reset_carry() methods to TRM (DONE)
- [x] Add step() method that runs ONE ACT iteration with carry (DONE)
- [ ] TRM wrapper handles carry initialization and halting logic (via training loop)

### Step 2: Create Carry dataclass
- [x] `TRMInnerCarry`: z_H, z_L tensors (DONE)
- [x] `TRMCarry`: inner_carry, steps, halted (DONE - current_data handled by training loop)

### Step 3: Refactor forward to single-step
- [ ] TRM.forward(carry, batch) → runs ONE ACT iteration
- [ ] Returns new_carry and outputs dict
- [ ] Carry persists across training loop iterations

### Step 4: Update training loop
- [ ] Move ACT loop from model.forward() to train.py
- [ ] Each ACT iteration: forward → loss → backward
- [ ] Replace halted sequences with new puzzles from batch

### Step 5: Fix loss computation
- [ ] Compute loss for ALL sequences at each step (not just halted)
- [ ] Use full batch for lm_loss and q_halt_loss

## Reference: trm_quest/sudoku/trm.py structure
```
TRMInner:
  - embed_tokens, context embedding, RoPE
  - reasoner (H_cycles × L_cycles)
  - lm_head, q_head
  - forward(carry, inputs) → (new_carry, logits, (q_halt, q_continue))

TRM:
  - initial_carry(inputs)
  - forward(carry, inputs) → (new_carry, outputs_dict)
    - Handles carry reset on halted sequences
    - Halting logic with exploration
```
