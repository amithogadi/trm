# Plan: Implement Carry-State Training for TRM Sudoku

## Goal
Reduce training time from ~2.5 years to ~4 weeks by implementing carry-state training (1 ACT iteration per step instead of 16).

## Current vs Target

| Aspect | Current | Target |
|--------|---------|--------|
| ACT iterations per step | 16 | 1 |
| Reasoner calls per step | 336 | 21 |
| State management | Fresh each forward | Carry across steps |
| Expected speedup | 1x | ~16x |

---

## Key Insight

Each batch position is a "slot" that works on ONE puzzle until it halts. When it halts:
1. Its z_H, z_L reset to H_init, L_init
2. It gets a NEW puzzle from the dataloader
3. Non-halted positions keep their puzzle and hidden state

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
    steps: torch.Tensor       # (B,) int32 - ACT step count per position
    halted: torch.Tensor      # (B,) bool - which positions have halted
    inputs: torch.Tensor      # (B, 81) - current puzzle inputs per position
    labels: torch.Tensor      # (B, 81) - current puzzle labels per position
```

### Step 2: Add Methods to TRM Class

**`initial_carry(inputs, labels, device)`** - Create fresh carry:
```python
def initial_carry(self, inputs, labels, device):
    B = inputs.shape[0]
    total_seq_len = 81 + self.context_len
    return TRMCarry(
        inner_carry=TRMInnerCarry(
            z_H=torch.empty(B, total_seq_len, self.input_dim, device=device),
            z_L=torch.empty(B, total_seq_len, self.input_dim, device=device),
        ),
        steps=torch.zeros(B, dtype=torch.int32, device=device),
        halted=torch.ones(B, dtype=torch.bool, device=device),  # All halted initially!
        inputs=inputs,
        labels=labels,
    )
```

**`_reset_carry(inner_carry, halted_mask)`** - Selectively reset z_H/z_L:
```python
def _reset_carry(self, inner_carry, halted_mask, total_seq_len):
    # Only reset positions where halted=True
    mask = halted_mask.view(-1, 1, 1)
    H_init = self.H_init.expand(inner_carry.z_H.shape[0], total_seq_len, -1)
    L_init = self.L_init.expand(inner_carry.z_L.shape[0], total_seq_len, -1)

    z_H = torch.where(mask, H_init, inner_carry.z_H)
    z_L = torch.where(mask, L_init, inner_carry.z_L)
    return TRMInnerCarry(z_H=z_H, z_L=z_L)
```

### Step 3: Replace TRM.forward() with Carry-State Mode

```python
def forward(self, carry, halt_exploration_prob=0.1):
    """Single ACT step. Uses puzzles from carry.inputs.

    Returns: (new_carry, outputs_dict)
    """
    # 1. Reset z_H, z_L for halted positions
    inner = self._reset_carry(carry.inner_carry, carry.halted, total_seq_len)

    # 2. Reset step counter for halted positions
    steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)

    # 3. Embed inputs and run single _inner_forward()
    input_emb = self.embedder(carry.inputs)  # embedder moves to TRM
    z_H, z_L, logits, q_halt = self._inner_forward(inner.z_H, inner.z_L, input_emb)

    # 4. Detach for next step
    new_inner = TRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())

    # 5. Update steps, determine halting
    new_steps = steps + 1
    with torch.no_grad():
        is_last = new_steps >= self.halt_max_steps
        halted = is_last | (q_halt > 0)
        # Exploration: randomly prevent early halt
        if self.training:
            explore = torch.rand_like(q_halt) < halt_exploration_prob
            min_steps = torch.randint_like(new_steps, 2, self.halt_max_steps + 1)
            halted = halted & (~explore | (new_steps >= min_steps))

    new_carry = TRMCarry(new_inner, new_steps, halted, carry.inputs, carry.labels)
    outputs = {"logits": logits, "q_halt": q_halt}
    return new_carry, outputs
```

### Step 4: Update SudokuModel Wrapper

Move embedder to TRM or simplify SudokuModel to just delegate to TRM.

### Step 5: Modify Training Loop in `train.py`

```python
carry = None
data_iter = iter(train_loader)

global_step = 0
while global_step < total_steps:
    # Initialize carry with first batch
    if carry is None:
        inputs, labels = next(data_iter)
        inputs, labels = inputs.to(device), labels.to(device)
        carry = model.initial_carry(inputs, labels, device)

    # Forward (uses carry.inputs, carry.labels)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        carry, outputs = model(carry)

    # Compute loss only for halted positions
    if carry.halted.any():
        halted_mask = carry.halted
        logits = outputs["logits"][halted_mask]
        labels_halted = carry.labels[halted_mask]
        q_halt = outputs["q_halt"][halted_mask]

        # Check correctness for q_halt loss
        with torch.no_grad():
            correct = (logits.argmax(-1) == labels_halted).all(-1)

        lm_loss = stablemax_cross_entropy(logits, labels_halted)
        q_loss = F.binary_cross_entropy_with_logits(q_halt, correct.float())
        loss = lm_loss + 0.5 * q_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Replace halted positions with NEW puzzles
        num_halted = halted_mask.sum().item()
        new_inputs, new_labels = next(data_iter)
        new_inputs, new_labels = new_inputs.to(device), new_labels.to(device)

        # Fill halted slots with new puzzles
        carry.inputs[halted_mask] = new_inputs[:num_halted]
        carry.labels[halted_mask] = new_labels[:num_halted]

    scheduler.step()
    global_step += 1
```

### Step 6: Update Evaluation

Evaluation runs until all positions halt:

```python
@torch.no_grad()
def evaluate(model, dataloader, device):
    correct_puzzles = 0
    total_puzzles = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        carry = model.initial_carry(inputs, labels, device)

        # Run until all halt
        while not carry.halted.all():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                carry, outputs = model(carry)

        preds = outputs["logits"].argmax(-1)
        correct_puzzles += (preds == labels).all(-1).sum().item()
        total_puzzles += labels.shape[0]

    return correct_puzzles / total_puzzles
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/model/trm.py` | Add dataclasses, `initial_carry()`, `_reset_carry()`, replace `forward()` |
| `src/model/sudoku.py` | Simplify or merge into TRM |
| `train.py` | New training loop with carry management, puzzle replacement |

## Notes

- **Carry includes puzzles**: inputs/labels stored in carry, replaced when halted
- **Selective reset**: Only halted positions get z_H, z_L reset to H_init, L_init
- **torch.where**: Used for selective updates without breaking gradients
- **Dataloader as stream**: Training loop pulls new puzzles to fill halted slots
- **Distributed**: Each GPU has its own carry state, no sync needed

## Expected Outcome

- Training time: ~2.5 years → ~4 weeks (with bfloat16 + carry-state)
- Same results as trm_quest approach
