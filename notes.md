# How TRM Solves Sudoku

## The Core Idea

The model doesn't solve Sudoku in one pass. It **thinks iteratively** - like a human scribbling notes, refining guesses, and re-examining the board multiple times.

---

## Key Components

### 1. Input Representation

```
Input: (B, 81) integer tokens in {1..10}
       └── 81 = 9x9 Sudoku board, flattened
```

Each cell value is looked up in a learned embedding table:

```
embed_tokens: (11, 512)
├── 11 possible token values (0-10)
└── Each maps to a learned 512-dim vector
```

This embedding is **value-only** - no position or neighbor info yet.

### 2. Puzzle Context

A single learned vector `(1, 512)` is:
1. Expanded to `(B, 1, 512)`
2. Prepended to the 81 cell embeddings

Result: `input_embeddings` of shape `(B, 82, 512)`

```
position 0:     puzzle context (1 learned token)
positions 1-81: cell embeddings (the actual puzzle)
```

### 3. The Scratchpad (Carry)

Two latent tensors the model uses to "think":

```
z_H: (B, 82, 512)  <- high-level state
z_L: (B, 82, 512)  <- low-level state
```

Both initialized from learned parameters `H_init` and `L_init`.

Think of these as the model's working memory - notes it updates while reasoning.

### 4. Bookkeeping

```
steps:  (B,)  <- counter per batch element
halted: (B,)  <- boolean: done thinking?
```

---

## The Algorithm

### Outer Loop (16 iterations)

```python
for step in range(16):
    carry, outputs = TRM(carry, inputs)
    logits = outputs["logits"]  # (B, 81, 11)
```

Each iteration refines the scratchpad. Only the final logits are returned.

### Inner Loop (per outer step)

Each outer step runs **3 H-cycles**, each containing **6 L-cycles**:

```python
for h in range(3):           # H-cycles
    for l in range(6):       # L-cycles
        z_L = reasoner(z_L, z_H + input_embeddings)
    z_H = reasoner(z_H, z_L)
```

**Pattern:**
- Update `z_L` 6 times using `z_H + input_embeddings` as context
- Update `z_H` once using the refined `z_L`
- Repeat 3 times

This interleaving lets low-level reasoning inform high-level, and vice versa.

### The Reasoner

The reasoner takes 3 parameters:

```python
reasoner(hidden_states, input_injection, cos_sin)
```

| Parameter | Shape | Purpose |
|-----------|-------|---------|
| `hidden_states` | (B, 82, 512) | The tensor being updated (z_L or z_H) |
| `input_injection` | (B, 82, 512) | Context to add in |
| `cos_sin` | tuple of two (82, 64) | RoPE embeddings for position info |

Processing:

```
hidden_states + input_injection   <- elementwise add
  │
  ├── Attention (uses cos_sin for RoPE) + RMSNorm
  ├── SwiGLU MLP + RMSNorm
  │
  ├── Attention (uses cos_sin for RoPE) + RMSNorm
  ├── SwiGLU MLP + RMSNorm
  │
Output: (B, 82, 512)
```

**Where context comes from:**
- The two input tensors are added before processing
- RoPE (via cos_sin) injects position information in attention
- Attention lets every cell see every other cell (non-causal)
- After one attention layer, each position knows about all 82 positions

### RoPE (Rotary Position Embeddings)

`cos_sin` is a tuple of two precomputed tensors:

```python
cos_sin = (cos, sin)
#          │     │
#          │     └── sin: (82, 64) <- sine of rotation angles
#          └── cos: (82, 64) <- cosine of rotation angles
```

Each row = a position (0-81). Each column = a frequency. These encode "where am I?" via rotation angles.

**How RoPE is applied in attention:**

Inside attention, query and key are projected from hidden states:

```python
query, key, value = project(hidden_states)  # each (B, 82, 8, 64)
```

Then RoPE rotates only query and key (not value):

```python
cos, sin = cos_sin  # unpack tuple

# rotate_half swaps and negates halves: [a,b,c,d] -> [-c,-d,a,b]
def rotate_half(x):  # x: (B, 82, 8, 64)
    x1 = x[..., :32]   # first 32 dims
    x2 = x[..., 32:]   # last 32 dims
    return cat([-x2, x1], dim=-1)  # (B, 82, 8, 64)

# Apply rotation formula: x' = x*cos(θ) + rotate(x)*sin(θ)
q_rotated = (query * cos) + (rotate_half(query) * sin)  # (B, 82, 8, 64)
k_rotated = (key * cos) + (rotate_half(key) * sin)      # (B, 82, 8, 64)
```

The rotation angles differ per position, so when attention computes `query @ key.T`,
the dot product is affected by the relative distance between positions. This lets the
model know "how far apart are these two cells?" without explicitly adding position to
the hidden states.

---

## Output

After 16 outer steps:

```python
logits = lm_head(z_H)[:, 1:]  # (B, 82, 512) -> (B, 82, 11) -> (B, 81, 11)
                               #                └── discard puzzle context position
```

`lm_head` is a learned linear projection `(512 -> 11)` that converts each 512-dim
hidden state into 11 logits (one score per possible digit).

### Halting logits

The puzzle context position (position 0) is also used to decide when to stop thinking:

```python
q_logits = q_head(z_H[:, 0])  # (B, 512) -> (B, 2)
q_halt_logits = q_logits[..., 0]      # (B,)
q_continue_logits = q_logits[..., 1]  # (B,)
```

`q_head` is a learned linear projection `(512 -> 2)`. In training mode, if `q_halt_logits > 0`,
the model can stop early. In eval mode, this is ignored and the model always runs 16 steps.

Note: `q_continue_logits` is never used - only `q_halt_logits` matters. The projection could
be simplified to `(512 -> 1)`. The extra output is likely a leftover from an earlier design.

The puzzle context acts as a "summary" token - through attention it sees all 81 cells and
learns to judge "am I done thinking?"

### Prediction

```python
preds = torch.argmax(logits, dim=-1)  # (B, 81)
```

---

## Compute Summary

Per forward pass:

```
Outer steps:     16
H-cycles each:   3
L-cycles each:   6
Reasoner layers: 2

Total reasoner calls per outer step: 3 * (6 + 1) = 21
Total reasoner calls overall:        16 * 21 = 336
Total transformer blocks:            336 * 2 = 672
```

---

## Why It Works

1. **Iterative refinement** - Hard Sudoku puzzles require multiple deduction steps. The model gets 16 chances to refine its answer.

2. **Two-level thinking** - `z_H` captures global structure, `z_L` handles details. They inform each other.

3. **Persistent state** - The scratchpad carries forward between iterations. The model can build on previous reasoning.

4. **Full attention** - Every cell can attend to every other cell. Sudoku constraints (rows, columns, boxes) require global awareness.
