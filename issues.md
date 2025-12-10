# Issues: Deviations from Original trm_quest

## Verified Correct (No Issues)

| Component | Status |
|-----------|--------|
| RMSNorm (`x * rsqrt(var + eps)`, float32, no weight) | ✅ |
| SwiGLU (`inter = round(exp*dim*2/3)` to mult of 256, fused gate_up) | ✅ |
| Reasoner structure (Post-LN: norm after residual) | ✅ |
| H/L cycle structure (`H_cycles-1` frozen, last with grad) | ✅ |
| RoPE math (freqs doubled then cos/sin - we repeat at apply, equivalent) | ✅ |
| rotate_half (`cat(-x2, x1)`) | ✅ |
| q_head init (weight=0, bias=-5) | ✅ |
| Token embedding variance (init std=1/sqrt(512), scale by sqrt(512)) | ✅ Fixed |
| Gradient flow between ACT steps (detach z_H/z_L) | ✅ Fixed |
| Loss only on halted sequences | ✅ Fixed |

---

## Remaining Issues

### 1. H_init/L_init Shape (Position-Dependent vs Uniform)

**Original:**
```python
self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size), std=1), ...)
# Shape: (512,) → broadcasts to (B, 97, 512)
```
ALL positions start with the **SAME** random vector.

**Ours:**
```python
init_shape = (1, config.seq_len + 1, config.hidden_size)  # (1, 82, 512)
self.H_init = nn.Parameter(trunc_normal_init_(torch.empty(init_shape), std=1.0))
```
EACH position has a **DIFFERENT** random vector.

**Fix:** Use shape `(hidden_size,)` and broadcast.

---

### 2. H_init/L_init Trainability

**Original:** `nn.Buffer` → **NOT trainable** (fixed after init)

**Ours:** `nn.Parameter` → **trainable**

**Fix:** Use `register_buffer` instead of `nn.Parameter`

---

### 3. Number of Puzzle Context Tokens

**Original config:** `puzzle_emb_len: 16` → 16 context tokens prepended

**Ours:** 1 token prepended

Total seq length:
- Original: 81 + 16 = **97**
- Ours: 81 + 1 = **82**

**Fix:** Add `puzzle_emb_len` config, default to 16

---

### 4. Puzzle Context Initialization

**Original:**
```python
self.puzzle_emb = CastedSparseEmbedding(..., init_std=0, ...)  # ZERO init
```

**Ours:**
```python
self.puzzle_context = nn.Parameter(torch.zeros(...))
trunc_normal_init_(self.puzzle_context, std=1.0)  # Random init
```

**Fix:** Initialize puzzle context to zero (and keep it learnable)

---

## Summary

| Issue | Impact | Difficulty |
|-------|--------|------------|
| H_init/L_init shape (per-position vs uniform) | **High** | Easy |
| H_init/L_init trainable vs fixed | **Medium** | Easy |
| 1 vs 16 puzzle context tokens | **High** | Medium |
| Puzzle context random vs zero init | **Medium** | Easy |
