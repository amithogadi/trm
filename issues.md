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
| H_init/L_init shape (512,) broadcast + non-trainable buffer | ✅ Fixed |

---

## Remaining Issues

### 1. Number of Puzzle Context Tokens

**Original config:** `puzzle_emb_len: 16` → 16 context tokens prepended

**Ours:** 1 token prepended

Total seq length:
- Original: 81 + 16 = **97**
- Ours: 81 + 1 = **82**

**Fix:** Add `puzzle_emb_len` config, default to 16

---

### 2. Puzzle Context Initialization

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
| 1 vs 16 puzzle context tokens | **High** | Medium |
| Puzzle context random vs zero init | **Medium** | Easy |
