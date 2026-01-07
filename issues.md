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
| Token embedding variance (init std=1/sqrt(512), scale by sqrt(512)) | ✅ |
| Gradient flow between ACT steps (detach z_H/z_L) | ✅ |
| H_init/L_init shape (512,) broadcast + non-trainable buffer | ✅ |
| Context token init (zero) | ✅ |
| Linear layer init (trunc_normal) | ✅ |
| Loss function (stablemax) | ✅ |
| Context token count (1 for sudoku) | ✅ |
| q_head output dim (1 is fine - sudoku uses only halt_logit) | ✅ |

---

## Resolved Issues

### 2. Training Loop Architecture ✅ FIXED

**Original (trm_quest):**
- Carry persists across training steps
- Each `model.forward()` = ONE ACT iteration
- Loss computed for ALL sequences at each step (not just halted)
- `.backward()` called after each iteration

**Ours (NOW MATCHES):**
- ✅ `TRM.forward(carry, input_emb)` runs ONE ACT step
- ✅ Carry persists across batches in `train.py`
- ✅ `compute_act_loss()` computes loss for ALL sequences
- ✅ `.backward()` called after each batch

**Verified:** Training smoke test shows loss decreasing, halting mechanism working.

---

## Summary

| Issue | Impact | Status |
|-------|--------|--------|
| 1. Context token count (1 for sudoku) | N/A | ✅ Verified correct |
| 2. Training loop architecture | CRITICAL | ✅ FIXED |
| 3. Loss normalization | MEDIUM | ✅ Subsumed by #2 |
| 4. q_head output dim (1 vs 2) | N/A | ✅ Verified (1 is fine) |
