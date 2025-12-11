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

---

## Open Issues

### 1. Context Token Count

**Original:** `puzzle_emb_len: 16` → 16 context tokens prepended

**Ours:** `context_len: 1` → 1 context token prepended

**Impact:** HIGH - Total sequence length differs (97 vs 82), affects RoPE frequencies

**Fix:** Change default `context_len=16`

---

### 2. Loss Scope

**Original:**
```python
# Every step, compute loss for ALL sequences
lm_loss = stablemax_cross_entropy(logits, labels)  # All B sequences
q_loss = F.binary_cross_entropy_with_logits(q_halt, (logits.argmax(-1) == labels).all(-1).float())
```

**Ours:**
```python
# Only compute loss for NEWLY HALTED sequences
if newly_halted.any():
    halted_logits = logits[newly_halted]  # Only halted ones
    halted_labels = labels[newly_halted]
    lm_loss = F.cross_entropy(halted_logits, halted_labels)
```

**Impact:** HIGH - Different training signal: original trains all sequences every step, ours only at halt time

**Fix:** Compute loss for all sequences at each step, accumulate weighted by step

---

### 3. Loss Normalization

**Original:**
```python
# Weighted average across steps
total_loss = sum(step_losses) / halt_max_steps
# Each step loss is mean over all B sequences
```

**Ours:**
```python
# Sum losses only for halted, divide by num_halted
loss = (total_lm_loss + 0.5 * total_q_loss) / num_halted
```

**Impact:** MEDIUM - Different effective learning rate per sequence

**Fix:** Match original: average over steps, each step averages over batch

---

### 4. q_head Output Dimension

**Original:**
```python
self.q_head = CastedLinear(dim, 2, ...)  # 2 outputs: [halt_logit, continue_logit]
q_halt = q_head_out[:, 0] - q_head_out[:, 1]  # halt decision = halt - continue
```

**Ours:**
```python
self.q_head = nn.Linear(input_dim, 1, bias=True)  # 1 output
q_halt = self.q_head(z_H[:, 0]).squeeze(-1)  # Direct halt score
```

**Impact:** MEDIUM - Mathematically similar but different parameterization

**Fix:** Change q_head to output 2, compute `q_halt = out[:, 0] - out[:, 1]`

---

## Summary

| Issue | Impact | Status |
|-------|--------|--------|
| 1. Context token count (1 vs 16) | HIGH | Open |
| 2. Loss scope (halted-only vs all sequences) | HIGH | Open |
| 3. Loss normalization | MEDIUM | Open |
| 4. q_head output dim (1 vs 2) | MEDIUM | Open |
