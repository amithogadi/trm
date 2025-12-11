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

---

## Open Issues

### 1. Context Token Count

**Original:** `puzzle_emb_len: 16` → 16 context tokens prepended

**Ours:** `context_len: 1` → 1 context token prepended

**Impact:** HIGH - Total sequence length differs (97 vs 82), affects RoPE frequencies

**Fix:** Change default `context_len=16`

---

### 2. Context Token Initialization

**Original:**
```python
self.puzzle_emb = CastedSparseEmbedding(..., init_std=0, ...)  # ZERO init
```

**Ours:**
```python
self.context_tokens = nn.Parameter(torch.zeros(1, context_len, input_dim))
trunc_normal_init_(self.context_tokens, std=1.0)  # Random init
```

**Impact:** HIGH - Different starting point for learned context

**Fix:** Remove `trunc_normal_init_` call, keep zeros

---

### 3. Linear Layer Initialization

**Original (layers.py):**
```python
class CastedLinear(nn.Linear):
    def __init__(...):
        nn.init.trunc_normal_(self.weight, std=init_std)  # init_std = 1/sqrt(in_features)
```

**Ours:** PyTorch default uniform initialization

**Impact:** HIGH - Different weight distribution affects training dynamics

**Fix:** Initialize all Linear layers (lm_head, q_head, Reasoner MLP/attn projections) with `trunc_normal_(std=1/sqrt(in_features))`

---

### 4. Loss Function

**Original (losses.py):**
```python
def stablemax_cross_entropy(logits, labels, gamma=4):
    logits_norm = logits - logits.max(dim=-1, keepdim=True).values  # subtract max
    probs = torch.pow(gamma, logits_norm)  # gamma^x instead of exp(x)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    loss = -torch.log(probs.gather(-1, labels.unsqueeze(-1)) + 1e-12)
```

**Ours:**
```python
F.cross_entropy(logits, labels)  # Standard softmax
```

**Impact:** HIGH - Different gradient behavior, stablemax more numerically stable

**Fix:** Implement `stablemax_cross_entropy` with `gamma=4`

---

### 5. Loss Scope - Which Sequences

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

### 6. Loss Normalization

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

### 7. q_head Output Dimension

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
| 2. Context token init (random vs zero) | HIGH | Open |
| 3. Linear layer init (uniform vs trunc_normal) | HIGH | Open |
| 4. Loss function (softmax vs stablemax) | HIGH | Open |
| 5. Loss scope (halted-only vs all sequences) | HIGH | Open |
| 6. Loss normalization | MEDIUM | Open |
| 7. q_head output dim (1 vs 2) | MEDIUM | Open |
