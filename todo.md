# TRM Implementation TODO

## Goal
Implement the complete TRM (Tiny Recursive Model) with clean code that reproduces the original `trm_quest` results for Sudoku.

---

## Already Done
- [x] `src/model/layers/attention.py` - Attention with RoPE
- [x] `src/model/layers/rotary.py` - RoPE, PositionEmbedding base class
- [x] `src/model/layers/rmsnorm.py` - RMSNorm (no learnable weight)
- [x] `src/model/layers/swiglu.py` - SwiGLU with fused projection
- [x] `src/model/reasoner.py` - Reasoner (stack of transformer blocks)
- [x] `src/dataset/sudoku_dataset.py` - Basic Sudoku dataset loader
- [x] `scripts/download_sudoku.py` - Data download script

---

## TODO

### 1. Configuration (`src/config.py`)
- [ ] Create `TRMConfig` dataclass with:
  - Model: hidden_size=512, num_heads=8, num_layers=2, expansion=4.0, vocab_size=11, seq_len=81
  - Reasoning: H_cycles=3, L_cycles=6, halt_max_steps=16
  - Training: batch_size=768, lr=1e-4, weight_decay=0.1, warmup_steps=2000, total_steps=100000
  - Misc: norm_eps=1e-5, rope_base=10000.0

### 2. Main TRM Model (`src/model/trm.py`)
- [ ] Token embedding (vocab_size=11 → hidden_size=512) with embed_scale=sqrt(512)
- [ ] Puzzle context token (learned, prepended to sequence)
- [ ] H_init, L_init (learned initial states, truncated normal std=1.0)
- [ ] Reasoner instance (reused for all cycles)
- [ ] lm_head (Linear 512→11) for digit prediction
- [ ] q_head (Linear 512→2) for halting, init weight=0, bias=-5
- [ ] Inner forward with gradient truncation:
  - H_cycles-1 iterations with torch.no_grad()
  - Last H_cycle with gradients
- [ ] ACT logic:
  - Carry state (z_H, z_L, steps, halted)
  - Reset carry on halted sequences
  - Halt decision: q_halt_logits > 0 OR steps >= 16
  - Exploration: 10% chance for random early halt

### 3. Dataset Enhancement (`src/dataset/sudoku_dataset.py`)
- [ ] Add collate function for batching
- [ ] Handle padding with ignore_index=-100

### 4. Training Script (`train.py`)
- [ ] Load config and create model
- [ ] AdamW optimizer (betas=0.9/0.95, weight_decay=0.1)
- [ ] Cosine warmup scheduler (2000 steps warmup, min_ratio=1.0)
- [ ] Training loop with ACT:
  - Initialize carry
  - Run up to halt_max_steps
  - Accumulate loss (CrossEntropy + q_halt BCE)
  - Backward and optimize
- [ ] Logging: loss, token_accuracy, exact_accuracy
- [ ] Periodic eval on test set
- [ ] Checkpoint saving

### 5. Evaluation Script (`eval.py`)
- [ ] Load model from checkpoint
- [ ] Run inference (fixed 16 steps in eval mode)
- [ ] Compute token_accuracy and exact_accuracy
- [ ] Print results

### 6. Update Exports (`src/model/__init__.py`)
- [ ] Export TRM class

---

## Simplifications vs Original
- Skip sparse puzzle embeddings (not needed for Sudoku-only)
- Skip distributed training (single GPU)
- Use standard nn.Linear/nn.Embedding (not CastedLinear)
- Single evaluator (just accuracy metrics)

## Keep
- Full ACT with q_head halting logic
- Gradient truncation (only last H_cycle has gradients)

---

## Expected Results
- ~95%+ exact accuracy after full training
- Converges within ~100k steps
