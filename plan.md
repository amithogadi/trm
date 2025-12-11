# Plan: Refactor TRM - Explicit Parameters, Move lm_head to SudokuModel

## Goal
- TRM: General reasoning module with explicit parameters (no config), no vocab_size
- SudokuModel: Task-specific - has embedder + lm_head, calls TRM

## What's Where

| Component | Location | Reason |
|-----------|----------|--------|
| Token embedding | SudokuModel | vocab_size is task-specific |
| Context tokens | SudokuModel | Task-specific padding |
| lm_head | SudokuModel | vocab_size is task-specific |
| H_init, L_init | TRM | General reasoning state |
| q_head | TRM | Part of ACT halting mechanism |
| ACT loop + halting | TRM | General adaptive computation |
| Reasoner | TRM | General transformer reasoning |

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/model/trm.py` | Explicit params, remove lm_head, accept output_head callable |
| `src/model/sudoku.py` | Add lm_head, pass to TRM |

---

## TRM (`src/model/trm.py`)

**Changes:**
- Replace `config: TRMConfig` with explicit parameters
- Remove `lm_head` (moves to SudokuModel)
- Keep `q_head` (for halting decision)
- Accept `output_head` callable for computing logits during loss
- Return `(logits, loss)` same as before

```python
class TRM(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        seq_len: int = 128,  # max seq_len for RoPE
        num_heads: int = 8,
        num_layers: int = 2,
        expansion: float = 4.0,
        norm_eps: float = 1e-5,
        H_cycles: int = 3,
        L_cycles: int = 6,
        halt_max_steps: int = 16,
    ):
        super().__init__()
        # H_init, L_init buffers
        # Reasoner
        # q_head (for halting)
        # NO lm_head - passed at forward time

    def forward(
        self,
        input_emb: torch.Tensor,
        labels: torch.Tensor | None = None,
        output_head: nn.Module | None = None,
        num_context: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            input_emb: (B, total_seq, hidden_size)
            labels: (B, num_tokens) targets
            output_head: callable that maps hidden -> logits (e.g., lm_head)
            num_context: context tokens to skip

        Returns:
            logits: (B, num_tokens, vocab_size)
            loss: scalar if labels provided
        """
        # ACT loop - at each step:
        #   hidden = z_H[:, num_context:]
        #   logits = output_head(hidden)
        #   compute loss for halted sequences (same as current)
```

**Key:** `output_head` is passed in, not owned by TRM. No vocab_size in TRM.

---

## SudokuModel (`src/model/sudoku.py`)

**Changes:**
- Has `embedder` and `lm_head`
- Passes `lm_head` to TRM for loss computation
- TRM does all the work, SudokuModel just wires things together

```python
class SudokuModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 11,
        hidden_size: int = 512,
        num_context_tokens: int = 1,
        # ... TRM params passed through
    ):
        super().__init__()
        self.num_context_tokens = num_context_tokens
        self.embedder = InputEmbedder(vocab_size, hidden_size, num_context_tokens)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.trm = TRM(hidden_size=hidden_size, ...)

    def forward(self, inputs, labels=None):
        input_emb = self.embedder(inputs)
        # Pass lm_head to TRM - it computes logits and loss internally
        logits, loss = self.trm(
            input_emb,
            labels=labels,
            output_head=self.lm_head,
            num_context=self.num_context_tokens,
        )
        return logits, loss
```

---

## Loss Computation (Original Behavior - MUST PRESERVE)

**Original loss has two components:**
1. **CE loss** - `F.cross_entropy(logits, labels)` per halted sequence
2. **q_halt loss** - `F.binary_cross_entropy_with_logits(q_halt, seq_correct)` - trains halting head

**Per-step behavior:**
- Each ACT step, check which sequences halt (`q_halt > 0` or last step)
- Compute loss ONLY for newly halted sequences
- Final: `(total_lm_loss + 0.5 * total_q_loss) / num_halted`

**Solution:** Pass `output_head` callable to TRM.forward():

```python
# TRM.forward - at each ACT step:
hidden = z_H[:, num_context:]
logits = output_head(hidden) if output_head else None
# Use logits for loss computation (same as current)

# SudokuModel calls:
logits, loss = self.trm(input_emb, labels, output_head=self.lm_head, num_context=...)
```

This preserves:
- TRM doesn't own lm_head (no vocab_size in TRM)
- TRM computes per-step loss using provided output_head
- Original behavior exactly preserved

---

## Summary

```
SudokuModel (task-specific)
├── embedder: tokens → embeddings
├── lm_head: hidden → logits
└── trm: embeddings → logits (uses lm_head internally for loss)

TRM (general, explicit params)
├── H_init, L_init
├── reasoner
├── q_head (halting)
└── ACT loop
```

TRM parameters:
- hidden_size, seq_len, num_heads, num_layers, expansion, norm_eps
- H_cycles, L_cycles, halt_max_steps

No vocab_size in TRM - it's purely a reasoning module.
