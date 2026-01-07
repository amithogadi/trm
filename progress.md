# Progress Log

- EXPLORE: Verified context_len=1 is correct for sudoku (trm_quest/sudoku/trm.py uses puzzle_emb_len=1, not 16). Updated issues.md.
- EXPLORE: Verified q_head output=1 is fine (sudoku/trm.py outputs 2 but only uses halt_logit, q_continue unused). Updated issues.md.
- EXPLORE: Found CRITICAL issue - trm_quest runs 16 optimizer steps per batch (one per ACT iteration with carry), ours runs 1. Updated issues.md.
- EXPLORE: Created todo.md with 5-step plan to refactor ACT loop architecture to match trm_quest.
- IMPLEMENT: Added TRMInnerCarry and TRMCarry dataclasses to src/model/trm.py (Step 2 of todo.md).
- DEEP REVIEW: Verified dataclass implementation correct, all 11 tests pass, SudokuModel forward works.
- IMPLEMENT: Added empty_carry() and reset_carry() methods to TRM class (partial Step 1).
- TEST: Added tests/test_trm_carry.py with 4 tests for carry dataclasses and methods. All 15 tests pass.
- IMPLEMENT: Added step(carry, input_emb) method to TRM - runs ONE ACT iteration, returns (new_carry, logits, q_halt).
- TEST: Added 3 tests for step() method (output shapes, carry detached, q_halt initial value). All 18 tests pass.
