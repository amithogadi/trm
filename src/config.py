from dataclasses import dataclass


@dataclass
class TRMConfig:
    """Configuration for Tiny Recursive Model."""

    # Model architecture
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 2
    expansion: float = 4.0
    vocab_size: int = 11  # 0-9 + blank
    seq_len: int = 81  # 9x9 sudoku

    # Reasoning cycles
    H_cycles: int = 3
    L_cycles: int = 6
    halt_max_steps: int = 16

    # Training
    batch_size: int = 768
    lr: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    total_steps: int = 100000

    # Misc
    norm_eps: float = 1e-5
    rope_base: float = 10000.0
    halt_exploration_prob: float = 0.1
