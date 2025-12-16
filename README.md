# Tiny Recursive Model for Sudoku

This is a reproduction of the Tiny Recursive Model (TRM) repository for Sudoku solving.

## Setup

Requires Python 3.13 with PyTorch already installed.

```bash
uv venv --python 3.13 --system-site-packages
uv sync
source .venv/bin/activate
```

## Download Data

Downloads full dataset and creates augmented `train_1k` (1k puzzles × 1000 augmentations):

```bash
python scripts/download_sudoku.py
```

## Training

```bash
python train.py

# Multi-GPU (same node)
torchrun --nproc_per_node=4 train.py
```

## Evaluation

```bash
python eval.py --checkpoint checkpoints/best.pt
```
