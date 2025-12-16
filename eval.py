import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.sudoku_dataset import SudokuDataset
from src.model.sudoku import SudokuModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Sudoku TRM model")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_dir", type=str, default="data/train_1k")
    parser.add_argument("--split", type=str, default="eval", choices=["train", "eval"])
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SudokuModel().to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    step = checkpoint.get("step", "unknown")
    print(f"Loaded checkpoint from step {step}")

    dataset = SudokuDataset(data_dir=args.data_dir, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Evaluating on {len(dataset)} {args.split} examples...")

    total_tokens = 0
    correct_tokens = 0
    total_puzzles = 0
    correct_puzzles = 0

    for inputs, labels in tqdm(loader, desc="Evaluating"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits, _ = model(inputs)
        preds = logits.argmax(dim=-1)

        correct_tokens += (preds == labels).sum().item()
        total_tokens += labels.numel()

        puzzle_correct = (preds == labels).all(dim=-1)
        correct_puzzles += puzzle_correct.sum().item()
        total_puzzles += labels.size(0)

    token_accuracy = correct_tokens / total_tokens
    puzzle_accuracy = correct_puzzles / total_puzzles

    print()
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Split:           {args.split}")
    print(f"Total puzzles:   {total_puzzles}")
    print(f"Token Accuracy:  {token_accuracy:.4f} ({correct_tokens}/{total_tokens})")
    print(f"Puzzle Accuracy: {puzzle_accuracy:.4f} ({correct_puzzles}/{total_puzzles})")
    print("=" * 50)


if __name__ == "__main__":
    main()
