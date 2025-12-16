import argparse
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.dataset.sudoku_dataset import SudokuDataset
from src.model.sudoku import SudokuModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sudoku TRM model")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.95])

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=1000)

    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=100)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--data_dir", type=str, default="data")

    return parser.parse_args()


def setup_ddp() -> tuple[int, int, torch.device]:
    """Initialize DDP and return (rank, world_size, device)."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def cleanup_ddp():
    """Clean up DDP resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Cosine schedule with linear warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    best_accuracy: float,
    checkpoint_dir: str,
    is_best: bool = False,
):
    """Save training checkpoint (only on rank 0)."""
    model_to_save = model.module if hasattr(model, "module") else model

    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "best_accuracy": best_accuracy,
    }

    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
    torch.save(checkpoint, path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pt")
        torch.save(checkpoint, best_path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    device: torch.device = None,
) -> dict:
    """Load checkpoint and return metadata."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "step": checkpoint["step"],
        "best_accuracy": checkpoint["best_accuracy"],
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    world_size: int = 1,
) -> dict[str, float]:
    """Evaluate model on dataloader."""
    model.eval()

    total_tokens = 0
    correct_tokens = 0
    total_puzzles = 0
    correct_puzzles = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits, _ = model(inputs)
        preds = logits.argmax(dim=-1)

        correct_tokens += (preds == labels).sum().item()
        total_tokens += labels.numel()

        puzzle_correct = (preds == labels).all(dim=-1)
        correct_puzzles += puzzle_correct.sum().item()
        total_puzzles += labels.size(0)

    if world_size > 1:
        stats = torch.tensor(
            [correct_tokens, total_tokens, correct_puzzles, total_puzzles],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        correct_tokens, total_tokens, correct_puzzles, total_puzzles = stats.tolist()

    model.train()

    return {
        "token_accuracy": correct_tokens / total_tokens,
        "puzzle_accuracy": correct_puzzles / total_puzzles,
    }


def main():
    args = parse_args()

    rank, world_size, device = setup_ddp()
    is_main = rank == 0

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if is_main:
        print(f"Training on {world_size} GPU(s)")
        print(f"Device: {device}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")

    train_dataset = SudokuDataset(data_dir=args.data_dir, split="train")
    test_dataset = SudokuDataset(data_dir=args.data_dir, split="test")

    if is_main:
        print(f"Train dataset: {len(train_dataset)} examples")
        print(f"Test dataset: {len(test_dataset)} examples")

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        if world_size > 1
        else None
    )

    test_sampler = (
        DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        if world_size > 1
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = SudokuModel().to(device)

    if args.compile:
        if is_main:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, total_steps)

    if is_main:
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total steps: {total_steps}")
        print(f"Warmup steps: {args.warmup_steps}")

    start_step = 0
    best_accuracy = 0.0
    if args.resume_from:
        metadata = load_checkpoint(args.resume_from, model, optimizer, scheduler, device)
        start_step = metadata["step"]
        best_accuracy = metadata["best_accuracy"]
        if is_main:
            print(f"Resumed from step {start_step}, best_accuracy={best_accuracy:.4f}")

    model.train()
    global_step = start_step

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            current_batch = epoch * steps_per_epoch + batch_idx
            if current_batch < start_step:
                continue

            inputs = inputs.to(device)
            labels = labels.to(device)

            logits, loss = model(inputs, labels)

            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()

            global_step += 1

            if is_main and global_step % args.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"Step {global_step} | Epoch {epoch+1}/{args.epochs} | Loss: {loss.item():.4f} | LR: {lr:.2e}")

            if global_step % args.eval_interval == 0:
                metrics = evaluate(model, test_loader, device, world_size)

                if is_main:
                    print(
                        f"Step {global_step} | Eval | "
                        f"Token Acc: {metrics['token_accuracy']:.4f} | "
                        f"Puzzle Acc: {metrics['puzzle_accuracy']:.4f}"
                    )

                is_best = metrics["puzzle_accuracy"] > best_accuracy
                if is_best:
                    best_accuracy = metrics["puzzle_accuracy"]

                if is_main:
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        global_step,
                        best_accuracy,
                        args.checkpoint_dir,
                        is_best,
                    )

            if is_main and global_step % args.save_every == 0 and global_step % args.eval_interval != 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    best_accuracy,
                    args.checkpoint_dir,
                )

    cleanup_ddp()
    if is_main:
        print(f"Training complete. Best puzzle accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
