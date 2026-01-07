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
from src.model.loss import compute_act_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sudoku TRM model")

    parser.add_argument("--global_batch_size", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.95])

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=1000)

    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=100)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--num_workers", type=int, default=4)

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


def get_base_model(model: nn.Module) -> nn.Module:
    """Get the base model, unwrapping DDP/compiled wrappers."""
    if hasattr(model, "module"):
        return model.module
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    world_size: int = 1,
    max_steps: int = 16,
) -> dict[str, float]:
    """Evaluate model on dataloader using ACT loop."""
    model.eval()
    base_model = get_base_model(model)

    total_tokens = 0
    correct_tokens = 0
    total_puzzles = 0
    correct_puzzles = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        carry = base_model.initial_carry(inputs)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(max_steps):
                carry, outputs = model(carry, inputs)
                if carry.halted.all():
                    break

        preds = outputs["logits"].argmax(dim=-1)

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

    batch_size = args.global_batch_size // world_size

    torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if is_main:
        print(f"Training on {world_size} GPU(s)")
        print(f"Device: {device}")
        print(f"Global batch size: {args.global_batch_size}")
        print(f"Batch size per GPU: {batch_size}")

    # Training and eval datasets
    train_dataset = SudokuDataset(data_dir="data", split="train")
    eval_dataset = SudokuDataset(data_dir="data", split="test")

    if is_main:
        print(f"Train dataset: {len(train_dataset)} examples")
        print(f"Eval dataset: {len(eval_dataset)} examples")

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

    def make_eval_sampler(dataset):
        return DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        sampler=make_eval_sampler(eval_dataset),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = SudokuModel().to(device)

    if not args.no_compile:
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
    base_model = get_base_model(model)

    # Persistent carry across batches (key architectural change)
    carry = None

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            current_batch = epoch * steps_per_epoch + batch_idx
            if current_batch < start_step:
                continue

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Initialize carry if None (first batch)
            if carry is None:
                carry = base_model.initial_carry(inputs)

            # ONE ACT step per batch (matching trm_quest)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                carry, outputs = model(carry, inputs)

            # Compute loss for ALL sequences, metrics for halted only
            loss, metrics = compute_act_loss(
                outputs["logits"],
                labels,
                outputs["q_halt_logits"],
                carry.halted,
                carry.steps,
            )

            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()

            global_step += 1

            if is_main and global_step % args.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                count = metrics["count"].item()
                exact_acc = metrics["exact_accuracy"].item() / max(count, 1)
                avg_steps = metrics["steps"].item() / max(count, 1)
                print(
                    f"Step {global_step} | Epoch {epoch+1}/{args.epochs} | "
                    f"Loss: {loss.item():.4f} | ExactAcc: {exact_acc:.4f} | "
                    f"AvgSteps: {avg_steps:.1f} | Halted: {count} | LR: {lr:.2e}"
                )

            if global_step % args.eval_interval == 0:
                eval_metrics = evaluate(model, eval_loader, device, world_size)

                if is_main:
                    print(
                        f"Step {global_step} | "
                        f"Eval: {eval_metrics['puzzle_accuracy']:.4f}"
                    )

                is_best = eval_metrics["puzzle_accuracy"] > best_accuracy
                if is_best:
                    best_accuracy = eval_metrics["puzzle_accuracy"]

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
