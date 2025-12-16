import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import os

import numpy as np
from argdantic import ArgParser
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from tqdm import tqdm


REPO = "sapientinc/sudoku-extreme"
OUTPUT_DIR = "data"

cli = ArgParser()


class Config(BaseModel):
    force: bool = False
    subsample_size: int = 1000
    num_aug: int = 1000
    seed: int = 42


def data_exists(set_name: str) -> bool:
    save_dir = os.path.join(OUTPUT_DIR, set_name)
    inputs_path = os.path.join(save_dir, "inputs.npy")
    labels_path = os.path.join(save_dir, "labels.npy")

    if not os.path.exists(inputs_path) or not os.path.exists(labels_path):
        return False

    inputs = np.load(inputs_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")

    return len(inputs) > 0 and len(inputs) == len(labels)


def download_and_process(set_name: str, force: bool):
    if not force and data_exists(set_name):
        print(f"Skipping {set_name}: data already exists with correct size. Use --force to re-download.")
        return

    csv_path = hf_hub_download(REPO, f"{set_name}.csv", repo_type="dataset")

    inputs, labels = [], []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header: source, q, a, rating
        for source, q, a, rating in tqdm(reader, desc=f"Processing {set_name}"):
            inputs.append([int(c) if c != "." else 0 for c in q])
            labels.append([int(c) for c in a])

    save_dir = os.path.join(OUTPUT_DIR, set_name)
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "inputs.npy"), np.array(inputs, dtype=np.uint8))
    np.save(os.path.join(save_dir, "labels.npy"), np.array(labels, dtype=np.uint8))

    print(f"Saved {len(inputs)} examples to {save_dir}/")


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray, rng: np.random.Generator):
    """Apply symmetry-preserving transformations to a Sudoku puzzle."""
    digit_map = np.concatenate([[0], rng.permutation(np.arange(1, 10))])
    transpose_flag = rng.random() < 0.5
    bands = rng.permutation(3)
    row_perm = np.concatenate([b * 3 + rng.permutation(3) for b in bands])
    stacks = rng.permutation(3)
    col_perm = np.concatenate([s * 3 + rng.permutation(3) for s in stacks])
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        x_2d = x.reshape(9, 9)
        if transpose_flag:
            x_2d = x_2d.T
        new_board = x_2d.flatten()[mapping]
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


def save_split(output_dir: str, name: str, inputs: np.ndarray, labels: np.ndarray):
    """Save a dataset split to output_dir/name/."""
    split_dir = os.path.join(output_dir, name)
    os.makedirs(split_dir, exist_ok=True)
    np.save(os.path.join(split_dir, "inputs.npy"), inputs)
    np.save(os.path.join(split_dir, "labels.npy"), labels)


def create_augmented_dataset(config: Config, force: bool):
    output_dir = os.path.join(OUTPUT_DIR, "train_1k")
    if not force and data_exists("train_1k"):
        print("Skipping train_1k: augmented data already exists. Use --force to regenerate.")
        return

    rng = np.random.default_rng(config.seed)

    inputs = np.load(os.path.join(OUTPUT_DIR, "train", "inputs.npy"), mmap_mode="r")
    labels = np.load(os.path.join(OUTPUT_DIR, "train", "labels.npy"), mmap_mode="r")

    # Sample 2k distinct puzzles: 1k for train, 1k for eval
    all_indices = rng.choice(len(inputs), size=config.subsample_size * 2, replace=False)
    train_indices = np.sort(all_indices[:config.subsample_size])
    eval_indices = np.sort(all_indices[config.subsample_size:])

    print(f"Creating dataset: {config.subsample_size} train + {config.subsample_size} eval puzzles")
    print(f"Augmenting train: {config.subsample_size} x {config.num_aug + 1} = {config.subsample_size * (config.num_aug + 1)} examples")

    # Collect non-augmented train and eval samples
    train_inputs = np.stack([inputs[i].copy() for i in train_indices])
    train_labels = np.stack([labels[i].copy() for i in train_indices])
    eval_inputs = np.stack([inputs[i].copy() for i in eval_indices])
    eval_labels = np.stack([labels[i].copy() for i in eval_indices])

    # Create augmented training set
    aug_inputs, aug_labels = [], []
    for i in tqdm(range(len(train_indices)), desc="Augmenting train"):
        orig_inp, orig_out = train_inputs[i], train_labels[i]
        aug_inputs.append(orig_inp)
        aug_labels.append(orig_out)
        for _ in range(config.num_aug):
            aug_inp, aug_out = shuffle_sudoku(orig_inp, orig_out, rng)
            aug_inputs.append(aug_inp)
            aug_labels.append(aug_out)

    # Save all splits
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "inputs.npy"), np.stack(aug_inputs))
    np.save(os.path.join(output_dir, "labels.npy"), np.stack(aug_labels))
    save_split(output_dir, "train", train_inputs, train_labels)
    save_split(output_dir, "eval", eval_inputs, eval_labels)

    print(f"Saved to {output_dir}/")
    print(f"  inputs.npy: {len(aug_inputs)} augmented examples (training)")
    print(f"  train/: {len(train_inputs)} examples (eval on train puzzles)")
    print(f"  eval/: {len(eval_inputs)} examples (eval on unseen puzzles)")


@cli.command(singleton=True)
def main(config: Config):
    download_and_process("train", config.force)
    download_and_process("test", config.force)
    create_augmented_dataset(config, config.force)


if __name__ == "__main__":
    cli()
