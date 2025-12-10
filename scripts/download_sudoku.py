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


@cli.command(singleton=True)
def main(config: Config):
    download_and_process("train", config.force)
    download_and_process("test", config.force)


if __name__ == "__main__":
    cli()
