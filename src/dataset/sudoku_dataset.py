from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SudokuDataset(Dataset):
    def __init__(self, data_dir: str = "data", split: str = "train"):
        data_path = Path(data_dir) / split
        self.inputs = np.load(data_path / "inputs.npy", mmap_mode="r")
        self.labels = np.load(data_path / "labels.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.from_numpy(self.inputs[idx].copy()).long()
        labels = torch.from_numpy(self.labels[idx].copy()).long()
        return inputs, labels
