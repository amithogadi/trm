import pytest
import torch

from src.sudoku_dataset import SudokuDataset


@pytest.fixture(params=["train", "test"])
def dataset(request):
    return SudokuDataset(split=request.param)


def test_dataset_not_empty(dataset):
    assert len(dataset) > 0


def test_item_shape(dataset):
    inputs, labels = dataset[0]
    assert inputs.shape == (81,)
    assert labels.shape == (81,)


def test_item_dtype(dataset):
    inputs, labels = dataset[0]
    assert inputs.dtype == torch.long
    assert labels.dtype == torch.long


def test_inputs_range(dataset):
    inputs, _ = dataset[0]
    assert inputs.min() >= 0
    assert inputs.max() <= 9


def test_labels_range(dataset):
    _, labels = dataset[0]
    assert labels.min() >= 1
    assert labels.max() <= 9


def test_missing_data_raises():
    with pytest.raises(FileNotFoundError):
        SudokuDataset(data_dir="nonexistent", split="train")
