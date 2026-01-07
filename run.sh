#!/bin/bash
#SBATCH -p iiser_gpu                # Partition to submit to
#SBATCH -w gpu1                     # Specific node if needed
#SBATCH --gpus=2                    # Number of GPUs
#SBATCH --cpus-per-task=8           # Number of CPU cores
#SBATCH --mem=64G                   # Memory
#SBATCH --time=48:00:00             # Time limit (2 days)
#SBATCH --output=output_%j.log

# Load the modules
module load NGC_container
module load python3.13.7

# Setup venv (inherits torch from system python 3.13)
uv venv --python 3.13 --system-site-packages
uv sync --inexact
source .venv/bin/activate

# Download data and create augmented dataset
python scripts/download_sudoku.py

# Run training with auto-detected GPU count
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
torchrun --nproc_per_node=$NUM_GPUS train.py