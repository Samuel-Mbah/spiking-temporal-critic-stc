#!/bin/bash
#SBATCH --job-name=ablation_window_sizes
#SBATCH --time=72:00:00
#SBATCH --partition=bigbatch
#SBATCH --output=logs/ablation_window_sizes_%j.out

source ~/.bashrc
conda activate snn

echo "Starting ablation_window_sizes | node=$(hostname)"

python scripts/ablation_window_sizes.py --seeds 1 2 3
