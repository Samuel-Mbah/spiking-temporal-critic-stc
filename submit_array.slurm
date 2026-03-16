#!/bin/bash
#SBATCH --job-name=cartpole_exps
#SBATCH --array=1-5                  # Run seeds 1-5
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out   # Log naming: jobname_jobID_arrayIndex

# 1. Accept arguments passed from the dispatcher
# usage: sbatch submit_array.slurm <script_name> <config_name> <experiment_key>
SCRIPT_NAME=$1
CONFIG_NAME=$2
EXP_KEY=$3

source ~/.bashrc
conda activate snn

# 2. Derive variables
SEED=$SLURM_ARRAY_TASK_ID
# CRITICAL: This path must match what the plotting script expects!
LOG_DIR="results/logs/${EXP_KEY}/seed_${SEED}"

echo "Starting $EXP_KEY | Seed $SEED"

# 3. Run the python script
python "experiments/${SCRIPT_NAME}" \
    --config "configs/${CONFIG_NAME}" \
    --seed "$SEED" \
    --run-name "seed_${SEED}" \
    --log-dir "$LOG_DIR"