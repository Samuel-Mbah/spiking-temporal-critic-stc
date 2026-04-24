#!/bin/bash
#SBATCH --job-name=snn_exp
#SBATCH --array=1-5                  # Run seeds 1-5
#SBATCH --time=72:00:00
#SBATCH --partition=bigbatch
#SBATCH --nodelist=mscluster[42,44,72,73,75-82,84-89]
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out   # Log naming: jobname_jobID_arrayIndex

# Usage: sbatch submit_array.slurm <script_name> <config_path> <exp_key> [env_active]
SCRIPT_NAME=$1       # e.g. ann_baseline.py
CONFIG_PATH=$2       # e.g. configs/cartpole/ann_baseline.yaml
EXP_KEY=$3           # e.g. cartpole/ann_baseline  (used to build log path)
ENV_ACTIVE=${4:-""}  # optional: "true" or "false" — only passed for tmaze

source ~/.bashrc
conda activate SNNAC

SEED=$SLURM_ARRAY_TASK_ID
LOG_DIR="results/logs/${EXP_KEY}/seed_${SEED}"

echo "Starting $EXP_KEY | Seed $SEED | active=${ENV_ACTIVE:-N/A}"

CMD=(python "experiments/${SCRIPT_NAME}"
    --config "$CONFIG_PATH"
    --seed "$SEED"
    --run-name "seed_${SEED}"
    --log-dir "$LOG_DIR")

if [[ -n "$ENV_ACTIVE" ]]; then
    CMD+=(--env-active "$ENV_ACTIVE")
fi

"${CMD[@]}"
