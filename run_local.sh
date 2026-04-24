#!/bin/bash
# Local equivalent of submit_all.sh — runs experiments sequentially without SLURM.
#
# Usage:
#   bash run_local.sh            → run all envs
#   bash run_local.sh cartpole   → cartpole only
#   bash run_local.sh poc        → poc only
#   bash run_local.sh tmaze      → tmaze active + passive
#   bash run_local.sh cartpole 1 → cartpole, seed 1 only (for quick testing)

SEEDS="${2:-1 2 3 4 5}"   # override with single seed for quick runs

declare -A MODELS
# format: script|config_file|dir_name  (dir_name matches the config log_dir short name)
MODELS["ann_baseline"]="ann_baseline.py|ann_baseline.yaml|ann"
MODELS["snn_actor_ann_critic"]="snn_actor_ann_critic.py|snn_actor_ann_critic.yaml|snn_ann_critic"
MODELS["snn_actor_snn_timing_critic"]="snn_actor_snn_timing_critic.py|snn_actor_snn_timing_critic.yaml|snn_timing_critic"
MODELS["ann2snn_actor"]="ann2snn_actor.py|ann2snn_actor.yaml|ann2snn_actor"
MODELS["ann2snn_both"]="ann2snn_both.py|ann2snn_both.yaml|ann2snn_both"

run_env() {
    local config_dir=$1
    local log_key=$2
    local env_active=${3:-""}

    for model_key in "${!MODELS[@]}"; do
        IFS='|' read -r script config_file dir_name <<< "${MODELS[$model_key]}"
        local config_path="configs/${config_dir}/${config_file}"

        for seed in $SEEDS; do
            local log_dir="results//neurips/${log_key}/${dir_name}/seed_${seed}"
            echo "Running: neurips/${log_key}/${dir_name} | seed=${seed} | active=${env_active:-N/A}"

            CMD=(python "experiments/${script}"
                --config "$config_path"
                --seed "$seed"
                --run-name "seed_${seed}"
                --log-dir "$log_dir")

            [[ -n "$env_active" ]] && CMD+=(--env-active "$env_active")

            "${CMD[@]}"

            if [[ $? -ne 0 ]]; then
                echo "ERROR: ${log_key}/${model_key} seed=${seed} failed — stopping."
                exit 1
            fi
        done
    done
}

case "${1:-all}" in
    cartpole) run_env "cartpole" "cartpole" ;;
    poc)      run_env "poc"      "poc"      ;;
    tmaze)
        run_env "tmaze" "tmaze_passive" "false"
        run_env "tmaze" "tmaze_active"  "true"
        ;;
    all)
        run_env "cartpole" "cartpole"
        run_env "poc"      "poc"
        run_env "tmaze"    "tmaze_passive" "false"
        run_env "tmaze"    "tmaze_active"  "true"
        ;;
    *)
        echo "Usage: $0 [all|cartpole|poc|tmaze] [seed]"
        exit 1
        ;;
esac
