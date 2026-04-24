#!/bin/bash
# Dispatch all SLURM array jobs for NeurIPS experiments.
#
# Usage:
#   bash submit_all.sh            → submit all envs (cartpole + poc + tmaze active + passive)
#   bash submit_all.sh cartpole   → cartpole only
#   bash submit_all.sh poc        → poc only
#   bash submit_all.sh tmaze      → tmaze active + passive

# Model registry: model_key → script|config_filename
declare -A MODELS
MODELS["ann_baseline"]="ann_baseline.py|ann_baseline.yaml"
MODELS["snn_actor_ann_critic"]="snn_actor_ann_critic.py|snn_actor_ann_critic.yaml"
MODELS["snn_actor_snn_timing_critic"]="snn_actor_snn_timing_critic.py|snn_actor_snn_timing_critic.yaml"
MODELS["ann2snn_actor"]="ann2snn_actor.py|ann2snn_actor.yaml"
MODELS["ann2snn_both"]="ann2snn_both.py|ann2snn_both.yaml"

mkdir -p logs   # SLURM writes output here

# submit_env <config_subdir> <log_key> [env_active]
#   config_subdir : subfolder under configs/       (cartpole | poc | tmaze)
#   log_key       : prefix under results/logs/     (cartpole | poc | tmaze_passive | tmaze_active)
#   env_active    : "true" | "false" | ""          (empty = don't pass --env-active)
submit_env() {
    local config_dir=$1
    local log_key=$2
    local env_active=${3:-""}

    for model_key in "${!MODELS[@]}"; do
        IFS='|' read -r script config_file <<< "${MODELS[$model_key]}"
        local config_path="configs/${config_dir}/${config_file}"
        local exp_key="${log_key}/${model_key}"

        echo "Submitting: $exp_key (active=${env_active:-N/A})"
        sbatch --job-name="${log_key}_${model_key}" \
               submit_array.slurm \
               "$script" "$config_path" "$exp_key" "$env_active"
    done
}

case "${1:-all}" in
    cartpole)
        submit_env "cartpole" "cartpole"
        ;;
    poc)
        submit_env "poc" "poc"
        ;;
    tmaze)
        submit_env "tmaze" "tmaze_passive" "false"
        submit_env "tmaze" "tmaze_active"  "true"
        ;;
    all)
        submit_env "cartpole" "cartpole"
        submit_env "poc"      "poc"
        submit_env "tmaze"    "tmaze_passive" "false"
        submit_env "tmaze"    "tmaze_active"  "true"
        ;;
    *)
        echo "Usage: $0 [all|cartpole|poc|tmaze]"
        exit 1
        ;;
esac
