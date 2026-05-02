#!/bin/bash
# Dispatch all SLURM array jobs for NeurIPS experiments.
#
# Usage:
#   bash submit_all.sh            → submit all envs (cartpole + poc + tmaze active + passive)
#   bash submit_all.sh cartpole   → cartpole only
#   bash submit_all.sh poc        → poc only
#   bash submit_all.sh tmaze      → tmaze active + passive

# GPU nodes on the WITS bigbatch cluster
GPU_NODES=(
    mscluster42 mscluster44
    mscluster72 mscluster73 mscluster75 mscluster76 mscluster77
    mscluster78 mscluster79 mscluster80 mscluster81 mscluster82
    mscluster84 mscluster85 mscluster86 mscluster87 mscluster88 mscluster89
)
NODE_COUNT=${#GPU_NODES[@]}
node_idx=0

# Model registry: model_key → script|config_filename
declare -A MODELS
# format: script|config_file|dir_name  (dir_name matches the config log_dir short name)
MODELS["ann_baseline"]="ann_baseline.py|ann_baseline.yaml|ann"
MODELS["snn_actor_ann_critic"]="snn_actor_ann_critic.py|snn_actor_ann_critic.yaml|snn_ann_critic"
MODELS["snn_actor_snn_timing_critic"]="snn_actor_snn_timing_critic.py|snn_actor_snn_timing_critic.yaml|snn_timing_critic"
MODELS["ann2snn_actor"]="ann2snn_actor.py|ann2snn_actor.yaml|ann2snn_actor"
MODELS["ann2snn_both"]="ann2snn_both.py|ann2snn_both.yaml|ann2snn_both"
MODELS["snn_actor_snn_rate_critic"]="snn_actor_snn_timing_critic.py|snn_actor_snn_rate_critic.yaml|snn_rate_critic"
MODELS["popsan_snn"]="snn_actor_ann_critic.py|popsan_snn.yaml|popsan_snn"


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
        IFS='|' read -r script config_file dir_name <<< "${MODELS[$model_key]}"
        local config_path="configs/${config_dir}/${config_file}"
        local exp_key="neurips/${log_key}/${dir_name}"
        local node="${GPU_NODES[$((node_idx % NODE_COUNT))]}"

        echo "Submitting: $exp_key → $node (active=${env_active:-N/A})"
        sbatch --job-name="${log_key}_${dir_name}" \
               --nodelist="$node" \
               submit_array.slurm \
               "$script" "$config_path" "$exp_key" "$env_active"

        ((node_idx++))
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
