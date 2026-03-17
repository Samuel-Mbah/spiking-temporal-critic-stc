#!/bin/bash

# Define experiments exactly as you did in run_experiments.sh
declare -A experiments
experiments["ann_baseline"]="ann_baseline.py|ann_baseline.yaml"
experiments["snn_actor_ann_critic"]="snn_actor_ann_critic.py|snn_actor_ann_critic.yaml"
experiments["snn_actor_snn_timing_critic"]="snn_actor_snn_timing_critic.py|snn_actor_snn_timing_critic.yaml"
experiments["ann2snn_full"]="ann2snn_both.py|ann2snn_both.yaml"
experiments["ann2snn_actor"]="ann2snn_actor.py|ann2snn_actor.yaml"

# Loop through and submit
for experiment_key in "${!experiments[@]}"; do
    IFS='|' read -r script_name config_name <<< "${experiments[$experiment_key]}"
    
    echo "Submitting: $experiment_key"
    
    # Submit to Slurm, passing the 3 arguments
    sbatch --job-name="$experiment_key" \
           submit_array.slurm "$script_name" "$config_name" "$experiment_key"
done