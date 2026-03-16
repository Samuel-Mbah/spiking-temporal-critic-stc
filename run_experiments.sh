#!/bin/bash

# --- Configuration ---
# Research-grade default: 5 independent seeds (1..SEED_COUNT)
SEED_COUNT="${SEED_COUNT:-5}"
SEEDS=($(seq 1 "$SEED_COUNT"))
PYTHON_EXEC="python3" 
SUITE="${SUITE:-tmaze}"  # use: SUITE=tmaze ./run_experiments.sh
TMAZE_ACTIVE="${TMAZE_ACTIVE:-}"

# Define experiments: "Script Name | Config Name"
# Ensure these keys match the folder names you want in results/logs/
declare -A experiments
if [ "$SUITE" = "poc" ]; then
    experiments["poc_ann_baseline_no_fs"]="ann_baseline.py|poc/ann_baseline.yaml"
    experiments["poc_snn_actor_ann_critic_no_fs"]="snn_actor_ann_critic.py|poc/snn_actor_ann_critic.yaml"
    experiments["poc_snn_actor_snn_timing_critic_no_fs"]="snn_actor_snn_timing_critic.py|poc/snn_actor_snn_timing_critic.yaml"
    experiments["poc_ann2snn_full_no_fs"]="ann2snn_both.py|poc/ann2snn_both.yaml"
    experiments["poc_ann2snn_actor_no_fs"]="ann2snn_actor.py|poc/ann2snn_actor.yaml"
else
    experiments["tmaze_ann_baseline"]="ann_baseline.py|tmaze/ann_baseline.yaml"
    # experiments["tmaze_snn_actor_ann_critic"]="snn_actor_ann_critic.py|tmaze/snn_actor_ann_critic.yaml"
    # experiments["tmaze_snn_actor_snn_timing_critic"]="snn_actor_snn_timing_critic.py|tmaze/snn_actor_snn_timing_critic.yaml"
    # experiments["tmaze_ann2snn_full"]="ann2snn_both.py|tmaze/ann2snn_both.yaml"
    # experiments["tmaze_ann2snn_actor"]="ann2snn_actor.py|tmaze/ann2snn_actor.yaml"
fi
# --- Main Loop ---
for experiment_key in "${!experiments[@]}"; do
    IFS='|' read -r script_name config_name <<< "${experiments[$experiment_key]}"
    run_key="$experiment_key"
    if [ "$SUITE" = "tmaze" ] && [ -n "$TMAZE_ACTIVE" ]; then
        active_lc="$(echo "$TMAZE_ACTIVE" | tr '[:upper:]' '[:lower:]')"
        if [ "$active_lc" = "1" ] || [ "$active_lc" = "true" ] || [ "$active_lc" = "yes" ] || [ "$active_lc" = "y" ]; then
            run_key="${experiment_key}_active"
        elif [ "$active_lc" = "0" ] || [ "$active_lc" = "false" ] || [ "$active_lc" = "no" ] || [ "$active_lc" = "n" ]; then
            run_key="${experiment_key}_passive"
        fi
    fi
    
    echo "========================================================"
    echo "Starting Research Run: $run_key"
    echo "Script: $script_name | Config: $config_name"
    echo "========================================================"

    for seed in "${SEEDS[@]}"; do
        echo "  > Running Seed: $seed..."
        
        # Define output directory for this specific seed
        OUTPUT_DIR="results/logs/${run_key}/seed_${seed}"
        
        # Ensure the directory exists before python runs (safety)
        mkdir -p "$OUTPUT_DIR"
        
        # Run the training script with explicit arguments
        # The Python scripts have been updated to accept these flags
        CMD=(
            "$PYTHON_EXEC" "experiments/${script_name}"
            --config "configs/${config_name}"
            --seed "$seed"
            --run-name "seed_${seed}"
            --log-dir "$OUTPUT_DIR"
        )
        if [ -n "$TMAZE_ACTIVE" ] && [ "$SUITE" = "tmaze" ]; then
            CMD+=(--env-active "$TMAZE_ACTIVE")
        fi

        "${CMD[@]}"

        if [ $? -eq 0 ]; then
            echo "  > Seed $seed completed successfully."
        else
            echo "  > ERROR: Seed $seed failed."
        fi
    done

    # --- NEW: Generate aggregated multi-seed dashboard for this experiment ---
    if [ -f "post_analysis/plot_multiseed_dashboard.py" ]; then
        echo "  > Generating multi-seed dashboard for $run_key..."
        $PYTHON_EXEC post_analysis/plot_multiseed_dashboard.py \
            --experiment "$run_key" \
            --config "configs/${config_name}"
    else
        echo "Warning: plot_multiseed_dashboard.py not found. Skipping multi-seed dashboard."
    fi
done

echo "All seed runs completed."

# --- NEW: Generate Aggregated Research Plots ---
# This script scans results/logs/* and creates comparison graphs
if [ -f "post_analysis/plot_all_comparisons.py" ]; then
    echo "========================================================"
    echo "Generating Final Comparison Plots"
    echo "========================================================"
    $PYTHON_EXEC post_analysis/plot_all_comparisons.py
else
    echo "Warning: plot_all_comparisons.py not found. Skipping aggregation."
fi

echo "Research experiments finished successfully."
