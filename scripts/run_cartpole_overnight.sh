#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_ROOT="${LOG_ROOT:-results/RLC/cartpole}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-results/RLC/logs/cartpole}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

mkdir -p "$LOG_ROOT"
mkdir -p "$ARTIFACT_ROOT"

if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(42 43 44 45 46 47 48 49 50 51)
fi

declare -a RUNS=(
  "experiments/ann_baseline.py|configs/cartpole/ann_baseline.yaml|ann_baseline"
  "experiments/snn_actor_ann_critic.py|configs/cartpole/snn_actor_ann_critic.yaml|snn_actor_ann_critic"
  "experiments/snn_actor_snntiming_critic.py|configs/cartpole/snn_actor_snntiming_critic.yaml|snn_actor_snntiming_critic"
  "experiments/ann2snn_actor.py|configs/cartpole/ann2snn_actor.yaml|ann2snn_actor"
  "experiments/ann2snn_both.py|configs/cartpole/ann2snn_both.yaml|ann2snn_both"
)

overall_fail=0

for seed in "${SEEDS[@]}"; do
  echo "============================================================"
  echo "Starting seed ${seed} at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "============================================================"

  pids=()
  names=()

  for run in "${RUNS[@]}"; do
    IFS='|' read -r script_path config_path run_name <<< "$run"
    log_file="${LOG_ROOT}/${run_name}_seed${seed}.log"
    run_log_dir="${ARTIFACT_ROOT}/${run_name}/seed_${seed}"
    mkdir -p "$run_log_dir"

    echo "Launching ${run_name} (seed ${seed}) -> ${log_file}"
    echo "  artifacts: ${run_log_dir}"
    "$PYTHON_BIN" "$script_path" \
      --config "$config_path" \
      --seed "$seed" \
      --run-name "seed_${seed}" \
      --log-dir "$run_log_dir" \
      >"$log_file" 2>&1 &

    pids+=("$!")
    names+=("$run_name")
  done

  batch_fail=0
  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    run_name="${names[$i]}"

    if wait "$pid"; then
      echo "Completed ${run_name} (seed ${seed})"
    else
      rc=$?
      echo "FAILED ${run_name} (seed ${seed}) with exit code ${rc}"
      log_file="${LOG_ROOT}/${run_name}_seed${seed}.log"
      if [ -f "$log_file" ]; then
        echo "---- Last 30 lines: ${log_file} ----"
        tail -n 30 "$log_file"
        echo "-------------------------------------"
      fi
      batch_fail=1
      overall_fail=1
    fi
  done

  if [ "$batch_fail" -eq 0 ]; then
    echo "Seed ${seed} batch completed successfully."
  else
    echo "Seed ${seed} batch finished with failures."
    if [ "$STOP_ON_FAIL" = "1" ]; then
      echo "STOP_ON_FAIL=1, exiting early."
      exit 1
    fi
  fi
done

echo "============================================================"
if [ "$overall_fail" -eq 0 ]; then
  echo "All batches completed successfully."
else
  echo "All batches finished, but at least one run failed."
fi
echo "Logs are in ${LOG_ROOT}"
echo "Per-seed artifacts are in ${ARTIFACT_ROOT}"
echo "============================================================"

exit "$overall_fail"
