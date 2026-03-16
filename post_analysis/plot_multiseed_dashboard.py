#!/usr/bin/env python3
"""
Generate aggregated (multi-seed) dashboards from seed log folders.
Usage:
  python3 plot_multiseed_dashboard.py --experiment ann_baseline --config configs/ann_baseline.yaml
"""

import argparse
import glob
import logging
import os
import sys
import yaml

# --- Path Setup ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.report import create_training_dashboard

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser("Multi-seed dashboard generator")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment key (folder name in results/logs)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--logs-root", type=str, default="results/logs", help="Root logs directory")
    parser.add_argument("--plots-root", type=str, default="results/plots", help="Root plots directory")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    log_glob = os.path.join(args.logs_root, args.experiment, "seed_*")
    log_dirs = sorted(glob.glob(log_glob))
    if len(log_dirs) < 2:
        logger.warning(f"Found {len(log_dirs)} seed dirs at {log_glob}. Skipping multiseed dashboard.")
        return 0

    output_dir = os.path.join(args.plots_root, args.experiment, "multiseed")
    os.makedirs(output_dir, exist_ok=True)

    threshold = config.get("ppo", {}).get("reward_threshold", 475.0)
    exp_lower = args.experiment.lower()
    if "ann2snn_actor" in exp_lower:
        dashboard_mode = "ann2snn_actor_conversion"
    elif "ann2snn_full" in exp_lower or "ann2snn_both" in exp_lower:
        dashboard_mode = "ann2snn_both_conversion"
    else:
        dashboard_mode = "standard"
    create_training_dashboard(
        log_dir=log_dirs,
        output_dir=output_dir,
        title_prefix=f"{args.experiment} (multi-seed)",
        threshold=threshold,
        dashboard_mode=dashboard_mode,
        config=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
