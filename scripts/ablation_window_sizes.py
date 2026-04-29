"""
Ablation: actor_T vs critic_T window size sweep.

Tests whether the asymmetric choice Ta=16, Tc=32 is justified by comparing
symmetric and shifted window sizes. Results go under results/ablations/window_sizes/.

Usage:
    python scripts/ablation_window_sizes.py [--seeds 1 2 3] [--dry-run]
"""

import argparse
import copy
import csv
import importlib
import json
import logging
import os
import sys

import yaml

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.training.surrogate_trainer import run_surrogate
from src.training.envs import set_global_seeds

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_CONFIG = os.path.join(
    repo_root, "configs/cartpole/snn_actor_snn_timing_critic.yaml"
)
OUT_ROOT = os.path.join(repo_root, "results/neurips/ablations/window_sizes")

# (actor_T, critic_T) pairs to sweep
SWEEP = [
    (8,  32),   # smaller actor
    (16, 16),   # symmetric at actor level
    (16, 32),   # default
    (32, 32),   # symmetric at critic level
    (32, 64),   # both doubled
]


def register_custom_envs():
    for mod in ["src.envs.t_maze"]:
        try:
            importlib.import_module(mod)
        except Exception:
            pass


def run_condition(base_cfg: dict, actor_T: int, critic_T: int, seed: int, dry_run: bool) -> dict:
    cfg = copy.deepcopy(base_cfg)

    cfg["snn"]["actor_T"] = actor_T
    cfg["snn"]["critic_T"] = critic_T

    tag = f"Ta{actor_T}_Tc{critic_T}"
    run_dir = os.path.join(OUT_ROOT, tag, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    cfg["run_name"] = f"window_ablation_{tag}_seed{seed}"
    cfg["log_dir"]   = run_dir
    cfg["plots_dir"] = os.path.join(run_dir, "plots")
    cfg["env_seed"]  = seed

    # Disable expensive side-outputs for the ablation
    cfg.setdefault("reporting", {}).setdefault("wandb", {})["use"] = False

    os.makedirs(cfg["plots_dir"], exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    if dry_run:
        log.info(f"[DRY RUN] would run Ta={actor_T} Tc={critic_T} seed={seed}")
        return {"actor_T": actor_T, "critic_T": critic_T, "seed": seed, "mean_return": None, "solved": None}

    log.info(f"=== Ta={actor_T}  Tc={critic_T}  seed={seed} ===")
    seed_cfg = cfg.get("seed_control", {})
    set_global_seeds(
        seed,
        deterministic_torch=seed_cfg.get("deterministic_torch", True),
        cudnn_benchmark=seed_cfg.get("cudnn_benchmark", False),
    )

    result = run_surrogate(config=cfg)

    test_rewards = result.get("test_rewards", []) or []
    reward_threshold = cfg["ppo"].get("reward_threshold", 475.0)
    success_rate_threshold = cfg["ppo"].get("success_rate_threshold", 95.0)

    tail = test_rewards[-20:] if len(test_rewards) >= 20 else test_rewards
    mean_return = float(sum(tail) / len(tail)) if tail else float("nan")
    solved = (
        mean_return >= reward_threshold
        and (sum(r >= reward_threshold for r in tail) / len(tail) * 100) >= success_rate_threshold
    ) if tail else False

    row = {
        "actor_T":     actor_T,
        "critic_T":    critic_T,
        "seed":        seed,
        "mean_return": round(mean_return, 2),
        "solved":      solved,
    }
    log.info(f"  → mean_return={mean_return:.1f}  solved={solved}")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--total-updates", type=int, default=None,
                        help="Override training.total_updates (e.g. 100 for a quick pass)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    register_custom_envs()

    with open(BASE_CONFIG) as f:
        base_cfg = yaml.safe_load(f)

    env_cfg = base_cfg.setdefault("env", {})
    if "kwargs" not in env_cfg or env_cfg["kwargs"] is None:
        env_cfg["kwargs"] = {}

    if args.total_updates is not None:
        base_cfg.setdefault("training", {})["total_updates"] = args.total_updates
        log.info(f"total_updates overridden → {args.total_updates}")

    os.makedirs(OUT_ROOT, exist_ok=True)
    results = []

    for actor_T, critic_T in SWEEP:
        for seed in args.seeds:
            row = run_condition(base_cfg, actor_T, critic_T, seed, args.dry_run)
            results.append(row)

    # Write summary CSV
    csv_path = os.path.join(OUT_ROOT, "summary.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        log.info(f"Summary written to {csv_path}")

    # Print table
    print("\n=== Window Size Ablation Results ===")
    print(f"{'Ta':>4}  {'Tc':>4}  {'seed':>4}  {'mean_return':>12}  {'solved':>6}")
    for r in results:
        print(f"{r['actor_T']:>4}  {r['critic_T']:>4}  {r['seed']:>4}  "
              f"{str(r['mean_return']):>12}  {str(r['solved']):>6}")


if __name__ == "__main__":
    main()
