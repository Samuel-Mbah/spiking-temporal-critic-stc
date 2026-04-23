"""
Standalone Energy Benchmark Runner
===================================
Loads trained ANN and SNN agents from checkpoints and runs the full energy
benchmark, producing a combined GPU + SOP report and saving JSON results.

This is the recommended way to run benchmarks for paper results.  Train all
your seeds first, then run this script once to compare them cleanly — rather
than embedding benchmark runs inside every training script.

Usage
-----
  # Compare one ANN checkpoint against one SNN checkpoint:
  python benchmark.py \\
      --ann-checkpoint  logs/ann_run/checkpoints/agent_best.pt \\
      --snn-checkpoint  logs/snn_run/checkpoints/agent_best.pt \\
      --ann-config      configs/ann.yaml \\
      --snn-config      configs/snn.yaml \\
      --num-episodes    100 \\
      --output-dir      results/

  # Benchmark a single model (no comparison report):
  python benchmark.py \\
      --ann-checkpoint  logs/ann_run/checkpoints/agent_best.pt \\
      --ann-config      configs/ann.yaml \\
      --output-dir      results/

Why a separate script?
----------------------
  - Training is slow; benchmarks should not block or slow it down.
  - You often want to compare across many seeds / environments without
    re-training.
  - The inline benchmark inside universal_baseline.py is a quick sanity
    check only.  This script is the authoritative benchmark for papers.
"""

import argparse
import dataclasses
import json
import logging
import os
import sys

import numpy as np
import torch
import yaml

# --- Path Setup ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.tools.energy_benchmark import EnergyBenchmark, sop_from_saved_metrics
from src.training.agents import make_agent, resolve_cartpole_types
from src.utils.plotting import (
    plot_energy_efficiency_comparison,
    plot_steps_to_solve_comparison,
    plot_sparsity_breakdown,
)
from src.utils.metrics import load_training_data
from src.training.envs import make_envs, VecNormalize, set_global_seeds
from src.training.evaluate import evaluate_snn
from src.utils.checkpoint import load_checkpoint

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# =======================================================================
#  HELPERS
# =======================================================================
def _build_agent_from_config(config: dict, device: torch.device) -> torch.nn.Module:
    """Reconstruct an agent from a YAML config dict (no weights loaded yet)."""
    env_cfg   = config.get("env", {})
    model_cfg = config.get("model", {})
    ppo_cfg   = config.get("ppo", {})

    # Build a throwaway env just to get action-space size, then close it.
    _, tmp_env = make_envs(
        seed=config.get("env_seed", 42),
        env_id=env_cfg.get("id", "CartPole-v1"),
        n_envs=1,
        env_kwargs=env_cfg.get("kwargs", {}),
        partial_obs=env_cfg.get("partial_obs"),
        frame_stack=env_cfg.get("frame_stack"),
        frame_stack_flatten=env_cfg.get("frame_stack_flatten", True),
    )
    act_dim = (
        tmp_env.single_action_space.n
        if hasattr(tmp_env, "single_action_space")
        else tmp_env.action_space.n
    )
    tmp_env.close()

    actor_type, critic_type = resolve_cartpole_types(model_cfg.get("mode", "ann"))

    agent = make_agent(
        actor_type=actor_type,
        critic_type=critic_type,
        hidden_dim=model_cfg.get("hidden_dim", 128),
        dropout=model_cfg.get("dropout", 0.0),
        in_dim=model_cfg.get("in_features", 4),
        act_dim=act_dim,
        gamma=ppo_cfg.get("gamma", 0.99),
        critic_informs_actor=bool(model_cfg.get("critic_informs_actor", False)),
        detach_critic_for_actor=bool(model_cfg.get("detach_critic_for_actor", True)),
        normalize_critic_for_actor=bool(model_cfg.get("normalize_critic_for_actor", True)),
        critic_actor_value_clip=model_cfg.get("critic_actor_value_clip", 5.0),
        critic_actor_norm_momentum=float(model_cfg.get("critic_actor_norm_momentum", 0.01)),
    ).to(device)

    return agent


def _load_agent(checkpoint_path: str, config: dict, device: torch.device) -> torch.nn.Module:
    """Build architecture, load weights from checkpoint, set eval mode."""
    agent = _build_agent_from_config(config, device)
    data  = load_checkpoint(checkpoint_path, agent=agent, map_location=device)

    # Restore VecNormalize obs stats if present (needed for normalised envs)
    vec_state = (data or {}).get("vecnorm_state") or {}
    if vec_state and hasattr(agent, "obs_rms"):
        obs_state = vec_state.get("obs_rms", {})
        if obs_state:
            agent.obs_rms.mean  = np.asarray(obs_state.get("mean",  agent.obs_rms.mean),  dtype=np.float64)
            agent.obs_rms.var   = np.asarray(obs_state.get("var",   agent.obs_rms.var),   dtype=np.float64)
            agent.obs_rms.count = float(obs_state.get("count", agent.obs_rms.count))

    agent.eval()
    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    return agent


def _make_bench_env(config: dict, seed_offset: int = 1000):
    """Create a single-env copy suitable for benchmarking."""
    env_cfg = config.get("env", {})
    _, bench_env = make_envs(
        seed=config.get("env_seed", 42) + seed_offset,
        env_id=env_cfg.get("id", "CartPole-v1"),
        n_envs=1,
        env_kwargs=env_cfg.get("kwargs", {}),
        partial_obs=env_cfg.get("partial_obs"),
        frame_stack=env_cfg.get("frame_stack"),
        frame_stack_flatten=env_cfg.get("frame_stack_flatten", True),
    )
    return bench_env


def _maybe_wrap_vecnorm(env, agent, config: dict):
    env_cfg = config.get("env", {})
    if env_cfg.get("vec_normalize", False) and hasattr(agent, "obs_rms"):
        env = VecNormalize(env, training=False, norm_obs=True, norm_reward=False)
        env.obs_rms = agent.obs_rms
    return env


def _save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved: {path}")


# =======================================================================
#  MAIN
# =======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Standalone energy benchmark — compare ANN vs SNN from checkpoints."
    )
    # ANN args
    parser.add_argument("--ann-checkpoint", type=str, default=None,
                        help="Path to trained ANN checkpoint (.pt)")
    parser.add_argument("--ann-config",     type=str, default=None,
                        help="YAML config used to train the ANN")
    # SNN args
    parser.add_argument("--snn-checkpoint", type=str, default=None,
                        help="Path to trained SNN checkpoint (.pt)")
    parser.add_argument("--snn-config",     type=str, default=None,
                        help="YAML config used to train the SNN")
    # Benchmark settings
    parser.add_argument("--num-episodes",   type=int,   default=100,
                        help="Episodes to run per model (default: 100)")
    parser.add_argument("--warmup-runs",    type=int,   default=2,
                        help="Warmup episodes before measuring (default: 2)")
    parser.add_argument("--output-dir",     type=str,   default="results",
                        help="Directory to write JSON results and report")
    parser.add_argument("--device",         type=str,   default=None,
                        help="Override device (cpu / cuda:0 etc.)")
    # SOP energy constants (override Horowitz defaults if needed)
    parser.add_argument("--sop-e-ac-pj",   type=float, default=0.9,
                        help="AC energy constant in pJ (default: 0.9, Horowitz 2014)")
    parser.add_argument("--sop-e-mac-pj",  type=float, default=4.6,
                        help="MAC energy constant in pJ (default: 4.6, Horowitz 2014)")
    # Training log dirs for steps-to-solve plot (optional; can be multi-seed glob patterns)
    parser.add_argument("--ann-log-dirs",  type=str, nargs="+", default=None,
                        help="One or more ANN training log dirs containing per_episode_metrics.csv")
    parser.add_argument("--snn-log-dirs",  type=str, nargs="+", default=None,
                        help="One or more SNN training log dirs containing per_episode_metrics.csv")
    parser.add_argument("--reward-threshold", type=float, default=None,
                        help="Reward threshold for 'solved' (inferred from config if omitted)")
    args = parser.parse_args()

    # Require at least one model
    if not args.ann_checkpoint and not args.snn_checkpoint:
        parser.error("Provide at least one of --ann-checkpoint or --snn-checkpoint.")
    if args.ann_checkpoint and not args.ann_config:
        parser.error("--ann-config is required when --ann-checkpoint is set.")
    if args.snn_checkpoint and not args.snn_config:
        parser.error("--snn-config is required when --snn-checkpoint is set.")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # --- Shared benchmarker (single idle calibration for both models) ---
    bencher = EnergyBenchmark()
    bencher.calibrate_idle()

    ann_metrics = None
    snn_metrics = None

    # =================================================================
    #  ANN BENCHMARK
    # =================================================================
    if args.ann_checkpoint:
        logger.info("=== Benchmarking ANN ===")
        with open(args.ann_config) as f:
            ann_config = yaml.safe_load(f)

        set_global_seeds(ann_config.get("env_seed", 42))
        ann_agent = _load_agent(args.ann_checkpoint, ann_config, device)

        ann_env = _make_bench_env(ann_config, seed_offset=1000)
        ann_env = _maybe_wrap_vecnorm(ann_env, ann_agent, ann_config)

        ann_metrics = bencher.benchmark_model(
            model=ann_agent,
            episode_fn=lambda m: evaluate_snn(ann_env, m, sticky_action=False),
            num_episodes=args.num_episodes,
            model_type="ANN",
            warmup_runs=args.warmup_runs,
        )

        ann_out = os.path.join(args.output_dir, "ann_benchmark_metrics.json")
        _save_json(dataclasses.asdict(ann_metrics), ann_out)

        if hasattr(ann_env, "close"):
            ann_env.close()

    # =================================================================
    #  SNN BENCHMARK
    # =================================================================
    if args.snn_checkpoint:
        logger.info("=== Benchmarking SNN ===")
        with open(args.snn_config) as f:
            snn_config = yaml.safe_load(f)

        set_global_seeds(snn_config.get("env_seed", 42))
        snn_agent = _load_agent(args.snn_checkpoint, snn_config, device)

        snn_env = _make_bench_env(snn_config, seed_offset=1000)
        snn_env = _maybe_wrap_vecnorm(snn_env, snn_agent, snn_config)

        snn_metrics = bencher.benchmark_model(
            model=snn_agent,
            episode_fn=lambda m: evaluate_snn(snn_env, m, sticky_action=False),
            num_episodes=args.num_episodes,
            model_type="SNN",
            warmup_runs=args.warmup_runs,
            sop_e_ac_pJ=args.sop_e_ac_pj,
            sop_e_mac_pJ=args.sop_e_mac_pj,
        )

        snn_out = os.path.join(args.output_dir, "snn_benchmark_metrics.json")
        _save_json(dataclasses.asdict(snn_metrics), snn_out)

        if hasattr(snn_env, "close"):
            snn_env.close()

    # =================================================================
    #  COMBINED REPORT
    # =================================================================
    if ann_metrics is not None and snn_metrics is not None:
        report = bencher.generate_report(snn_metrics, ann_metrics)
        report_path = os.path.join(args.output_dir, "benchmark_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"Combined report saved to {report_path}")

        plot_energy_efficiency_comparison(
            ann_metrics=dataclasses.asdict(ann_metrics),
            snn_metrics=dataclasses.asdict(snn_metrics),
            save_path=os.path.join(args.output_dir, "energy_comparison.png"),
            env_name=snn_config.get("env", {}).get("id", "Unknown"),
            ann_label="ANN Baseline",
            snn_label="SNN Timing Critic",
        )
        logger.info(f"Saved comparison plot: {args.output_dir}/energy_comparison.png")

        plot_sparsity_breakdown(
            snn_metrics=dataclasses.asdict(snn_metrics),
            ann_metrics=dataclasses.asdict(ann_metrics),
            save_path=os.path.join(args.output_dir, "sparsity_breakdown.png"),
            env_name=snn_config.get("env", {}).get("id", "Unknown"),
            ann_label="ANN Baseline",
            snn_label="SNN Timing Critic",
        )
        logger.info(f"Saved sparsity plot: {args.output_dir}/sparsity_breakdown.png")

    # ------------------------------------------------------------------
    # Steps-to-solve plot (requires --ann-log-dirs / --snn-log-dirs)
    # ------------------------------------------------------------------
    if args.ann_log_dirs or args.snn_log_dirs:
        active_config = snn_config if args.snn_checkpoint else ann_config
        threshold = args.reward_threshold or active_config.get("ppo", {}).get("reward_threshold", 475.0)
        env_id = active_config.get("env", {}).get("id", "Unknown")

        def _load_dfs(dirs):
            dfs = []
            for d in (dirs or []):
                try:
                    _, per_ep = load_training_data(d)
                    if not per_ep.empty:
                        dfs.append(per_ep)
                    else:
                        logger.warning(f"Empty per_episode_metrics.csv in {d!r}, skipping.")
                except Exception as e:
                    logger.warning(f"Could not load log dir {d!r}: {e}")
            return dfs

        ann_dfs = _load_dfs(args.ann_log_dirs)
        snn_dfs = _load_dfs(args.snn_log_dirs)

        if ann_dfs or snn_dfs:
            plot_steps_to_solve_comparison(
                ann_dfs=ann_dfs or [next(iter(snn_dfs))],  # fallback keeps function happy
                snn_dfs=snn_dfs or [next(iter(ann_dfs))],
                save_path=os.path.join(args.output_dir, "steps_to_solve.png"),
                reward_threshold=float(threshold),
                env_name=env_id,
                ann_label="ANN Baseline",
                snn_label="SNN Timing Critic",
            )
            logger.info(f"Saved steps-to-solve plot: {args.output_dir}/steps_to_solve.png")
        else:
            logger.warning("--ann-log-dirs / --snn-log-dirs provided but no valid CSVs found.")

    # Solo-model summary logs (independent of the steps-to-solve block above)
    if ann_metrics is None and snn_metrics is None:
        pass
    elif ann_metrics is not None and snn_metrics is None:
        logger.info(
            f"\nANN-only benchmark complete.\n"
            f"  Total GPU J    : {ann_metrics.total_energy_joules:.4f}\n"
            f"  Avg power (W)  : {ann_metrics.avg_power_watts:.3f}\n"
            f"  J / env-step   : {ann_metrics.raw_joules_per_env_step:.6f}\n"
        )
    elif snn_metrics is not None and ann_metrics is None:
        sop = snn_metrics.sop
        sop_line = sop.summary() if sop else "SOP: N/A (no spikes recorded)"
        logger.info(
            f"\nSNN-only benchmark complete.\n"
            f"  Total GPU J    : {snn_metrics.total_energy_joules:.4f}\n"
            f"  Avg power (W)  : {snn_metrics.avg_power_watts:.3f}\n"
            f"  Sparsity       : {snn_metrics.sparsity_factor:.2%}\n"
            f"  {sop_line}\n"
        )
        plot_sparsity_breakdown(
            snn_metrics=dataclasses.asdict(snn_metrics),
            save_path=os.path.join(args.output_dir, "sparsity_breakdown.png"),
            env_name=snn_config.get("env", {}).get("id", "Unknown"),
            snn_label="SNN Timing Critic",
        )
        logger.info(f"Saved sparsity plot: {args.output_dir}/sparsity_breakdown.png")

    logger.info("Done.")


if __name__ == "__main__":
    main()