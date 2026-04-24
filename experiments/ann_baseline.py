"""
Experiment: ANN Baseline
- Mode: Single Seed Execution (CLI Driven)
- Logic: Trains a standard PPO agent with an ANN architecture on ANY Gym environment.
- Usage: python ann_baseline.py --config path/to/config.yaml --seed 1
"""

import argparse
import dataclasses
import os
import sys
import yaml
import wandb
import logging
import json
import gymnasium as gym
import matplotlib
import torch
import numpy as np

# Force non-interactive backend for server/headless video generation
matplotlib.use('Agg')

# --- Path Setup ---
# Assumes this script is in `experiments/` folder, one level below repo root.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# --- Imports ---
from src.training.baseline_trainer import run_baseline
from src.utils.report import create_training_dashboard
from src.utils.metrics import calculate_and_save_metrics_csv
from src.training.envs import set_global_seeds, make_envs, VecNormalize
from src.tools.energy_benchmark import EnergyBenchmark
from src.training.evaluate import evaluate_snn

# Logging Setup
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# =======================================================================
#  SERIALISATION HELPER
# =======================================================================
def _metrics_to_dict(metrics) -> dict:
    """
    Safely serialise an EnergyMetrics dataclass to a plain dict.

    The `sop` field is itself a dataclass (SOPEnergyResult) and therefore not
    directly JSON-serialisable.  We convert it to a nested dict here so that
    `json.dump` never raises a TypeError.
    """
    d = dataclasses.asdict(metrics)  # recursively converts nested dataclasses
    return d


# =======================================================================
#  INTRA-EPISODE VALUE HELPERS  (unchanged)
# =======================================================================
@torch.no_grad()
def _collect_intra_episode_values(agent, env, max_steps: int = 1000):
    """Collect critic values over one deterministic evaluation episode."""
    agent.eval()
    device = next(agent.parameters()).device
    values = []
    obs, _ = env.reset()
    done = False
    steps = 0
    is_vector_env = hasattr(env, "num_envs")

    while not done and steps < max_steps:
        if hasattr(agent, "obs_rms"):
            rms = agent.obs_rms
            obs = np.clip((obs - rms.mean) / np.sqrt(rms.var + 1e-8), -10.0, 10.0)

        if is_vector_env:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        logits, val_tensor = agent(obs_t)
        action = int(torch.argmax(logits, dim=-1).item())

        if val_tensor is not None:
            values.append(float(val_tensor.reshape(-1)[0].item()))

        if is_vector_env:
            obs, _, terminated, truncated, _ = env.step([action])
            done = bool(terminated[0] or truncated[0])
        else:
            obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
        steps += 1

    return values


def _mean_intra_episode_profile_from_validation(validation_data):
    """
    Build a mean critic-value profile over step-in-episode from post-eval traces.
    Returns a list where index i corresponds to step (i+1) in an episode.
    """
    if not isinstance(validation_data, dict):
        return []

    traces = validation_data.get("step_traces", {}) or {}
    values = np.asarray(traces.get("critic_values", []), dtype=float)
    step_in_episode = np.asarray(traces.get("step_in_episode", []), dtype=int)
    episode_index = np.asarray(traces.get("episode_index", []), dtype=int)

    if values.size == 0 or step_in_episode.size == 0 or episode_index.size == 0:
        return []

    n = min(values.size, step_in_episode.size, episode_index.size)
    values = values[:n]
    step_in_episode = step_in_episode[:n]
    episode_index = episode_index[:n]

    num_completed = int((validation_data.get("episode_metrics", {}) or {}).get("num_completed", 0))
    if num_completed > 0:
        mask = episode_index < num_completed
        values = values[mask]
        step_in_episode = step_in_episode[mask]

    if values.size == 0:
        return []

    max_step = int(np.max(step_in_episode))
    profile = []
    for s in range(1, max_step + 1):
        step_vals = values[step_in_episode == s]
        if step_vals.size == 0:
            break
        profile.append(float(np.mean(step_vals)))
    return profile


def register_custom_envs():
    """Helper to ensure custom envs are registered before Gym uses them."""
    try:
        import src.envs.t_maze  # Registers 'tmaze-v0'
        logger.info("Custom Environment 'tmaze-v0' registered.")
    except ImportError:
        logger.warning("Could not import t_maze. If using T-Maze, ensure src/envs/t_maze.py exists.")


def main():
    # 0. Register Custom Envs
    register_custom_envs()

    # 1. Parsing Arguments
    parser = argparse.ArgumentParser("Universal ANN Baseline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--run-name", type=str, default=None, help="WandB/Log run name override")
    parser.add_argument("--log-dir", type=str, default=None, help="Output directory override")
    parser.add_argument("--env-active", type=str, default=None, help="Override env.kwargs.active (true/false)")
    args = parser.parse_args()

    # 2. Load Configuration
    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join(repo_root, args.config)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 3. Apply CLI Overrides
    if args.seed is not None:
        config['env_seed'] = args.seed
    if args.run_name is not None:
        config['run_name'] = args.run_name
    if args.log_dir is not None:
        config['log_dir'] = args.log_dir
        config['plots_dir'] = os.path.join(args.log_dir, "plots")
    if args.env_active is not None:
        val = str(args.env_active).strip().lower()
        if val in {"1", "true", "yes", "y"}:
            config.setdefault("env", {}).setdefault("kwargs", {})["active"] = True
        elif val in {"0", "false", "no", "n"}:
            config.setdefault("env", {}).setdefault("kwargs", {})["active"] = False
        else:
            raise ValueError(f"Invalid --env-active value: {args.env_active}")

    # Ensure directories exist
    os.makedirs(config['log_dir'], exist_ok=True)
    if 'plots_dir' in config:
        os.makedirs(config['plots_dir'], exist_ok=True)

    # Save resolved config (includes CLI overrides like --seed, --log-dir)
    with open(os.path.join(config['log_dir'], 'config.yaml'), 'w') as _f:
        yaml.dump(config, _f, default_flow_style=False, sort_keys=False)

    # 4. Weights & Biases
    if config.get("reporting", {}).get("wandb", {}).get("use", False):
        wandb.init(
            project=config.get("project_name", "neuroai-project"),
            config=config,
            name=config.get("run_name"),
            reinit=True,
            mode="online"
        )

    # 5. Set Global Seed
    seed_cfg = config.get("seed_control", {})
    set_global_seeds(
        config['env_seed'],
        deterministic_torch=seed_cfg.get("deterministic_torch", True),
    )

    env_cfg = config.get("env", {})
    env_kwargs = env_cfg.get("kwargs", {}) or {}
    logger.info(
        f"--- Starting Run: {config['run_name']} | Env: {env_cfg.get('id', 'unknown')} | "
        f"Seed: {config['env_seed']} | length: {env_kwargs.get('length', 'N/A')} | "
        f"active: {env_kwargs.get('active', 'N/A')} ---"
    )

    # 6. Run Training
    result = run_baseline(config=config)
    agent = result["agent"]

    # 7. Post-Training Analysis
    logger.info("--- Training Complete. Starting Analysis... ---")

    # A. Save CSV Metrics
    try:
        calculate_and_save_metrics_csv(
            result,
            config['log_dir'],
            env_name=config['env'].get('id'),
            reward_threshold=config['ppo'].get("reward_threshold", 0.0)
        )
    except Exception as e:
        logger.error(f"Failed to save metrics CSV: {e}")

    # B. Generate Dashboard
    if os.path.exists(config['plots_dir']):
        try:
            env_cfg = config.get("env", {})
            validation_data = result.get("validation_data", {}) or {}

            # Prefer post-eval aggregated profile (more statistically valid than a single rollout).
            intra_profile_mode = "single_episode"
            intra_values = _mean_intra_episode_profile_from_validation(validation_data)
            if intra_values:
                intra_profile_mode = "mean_post_eval"

            # Fallback to one deterministic episode if post-eval traces are unavailable.
            if not intra_values:
                _, plot_env = make_envs(
                    seed=config['env_seed'] + 2000,
                    env_id=env_cfg.get('id'),
                    n_envs=1,
                    env_kwargs=env_cfg.get("kwargs", {}),
                    partial_obs=env_cfg.get("partial_obs"),
                    frame_stack=env_cfg.get("frame_stack"),
                    frame_stack_flatten=env_cfg.get("frame_stack_flatten", True),
                )
                if env_cfg.get('vec_normalize', False) and hasattr(agent, 'obs_rms'):
                    plot_env = VecNormalize(plot_env, training=False, norm_obs=True, norm_reward=False)
                    plot_env.obs_rms = agent.obs_rms

                intra_values = _collect_intra_episode_values(
                    agent,
                    plot_env,
                    max_steps=int(env_cfg.get("max_episode_steps", 1000)),
                )
                if hasattr(plot_env, "close"):
                    plot_env.close()

            if intra_values:
                validation_data["intra_episode_values"] = intra_values
                validation_data["intra_episode_values_profile"] = intra_profile_mode

            create_training_dashboard(
                log_dir=config['log_dir'],
                output_dir=config['plots_dir'],
                title_prefix=f"{config['env']['id']} Seed {config['env_seed']}",
                threshold=config['ppo'].get("reward_threshold", 0.0),
                validation_data=validation_data if validation_data else None,
                config=config,
            )
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")

    # C. Inline Benchmarking
    # NOTE: For multi-experiment comparison (ANN vs SNN), prefer the standalone
    # benchmark.py script which loads from checkpoints and generates a combined
    # report.  This inline benchmark is kept for single-run sanity checks only.
    bench_cfg = config.get("benchmark", {}) or {}
    benchmark_enabled = bool(
        bench_cfg.get(
            "enabled",
            bench_cfg.get("measure_inference_energy", bench_cfg.get("run_for_ann", True)),
        )
    )
    if benchmark_enabled:
        try:
            env_cfg = config.get("env", {})
            _, bench_env = make_envs(
                seed=config['env_seed'] + 1000,
                env_id=env_cfg.get('id'),
                n_envs=1,
                # Pass env_kwargs so custom envs (T-Maze etc.) initialise correctly
                env_kwargs=env_cfg.get("kwargs", {}),
                partial_obs=env_cfg.get("partial_obs"),
                frame_stack=env_cfg.get("frame_stack"),
                frame_stack_flatten=env_cfg.get("frame_stack_flatten", True),
            )

            if env_cfg.get('vec_normalize', False) and hasattr(agent, 'obs_rms'):
                bench_env = VecNormalize(bench_env, training=False)
                bench_env.obs_rms = agent.obs_rms

            bencher = EnergyBenchmark()
            metrics = bencher.benchmark_model(
                model=agent,
                episode_fn=lambda m: evaluate_snn(bench_env, m, sticky_action=False),
                num_episodes=bench_cfg.get('num_episodes_for_benchmark', 50),
                model_type="ANN",
            )

            # Use dataclasses.asdict so the nested SOPEnergyResult serialises cleanly.
            # The old metrics.__dict__ call would raise TypeError on the `sop` field.
            out_path = os.path.join(config['log_dir'], "benchmark_metrics.json")
            with open(out_path, "w") as f:
                json.dump(_metrics_to_dict(metrics), f, indent=4)
            logger.info(f"Benchmark metrics saved to {out_path}")

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
    else:
        logger.info("Energy benchmark disabled by config.")

    if wandb.run:
        wandb.finish()

    logger.info("--- Finished Successfully ---")


if __name__ == "__main__":
    main()