"""
Experiment: SNN Actor + Timing Critic (Research Grade) - Universal
- Mode: Single Seed Execution
- Logic: Surrogate gradient training of SNN Actor with Temporal Critic.
- Supports: CartPole-v1, tmaze-v0, and other registered Gym environments.
"""

import argparse
import os
import sys
import yaml
import logging
import json
import importlib
import numpy as np
try:
    import wandb
except ImportError:
    wandb = None

# --- Path Setup ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# --- Imports ---
from src.training.surrogate_trainer import run_surrogate
from src.utils.report import create_training_dashboard
from src.utils.metrics import calculate_and_save_metrics_csv
from src.training.envs import set_global_seeds, make_envs, VecNormalize
from src.tools.energy_benchmark import EnergyBenchmark
from src.training.evaluate import evaluate_snn

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _save_validation_traces(log_dir: str, validation_data: dict) -> None:
    """
    Persist compact validation traces needed for post-hoc multi-seed plots.
    """
    if not isinstance(validation_data, dict) or not validation_data:
        return
    payload = {}
    for key in ("critic_values_single_episode", "intra_episode_values", "critic_values", "critic_timings"):
        if key in validation_data:
            arr = np.asarray(validation_data[key]).reshape(-1)
            if arr.size > 0:
                payload[key] = arr
    step_traces = validation_data.get("step_traces", {})
    if isinstance(step_traces, dict) and "episode_index" in step_traces:
        ep_idx = np.asarray(step_traces["episode_index"]).reshape(-1)
        if ep_idx.size > 0:
            payload["step_traces_episode_index"] = ep_idx
    if not payload:
        return
    out_path = os.path.join(log_dir, "validation_data.npz")
    try:
        np.savez_compressed(out_path, **payload)
        logger.info(f"Saved validation traces for multi-seed plotting: {out_path}")
    except Exception as e:
        logger.warning(f"Failed to save validation traces at {out_path}: {e}")

def register_custom_envs():
    """
    Dynamically imports custom environment modules to register them with Gymnasium.
    Add new custom environments to the 'custom_modules' list below.
    """
    custom_modules = [
        "src.envs.t_maze",  # Registers 'tmaze-v0'
        # "src.envs.snake_game",  # Example for future
    ]
    
    for module in custom_modules:
        try:
            importlib.import_module(module)
            logger.info(f"Successfully registered custom module: {module}")
        except ImportError:
            # It's okay if a module is missing (e.g. running on a different project)
            logger.debug(f"Could not import {module} - skipping registration.")
        except Exception as e:
            logger.warning(f"Error registering {module}: {e}")

def main():
    # 0. Register Custom Environments
    register_custom_envs()

    # 1. Parse Arguments
    parser = argparse.ArgumentParser("Universal SNN Timing Critic")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--run-name", type=str, default=None, help="Override run name")
    parser.add_argument("--log-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--env-active", type=str, default=None, help="Override env.kwargs.active (true/false)")
    args = parser.parse_args()

    # 2. Load & Override Config
    if not os.path.exists(args.config):
        args.config = os.path.join(repo_root, args.config)
        
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Backward compatibility: some configs use env_kwargs instead of kwargs.
    env_cfg = config.setdefault("env", {})
    if "kwargs" not in env_cfg:
        env_cfg["kwargs"] = env_cfg.get("env_kwargs", {}) or {}
    elif env_cfg.get("kwargs") is None:
        env_cfg["kwargs"] = {}

    if args.seed is not None:
        config['env_seed'] = args.seed
    if args.run_name is not None:
        config['run_name'] = args.run_name
    if args.log_dir is not None:
        config['log_dir'] = args.log_dir
        # Redirect plots to subfolder to keep root clean
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

    # 3. Weights & Biases
    use_wandb = bool(config.get("reporting", {}).get("wandb", {}).get("use", False))
    if use_wandb and wandb is None:
        raise ModuleNotFoundError(
            "reporting.wandb.use is true but 'wandb' is not installed. "
            "Install wandb or set reporting.wandb.use to false."
        )
    if use_wandb:
        wandb.init(
            project=config.get("project_name", "universal-snn-project"), 
            config=config, 
            name=config.get("run_name"), 
            reinit=True
        )

    # 4. Global Seeding
    seed_cfg = config.get("seed_control", {})
    set_global_seeds(
        config['env_seed'],
        deterministic_torch=seed_cfg.get("deterministic_torch", True),
        cudnn_benchmark=seed_cfg.get("cudnn_benchmark", False),
    )
    
    env_kwargs = config.get("env", {}).get("kwargs", {})
    logger.info(
        f"--- Running SNN Timing Critic | Env: {config['env']['id']} | Seed: {config['env_seed']} "
        f"|length: {env_kwargs.get('length', 'N/A')} | active: {env_kwargs.get('active', 'N/A')} ---"
    )
    
    # 5. Run Surrogate Trainer
    result = run_surrogate(config=config)
    agent = result["agent"]

    # 6. Save Metrics & Dashboard
    env_name = config.get('env', {}).get('id', 'unknown_env')
    
    calculate_and_save_metrics_csv(
        result, 
        config['log_dir'], 
        env_name=env_name,
        reward_threshold=config['ppo'].get("reward_threshold", 0.0)
    )
    _save_validation_traces(config["log_dir"], result.get("validation_data", {}) or {})
    
    if os.path.exists(config.get('plots_dir', '')):
        try:
            create_training_dashboard(
                log_dir=config['log_dir'],
                output_dir=config['plots_dir'],
                title_prefix=f"{env_name} Seed {config['env_seed']}",
                threshold=float(config.get("ppo", {}).get("reward_threshold", 475.0)),
                validation_data=result.get('validation_data'),
                config=config,
            )
        except Exception as e:
            logger.error(f"Reporting failed: {e}")

    # 7. Energy Benchmarking
    bench_cfg = config.get("benchmark", {}) or {}
    benchmark_enabled = bool(
        bench_cfg.get(
            "enabled",
            bench_cfg.get("measure_inference_energy", bench_cfg.get("run_for_ann", True)),
        )
    )
    if benchmark_enabled:
        logger.info("Starting Energy Benchmark...")
        try:
            # Ideally use the trained agent from result['agent']
            agent = result.get('agent')
            
            if agent:
                env_cfg = config.get("env", {})
                # make_envs returns (train_env, eval_env). We use the eval_env for benchmarking.
                _, bench_env = make_envs(
                    seed=config['env_seed'], 
                    env_id=env_name,
                    n_envs=1,
                    partial_obs=env_cfg.get("partial_obs"),
                    frame_stack=env_cfg.get("frame_stack"),
                    frame_stack_flatten=env_cfg.get("frame_stack_flatten", True),
                )
                
                # Create Benchmark Tool
                benchmark = EnergyBenchmark()
                
                # Wrapper for episode execution
                def episode_fn(model):
                     return evaluate_snn(bench_env, model, sticky_action=config['training'].get('sticky_action', True))

                # Run
                metrics_snn = benchmark.benchmark_model(
                    agent, 
                    episode_fn, 
                    num_episodes=bench_cfg.get('num_episodes_for_benchmark', 100),
                    model_type="SNN",
                    prev_train_energy=0.0, # We only benchmark inference here
                    warmup_runs=int(bench_cfg.get('warmup_runs_per_measure', 1)),
                    active_repeat=int(bench_cfg.get('energy_runs_per_measure', 1)),
                )
                
                # Generate Report
                report = benchmark.generate_report(metrics_snn, metrics_snn)
                
                # Save to text file
                with open(os.path.join(config['log_dir'], "energy_report.txt"), "w") as f:
                    f.write(report)
                    
                bench_env.close()
            else:
                logger.warning("Agent not found in result, skipping benchmark.")

        except Exception as e:
            logger.error(f"Energy benchmark failed: {e}")
    else:
        logger.info("Energy benchmark disabled by config.")

    if wandb is not None and wandb.run:
        wandb.finish()

    logger.info("--- Experiment Complete ---")

if __name__ == "__main__":
    main()
