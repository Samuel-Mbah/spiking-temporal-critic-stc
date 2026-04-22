"""
Experiment: Full ANN to SNN Conversion (Universal)
- Mode: Single Seed
- Stages: 
  1. Train ANN (or load)
  2. Convert Actor & Critic
  3. Finetune
- Supports: CartPole-v1, tmaze-v0, and other registered Gym environments.
"""
import argparse
import os
import sys
import yaml
import wandb
import logging
import json
import importlib

# --- Path Setup ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# --- Imports ---
from src.training.conversion_trainer import run_conversion
from src.utils.report import create_training_dashboard
from src.utils.metrics import calculate_and_save_metrics_csv
from src.training.envs import set_global_seeds

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def register_custom_envs():
    """
    Dynamically imports custom environment modules to register them with Gymnasium.
    Add new custom environments to the 'custom_modules' list below.
    """
    custom_modules = [
        "src.envs.t_maze",  # Registers 'tmaze-v0'
        # "src.envs.snake_game",  # Future envs
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
    parser = argparse.ArgumentParser("Universal ANN2SNN Full")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--run-name", type=str, default=None, help="Override run name")
    parser.add_argument("--log-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--env-active", type=str, default=None, help="Override env.kwargs.active (true/false)")
    args = parser.parse_args()

    # 2. Load & Override Config
    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join(repo_root, args.config)
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

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

    # 3. Weights & Biases
    if config.get("reporting", {}).get("wandb", {}).get("use", False):
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

    env_cfg = config.get("env", {})
    env_kwargs = env_cfg.get("kwargs", {}) or {}
    logger.info(
        f"--- Starting Full Conversion: {config['run_name']} | Env: {env_cfg.get('id', 'unknown')} | "
        f"Seed: {config['env_seed']} Len: {env_kwargs.get('length', 'N/A')} "
        f"Active: {env_kwargs.get('active', 'N/A')} ---"
    )

    # 5. Run Conversion Pipeline
    # run_conversion handles: Train ANN -> Convert -> FineTune -> Return Result
    result = run_conversion(config=config)
    
    # 6. Reporting & Metrics
    # Safely get env name
    env_name = config.get('env', {}).get('id', 'unknown_env')
    
    calculate_and_save_metrics_csv(result, config['log_dir'], env_name=env_name)
    
    if os.path.exists(config.get('plots_dir', '')):
        try:
            val_data = result.get("validation_data") or {}
            if result.get("comparison_metrics"):
                val_data["comparison_metrics"] = result["comparison_metrics"]
            create_training_dashboard(
                log_dir=config['log_dir'],
                output_dir=config['plots_dir'],
                threshold=config['ppo'].get("reward_threshold", 0.0),
                title_prefix=f"{env_name} Seed {config['env_seed']}",
                validation_data=val_data,
                dashboard_mode="ann2snn_both_conversion",
                config=config,
            )
        except Exception as e:
            logger.error(f"Reporting failed: {e}")

    if wandb.run:
        wandb.finish()
        
    logger.info("--- Experiment Complete ---")

if __name__ == "__main__":
    main()
