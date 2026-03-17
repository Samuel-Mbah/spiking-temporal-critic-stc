# """
# Experiment: ANN Baseline (Research Grade)
# - Mode: Single Seed Execution (CLI Driven)
# - Logic: Trains a standard PPO agent with an ANN architecture.
# - Usage: python ann_baseline.py --config path/to/yaml --seed 1 --log-dir results/seed_1
# """

# import argparse
# import os
# import sys
# import yaml
# import wandb
# import logging
# import json

# # --- Path Setup ---
# repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if repo_root not in sys.path:
#     sys.path.insert(0, repo_root)

# from src.training.baseline_trainer import run_baseline
# from src.utils.report import create_training_dashboard
# from src.utils.metrics import calculate_and_save_metrics_csv
# from src.training.envs import set_global_seeds
# from src.tools.energy_benchmark import EnergyBenchmark
# from src.training.evaluate import evaluate_snn # Use universal evaluator
# import cartpole.src.envs.t_maze

# # Logging Setup
# logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
# logger = logging.getLogger(__name__)

# def main():
#     # 1. Parsing Arguments (Matches run_experiments.sh)
#     parser = argparse.ArgumentParser("CartPole ANN Baseline")
#     parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
#     parser.add_argument("--seed", type=int, default=None, help="Random seed for this run")
#     parser.add_argument("--run-name", type=str, default=None, help="Unique identifier for WandB/Logging")
#     parser.add_argument("--log-dir", type=str, default=None, help="Specific output directory for this seed")
#     args = parser.parse_args()

#     # 2. Load Configuration
#     if not os.path.exists(args.config):
#         # Fallback to looking in repo root
#         args.config = os.path.join(repo_root, args.config)
        
#     with open(args.config, 'r') as f:
#         config = yaml.safe_load(f)

#     # 3. Apply CLI Overrides (Crucial for shell script compatibility)
#     if args.seed is not None:
#         config['env_seed'] = args.seed
#     if args.run_name is not None:
#         config['run_name'] = args.run_name
#     if args.log_dir is not None:
#         config['log_dir'] = args.log_dir
#         config['plots_dir'] = os.path.join(args.log_dir, "plots") # Redirect plots inside log_dir

#     # Ensure both log and plots directories exist
#     os.makedirs(config['log_dir'], exist_ok=True)
#     if 'plots_dir' in config:
#         os.makedirs(config['plots_dir'], exist_ok=True)

#     # 4. Weights & Biases (Optional Industry Standard Tracking)
#     if config.get("reporting", {}).get("wandb", {}).get("use", False):
#         wandb.init(
#             project=config.get("project_name", "cartpole-research"),
#             config=config,
#             name=config.get("run_name"),
#             reinit=True,
#             mode="online"
#         )

#     # 5. Set Global Seed (Reproducibility)
#     seed_cfg = config.get("seed_control", {})
#     set_global_seeds(
#         config['env_seed'],
#         deterministic_torch=seed_cfg.get("deterministic_torch", True),
#         cudnn_benchmark=seed_cfg.get("cudnn_benchmark", False),
#     )

#     logger.info(f"--- Starting Run: {config['run_name']} | Seed: {config['env_seed']} ---")

#     # 6. Run Training
#     # run_baseline returns a dictionary with 'agent', 'logger', etc.
#     result = run_baseline(config=config)
#     agent = result["agent"]

#     # 7. Post-Training Analysis
#     logger.info("--- Training Complete. Starting Analysis... ---")

#     # A. Save CSV Metrics
#     try:
#         calculate_and_save_metrics_csv(
#             result, 
#             config['log_dir'], 
#             env_name=config['env'].get('id', 'CartPole-v1'),
#             reward_threshold=config['ppo'].get("reward_threshold", 475.0)
#         )
#     except Exception as e:
#         logger.error(f"Failed to save metrics CSV: {e}")

#     # B. Generate Single-Seed Dashboard
#     if os.path.exists(config['plots_dir']):
#         create_training_dashboard(
#             log_dir=config['log_dir'],
#             output_dir=config['plots_dir'],
#             title_prefix=f"Seed {config['env_seed']}",
#             threshold=config['ppo'].get("reward_threshold", 475.0),
#             config=config,
#         )

#     # C. Energy Benchmarking (If enabled)
#     if config.get("benchmark", {}).get("run_for_ann", True):
#         try:
#             from src.training.envs import make_envs, VecNormalize
#             env_cfg = config.get("env", {})
#             # Create a clean env for benchmarking
#             _, bench_env = make_envs(
#                 seed=config['env_seed'] + 1000, 
#                 env_id=env_cfg.get('id', 'CartPole-v1'),
#                 n_envs=1,
#                 partial_obs=env_cfg.get("partial_obs"),
#                 frame_stack=env_cfg.get("frame_stack"),
#                 frame_stack_flatten=env_cfg.get("frame_stack_flatten", True),
#             )
            
#             # Handle normalization
#             if config['env'].get('vec_normalize', False) and hasattr(agent, 'obs_rms'):
#                  bench_env = VecNormalize(bench_env, training=False)
#                  bench_env.obs_rms = agent.obs_rms

#             bencher = EnergyBenchmark()
#             metrics = bencher.benchmark_model(
#                 model=agent, 
#                 episode_fn=lambda m: evaluate_snn(bench_env, m, sticky_action=False),
#                 num_episodes=config['benchmark'].get('num_episodes_for_benchmark', 50),
#                 model_type="ANN"
#             )
#             # Save to JSON
#             with open(os.path.join(config['log_dir'], "benchmark_metrics.json"), "w") as f:
#                 json.dump(metrics.__dict__, f, indent=4)
#         except Exception as e:
#             logger.error(f"Benchmark failed: {e}")

#     if wandb.run:
#         wandb.finish()

#     logger.info(f"--- Seed {config['env_seed']} Finished Successfully ---")

# if __name__ == "__main__":
#     main()






# """
# Experiment: Actor-Only ANN to SNN Conversion
# - Mode: Single Seed
# """
# import argparse, os, sys, yaml, wandb, logging, json

# repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if repo_root not in sys.path:
#     sys.path.insert(0, repo_root)

# from src.training.conversion_trainer import run_conversion
# from src.utils.report import create_training_dashboard
# from src.utils.metrics import calculate_and_save_metrics_csv
# from src.training.envs import set_global_seeds
# from src.utils.plotting import plot_snn_phase

# logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
# logger = logging.getLogger(__name__)

# def main():
#     parser = argparse.ArgumentParser("ANN2SNN Actor Only")
#     parser.add_argument("--config", type=str, required=True)
#     parser.add_argument("--seed", type=int, default=None)
#     parser.add_argument("--run-name", type=str, default=None)
#     parser.add_argument("--log-dir", type=str, default=None)
#     args = parser.parse_args()

#     # Load & Override
#     if not os.path.exists(args.config): args.config = os.path.join(repo_root, args.config)
#     with open(args.config, 'r') as f: config = yaml.safe_load(f)

#     if args.seed: config['env_seed'] = args.seed
#     if args.run_name: config['run_name'] = args.run_name
#     if args.log_dir: 
#         config['log_dir'] = args.log_dir
#         config['plots_dir'] = os.path.join(args.log_dir, "plots")

#     # Ensure directories exist
#     os.makedirs(config['log_dir'], exist_ok=True)
#     if 'plots_dir' in config:
#         os.makedirs(config['plots_dir'], exist_ok=True)

#     if config.get("reporting", {}).get("wandb", {}).get("use", False):
#         wandb.init(project=config.get("project_name"), config=config, name=config.get("run_name"), reinit=True)

#     seed_cfg = config.get("seed_control", {})
#     set_global_seeds(
#         config['env_seed'],
#         deterministic_torch=seed_cfg.get("deterministic_torch", True),
#         cudnn_benchmark=seed_cfg.get("cudnn_benchmark", False),
#     )

#     # Run Pipeline
#     result = run_conversion(config=config)
    
#     calculate_and_save_metrics_csv(result, config['log_dir'], env_name=config['env']['id'])
    
#     if os.path.exists(config.get('plots_dir', '')):
#         plot_snn_phase(result, config['plots_dir'])
#         create_training_dashboard(
#             config['log_dir'],
#             config['plots_dir'],
#             title_prefix=f"Seed {config['env_seed']}",
#             validation_data=result.get("validation_data"),
#             config=config,
#         )
       

#     if wandb.run: wandb.finish()

# if __name__ == "__main__":
#     main()



# """
# Experiment: Full ANN to SNN Conversion
# - Mode: Single Seed
# - Stages: 
#   1. Train ANN (or load)
#   2. Convert Actor & Critic
#   3. Finetune
# """
# import argparse, os, sys, yaml, wandb, logging, json

# repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if repo_root not in sys.path:
#     sys.path.insert(0, repo_root)

# from src.training.conversion_trainer import run_conversion
# from src.utils.report import create_training_dashboard
# from src.utils.metrics import calculate_and_save_metrics_csv
# from src.training.envs import set_global_seeds
# from src.utils.plotting import plot_snn_phase

# logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
# logger = logging.getLogger(__name__)

# def main():
#     parser = argparse.ArgumentParser("ANN2SNN Full")
#     parser.add_argument("--config", type=str, required=True)
#     parser.add_argument("--seed", type=int, default=None)
#     parser.add_argument("--run-name", type=str, default=None)
#     parser.add_argument("--log-dir", type=str, default=None)
#     args = parser.parse_args()

#     # Load & Override
#     if not os.path.exists(args.config): args.config = os.path.join(repo_root, args.config)
#     with open(args.config, 'r') as f: config = yaml.safe_load(f)

#     if args.seed: config['env_seed'] = args.seed
#     if args.run_name: config['run_name'] = args.run_name
#     if args.log_dir: 
#         config['log_dir'] = args.log_dir
#         config['plots_dir'] = os.path.join(args.log_dir, "plots")

#     # Ensure directories exist
#     os.makedirs(config['log_dir'], exist_ok=True)
#     if 'plots_dir' in config:
#         os.makedirs(config['plots_dir'], exist_ok=True)

#     if config.get("reporting", {}).get("wandb", {}).get("use", False):
#         wandb.init(project=config.get("project_name"), config=config, name=config.get("run_name"), reinit=True)

#     seed_cfg = config.get("seed_control", {})
#     set_global_seeds(
#         config['env_seed'],
#         deterministic_torch=seed_cfg.get("deterministic_torch", True),
#         cudnn_benchmark=seed_cfg.get("cudnn_benchmark", False),
#     )

#     # Run Conversion Pipeline
#     # run_conversion handles: Train ANN -> Convert -> FineTune -> Return Result
#     result = run_conversion(config=config)
    
#     # Save Metrics for the *Finetuning* phase usually
#     calculate_and_save_metrics_csv(result, config['log_dir'], env_name=config['env']['id'])
    
#     # Generate dashboard for this specific seed
#     if os.path.exists(config.get('plots_dir', '')):
#         plot_snn_phase(result, config['plots_dir'])
#         create_training_dashboard(
#             config['log_dir'],
#             config['plots_dir'],
#             title_prefix=f"Seed {config['env_seed']}",
#             validation_data=result.get("validation_data"),
#             config=config,
#         )

#     if wandb.run: wandb.finish()

# if __name__ == "__main__":
#     main()




# """
# Experiment: SNN Actor + ANN Critic (Direct Training)
# - Mode: Single Seed Execution
# - Logic: SNN Actor (Surrogate) + Standard ANN Critic
# """
# import argparse, os, sys, yaml, wandb, logging, json

# repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if repo_root not in sys.path:
#     sys.path.insert(0, repo_root)

# from src.training.surrogate_trainer import run_surrogate
# from src.utils.report import create_training_dashboard
# from src.utils.metrics import calculate_and_save_metrics_csv
# from src.training.envs import set_global_seeds, make_envs, VecNormalize
# from src.tools.energy_benchmark import EnergyBenchmark
# from src.training.evaluate import evaluate_snn

# logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
# logger = logging.getLogger(__name__)

# def main():
#     parser = argparse.ArgumentParser("SNN Actor ANN Critic")
#     parser.add_argument("--config", type=str, required=True)
#     parser.add_argument("--seed", type=int, default=None)
#     parser.add_argument("--run-name", type=str, default=None)
#     parser.add_argument("--log-dir", type=str, default=None)
#     args = parser.parse_args()

#     # Load Config
#     if not os.path.exists(args.config): args.config = os.path.join(repo_root, args.config)
#     with open(args.config, 'r') as f: config = yaml.safe_load(f)

#     # CLI Overrides
#     if args.seed: config['env_seed'] = args.seed
#     if args.run_name: config['run_name'] = args.run_name
#     if args.log_dir: 
#         config['log_dir'] = args.log_dir
#         config['plots_dir'] = os.path.join(args.log_dir, "plots")

#     # Ensure directories exist
#     os.makedirs(config['log_dir'], exist_ok=True)
#     if 'plots_dir' in config:
#         os.makedirs(config['plots_dir'], exist_ok=True)

#     # Run WandB
#     if config.get("reporting", {}).get("wandb", {}).get("use", False):
#         wandb.init(project=config.get("project_name"), config=config, name=config.get("run_name"), reinit=True)

#     seed_cfg = config.get("seed_control", {})
#     set_global_seeds(
#         config['env_seed'],
#         deterministic_torch=seed_cfg.get("deterministic_torch", True),
#         cudnn_benchmark=seed_cfg.get("cudnn_benchmark", False),
#     )
    
#     # Train
#     result = run_surrogate(config=config)
    
#     # Save Metrics
#     calculate_and_save_metrics_csv(result, config['log_dir'], env_name=config['env']['id'])
    
#     # Generate Dashboard
#     val_data = result.get('validation_data', {})
#     if val_data and "critic_timings" in val_data:
#         val_data.pop("critic_timings")  # Remove timing data to suppress that specific plot
        
#     if os.path.exists(config.get('plots_dir', '')):
#         create_training_dashboard(
#             config['log_dir'],
#             config['plots_dir'],
#             title_prefix=f"Seed {config['env_seed']}",
#             validation_data=val_data,
#             config=config,
#         )
    
#     # Energy Benchmarking
#     logger.info("Starting Energy Benchmark...")
#     try:
#         # Re-create env/agent from result or new for benchmarking
#         # Ideally use the trained agent from result['agent']
#         agent = result.get('agent')
        
#         if agent:
#             env_cfg = config.get("env", {})
#             # make_envs returns (train_env, eval_env). We use the eval_env for benchmarking.
#             _, bench_env = make_envs(
#                 seed=config['env_seed'], 
#                 env_id=env_cfg.get('id', 'CartPole-v1'),
#                 n_envs=1,
#                 partial_obs=env_cfg.get("partial_obs"),
#                 frame_stack=env_cfg.get("frame_stack"),
#                 frame_stack_flatten=env_cfg.get("frame_stack_flatten", True),
#             )
            
#             # Create Benchmark Tool
#             benchmark = EnergyBenchmark()
            
#             # Wrapper for episode execution
#             def episode_fn(model):
#                  return evaluate_snn(bench_env, model, sticky_action=config['training'].get('sticky_action', True))

#             # Run
#             metrics_snn = benchmark.benchmark_model(
#                 agent, 
#                 episode_fn, 
#                 num_episodes=config.get('benchmark', {}).get('num_episodes_for_benchmark', 100),
#                 model_type="SNN",
#                 prev_train_energy=0.0 # We only benchmark inference here
#             )
            
#             # Create Dummy ANN metrics for comparison (or load if available)
#             # For now, we pass SNN metrics as ANN just to see the SNN report part, 
#             # or you can implement a baseline load here.
#             report = benchmark.generate_report(metrics_snn, metrics_snn)
#             # print(report) # FORCE PRINT TO STDOUT
            
#             # Save to text file
#             with open(os.path.join(config['log_dir'], "energy_report.txt"), "w") as f:
#                 f.write(report)
                
#             bench_env.close()
#         else:
#             logger.warning("Agent not found in result, skipping benchmark.")

#     except Exception as e:
#         logger.error(f"Energy benchmark failed: {e}")

#     if wandb.run: wandb.finish()

# if __name__ == "__main__":
#     main()











# """
# Experiment: SNN Actor + Timing Critic (Research Grade)
# - Mode: Single Seed Execution
# - Logic: Surrogate gradient training of SNN Actor with Temporal Critic.
# """

# import argparse
# import os
# import sys
# import yaml
# import wandb
# import logging
# import json

# repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if repo_root not in sys.path:
#     sys.path.insert(0, repo_root)

# from src.training.surrogate_trainer import run_surrogate
# from src.utils.report import create_training_dashboard
# from src.utils.metrics import calculate_and_save_metrics_csv
# from src.training.envs import set_global_seeds, make_envs, VecNormalize
# from src.tools.energy_benchmark import EnergyBenchmark
# from src.training.evaluate import evaluate_snn

# logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
# logger = logging.getLogger(__name__)

# def main():
#     parser = argparse.ArgumentParser("SNN Timing Critic")
#     parser.add_argument("--config", type=str, required=True)
#     parser.add_argument("--seed", type=int, default=None)
#     parser.add_argument("--run-name", type=str, default=None)
#     parser.add_argument("--log-dir", type=str, default=None)
#     args = parser.parse_args()

#     # Load & Override Config
#     if not os.path.exists(args.config): args.config = os.path.join(repo_root, args.config)
#     with open(args.config, 'r') as f: config = yaml.safe_load(f)

#     if args.seed is not None: config['env_seed'] = args.seed
#     if args.run_name is not None: config['run_name'] = args.run_name
#     if args.log_dir is not None:
#         config['log_dir'] = args.log_dir
#         config['plots_dir'] = os.path.join(args.log_dir, "plots")

#     # Ensure directories exist
#     os.makedirs(config['log_dir'], exist_ok=True)
#     if 'plots_dir' in config:
#         os.makedirs(config['plots_dir'], exist_ok=True)

#     # WandB
#     if config.get("reporting", {}).get("wandb", {}).get("use", False):
#         wandb.init(project=config.get("project_name"), config=config, name=config.get("run_name"), reinit=True)

#     # Execution
#     seed_cfg = config.get("seed_control", {})
#     set_global_seeds(
#         config['env_seed'],
#         deterministic_torch=seed_cfg.get("deterministic_torch", True),
#         cudnn_benchmark=seed_cfg.get("cudnn_benchmark", False),
#     )
#     logger.info(f"--- Running SNN Timing Critic | Seed: {config['env_seed']} ---")
    
#     # Run Surrogate Trainer
#     result = run_surrogate(config=config)
#     agent = result["agent"]

#     # Save Metrics
#     calculate_and_save_metrics_csv(
#         result, 
#         config['log_dir'], 
#         env_name=config['env'].get('id', 'CartPole-v1'),
#         reward_threshold=config['ppo'].get("reward_threshold", 475.0)
#     )
    
#     if os.path.exists(config.get('plots_dir', '')):
#         create_training_dashboard(
#             config['log_dir'],
#             config['plots_dir'],
#             title_prefix=f"Seed {config['env_seed']}",
#             validation_data=result.get('validation_data'),
#             config=config,
#         )

#     # Energy Benchmarking
#     logger.info("Starting Energy Benchmark...")
#     try:
#         # Re-create env/agent from result or new for benchmarking
#         # Ideally use the trained agent from result['agent']
#         agent = result.get('agent')
        
#         if agent:
#             env_cfg = config.get("env", {})
#             # make_envs returns (train_env, eval_env). We use the eval_env for benchmarking.
#             _, bench_env = make_envs(
#                 seed=config['env_seed'], 
#                 env_id=env_cfg.get('id', 'CartPole-v1'),
#                 n_envs=1,
#                 partial_obs=env_cfg.get("partial_obs"),
#                 frame_stack=env_cfg.get("frame_stack"),
#                 frame_stack_flatten=env_cfg.get("frame_stack_flatten", True),
#             )
            
#             # Create Benchmark Tool
#             benchmark = EnergyBenchmark()
            
#             # Wrapper for episode execution
#             def episode_fn(model):
#                  return evaluate_snn(bench_env, model, sticky_action=config['training'].get('sticky_action', True))

#             # Run
#             metrics_snn = benchmark.benchmark_model(
#                 agent, 
#                 episode_fn, 
#                 num_episodes=config.get('benchmark', {}).get('num_episodes_for_benchmark', 100),
#                 model_type="SNN",
#                 prev_train_energy=0.0 # We only benchmark inference here
#             )
            
#             # Create Dummy ANN metrics for comparison (or load if available)
#             # For now, we pass SNN metrics as ANN just to see the SNN report part, 
#             # or you can implement a baseline load here.
#             report = benchmark.generate_report(metrics_snn, metrics_snn)
#             # print(report) # FORCE PRINT TO STDOUT
            
#             # Save to text file
#             with open(os.path.join(config['log_dir'], "energy_report.txt"), "w") as f:
#                 f.write(report)
                
#             bench_env.close()
#         else:
#             logger.warning("Agent not found in result, skipping benchmark.")

#     except Exception as e:
#         logger.error(f"Energy benchmark failed: {e}")

#     if wandb.run: wandb.finish()

# if __name__ == "__main__":
#     main()

