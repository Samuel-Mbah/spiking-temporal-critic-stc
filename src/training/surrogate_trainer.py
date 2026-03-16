import os
import time
import torch
import numpy as np
import logging
import sys
import glob  # <--- Essential for robust file matching
import traceback
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any

from src.training.core_trainer import CoreTrainer
from src.training.envs import make_envs, VecNormalize
from src.training.agents import make_agent, resolve_cartpole_types
from src.training.device import coerce_module_to_device
from src.training.hooks import EnergyHook
from src.training.record import record_best_agent
from src.training.evaluate import evaluate_snn, get_last_latency

from src.utils.logger import PPOLogger
from src.utils.torch_utils import first_output
from src.utils.checkpoint import save_checkpoint, load_checkpoint

# Configure module-level logger
logger_mod = logging.getLogger(__name__)


def _get_git_commit_hash() -> str:
    """Best-effort retrieval of current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"

def run_surrogate(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes Direct SNN training using surrogate gradients.
    
    Features:
    - Auto-resumption from checkpoints (glob-based)
    - Energy and Latency benchmarking hooks
    - "Best Model" preservation logic
    - Research-grade logging and exception handling
    """
    # --- 1. Config Unpacking & Path Setup ---
    env_cfg = config.get("env", {})
    train_cfg = config.get("training", {})
    ppo_cfg = config.get("ppo", {})
    model_cfg = config.get("model", {})
    log_cfg = config.get("logging", {})
    snn_cfg = config.get("snn", {})
    post_eval_cfg = config.get("post_eval", ppo_cfg.get("post_eval", {})) or {}

    # Consistency Fix: Define log_dir_root once to use everywhere
    log_dir_root = config.get("log_dir", "logs")
    ckpt_dir = os.path.join(log_dir_root, "checkpoints")

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # --- 2. Environment Setup ---
    env_train, env_eval = make_envs(
        seed=config.get("env_seed", 42),
        env_id=env_cfg.get("id", "CartPole-v1"),
        n_envs=env_cfg.get("n_envs", 8),
        env_kwargs=env_cfg.get("kwargs", {}),
        partial_obs=env_cfg.get("partial_obs"),
        frame_stack=env_cfg.get("frame_stack"),
        frame_stack_flatten=env_cfg.get("frame_stack_flatten", True),
    )

    # Apply Normalization if configured
    if env_cfg.get("vec_normalize", False):
        env_train = VecNormalize(env_train, training=True, norm_obs=True, norm_reward=True, clip_reward=10.0)
        env_eval_wrapper = VecNormalize(env_eval, training=False, norm_obs=True, norm_reward=False, clip_reward=10.0)
        # Sync stats from training env to eval env
        env_eval_wrapper.obs_rms = env_train.obs_rms
        env_eval = env_eval_wrapper

    # --- 3. Agent Setup ---
    mode = model_cfg.get("mode", "snn_actor_ann_critic")
    actor_type, critic_type = resolve_cartpole_types(mode)

    agent = make_agent(
        actor_type=actor_type,
        critic_type=critic_type,
        hidden_dim=model_cfg.get("hidden_dim", 64),
        in_dim=model_cfg.get("in_features", 4),
        act_dim=env_train.single_action_space.n if hasattr(env_train, 'single_action_space') else env_train.action_space.n,
        gamma=ppo_cfg.get("gamma", 0.99),
        critic_informs_actor=bool(model_cfg.get("critic_informs_actor", False)),
        detach_critic_for_actor=bool(model_cfg.get("detach_critic_for_actor", True)),
        normalize_critic_for_actor=bool(model_cfg.get("normalize_critic_for_actor", True)),
        critic_actor_value_clip=model_cfg.get("critic_actor_value_clip", 5.0),
        critic_actor_norm_momentum=float(model_cfg.get("critic_actor_norm_momentum", 0.01)),
        **snn_cfg
    )

    agent = agent.to(device)
    coerce_module_to_device(agent, device)
    
    # Attach normalization stats to agent for portability
    if env_cfg.get("vec_normalize", False):
        agent.obs_rms = env_train.obs_rms

    # --- 4. Optimization & Logging ---
    optimizer = torch.optim.Adam(agent.parameters(), lr=float(ppo_cfg.get("lr", 1e-3)))
    initial_lr = float(ppo_cfg.get("lr", 1e-3))
    
    n_eval_episodes = int(ppo_cfg.get("eval_episodes", 5))
    solved_window_eval_episodes = int(ppo_cfg.get("solved_window_eval_episodes", 100))
    window_size = max(1, solved_window_eval_episodes // n_eval_episodes)

    logger = PPOLogger(log_dir=log_dir_root, window=window_size)

    # Instrumentation
    energy_cfg = config.get("energy", {}) or {}
    energy_hook = EnergyHook(
        sample_interval=float(energy_cfg.get("sample_interval", 0.02)),
        gpu_index=int(energy_cfg.get("gpu_index", 0)),
    )
    if bool(energy_cfg.get("calibrate_idle", True)):
        energy_hook.calibrate_idle(float(energy_cfg.get("idle_calibration_seconds", 2.0)))
    logger.record("energy/idle_power_watts", float(energy_hook.idle_power_watts), exclude_from_console=True)
    hooks = {
        "on_rollout_start": energy_hook.on_rollout_start,
        "on_rollout_end": energy_hook.on_rollout_end,
    }

    trainer = CoreTrainer(
        agent=agent,
        env_train=env_train,
        env_eval=env_eval,
        optimizer=optimizer,
        logger=logger,
        config=config,
        hooks=hooks,
    )
    try:
        actor_T = getattr(agent.actor, "T", None)
        critic_T = getattr(agent.critic, "T", None)
        logger_mod.info(f"Actor T={actor_T}, Critic T={critic_T}")
        if actor_T is not None:
            logger.record("snn/actor_T", actor_T, exclude_from_console=True)
        if critic_T is not None:
            logger.record("snn/critic_T", critic_T, exclude_from_console=True)
    except Exception:
        pass

    # --- 5. Training Parameters ---
    total_updates = int(train_cfg.get("total_updates", 300))
    ckpt_interval = int(log_cfg.get("checkpoint_interval_updates", 50))
    reward_threshold = float(ppo_cfg.get("reward_threshold", 475.0))
    success_rate_threshold = float(ppo_cfg.get("success_rate_threshold", 95.0))
    save_ckpt_flag = config.get("save_ckpt", False)
    eval_interval = int(ppo_cfg.get("eval_interval_updates", 25))

    train_rewards = []
    test_rewards = []
    resume_checkpoint_path = None
    last_checkpoint_path = None
    
    # --- 6. RESUME LOGIC (Robust) ---
    start_update = 1
    cumulative_energy = 0.0
    cumulative_dynamic_energy = 0.0
    
    # Setup variables to track "Best" state
    best_rolling_avg = -float('inf')
    has_logged_solution = False

    if os.path.exists(ckpt_dir):
        # Use glob to find all .pt files
        ckpts = glob.glob(os.path.join(ckpt_dir, "*.pt"))
        if ckpts:
            # Sort by modification time to get the absolute latest
            latest_ckpt = max(ckpts, key=os.path.getctime)
            
            logger_mod.info(f"Found checkpoint: {latest_ckpt}. Attempting resume...")
            try:
                # Load state dicts into agent/optimizer
                resume_data = load_checkpoint(
                    latest_ckpt, 
                    agent=agent, 
                    optimizer=optimizer, 
                    logger=logger,
                    map_location=device
                )
                
                # Restore Training Loop State
                start_update = resume_data.get('episode', 0) + 1
                resume_checkpoint_path = latest_ckpt
                if 'num_timesteps' in resume_data:
                    logger.num_timesteps = resume_data['num_timesteps']
                
                # Restore Metric Accumulators
                if resume_data.get('energy_metrics'):
                    cumulative_energy = resume_data['energy_metrics'].get('total', 0.0)
                    cumulative_dynamic_energy = resume_data['energy_metrics'].get('total_dynamic', 0.0)
                
                # Restore Best Rolling Avg (Prevents overwriting best model with a recovering one)
                if "best_rolling_avg" in resume_data:
                    best_rolling_avg = resume_data["best_rolling_avg"]

                logger_mod.info(f"Resuming SNN training at Update {start_update} | Best Avg: {best_rolling_avg:.2f}")
                
                if start_update > total_updates:
                    logger_mod.info(
                        "Checkpoint already reached/exceeded configured total_updates; "
                        "skipping additional training updates and proceeding to post-eval analysis."
                    )
            except Exception as e:
                logger_mod.error(f"Failed to load checkpoint: {e}")

    # --- 7. Main Training Loop ---
    start_time = time.time()
    logger_mod.info(f"Starting SNN surrogate training: {mode} (Updates: {start_update}->{total_updates})")
    print(f"\n[Trainer INTERNAL] Configured Total Updates: {total_updates}")
    sys.stdout.flush()

    update = max(0, start_update - 1)
    try:
        # Loop range handles start_update automatically
        for update in range(start_update, total_updates + 1):
            
            # Learning Rate Schedule (Linear Decay)
            progress = 1.0 - (update - 1.0) / total_updates
            current_lr = initial_lr * progress
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
            logger.record("train/lr", current_lr)

            logger.iteration = update

            # A. Training Step
            train_meter, train_t0 = energy_hook.start_span()
            tr, metrics = trainer.train_episode()
            train_full_stats = energy_hook.stop_span(train_meter, train_t0)
            train_rewards.append(tr)

            # PPO update diagnostics (helps inspect late-stage under-updating).
            if metrics:
                if "approx_kl" in metrics and "clip_fraction" in metrics:
                    logger.record("train/update_strength", float(metrics["approx_kl"]) * float(metrics["clip_fraction"]))
                if "grad_critic_total" in metrics:
                    logger.record("train/critic_grad_total", float(metrics["grad_critic_total"]))
                if "grad_critic_block_out_linear_weight" in metrics:
                    logger.record("train/critic_grad_output_head", float(metrics["grad_critic_block_out_linear_weight"]))

            # B. Metrics Logging
            # Energy
            train_joules = 0.0
            train_rollout_dynamic = 0.0
            if hasattr(energy_hook, 'energy') and energy_hook.energy:
                train_joules = energy_hook.energy.get("total_joules", 0.0)
                train_rollout_dynamic = energy_hook.energy.get("dynamic_joules", train_joules)

            train_full_joules = float(train_full_stats.get("total_joules", 0.0))
            train_full_dynamic = float(train_full_stats.get("dynamic_joules", train_full_joules))
            cumulative_energy += train_full_joules
            cumulative_dynamic_energy += train_full_dynamic
            logger.record("energy/train_rollout", train_joules)
            logger.record("energy/train_rollout_dynamic", train_rollout_dynamic)
            logger.record("energy/train_full_update", train_full_joules)
            logger.record("energy/train_full_update_dynamic", train_full_dynamic)
            logger.record("energy/total", cumulative_energy)
            logger.record("energy/total_dynamic", cumulative_dynamic_energy)

            # Latency (Algorithmic vs Wall Clock)
            snn_algo_latency = get_last_latency(agent.actor)
            critic_algo_latency = get_last_latency(agent.critic)
            steps_this_update = int(train_cfg.get("rollout_length", 2048) * env_train.num_envs)
            
            if snn_algo_latency > 0:
                logger.record("latency/mean_ms", snn_algo_latency)
                logger.record("latency/actor_spike_timing_steps", snn_algo_latency)
            if critic_algo_latency > 0:
                logger.record("latency/critic_spike_timing_steps", critic_algo_latency)
            else:
                # Fallback to wall clock
                duration = energy_hook.energy.get('duration_seconds', 0.0)
                if steps_this_update > 0:
                    logger.record("latency/mean_ms", (duration / steps_this_update) * 1000.0)
                    
            logger.num_timesteps += steps_this_update
            
            logger.record("train/rollout_steps", steps_this_update)
            logger.record("train/rollout_reward", float(tr))
            
            # C. Periodic Evaluation
            if update % eval_interval == 0 or update == total_updates:
                
                # Evaluation (w/ separate energy metering)
                eval_meter, eval_t0 = energy_hook.start_span()
                
                sticky_cfg = config.get("sticky_action", {})
                sticky = sticky_cfg.get("eval", train_cfg.get("sticky_action", False))
                
                # Accumulators
                acc_rewards, acc_lengths = [], []
                acc_spikes, acc_actor_spikes, acc_critic_spikes, acc_latency, acc_critic_latency, acc_no_spike = [], [], [], [], [], []

                for _ in range(n_eval_episodes):
                    r, l, m = evaluate_snn(env_eval, agent, sticky_action=sticky)
                    acc_rewards.append(r)
                    acc_lengths.append(l)
                    acc_spikes.append(m.get("total_spikes", 0))
                    acc_actor_spikes.append(m.get("actor_spikes", 0))
                    acc_critic_spikes.append(m.get("critic_spikes", 0))
                    acc_latency.append(m.get("mean_latency", 0))
                    acc_critic_latency.append(m.get("critic_mean_latency", 0))
                    acc_no_spike.append(m.get("no_spike_rate", 0))

                # Stop Energy Meter
                eval_stats = energy_hook.stop_span(eval_meter, eval_t0)
                eval_joules = eval_stats.get("total_joules", 0.0)
                eval_dynamic_joules = eval_stats.get("dynamic_joules", eval_joules)
                cumulative_energy += eval_joules
                cumulative_dynamic_energy += eval_dynamic_joules
                
                logger.record("energy/inference", eval_joules)
                logger.record("energy/inference_dynamic", eval_dynamic_joules)
                logger.record("energy/total", cumulative_energy)
                logger.record("energy/total_dynamic", cumulative_dynamic_energy)

                # Compute Aggregates
                te = np.mean(acc_rewards)
                avg_len = np.mean(acc_lengths)
                avg_spikes = np.mean(acc_spikes)
                avg_actor_spikes = np.mean(acc_actor_spikes) if acc_actor_spikes else 0.0
                avg_critic_spikes = np.mean(acc_critic_spikes) if acc_critic_spikes else 0.0
                avg_nsr = np.mean(acc_no_spike)
                avg_critic_latency = np.mean(acc_critic_latency) if acc_critic_latency else 0.0

                # Sparsity Calculation (Approximate)
                snn_module = agent.actor
                if hasattr(snn_module, "backbone"): snn_module = snn_module.backbone
                
                n_neurons = 0
                for block_name in ["block1", "block2", "block_out"]:
                    if hasattr(snn_module, block_name):
                        n_neurons += getattr(snn_module, block_name).linear.out_features

                T = getattr(snn_module, "T", snn_cfg.get("T", 32))
                total_capacity = n_neurons * T * avg_len
                sparsity = 1.0 - (avg_spikes / total_capacity) if total_capacity > 0 else 0.0

                # Log Evaluation Spike Metrics separately from rollout spikes
                logger.record("spikes/eval_total", avg_spikes)
                logger.record("spikes/eval_actor_total", avg_actor_spikes)
                logger.record("spikes/eval_critic_total", avg_critic_spikes)
                logger.record("eval/spikes_actor", avg_actor_spikes)
                logger.record("eval/spikes_critic", avg_critic_spikes)
                logger.record("eval/spikes_actor_per_step", avg_actor_spikes / avg_len if avg_len > 0 else 0.0)
                logger.record("eval/spikes_critic_per_step", avg_critic_spikes / avg_len if avg_len > 0 else 0.0)
                logger.record("spikes/eval_sparsity", sparsity)
                logger.record("spikes/eval_no_spike_rate", avg_nsr)
                if avg_critic_latency > 0:
                    logger.record("latency/critic_eval_spike_timing_steps", avg_critic_latency)

                test_rewards.append(te)
                
                success_rate = float(np.mean(np.asarray(acc_rewards) >= reward_threshold) * 100.0)
                success_count = int(np.sum(np.asarray(acc_rewards) >= reward_threshold))
                # Rolling Average for 'Solved' Condition & Best Model
                if len(test_rewards) >= window_size:
                    rolling_avg = np.mean(test_rewards[-window_size:])
                    is_solved = (rolling_avg >= reward_threshold) and (success_rate >= success_rate_threshold)
                else:
                    rolling_avg = np.mean(test_rewards)
                    is_solved = False
                # Legacy aliases (kept for backward compatibility with old plots)
                logger.record("eval/mean_100ep", float(rolling_avg))
                logger.record("eval/episode_reward_ma100", float(rolling_avg))
                logger.record("eval/solved_reward_avg", float(rolling_avg))
                logger.record("eval/solved_success_rate", float(success_rate))
                logger.record("eval/success_rate", float(success_rate))
                logger.record("eval/success_count", float(success_count))
                logger.record("eval/n_eval_episodes", float(n_eval_episodes))
                logger.record("eval/current_reward", float(te))
                logger.record("eval/rolling_reward", float(rolling_avg))
                logger.record("eval/rolling_window_eval_episodes", float(window_size * n_eval_episodes))
                logger.record("eval/reward_threshold", float(reward_threshold))
                logger.record("eval/success_rate_threshold", float(success_rate_threshold))
                logger.record("eval/is_solved", float(is_solved))
                logger.record_episode(reward=te, length=avg_len, success=min(100.0, success_rate), source='eval')

                # Critic value/timing probe (mean/std) for debugging scale collapse
                try:
                    probe_steps = int(log_cfg.get("critic_probe_steps", 128))
                    probe_vals = []
                    probe_vals_raw_map = []
                    probe_tau = []
                    probe_sat_lo = []
                    probe_sat_hi = []
                    obs_probe, _ = env_eval.reset()
                    for _ in range(probe_steps):
                        obs_t = torch.as_tensor(obs_probe, dtype=torch.float32, device=device)
                        if obs_t.dim() == 1:
                            obs_t = obs_t.unsqueeze(0)
                        with torch.no_grad():
                            if hasattr(agent.critic, "forward_detailed"):
                                v, tau = agent.critic.forward_detailed(obs_t)
                                probe_vals.append(float(v.mean().item()))
                                probe_tau.append(float(tau.mean().item()))
                                if hasattr(agent.critic, "_map_tau_to_value"):
                                    v_raw = agent.critic._map_tau_to_value(tau)
                                    probe_vals_raw_map.append(float(v_raw.mean().item()))
                                rmin = getattr(agent.critic, "Rmin", None)
                                rmax = getattr(agent.critic, "Rmax", None)
                                if rmin is not None:
                                    probe_sat_lo.append(float((v <= (float(rmin) + 1e-6)).float().mean().item()))
                                if rmax is not None:
                                    probe_sat_hi.append(float((v >= (float(rmax) - 1e-6)).float().mean().item()))
                            else:
                                v = agent.critic(obs_t)
                                probe_vals.append(float(v.mean().item()))
                        if hasattr(agent, "get_action"):
                            action = agent.get_action(obs_t, deterministic=True)
                        else:
                            logits = first_output(agent(obs_t))
                            action = torch.argmax(logits, dim=-1)
                        act_int = int(action.item())
                        obs_probe, _, done, _, _ = env_eval.step([act_int])
                        if done[0]:
                            obs_probe, _ = env_eval.reset()
                    if probe_vals:
                        v_mean = float(np.mean(probe_vals))
                        v_std = float(np.std(probe_vals))
                        v_min = float(np.min(probe_vals))
                        v_max = float(np.max(probe_vals))
                        logger.record("eval/critic_value_mean", v_mean)
                        logger.record("eval/critic_value_std", v_std)
                        logger.record("eval/critic_value_min", v_min)
                        logger.record("eval/critic_value_max", v_max)
                        logger_mod.info(f"[Eval Debug] Critic value mean/std/min/max: {v_mean:.4f}/{v_std:.4f}/{v_min:.4f}/{v_max:.4f}")
                    if probe_vals_raw_map:
                        rv_mean = float(np.mean(probe_vals_raw_map))
                        rv_std = float(np.std(probe_vals_raw_map))
                        rv_min = float(np.min(probe_vals_raw_map))
                        rv_max = float(np.max(probe_vals_raw_map))
                        logger.record("eval/critic_value_raw_mean", rv_mean)
                        logger.record("eval/critic_value_raw_std", rv_std)
                        logger.record("eval/critic_value_raw_min", rv_min)
                        logger.record("eval/critic_value_raw_max", rv_max)
                        logger_mod.info(f"[Eval Debug] Critic raw-map mean/std/min/max: {rv_mean:.4f}/{rv_std:.4f}/{rv_min:.4f}/{rv_max:.4f}")
                    if probe_tau:
                        t_mean = float(np.mean(probe_tau))
                        t_std = float(np.std(probe_tau))
                        logger.record("eval/critic_tau_mean", t_mean)
                        logger.record("eval/critic_tau_std", t_std)
                        logger_mod.info(f"[Eval Debug] Critic tau mean/std: {t_mean:.4f}/{t_std:.4f}")
                    if probe_sat_lo:
                        sat_lo = float(np.mean(probe_sat_lo))
                        logger.record("eval/critic_value_sat_lo_frac", sat_lo)
                        logger_mod.info(f"[Eval Debug] Critic clamp saturation @Rmin: {sat_lo:.4f}")
                    if probe_sat_hi:
                        sat_hi = float(np.mean(probe_sat_hi))
                        logger.record("eval/critic_value_sat_hi_frac", sat_hi)
                        logger_mod.info(f"[Eval Debug] Critic clamp saturation @Rmax: {sat_hi:.4f}")
                except Exception:
                    pass

                # 1. Checkpointing (Periodic)
                if save_ckpt_flag and (update % ckpt_interval == 0):
                    last_checkpoint_path = save_checkpoint(
                        agent=agent, optimizer=optimizer, logger=logger, episode=update,
                        num_timesteps=logger.num_timesteps, 
                        save_dir=ckpt_dir,
                        mean_reward=te, config=config, 
                        energy_metrics={'total': cumulative_energy, 'total_dynamic': cumulative_dynamic_energy},
                        extra_data={'best_rolling_avg': best_rolling_avg},
                        phase="surrogate_snn"
                    )

                # 2. Track "Best" Model (Research Standard)
                if rolling_avg > best_rolling_avg:
                    best_rolling_avg = rolling_avg
                    if save_ckpt_flag:
                        last_checkpoint_path = save_checkpoint(
                            agent=agent,
                            optimizer=optimizer,
                            logger=logger,
                            episode=update,
                            num_timesteps=logger.num_timesteps,
                            save_dir=ckpt_dir,
                            suffix="best", # Save as 'agent_best.pt'
                            energy_metrics={'total': cumulative_energy, 'total_dynamic': cumulative_dynamic_energy},
                            extra_data={'best_rolling_avg': best_rolling_avg},
                            phase="surrogate_snn"
                        )
                        logger_mod.info(f"New best SNN model saved! Avg Reward: {best_rolling_avg:.2f}")

                # 3. Solved Check & Video
                if is_solved and not has_logged_solution:
                    logger_mod.info(f"✅ SNN Solved at update {update} (Rolling Avg: {rolling_avg:.1f})!")
                    has_logged_solution = True
                    
                    try:
                        logger_mod.info("Recording solution video...")
                        record_best_agent(
                            agent, env_id=env_cfg.get("id", "CartPole-v1"),
                            video_root=os.path.join(log_dir_root, "videos"),
                            exp_name=config.get("run_name", "snn_surrogate"),
                            seed=config.get("env_seed", 42), sticky_action=sticky,
                            env_kwargs=env_cfg.get("kwargs", {}),
                            partial_obs=env_cfg.get("partial_obs"),
                            frame_stack=env_cfg.get("frame_stack"),
                            frame_stack_flatten=env_cfg.get("frame_stack_flatten", True),
                            max_steps=config.get("max_episode_steps", 500)
                        )
                    except Exception as e:
                        logger_mod.warning(f"Video recording failed: {e}")
                    
                    # We do NOT break here anymore, we continue to ensure stability/energy convergence
                    # unless explicitly desired. If you want to stop on solve, uncomment break below.
                    # break

            # D. Log Dump Interval
            if update % ppo_cfg.get("print_interval_updates", 10) == 0:
                elapsed = time.time() - start_time
                fps = int(logger.num_timesteps / (elapsed + 1e-6))
                logger.record_step_info(updates=update, timesteps=logger.num_timesteps, fps=fps)
                logger.dump()

    except KeyboardInterrupt:
        logger_mod.warning("Training interrupted by user.")
    except Exception as e:
        logger_mod.error(f"Surrogate trainer CRASHED at Update {update}: {e}")
        traceback.print_exc()
        raise e
    
    # --- 8. Validation Data Collection (Post-Training) ---
    val_data = {}
    try:
        val_n_steps = post_eval_cfg.get("n_eval_steps", 500)
        val_n_steps = int(val_n_steps) if val_n_steps is not None else None
        if val_n_steps is not None and val_n_steps <= 0:
            val_n_steps = None

        val_n_episodes = post_eval_cfg.get("n_eval_episodes", None)
        val_n_episodes = int(val_n_episodes) if val_n_episodes is not None else None
        if val_n_episodes is not None and val_n_episodes <= 0:
            val_n_episodes = None

        if val_n_steps is None and val_n_episodes is None:
            val_n_steps = 500

        val_deterministic = bool(post_eval_cfg.get("deterministic", True))
        val_max_episode_steps = post_eval_cfg.get("max_episode_steps", env_cfg.get("max_episode_steps"))
        val_max_episode_steps = int(val_max_episode_steps) if val_max_episode_steps else None

        # Simple safeguard: check if env exists
        obs, _ = env_eval.reset()
        step_count = 0
        episodes_completed = 0
        ep_return = 0.0
        ep_len = 0
        episode_returns = []
        episode_lengths = []
        
        collected_values = []
        collected_latencies = []
        collected_step_rewards = []
        collected_episode_index = []
        collected_step_in_episode = []
        collected_logits = []
        collected_output_spikes = []
        collected_actor_logits = []
        collected_actor_spikes = []
        actor_decision_potentials = None
        actor_decision_spikes = None
        # Store accumulated activations for each layer
        collected_activations = {
            "layer_0": [], 
            "layer_1": [], 
            "output": []
        }
        
        def _budget_reached() -> bool:
            step_cap = (val_n_steps is not None) and (step_count >= val_n_steps)
            episode_cap = (val_n_episodes is not None) and (episodes_completed >= val_n_episodes)
            return step_cap or episode_cap

        while not _budget_reached():
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                if obs_t.ndim == 1: obs_t = obs_t.unsqueeze(0)
                
                # 1. Get critic value/tau from the same forward pass when available.
                val = 0.0
                val_tau = None
                value_raw_t = None
                critic_ref = agent.critic if hasattr(agent, "critic") else None
                if critic_ref is not None and hasattr(critic_ref, "forward_detailed"):
                    v_det, tau_det = critic_ref.forward_detailed(obs_t)
                    value_raw_t = first_output(v_det)
                    try:
                        val_tau = float(tau_det.mean().item())
                    except Exception:
                        val_tau = None
                elif hasattr(agent, "critic_forward"):
                    value_raw_t = first_output(agent.critic_forward(obs_t))
                elif hasattr(agent, "critic"):
                    value_raw_t = first_output(agent.critic(obs_t))
                elif hasattr(agent, "get_value"):
                    value_raw_t = first_output(agent.get_value(obs_t))
                if value_raw_t is not None:
                    val = float(value_raw_t.mean().item())
                
                # 2. Run Actor with Activations
                # Check for direct forward_T or through agent structure
                actor_ref = agent.actor if hasattr(agent, "actor") else agent
                
                if hasattr(actor_ref, "forward_T"):
                    collect_temporal = actor_decision_potentials is None
                    actor_kwargs = {
                        "return_activations": True,
                        "return_temporal": collect_temporal,
                    }
                    if bool(getattr(agent, "critic_informs_actor", False)) and value_raw_t is not None:
                        value_for_actor = value_raw_t
                        if hasattr(agent, "_prepare_critic_value_for_actor"):
                            value_for_actor = agent._prepare_critic_value_for_actor(value_raw_t)
                        actor_kwargs["critic_value"] = value_for_actor
                    logits, _, acts = actor_ref.forward_T(obs_t, **actor_kwargs)
                    if logits is not None:
                        logits_vec = logits.detach().cpu().reshape(logits.shape[0], -1)[0]
                        collected_actor_logits.append(logits_vec.numpy())
                        collected_logits.append(float(logits.mean().item()))
                    
                    # Accumulate (mean over batch)
                    for k, v in acts.items():
                        if k in collected_activations:
                            collected_activations[k].append(v.mean().item())
                    if "output" in acts:
                        collected_output_spikes.append(float(acts["output"].mean().item()))
                    if "output_per_action" in acts:
                        spike_vec = acts["output_per_action"].detach().cpu().reshape(acts["output_per_action"].shape[0], -1)[0]
                        collected_actor_spikes.append(spike_vec.numpy())
                    # Store exactly one decision-window trace (shape: [actions, tau]).
                    if collect_temporal and "output_potential_trace" in acts and "output_spike_trace" in acts:
                        pot_trace = acts["output_potential_trace"]  # [tau, B, A]
                        spk_trace = acts["output_spike_trace"]      # [tau, B, A]
                        if pot_trace.ndim == 3 and spk_trace.ndim == 3 and pot_trace.shape[0] > 0:
                            expected_tau = int(getattr(actor_ref, "T", pot_trace.shape[0]))
                            tau_len = min(expected_tau, pot_trace.shape[0], spk_trace.shape[0])
                            actor_decision_potentials = pot_trace[:tau_len, 0, :].detach().cpu().numpy().T
                            actor_decision_spikes = spk_trace[:tau_len, 0, :].detach().cpu().numpy().T

                # Timing-critic latency (tau). Prefer same-pass tau; fallback to module state, then actor.
                lat = float(val_tau) if val_tau is not None else get_last_latency(critic_ref)
                if lat <= 0:
                    lat = get_last_latency(actor_ref)
                
                collected_values.append(val)
                collected_latencies.append(lat)
                
                # 3. Step
                if hasattr(agent, "get_action"):
                    try:
                        action = agent.get_action(obs_t, deterministic=val_deterministic)
                    except TypeError:
                        action = agent.get_action(obs_t)
                else:
                    logits = first_output(agent(obs_t))
                    if val_deterministic:
                        action = torch.argmax(logits, dim=-1)
                    else:
                        action = torch.distributions.Categorical(logits=logits).sample()
                
                act_int = int(action.item())
                reward_val = 0.0
                done_flag = False
                if hasattr(env_eval, "step"):
                    if hasattr(env_eval, "num_envs"):
                        obs_next, reward, terminated, truncated, _ = env_eval.step([act_int])
                        reward_val = float(reward[0])
                        done_flag = bool(terminated[0] or truncated[0])
                        obs = obs_next[0]
                    else:
                        obs, reward, terminated, truncated, _ = env_eval.step(act_int)
                        reward_val = float(reward)
                        done_flag = bool(terminated or truncated)

                step_count += 1
                ep_return += reward_val
                ep_len += 1
                collected_step_rewards.append(reward_val)
                collected_episode_index.append(episodes_completed)
                collected_step_in_episode.append(ep_len)

                hit_max_episode_steps = bool(val_max_episode_steps and ep_len >= val_max_episode_steps)
                if done_flag or hit_max_episode_steps:
                    episode_returns.append(ep_return)
                    episode_lengths.append(ep_len)
                    episodes_completed += 1
                    ep_return = 0.0
                    ep_len = 0
                    if not _budget_reached():
                        obs, _ = env_eval.reset()

        if ep_len > 0:
            episode_returns.append(ep_return)
            episode_lengths.append(ep_len)

        env_id_meta = str(env_cfg.get("id", "unknown"))
        if hasattr(env_eval, "spec") and getattr(env_eval.spec, "id", None):
            env_id_meta = str(env_eval.spec.id)

        step_traces = {
            "critic_values": np.array(collected_values),
            "critic_timings": np.array(collected_latencies),
            "rewards": np.array(collected_step_rewards),
            "episode_index": np.array(collected_episode_index),
            "step_in_episode": np.array(collected_step_in_episode),
            "activations": {k: np.array(v) for k, v in collected_activations.items()},
            "output_logits": np.array(collected_logits),
            "output_spikes": np.array(collected_output_spikes),
        }

        episode_metrics = {
            "returns": np.array(episode_returns),
            "lengths": np.array(episode_lengths),
            "num_completed": int(episodes_completed),
        }

        # Deterministic first post-eval episode trace for thesis-grade intra-episode plotting.
        critic_values_single_episode = []
        for v, ep_idx in zip(collected_values, collected_episode_index):
            if int(ep_idx) == 0:
                critic_values_single_episode.append(float(v))
            elif int(ep_idx) > 0 and critic_values_single_episode:
                break
        
        # Convert lists to arrays
        val_data = {
            "step_traces": step_traces,
            "episode_metrics": episode_metrics,
            "critic_values": step_traces["critic_values"],
            "critic_values_single_episode": np.array(critic_values_single_episode),
            "critic_timings": step_traces["critic_timings"],
            "activations": step_traces["activations"],
            "output_logits": step_traces["output_logits"],
            "output_spikes": step_traces["output_spikes"],
            "intra_episode_values": np.array(collected_values),
            "metadata": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "seed": int(config.get("env_seed", 42)),
                "checkpoint": last_checkpoint_path or resume_checkpoint_path,
                "env_id": env_id_meta,
                "git_commit": _get_git_commit_hash(),
                "evaluation_protocol": {
                    "deterministic_policy": bool(val_deterministic),
                    "reset_on_done": True,
                    "max_episode_steps": val_max_episode_steps,
                    "n_eval_steps": val_n_steps,
                    "n_eval_episodes": val_n_episodes,
                    "stop_rule": "first_budget_reached",
                },
            },
        }
        if collected_actor_logits:
            actor_pot = np.asarray(collected_actor_logits, dtype=float)  # [steps, actions]
            val_data["actor_output_potentials"] = actor_pot.T           # [actions, steps]
        if collected_actor_spikes:
            actor_spk = np.asarray(collected_actor_spikes, dtype=float)  # [steps, actions]
            val_data["actor_output_spikes"] = actor_spk.T               # [actions, steps]
        if actor_decision_potentials is not None and actor_decision_spikes is not None:
            val_data["actor_decision_potentials"] = np.asarray(actor_decision_potentials, dtype=float)
            val_data["actor_decision_spikes"] = np.asarray(actor_decision_spikes, dtype=float)
        if episode_returns:
            logger.record("post_eval/episode_return_mean", float(np.mean(episode_returns)))
            logger.record("post_eval/episode_length_mean", float(np.mean(episode_lengths)))
        logger.record("post_eval/steps_collected", float(step_count))
        logger.record("post_eval/episodes_completed", float(episodes_completed))
        logger_mod.info(
            "Collected validation data: %d steps, %d episodes (deterministic=%s).",
            step_count,
            episodes_completed,
            val_deterministic,
        )
        
    except Exception as e:
        logger_mod.warning(f"Validation data collection failed: {e}")
        
    return {
        "agent": agent,
        "logger": logger,
        "train_rewards": train_rewards,
        "test_rewards": test_rewards,
        "validation_data": val_data
    }






















































