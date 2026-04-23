"""Entry point for training ANN baseline agents with PPO.

Loads a YAML config, constructs the environment and agent, runs the training
loop, and writes checkpoints and evaluation plots.

"""
import os
import time
import torch
import logging
import numpy as np
import sys
import glob
import traceback
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any

from src.training.core_trainer import CoreTrainer
from src.training.hooks import EnergyHook
from src.training.agents import make_agent, resolve_cartpole_types
from src.training.envs import make_envs, VecNormalize
from src.training.record import record_best_agent

from src.utils.logger import PPOLogger
from src.utils.checkpoint import save_checkpoint, load_checkpoint

logger_mod = logging.getLogger(__name__)

# Floor fraction used in EnergyBenchmark._compute_dynamic_energy() —
# keep in sync with that value so training and benchmark numbers are comparable.
_DYNAMIC_ENERGY_FLOOR_FRACTION = 0.05


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


def _compute_dynamic_joules(
    total_joules: float,
    duration_seconds: float,
    idle_power_watts: float,
    floor_fraction: float = _DYNAMIC_ENERGY_FLOOR_FRACTION,
) -> float:
    """
    Subtract the idle-power baseline from a measured energy span.

    Mirrors EnergyBenchmark._compute_dynamic_energy() exactly so that
    dynamic-energy numbers in training logs are on the same scale as those
    produced by the post-training benchmark script.

    The floor prevents negative or near-zero values from noisy NVML readings
    on short spans (common during fast eval steps).
    """
    idle_energy = idle_power_watts * duration_seconds
    raw_dynamic = total_joules - idle_energy
    floor_value = total_joules * floor_fraction
    return max(raw_dynamic, floor_value)


def run_baseline(config: Dict[str, Any]) -> Dict[str, Any]:
    # --- 1. Config Unpacking & Path Setup ---
    env_cfg      = config.get("env", {})
    train_cfg    = config.get("training", {})
    ppo_cfg      = config.get("ppo", {})
    model_cfg    = config.get("model", {})
    log_cfg      = config.get("logging", {})
    post_eval_cfg = config.get("post_eval", ppo_cfg.get("post_eval", {})) or {}

    log_dir_root = config.get("log_dir", "logs")

    device_name = config.get("device", "cpu")
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        logger_mod.warning("CUDA requested but unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

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

    if env_cfg.get("vec_normalize", False):
        env_train = VecNormalize(env_train, training=True,  norm_obs=True, norm_reward=True,  clip_reward=10.0)
        env_eval_wrapper = VecNormalize(env_eval, training=False, norm_obs=True, norm_reward=False, clip_reward=10.0)
        env_eval_wrapper.obs_rms = env_train.obs_rms
        env_eval = env_eval_wrapper

    # --- 3. Agent Setup ---
    actor_type, critic_type = resolve_cartpole_types(model_cfg.get("mode", "ann"))

    agent = make_agent(
        actor_type=actor_type,
        critic_type=critic_type,
        hidden_dim=model_cfg.get("hidden_dim", 128),
        dropout=model_cfg.get("dropout", 0.0),
        in_dim=model_cfg.get("in_features", 4),
        act_dim=(
            env_train.single_action_space.n
            if hasattr(env_train, "single_action_space")
            else env_train.action_space.n
        ),
        gamma=ppo_cfg.get("gamma", 0.99),
        critic_informs_actor=bool(model_cfg.get("critic_informs_actor", False)),
        detach_critic_for_actor=bool(model_cfg.get("detach_critic_for_actor", True)),
        normalize_critic_for_actor=bool(model_cfg.get("normalize_critic_for_actor", True)),
        critic_actor_value_clip=model_cfg.get("critic_actor_value_clip", 5.0),
        critic_actor_norm_momentum=float(model_cfg.get("critic_actor_norm_momentum", 0.01)),
    ).to(device)

    if env_cfg.get("vec_normalize", False):
        agent.obs_rms = env_eval.obs_rms
        agent.ret_rms = env_train.ret_rms

    # --- 4. Optimization & Logging ---
    optimizer   = torch.optim.Adam(agent.parameters(), lr=float(ppo_cfg.get("lr", 3e-4)))
    initial_lr  = float(ppo_cfg.get("lr", 3e-4))

    n_eval_episodes_cfg        = int(ppo_cfg.get("eval_episodes", 5))
    solved_window_eval_episodes = int(ppo_cfg.get("solved_window_eval_episodes", 100))
    window_size = max(1, solved_window_eval_episodes // n_eval_episodes_cfg)

    logger = PPOLogger(log_dir=log_dir_root, window=window_size)

    energy_cfg   = config.get("energy", {}) or {}
    energy_hook  = EnergyHook(
        sample_interval=float(energy_cfg.get("sample_interval", 0.02)),
        gpu_index=int(energy_cfg.get("gpu_index", 0)),
    )
    if bool(energy_cfg.get("calibrate_idle", True)):
        energy_hook.calibrate_idle(float(energy_cfg.get("idle_calibration_seconds", 2.0)))

    # Expose idle power to the logger for downstream benchmark consistency.
    # Recorded once at startup — it is a constant for the run.
    idle_power_watts = float(getattr(energy_hook, "idle_power_watts", 0.0))
    logger.record("energy/idle_power_watts", idle_power_watts, exclude_from_console=True)

    # Record run-level constants once so they appear in logs without
    # polluting every eval step with flat lines.
    logger.record("config/reward_threshold",        float(reward_threshold),       exclude_from_console=True)
    logger.record("config/success_rate_threshold",  float(success_rate_threshold), exclude_from_console=True)
    logger.record("config/eval_window_episodes",    float(window_size * n_eval_episodes_cfg), exclude_from_console=True)

    # FIX 2: Pass hooks to CoreTrainer for internal reset/flush behaviour only.
    # We do NOT use hook-reported energy values for cumulative accounting —
    # the outer start_span/stop_span covers the full update and is the single
    # source of truth for energy numbers.
    hooks = {
        "on_rollout_start": energy_hook.on_rollout_start,
        "on_rollout_end":   energy_hook.on_rollout_end,
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

    # --- 5. Training Parameters ---
    total_updates          = int(train_cfg.get("total_updates", 1000))
    ckpt_interval          = int(log_cfg.get("checkpoint_interval_updates", 50))
    reward_threshold       = float(ppo_cfg.get("reward_threshold", 475.0))
    success_rate_threshold = float(ppo_cfg.get("success_rate_threshold", 95.0))
    save_ckpt_flag         = config.get("save_ckpt", False)

    train_rewards             = []
    test_rewards              = []
    cumulative_energy         = 0.0
    cumulative_dynamic_energy = 0.0
    resume_checkpoint_path    = None
    last_checkpoint_path      = None

    best_rolling_avg   = -float("inf")
    has_logged_solution = False

    # --- 6. Resume Logic ---
    start_update = 1
    ckpt_dir     = os.path.join(log_dir_root, "checkpoints")

    def _apply_rms_state(rms, state):
        if not rms or not state:
            return
        rms.mean  = np.asarray(state.get("mean",  rms.mean),  dtype=np.float64)
        rms.var   = np.asarray(state.get("var",   rms.var),   dtype=np.float64)
        rms.count = float(state.get("count", rms.count))

    if os.path.exists(ckpt_dir):
        ckpts = glob.glob(os.path.join(ckpt_dir, "*.pt"))
        if ckpts:
            latest_ckpt = max(ckpts, key=os.path.getctime)
            logger_mod.info(f"Found checkpoint: {latest_ckpt}. Attempting resume...")
            try:
                data = load_checkpoint(
                    latest_ckpt,
                    agent=agent,
                    optimizer=optimizer,
                    logger=logger,
                    map_location=device,
                )
                resume_checkpoint_path = latest_ckpt
                start_update           = data["episode"] + 1
                logger.num_timesteps   = data["num_timesteps"]
                logger.iteration       = start_update

                if "energy_metrics" in data and data["energy_metrics"]:
                    cumulative_energy         = data["energy_metrics"].get("total", 0.0)
                    cumulative_dynamic_energy = data["energy_metrics"].get("total_dynamic", 0.0)

                if "best_rolling_avg" in data:
                    best_rolling_avg = data["best_rolling_avg"]

                vec_state = data.get("vecnorm_state") or {}
                if env_cfg.get("vec_normalize", False) and vec_state:
                    _apply_rms_state(env_train.obs_rms, vec_state.get("obs_rms"))
                    _apply_rms_state(env_train.ret_rms, vec_state.get("ret_rms"))
                    env_eval.obs_rms  = env_train.obs_rms
                    agent.obs_rms     = env_train.obs_rms
                    agent.ret_rms     = env_train.ret_rms

                logger_mod.info(f"Resumed from update {start_update} (Energy: {cumulative_energy:.2f} J)")
            except Exception as e:
                logger_mod.error(f"Failed to load checkpoint: {e}")

    logger_mod.info(f"Starting training loop from update {start_update} to {total_updates}")
    if start_update > total_updates:
        logger_mod.info(
            "Checkpoint already reached/exceeded configured total_updates; "
            "skipping additional training updates and proceeding to analysis."
        )

    start_time = time.time()
    rollout_len = int(train_cfg.get("rollout_length", 2048))

    print(f"\n[Trainer INTERNAL] Configured Total Updates: {total_updates}")
    print(f"[Trainer INTERNAL] Reward Threshold: {reward_threshold}")
    sys.stdout.flush()

    update = max(0, start_update - 1)
    try:
        for update in range(start_update, total_updates + 1):

            # Linear LR Decay
            progress   = 1.0 - (update - 1.0) / total_updates
            current_lr = initial_lr * progress
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            logger.iteration = update

            # ----------------------------------------------------------------
            # A. Train Step — outer span is the single source of truth.
            #    CoreTrainer fires the hooks internally (rollout only), but we
            #    do NOT read energy back from energy_hook.energy here because:
            #      a) 'dynamic_joules' key does not exist in GPUEnergyMeter output
            #      b) it would double-count the rollout inside the outer span
            # ----------------------------------------------------------------
            train_meter, train_t0 = energy_hook.start_span()
            tr, metrics           = trainer.train_episode()
            train_full_stats      = energy_hook.stop_span(train_meter, train_t0)
            train_rewards.append(tr)

            # FIX 1: compute dynamic energy explicitly — GPUEnergyMeter never
            # emits a 'dynamic_joules' key, so .get('dynamic_joules', fallback)
            # always silently returned the raw total in the original code.
            train_full_joules   = float(train_full_stats.get("total_joules", 0.0))
            train_duration      = float(train_full_stats.get("duration_seconds", 0.0))
            train_full_dynamic_joules = _compute_dynamic_joules(
                train_full_joules, train_duration, idle_power_watts
            )

            cumulative_energy         += train_full_joules
            cumulative_dynamic_energy += train_full_dynamic_joules

            # Per-step latency from the outer span (more accurate than hook)
            latency_ms = 0.0
            steps_in_rollout = trainer.env_train.num_envs * rollout_len
            if steps_in_rollout > 0 and train_duration > 0:
                latency_ms = (train_duration / steps_in_rollout) * 1000.0
            elif "latency_mean" in metrics:
                latency_ms = metrics["latency_mean"] * 1000.0

            logger.record("train/rollout_reward",             float(tr))
            # Log the decaying LR so convergence issues are visible in charts
            logger.record("train/learning_rate",              current_lr)
            logger.record("energy/train_full_update",         train_full_joules)
            logger.record("energy/train_full_update_dynamic", train_full_dynamic_joules)
            # energy/total and energy/total_dynamic are written ONCE per update,
            # after the eval energy is also added on eval steps.  Writing them
            # here AND inside the eval block caused silent overwrites in WandB.
            if latency_ms > 0.0:
                logger.record("latency/mean_ms", latency_ms)

            # ----------------------------------------------------------------
            # C. Periodic Evaluation
            # ----------------------------------------------------------------
            eval_interval = int(ppo_cfg.get("eval_interval_updates", 25))

            if update % eval_interval == 0 or update == total_updates:
                eval_meter, eval_t0 = energy_hook.start_span()
                te, success_rate, avg_len, success_count, n_eval_episodes_actual = trainer.evaluate()
                eval_energy_stats = energy_hook.stop_span(eval_meter, eval_t0)

                eval_joules   = float(eval_energy_stats.get("total_joules", 0.0))
                eval_duration = float(eval_energy_stats.get("duration_seconds", 0.0))
                # FIX 1+3: same pattern — compute dynamic explicitly
                eval_dynamic_joules = _compute_dynamic_joules(
                    eval_joules, eval_duration, idle_power_watts
                )

                cumulative_energy         += eval_joules
                cumulative_dynamic_energy += eval_dynamic_joules

                logger.record("energy/eval_update",         eval_joules)
                logger.record("energy/eval_update_dynamic", eval_dynamic_joules)
                # energy/total and energy/total_dynamic are written unconditionally
                # after this block (every update) so they are not written here.

                test_rewards.append(te)

                if len(test_rewards) >= window_size:
                    rolling_avg = np.mean(test_rewards[-window_size:])
                    window_success_rate = float(
                        np.mean(np.asarray(test_rewards[-window_size:]) >= reward_threshold) * 100.0
                    )
                    is_solved = (rolling_avg >= reward_threshold) and (window_success_rate >= success_rate_threshold)
                else:
                    rolling_avg = np.mean(test_rewards)
                    is_solved   = False

                # Canonical eval keys — one name per concept.
                # Legacy aliases (eval/mean_100ep, eval/episode_reward_ma100,
                # eval/solved_reward_avg) have been removed; they all pointed
                # to the same rolling_avg value and produced duplicate chart lines.
                logger.record("eval/rolling_reward",    float(rolling_avg))
                logger.record("eval/current_reward",    float(te))
                logger.record("eval/episode_length",    float(avg_len))
                logger.record("eval/success_rate",      float(success_rate))
                logger.record("eval/success_count",     float(success_count))
                logger.record("eval/n_eval_episodes",   float(n_eval_episodes_actual))
                # eval/is_solved logged only on the update it first becomes True.
                # Logging it every step after that produces a permanent flat 1.0
                # line that is not useful as a chart signal.
                if is_solved and not has_logged_solution:
                    logger.record("eval/is_solved_at_update", float(update))
                logger.record_episode(reward=te, length=avg_len, success=min(100.0, success_rate), source="eval")

                # Periodic checkpoint
                if save_ckpt_flag and (update % ckpt_interval == 0):
                    last_checkpoint_path = save_checkpoint(
                        agent=agent,
                        optimizer=optimizer,
                        logger=logger,
                        episode=update,
                        num_timesteps=logger.num_timesteps,
                        save_dir=ckpt_dir,
                        mean_reward=te,
                        config=config,
                        energy_metrics={"total": cumulative_energy, "total_dynamic": cumulative_dynamic_energy},
                        extra_data={"best_rolling_avg": best_rolling_avg},
                    )

                # Best model checkpoint
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
                            suffix="best",
                            energy_metrics={"total": cumulative_energy, "total_dynamic": cumulative_dynamic_energy},
                            extra_data={"best_rolling_avg": best_rolling_avg},
                        )
                        logger_mod.info(f"New best model saved! Avg Reward: {best_rolling_avg:.2f}")

                # Log "solved" milestone once, but keep training
                if is_solved and not has_logged_solution:
                    logger_mod.info(
                        f"✅ Environment solved at update {update} "
                        f"(Avg: {rolling_avg:.2f})! Continuing training for stability..."
                    )
                    has_logged_solution = True
                    try:
                        logger_mod.info("Recording solution video...")
                        sticky_cfg = config.get("sticky_action", {})
                        record_best_agent(
                            agent,
                            env_id=env_cfg.get("id", "CartPole-v1"),
                            video_root=os.path.join(log_dir_root, "videos"),
                            exp_name=config.get("run_name", "ann_baseline"),
                            seed=config.get("env_seed", 42),
                            sticky_action=sticky_cfg.get("eval", False),
                            env_kwargs=env_cfg.get("kwargs"),
                            partial_obs=env_cfg.get("partial_obs"),
                            frame_stack=env_cfg.get("frame_stack"),
                            frame_stack_flatten=env_cfg.get("frame_stack_flatten", True),
                            max_steps=500,
                        )
                    except Exception as e:
                        logger_mod.warning(f"Video recording failed: {e}")

            # Write cumulative totals every update so the WandB curve is
            # continuous.  On eval steps the value already includes eval energy
            # (added above); on non-eval steps it reflects train energy only.
            logger.record("energy/total",         cumulative_energy)
            logger.record("energy/total_dynamic", cumulative_dynamic_energy)

            # ----------------------------------------------------------------
            # D. Log Dump Interval
            # ----------------------------------------------------------------
            if update % ppo_cfg.get("print_interval_updates", 10) == 0:
                elapsed = time.time() - start_time
                fps     = int(logger.num_timesteps / (elapsed + 1e-6))
                logger.record_step_info(updates=update, timesteps=logger.num_timesteps, fps=fps)
                logger.dump()
                sys.stdout.flush()

    except KeyboardInterrupt:
        logger_mod.warning("Interrupted by user.")
    except Exception as e:
        logger_mod.error(f"Trainer CRASHED at Update {update}: {e}")
        traceback.print_exc()
    else:
        logger_mod.info(f"Finished loop naturally after {update} updates.")

    # -----------------------------------------------------------------------
    # Post-training validation rollout
    # -----------------------------------------------------------------------
    validation_data = {}
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

        val_deterministic     = bool(post_eval_cfg.get("deterministic", True))
        val_max_episode_steps = post_eval_cfg.get("max_episode_steps", env_cfg.get("max_episode_steps"))
        val_max_episode_steps = int(val_max_episode_steps) if val_max_episode_steps else None

        obs, _ = env_eval.reset()
        step_count         = 0
        episodes_completed = 0
        ep_return          = 0.0
        ep_len             = 0
        episode_returns    = []
        episode_lengths    = []
        step_rewards       = []
        step_episode_index = []
        step_in_episode    = []
        critic_values      = []
        output_logits      = []

        def _budget_reached() -> bool:
            step_cap    = (val_n_steps    is not None) and (step_count         >= val_n_steps)
            episode_cap = (val_n_episodes is not None) and (episodes_completed >= val_n_episodes)
            return step_cap or episode_cap

        while not _budget_reached():
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                if obs_t.ndim == 1:
                    obs_t = obs_t.unsqueeze(0)

                logits, value = agent(obs_t)
                output_logits.append(float(logits.mean().item()))
                critic_values.append(float(value.mean().item()))

                if val_deterministic:
                    action = int(torch.argmax(logits, dim=-1).item())
                else:
                    action = int(torch.distributions.Categorical(logits=logits).sample().item())

                if hasattr(env_eval, "num_envs"):
                    obs_next, reward, terminated, truncated, _ = env_eval.step([action])
                    reward_val = float(reward[0])
                    done_flag  = bool(terminated[0] or truncated[0])
                    obs        = obs_next[0]
                else:
                    obs, reward, terminated, truncated, _ = env_eval.step(action)
                    reward_val = float(reward)
                    done_flag  = bool(terminated or truncated)

                step_count  += 1
                ep_return   += reward_val
                ep_len      += 1
                step_rewards.append(reward_val)
                step_episode_index.append(episodes_completed)
                step_in_episode.append(ep_len)

                hit_max = bool(val_max_episode_steps and ep_len >= val_max_episode_steps)
                if done_flag or hit_max:
                    episode_returns.append(ep_return)
                    episode_lengths.append(ep_len)
                    episodes_completed += 1
                    ep_return = 0.0
                    ep_len    = 0
                    if not _budget_reached():
                        obs, _ = env_eval.reset()

        if ep_len > 0:
            episode_returns.append(ep_return)
            episode_lengths.append(ep_len)

        env_id_meta = str(env_cfg.get("id", "unknown"))
        if hasattr(env_eval, "spec") and getattr(env_eval.spec, "id", None):
            env_id_meta = str(env_eval.spec.id)

        step_traces = {
            "rewards":        np.array(step_rewards),
            "episode_index":  np.array(step_episode_index),
            "step_in_episode": np.array(step_in_episode),
            "output_logits":  np.array(output_logits),
            "critic_values":  np.array(critic_values),
        }
        episode_metrics_out = {
            "returns":       np.array(episode_returns),
            "lengths":       np.array(episode_lengths),
            "num_completed": int(episodes_completed),
        }

        validation_data = {
            "step_traces":          step_traces,
            "episode_metrics":      episode_metrics_out,
            "output_logits":        step_traces["output_logits"],
            "critic_values":        step_traces["critic_values"],
            "intra_episode_values": step_traces["critic_values"],
            "metadata": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "seed":          int(config.get("env_seed", 42)),
                "checkpoint":    last_checkpoint_path or resume_checkpoint_path,
                "env_id":        env_id_meta,
                "git_commit":    _get_git_commit_hash(),
                "evaluation_protocol": {
                    "deterministic_policy": bool(val_deterministic),
                    "reset_on_done":        True,
                    "max_episode_steps":    val_max_episode_steps,
                    "n_eval_steps":         val_n_steps,
                    "n_eval_episodes":      val_n_episodes,
                    "stop_rule":            "first_budget_reached",
                },
            },
        }

        if episode_returns:
            logger.record("post_eval/episode_return_mean", float(np.mean(episode_returns)))
            logger.record("post_eval/episode_length_mean", float(np.mean(episode_lengths)))
        logger.record("post_eval/steps_collected",   float(step_count))
        logger.record("post_eval/episodes_completed", float(episodes_completed))

    except Exception as e:
        logger_mod.warning(f"Post-eval validation collection failed: {e}")

    return {
        "agent":           agent,
        "logger":          logger,
        "train_rewards":   train_rewards,
        "test_rewards":    test_rewards,
        "episode_lengths": list(validation_data.get("episode_metrics", {}).get("lengths", [])),
        "validation_data": validation_data,
    }