"""Orchestrates the ANN-to-SNN conversion training workflow.

Stages: train ANN baseline → calibrate activation ranges → convert to SNN
→ validate → fine-tune → evaluate and compare.
"""

import os
import time
import logging
import torch
import numpy as np
import sys
import traceback
import glob
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional

# --- Project Imports ---
from src.conversion.verification import verify_actor_conversion, verify_critic_conversion
from src.models.actor_critic import ActorCritic
from src.models.ann import Actor, Critic

from src.training.core_trainer import CoreTrainer
from src.training.envs import make_envs, VecNormalize
from src.training.agents import make_agent, ActorType, CriticType, resolve_cartpole_types

from src.training.evaluate import evaluate, gather_observations, get_last_spike_count, get_last_latency, evaluate_snn
from src.training.device import coerce_module_to_device
from src.training.record import record_best_agent

from src.conversion.scales import pick_scales
from src.conversion.calibration import estimate_ann_percentiles
from src.conversion.ann_to_snn import snn_from_ann, snn_critic_from_ann

from src.tools.energy_benchmark import GPUEnergyMeter
from src.utils.logger import PPOLogger
from src.utils.plotting import plot_conversion_validation
from src.training.hooks import EnergyHook
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.torch_utils import first_output

logger_mod = logging.getLogger(__name__)

# Keep in sync with EnergyBenchmark._compute_dynamic_energy()
_DYNAMIC_ENERGY_FLOOR_FRACTION = 0.05


def _get_git_commit_hash() -> str:
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
    Subtract idle baseline from a measured energy span.
    Mirrors EnergyBenchmark._compute_dynamic_energy() exactly.
    The floor prevents negative values from noisy NVML readings on short spans.
    """
    idle_energy = idle_power_watts * duration_seconds
    raw_dynamic = total_joules - idle_energy
    floor_value = total_joules * floor_fraction
    return max(raw_dynamic, floor_value)


def _resolve_local_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if os.path.exists(path):
        return path
    candidate = os.path.join(os.getcwd(), path)
    if os.path.exists(candidate):
        return candidate
    return None


def _infer_ann_dims_from_checkpoint(
    checkpoint_path: str,
    *,
    fallback_in_dim: int,
    fallback_act_dim: int,
    fallback_hidden_dim: int,
) -> Tuple[int, int, int]:
    try:
        data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        actor_state = data.get("actor_state") or data.get("actor_state_dict") or {}
        first_w = actor_state.get("backbone.layers.0.weight")
        head_w  = actor_state.get("policy_head.weight")
        if first_w is None or head_w is None:
            return fallback_in_dim, fallback_act_dim, fallback_hidden_dim
        return int(first_w.shape[1]), int(head_w.shape[0]), int(first_w.shape[0])
    except Exception:
        return fallback_in_dim, fallback_act_dim, fallback_hidden_dim


def _apply_rms_state(rms, state):
    if not rms or not state:
        return
    rms.mean  = np.asarray(state.get("mean",  rms.mean),  dtype=np.float64)
    rms.var   = np.asarray(state.get("var",   rms.var),   dtype=np.float64)
    rms.count = float(state.get("count", rms.count))


def run_conversion(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes the ANN -> SNN Conversion Pipeline.
    Research Standard: Continues training after solving to verify stability.
    """
    # --- 1. Setup & Config ---
    env_cfg   = config.get("env", {})
    train_cfg = config.get("training", {})
    ppo_cfg   = config.get("ppo", {})
    post_eval_cfg = config.get("post_eval", ppo_cfg.get("post_eval", {})) or {}
    log_cfg   = config.get("logging", {})

    reward_threshold       = float(ppo_cfg.get("reward_threshold", 475.0))
    success_rate_threshold = float(ppo_cfg.get("success_rate_threshold", 95.0))
    seed   = config.get("env_seed", 42)
    env_id = env_cfg.get("id", "CartPole-v1")
    n_envs = env_cfg.get("n_envs", 8)

    log_dir   = config.get("log_dir", "logs/conversion")
    plots_dir = config.get("plots_dir", os.path.join(log_dir, "plots"))
    ckpt_dir  = os.path.join(log_dir, "checkpoints")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(ckpt_dir,  exist_ok=True)

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    n_eval_episodes_cfg        = int(ppo_cfg.get("eval_episodes", 5))
    solved_window_eval_episodes = int(ppo_cfg.get("solved_window_eval_episodes", 100))
    window_size = max(1, solved_window_eval_episodes // n_eval_episodes_cfg)

    logger = PPOLogger(log_dir=log_dir, window=window_size)

    # --- 2. Environment ---
    env_train, env_eval = make_envs(
        seed=seed,
        env_id=env_id,
        n_envs=n_envs,
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

    # --- Energy instrumentation ---
    energy_cfg  = config.get("energy", {}) or {}
    energy_hook = EnergyHook(
        sample_interval=float(energy_cfg.get("sample_interval", 0.02)),
        gpu_index=int(energy_cfg.get("gpu_index", 0)),
    )
    if bool(energy_cfg.get("calibrate_idle", True)):
        energy_hook.calibrate_idle(float(energy_cfg.get("idle_calibration_seconds", 2.0)))

    # Captured once; used for all _compute_dynamic_joules() calls in this run.
    idle_power_watts = float(getattr(energy_hook, "idle_power_watts", 0.0))

    # Record run-level constants once — not as per-step chart lines.
    logger.record("energy/idle_power_watts",       idle_power_watts,       exclude_from_console=True)
    logger.record("config/reward_threshold",        reward_threshold,       exclude_from_console=True)
    logger.record("config/success_rate_threshold",  success_rate_threshold, exclude_from_console=True)
    logger.record("config/eval_window_episodes",    float(window_size * n_eval_episodes_cfg), exclude_from_console=True)

    hooks = {
        "on_rollout_start": energy_hook.on_rollout_start,
        "on_rollout_end":   energy_hook.on_rollout_end,
    }
    cumulative_energy         = 0.0
    cumulative_dynamic_energy = 0.0

    rollout_len = int(train_cfg.get("rollout_length", 2048))

    # ============================================================
    # Phase 1: Train ANN Baseline
    # ============================================================
    logger_mod.info("--- Phase 1: ANN Baseline ---")

    model_cfg = config.get("model", {})
    ann_agent = make_agent(
        actor_type=ActorType.ANN,
        critic_type=CriticType.ANN,
        hidden_dim=model_cfg.get("hidden_dim", 64),
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
        ann_agent.obs_rms = env_eval.obs_rms
        ann_agent.ret_rms = env_train.ret_rms

    initial_lr  = float(ppo_cfg.get("lr", 1e-3))
    optimizer   = torch.optim.Adam(ann_agent.parameters(), lr=initial_lr)
    save_ckpt_flag = config.get("save_ckpt", False)
    ckpt_interval  = int(log_cfg.get("checkpoint_interval_updates", 50))

    # --- Resume Logic ---
    start_update     = 1
    best_rolling_avg = -float("inf")

    if os.path.exists(ckpt_dir):
        ckpts = glob.glob(os.path.join(ckpt_dir, "*.pt"))
        if ckpts:
            latest_ckpt = max(ckpts, key=os.path.getctime)
            logger_mod.info(f"Found checkpoint: {latest_ckpt}. Attempting resume...")
            try:
                data = load_checkpoint(
                    latest_ckpt,
                    agent=ann_agent,
                    optimizer=optimizer,
                    logger=logger,
                    map_location=device,
                )
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
                    ann_agent.obs_rms = env_train.obs_rms
                    ann_agent.ret_rms = env_train.ret_rms

                logger_mod.info(f"Resumed from update {start_update} (Energy: {cumulative_energy:.2f} J)")
            except Exception as e:
                logger_mod.error(f"Failed to load checkpoint: {e}")

    ann_trainer = CoreTrainer(
        agent=ann_agent,
        env_train=env_train,
        env_eval=env_eval,
        optimizer=optimizer,
        logger=logger,
        config=config,
        hooks=hooks,
    )

    total_updates = int(train_cfg.get("total_updates", 300))
    train_rewards = []
    test_rewards  = []

    video_cfg = config.get("video", {})
    record_ann_solution_video      = bool(video_cfg.get("record_ann_solution_video",      True))
    record_zero_shot_video         = bool(video_cfg.get("record_zero_shot_video",         True))
    record_finetune_solution_video = bool(video_cfg.get("record_finetune_solution_video", True))

    has_logged_ann_solution = False
    eval_interval = int(ppo_cfg.get("eval_interval_updates", 25))

    start_time = time.time()
    logger_mod.info(f"Starting training loop from update {start_update} to {total_updates}")
    if start_update > total_updates:
        logger_mod.info(
            "Checkpoint already reached/exceeded total_updates; "
            "skipping ANN training and proceeding to conversion."
        )

    update = max(0, start_update - 1)
    for update in range(start_update, total_updates + 1):
        # Linear LR decay
        progress   = 1.0 - (update - 1.0) / total_updates
        current_lr = initial_lr * progress
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        logger.iteration = update

        # ---- A. Train (outer span is single source of truth) ----
        train_meter, train_t0 = energy_hook.start_span()
        tr, metrics           = ann_trainer.train_episode()
        train_full_stats      = energy_hook.stop_span(train_meter, train_t0)
        train_rewards.append(tr)

        train_full_joules         = float(train_full_stats.get("total_joules", 0.0))
        train_full_duration       = float(train_full_stats.get("duration_seconds", 0.0))
        train_full_dynamic_joules = _compute_dynamic_joules(
            train_full_joules, train_full_duration, idle_power_watts
        )
        cumulative_energy         += train_full_joules
        cumulative_dynamic_energy += train_full_dynamic_joules

        # Latency from outer span (covers rollout + update, not rollout-only)
        steps_in_rollout = rollout_len * n_envs
        latency_ms = (train_full_duration / steps_in_rollout * 1000.0) if steps_in_rollout > 0 else 0.0

        logger.record("train/rollout_reward",             float(tr))
        logger.record("train/learning_rate",              current_lr)
        logger.record("energy/train_full_update",         train_full_joules)
        logger.record("energy/train_full_update_dynamic", train_full_dynamic_joules)
        if latency_ms > 0.0:
            logger.record("latency/mean_ms", latency_ms)

        # ---- B. Periodic Evaluation ----
        if update % eval_interval == 0 or update == total_updates:
            eval_meter, eval_t0 = energy_hook.start_span()
            te, success_rate, avg_len, success_count, n_eval_episodes_actual = ann_trainer.evaluate()
            eval_stats = energy_hook.stop_span(eval_meter, eval_t0)

            eval_joules         = float(eval_stats.get("total_joules", 0.0))
            eval_duration       = float(eval_stats.get("duration_seconds", 0.0))
            eval_dynamic_joules = _compute_dynamic_joules(eval_joules, eval_duration, idle_power_watts)
            cumulative_energy         += eval_joules
            cumulative_dynamic_energy += eval_dynamic_joules

            logger.record("energy/eval_update",         eval_joules)
            logger.record("energy/eval_update_dynamic", eval_dynamic_joules)

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
            logger.record("eval/rolling_reward",  float(rolling_avg))
            logger.record("eval/current_reward",  float(te))
            logger.record("eval/episode_length",  float(avg_len))
            logger.record("eval/success_rate",    float(success_rate))
            logger.record("eval/success_count",   float(success_count))
            logger.record("eval/n_eval_episodes", float(n_eval_episodes_actual))
            # Fired once at the update it first becomes True.
            if is_solved and not has_logged_ann_solution:
                logger.record("eval/is_solved_at_update", float(update))
            logger.record_episode(reward=te, length=avg_len, success=min(100.0, success_rate), source="eval")

            # Periodic checkpoint
            if save_ckpt_flag and (update % ckpt_interval == 0):
                save_checkpoint(
                    agent=ann_agent, optimizer=optimizer, logger=logger,
                    episode=update, num_timesteps=logger.num_timesteps,
                    save_dir=ckpt_dir, mean_reward=te, config=config,
                    energy_metrics={"total": cumulative_energy, "total_dynamic": cumulative_dynamic_energy},
                    extra_data={"best_rolling_avg": best_rolling_avg},
                )

            if rolling_avg > best_rolling_avg:
                best_rolling_avg = rolling_avg
                if save_ckpt_flag:
                    save_checkpoint(
                        agent=ann_agent, optimizer=optimizer, logger=logger,
                        episode=update, num_timesteps=logger.num_timesteps,
                        save_dir=ckpt_dir, suffix="best", config=config,
                        energy_metrics={"total": cumulative_energy, "total_dynamic": cumulative_dynamic_energy},
                        extra_data={"best_rolling_avg": best_rolling_avg},
                    )
                    logger_mod.info(f"New best ANN model saved! Avg Reward: {best_rolling_avg:.2f}")

            if is_solved and not has_logged_ann_solution:
                logger_mod.info(
                    f"✅ ANN solved at update {update} (Avg: {rolling_avg:.2f})! "
                    "Continuing for stability..."
                )
                has_logged_ann_solution = True
                if record_ann_solution_video:
                    try:
                        sticky_cfg = config.get("sticky_action", {})
                        record_best_agent(
                            ann_agent,
                            env_id=env_cfg.get("id", "CartPole-v1"),
                            video_root=os.path.join(log_dir, "videos"),
                            exp_name=config.get("run_name", "ann_baseline"),
                            seed=config.get("env_seed", 42),
                            sticky_action=sticky_cfg.get("eval", False),
                            partial_obs=env_cfg.get("partial_obs"),
                            frame_stack=env_cfg.get("frame_stack"),
                            frame_stack_flatten=env_cfg.get("frame_stack_flatten", True),
                            max_steps=500,
                        )
                    except Exception as e:
                        logger_mod.warning(f"Video recording failed: {e}")

        # Write cumulative totals once per update — after all energy for this
        # step has been added (train only on non-eval steps, train+eval on eval steps).
        logger.record("energy/total",         cumulative_energy)
        logger.record("energy/total_dynamic", cumulative_dynamic_energy)

        if update % ppo_cfg.get("print_interval_updates", 10) == 0:
            elapsed = time.time() - start_time
            fps = int(logger.num_timesteps / (elapsed + 1e-6))
            logger.record_step_info(updates=update, timesteps=logger.num_timesteps, fps=fps)
            logger.dump()

    # ============================================================
    # Phase 2: ANN -> SNN Conversion
    # ============================================================
    logger_mod.info("--- Phase 2: Conversion ---")
    ann_agent.eval()

    actor_backbone = ann_agent.actor
    if hasattr(ann_agent.actor, "backbone"):
        actor_backbone = ann_agent.actor.backbone

    logger_mod.info("Collecting calibration data...")
    snn_convert_cfg    = config.get("snn_convert", {})
    calib_episodes     = int(snn_convert_cfg.get("calibration_episodes", 256))
    calib_stochastic   = bool(snn_convert_cfg.get("calibration_stochastic", True))
    calib_noise_std    = float(snn_convert_cfg.get("calibration_obs_noise_std", 0.01))
    calib_max_steps    = int(snn_convert_cfg.get("calibration_max_steps_per_episode", 0))
    obs_buf = gather_observations(
        env_eval,
        ann_agent.actor,
        episodes=calib_episodes,
        stochastic_actions=calib_stochastic,
        obs_noise_std=calib_noise_std,
        max_steps_per_episode=calib_max_steps,
    ).to(device)

    p90       = estimate_ann_percentiles(actor_backbone, obs_buf)
    input_p99 = torch.quantile(obs_buf.abs(), 0.99).item()
    p90["input"] = input_p99

    scales = pick_scales(
        percentiles=p90,
        V_th=snn_convert_cfg.get("V_th", 1.0),
        safety=0.99,
    )

    logger_mod.info("Converting Actor...")
    actor_snn_backbone = snn_from_ann(
        actor_backbone,
        actor_T=snn_convert_cfg.get("T", 100),
        beta=snn_convert_cfg.get("beta", 0.95),
        V_th=snn_convert_cfg.get("V_th", 1.0),
        scales=scales,
        poisson_encode=snn_convert_cfg.get("poisson_encode", False),
        rate_scale=snn_convert_cfg.get("rate_scale", 1.0),
        center_logits=snn_convert_cfg.get("center_logits", True),
    )

    if hasattr(ann_agent.actor, "backbone"):
        action_dim = int(ann_agent.actor.policy_head.out_features)
        actor_snn  = Actor(
            actor_snn_backbone,
            latent_dim=model_cfg.get("hidden_dim", 64),
            action_dim=action_dim,
            critic_informs_actor=getattr(ann_agent.actor, "critic_informs_actor", False),
        )
        actor_snn.policy_head.load_state_dict(ann_agent.actor.policy_head.state_dict())
        last_layer_key = f"layers.{len(ann_agent.actor.backbone.layers) - 1}"
        scale_last     = scales.get(last_layer_key, 1.0)
        T              = snn_convert_cfg.get("T", 100)
        scale_factor   = scale_last / T
        logger_mod.info(f"Scaling Actor Head by {scale_factor:.6f} (Scale: {scale_last:.2f}, T: {T})")
        actor_snn.policy_head.weight.data.mul_(scale_factor)
        actor_snn.policy_head.bias.data.fill_(0.0)
    else:
        actor_snn = actor_snn_backbone

    critic_snn = ann_agent.critic
    if snn_convert_cfg.get("convert_critic", False):
        logger_mod.info("Converting Critic...")
        ann_critic_backbone = ann_agent.critic.backbone
        critic_snn_backbone = snn_from_ann(
            ann_critic_backbone,
            actor_T=snn_convert_cfg.get("T", 100),
            beta=snn_convert_cfg.get("beta", 1.0),
            V_th=snn_convert_cfg.get("V_th", 1.0),
            scales=scales,
            poisson_encode=snn_convert_cfg.get("poisson_encode", False),
            rate_scale=snn_convert_cfg.get("rate_scale", 1.0),
            center_logits=False,
        )
        critic_snn = Critic(
            critic_snn_backbone,
            latent_dim=model_cfg.get("hidden_dim", 64),
        )
        critic_snn.value_head.load_state_dict(ann_agent.critic.value_head.state_dict())

        scale_key_candidates = ["critic.backbone.layers.2", "backbone.layers.2", "layers.2"]
        scale_last_critic = 1.0
        for k in scale_key_candidates:
            if k in scales:
                scale_last_critic = scales[k]
                break
        scale_factor_critic = scale_last_critic / snn_convert_cfg.get("T", 100)
        logger_mod.info(f"Scaling Critic Head by {scale_factor_critic:.6f}")
        critic_snn.value_head.weight.data.mul_(scale_factor_critic)
        critic_snn.value_head.bias.data.fill_(0.0)

    agent_snn = ActorCritic(actor_snn, critic_snn).to(device)
    coerce_module_to_device(agent_snn, device)

    if env_cfg.get("vec_normalize", False):
        agent_snn.obs_rms = env_eval.obs_rms

    # ============================================================
    # Verification
    # ============================================================
    logger_mod.info("Verifying Conversion Quality...")
    actor_metrics = verify_actor_conversion(ann_agent.actor, agent_snn.actor, obs_buf)

    print("\n" + "=" * 40)
    print(" [Verification] Actor Metrics")
    print("=" * 40)
    print(f" MSE:           {actor_metrics['mse_centered']:.6f}")
    print(f" Agreement:     {actor_metrics['argmax_agreement']:.2%}")
    print(f" Correlation:   {actor_metrics['mean_pearson']:.4f}")
    print("=" * 40 + "\n")

    logger.record("convert/actor_mse",         actor_metrics["mse_centered"])
    logger.record("convert/actor_agreement",   actor_metrics["argmax_agreement"])
    logger.record("convert/actor_correlation", actor_metrics["mean_pearson"])

    has_snn_critic = snn_convert_cfg.get("convert_critic", False)
    critic_metrics = {}
    if has_snn_critic:
        critic_metrics = verify_critic_conversion(ann_agent.critic, agent_snn.critic, obs_buf)
        print(" [Verification] Critic Metrics")
        print(f" MSE:           {critic_metrics['mse_centered']:.6f}")
        print(f" Correlation:   {critic_metrics['mean_pearson']:.4f}\n")
        print("=" * 40 + "\n")
        logger.record("convert/critic_mse",         critic_metrics["mse_centered"])
        if "max_error" in critic_metrics:
            logger.record("convert/critic_max_error", critic_metrics["max_error"])
        logger.record("convert/critic_correlation", critic_metrics["mean_pearson"])

    verification_metrics = {"actor": actor_metrics, "critic": critic_metrics}

    ann_actor_out_np  = None
    snn_actor_out_np  = None
    ann_critic_out_np = None
    snn_critic_out_np = None

    try:
        with torch.no_grad():
            if hasattr(ann_agent.actor, "get_action_logits"):
                ann_actor_out = ann_agent.actor.get_action_logits(obs_buf)
            else:
                ann_actor_out = first_output(ann_agent.actor(obs_buf))

            if hasattr(agent_snn.actor, "forward_T"):
                snn_actor_out = first_output(agent_snn.actor.forward_T(obs_buf))
            else:
                snn_actor_out = first_output(agent_snn.actor(obs_buf))

            ann_critic_out = None
            snn_critic_out = None
            if has_snn_critic:
                ann_critic_out = first_output(ann_agent.critic(obs_buf))
                if hasattr(agent_snn.critic, "forward_T"):
                    snn_critic_out = first_output(agent_snn.critic.forward_T(obs_buf))
                else:
                    snn_critic_out = first_output(agent_snn.critic(obs_buf))

        ann_actor_out_np = ann_actor_out.cpu().numpy()
        snn_actor_out_np = snn_actor_out.cpu().numpy()
        if ann_critic_out is not None:
            ann_critic_out_np = ann_critic_out.cpu().numpy()
        if snn_critic_out is not None:
            snn_critic_out_np = snn_critic_out.cpu().numpy()

        convert_actor  = bool(snn_convert_cfg.get("convert_actor", True))
        convert_critic = bool(snn_convert_cfg.get("convert_critic", False))
        conv_exp_name  = "ann2snn_both" if (convert_actor and convert_critic) else "ann2snn_actor"
        conv_env_name  = str(env_cfg.get("id", "Unknown Env"))

        actor_viz_path = os.path.join(plots_dir, f"conversion_actor_seed{seed}.png")
        plot_conversion_validation(
            ann_values=ann_actor_out_np, snn_values=snn_actor_out_np,
            save_path=actor_viz_path, exp_name=conv_exp_name,
            env_name=conv_env_name, component="Actor",
        )
        logger_mod.info(f"Saved Actor conversion plot to: {actor_viz_path}")

        if not has_snn_critic:
            logger_mod.info("Skipping critic conversion plot (convert_critic is false).")
        elif ann_critic_out is not None and snn_critic_out is not None:
            critic_viz_path = os.path.join(plots_dir, f"conversion_critic_seed{seed}.png")
            plot_conversion_validation(
                ann_values=ann_critic_out.cpu().numpy(), snn_values=snn_critic_out.cpu().numpy(),
                save_path=critic_viz_path, exp_name=conv_exp_name,
                env_name=conv_env_name, component="Critic",
            )
            logger_mod.info(f"Saved Critic conversion plot to: {critic_viz_path}")

    except Exception as e:
        logger_mod.warning(f"Failed to generate conversion plots: {e}")
        traceback.print_exc()

    # ============================================================
    # Phase 3: Zero-Shot Evaluation
    # ============================================================
    logger_mod.info("--- Phase 3: Zero-Shot Evaluation ---")

    if hasattr(logger, "_last_eval_rewards"): logger._last_eval_rewards.clear()
    if hasattr(logger, "_last_total_spikes"): logger._last_total_spikes.clear()

    zs_meter, zs_t0 = energy_hook.start_span()
    sticky_cfg = config.get("sticky_action", {})

    logger_mod.info(f"Zero-shot eval over {n_eval_episodes_cfg} episodes...")
    zs_rewards, zs_lengths = [], []
    zs_spikes, zs_actor_spikes, zs_critic_spikes, zs_sparsity, zs_latency = [], [], [], [], []

    for _ in range(n_eval_episodes_cfg):
        r, l, m = evaluate_snn(env_eval, agent_snn, sticky_action=sticky_cfg.get("eval", True))
        zs_rewards.append(r)
        zs_lengths.append(l)
        zs_spikes.append(m.get("total_spikes", 0.0))
        zs_actor_spikes.append(m.get("actor_spikes", 0.0))
        zs_critic_spikes.append(m.get("critic_spikes", 0.0))
        zs_sparsity.append(m.get("sparsity", 0.0))
        zs_latency.append(m.get("mean_latency", 0.0))

    zero_shot_reward = float(np.mean(zs_rewards))   if zs_rewards  else 0.0
    zero_shot_len    = float(np.mean(zs_lengths))   if zs_lengths  else 0.0
    zs_success_rate  = float(np.mean(np.asarray(zs_rewards) >= reward_threshold) * 100.0) if zs_rewards else 0.0

    snn_metrics = {
        "total_spikes":  float(np.mean(zs_spikes))        if zs_spikes       else 0.0,
        "actor_spikes":  float(np.mean(zs_actor_spikes))  if zs_actor_spikes else 0.0,
        "critic_spikes": float(np.mean(zs_critic_spikes)) if zs_critic_spikes else 0.0,
        "sparsity":      float(np.mean(zs_sparsity))      if zs_sparsity     else 0.0,
        "mean_latency":  float(np.mean(zs_latency))       if zs_latency      else 0.0,
    }

    zs_stats = energy_hook.stop_span(zs_meter, zs_t0)
    zs_joules         = float(zs_stats.get("total_joules", 0.0))
    zs_duration       = float(zs_stats.get("duration_seconds", 0.0))
    zs_dynamic_joules = _compute_dynamic_joules(zs_joules, zs_duration, idle_power_watts)
    cumulative_energy         += zs_joules
    cumulative_dynamic_energy += zs_dynamic_joules

    # All phases write to the SAME energy/total key for a continuous curve.
    # (Original code wrote to post_conversion/energy/total here, breaking the curve.)
    logger.record("energy/total",                    cumulative_energy)
    logger.record("energy/total_dynamic",            cumulative_dynamic_energy)
    logger.record("post_conversion/zs_energy",       zs_joules)
    logger.record("post_conversion/zs_energy_dynamic", zs_dynamic_joules)
    logger.record("post_conversion/zero_shot_reward",  zero_shot_reward)
    logger.record("post_conversion/zero_shot_length",  zero_shot_len)
    logger.record("post_conversion/zero_shot_success_rate", zs_success_rate)

    if snn_metrics:
        logger.record("post_conversion/total_spikes",  snn_metrics["total_spikes"])
        logger.record("post_conversion/actor_spikes",  snn_metrics["actor_spikes"])
        logger.record("post_conversion/critic_spikes", snn_metrics["critic_spikes"])
        # Sparsity kept as [0, 1] fraction — consistent with sparsity_factor in
        # EnergyMetrics.  The original * 100 here was the only place it was percent.
        logger.record("post_conversion/sparsity",      snn_metrics["sparsity"])
        logger.record("post_conversion/mean_latency",  snn_metrics["mean_latency"])

    zs_is_solved = (zero_shot_reward >= reward_threshold) and (zs_success_rate >= success_rate_threshold)
    logger.record_episode(
        reward=zero_shot_reward, length=zero_shot_len,
        success=min(100.0, zs_success_rate), source="snn_zero_shot",
    )

    # Side-by-side comparison: converted ANN vs SNN zero-shot
    comparison_metrics: Dict[str, float] = {}

    ann_rewards, ann_lengths, ann_wall_ms = [], [], []
    ann_sticky = sticky_cfg.get("eval", False)
    for _ in range(n_eval_episodes_cfg):
        r_ann, l_ann, m_ann = evaluate(
            env_eval, ann_agent,
            sticky_action=ann_sticky, return_metrics=True,
        )
        ann_rewards.append(r_ann)
        ann_lengths.append(l_ann)
        ann_wall_ms.append(m_ann.get("eval/wall_clock_ms", 0.0))

    converted_ann_reward       = float(np.mean(ann_rewards))   if ann_rewards else 0.0
    converted_ann_len          = float(np.mean(ann_lengths))   if ann_lengths else 0.0
    converted_ann_success_rate = float(np.mean(np.asarray(ann_rewards) >= reward_threshold) * 100.0) if ann_rewards else 0.0
    converted_ann_wall_ms      = float(np.mean(ann_wall_ms))   if ann_wall_ms else 0.0

    logger.record("comparison/converted_ann/reward",         converted_ann_reward)
    logger.record("comparison/converted_ann/success_rate",   converted_ann_success_rate)
    logger.record("comparison/converted_ann/episode_length", converted_ann_len)
    logger.record("comparison/converted_ann/wall_clock_ms",  converted_ann_wall_ms)
    logger.record("comparison/converted_snn/reward",         zero_shot_reward)
    logger.record("comparison/converted_snn/success_rate",   zs_success_rate)
    logger.record("comparison/converted_snn/episode_length", zero_shot_len)
    logger.record("comparison/converted_snn/mean_latency",   snn_metrics.get("mean_latency", 0.0))
    snn_minus_converted_ann = float(zero_shot_reward - converted_ann_reward)
    logger.record("comparison/delta/snn_minus_converted_ann_reward", snn_minus_converted_ann)

    comparison_metrics.update({
        "converted_ann_reward":           converted_ann_reward,
        "converted_ann_success_rate":     converted_ann_success_rate,
        "converted_ann_episode_length":   converted_ann_len,
        "converted_ann_wall_clock_ms":    converted_ann_wall_ms,
        "converted_snn_reward":           zero_shot_reward,
        "converted_snn_success_rate":     zs_success_rate,
        "converted_snn_episode_length":   zero_shot_len,
        "converted_snn_mean_latency":     snn_metrics.get("mean_latency", 0.0),
        "delta_snn_minus_converted_ann_reward": snn_minus_converted_ann,
    })

    # Optional: compare against a separately trained conventional ANN checkpoint
    conventional_ann_metrics: Dict[str, float] = {}
    comparison_cfg = config.get("comparison", {})
    conventional_ann_ckpt = _resolve_local_path(comparison_cfg.get("conventional_ann_checkpoint"))
    if conventional_ann_ckpt:
        logger_mod.info("Evaluating conventional ANN checkpoint: %s", conventional_ann_ckpt)
        try:
            fallback_in_dim     = int(model_cfg.get("in_features", 4))
            fallback_hidden_dim = int(model_cfg.get("hidden_dim", 64))
            fallback_act_dim    = int(
                env_train.single_action_space.n
                if hasattr(env_train, "single_action_space")
                else env_train.action_space.n
            )
            in_dim_conv, act_dim_conv, hidden_dim_conv = _infer_ann_dims_from_checkpoint(
                conventional_ann_ckpt,
                fallback_in_dim=fallback_in_dim,
                fallback_act_dim=fallback_act_dim,
                fallback_hidden_dim=fallback_hidden_dim,
            )
            conventional_ann_agent = make_agent(
                actor_type=ActorType.ANN, critic_type=CriticType.ANN,
                hidden_dim=hidden_dim_conv, dropout=model_cfg.get("dropout", 0.0),
                in_dim=in_dim_conv, act_dim=act_dim_conv,
                gamma=ppo_cfg.get("gamma", 0.99),
                critic_informs_actor=bool(model_cfg.get("critic_informs_actor", False)),
                detach_critic_for_actor=bool(model_cfg.get("detach_critic_for_actor", True)),
                normalize_critic_for_actor=bool(model_cfg.get("normalize_critic_for_actor", True)),
                critic_actor_value_clip=model_cfg.get("critic_actor_value_clip", 5.0),
                critic_actor_norm_momentum=float(model_cfg.get("critic_actor_norm_momentum", 0.01)),
            ).to(device)
            load_checkpoint(
                conventional_ann_ckpt, agent=conventional_ann_agent,
                optimizer=None, logger=None, map_location=device,
            )
            if env_cfg.get("vec_normalize", False):
                conventional_ann_agent.obs_rms = env_eval.obs_rms
                conventional_ann_agent.ret_rms = env_train.ret_rms

            conv_rewards, conv_lengths, conv_wall_ms = [], [], []
            for _ in range(n_eval_episodes_cfg):
                r_conv, l_conv, m_conv = evaluate(
                    env_eval, conventional_ann_agent,
                    sticky_action=ann_sticky, return_metrics=True,
                )
                conv_rewards.append(r_conv)
                conv_lengths.append(l_conv)
                conv_wall_ms.append(m_conv.get("eval/wall_clock_ms", 0.0))

            conventional_ann_reward       = float(np.mean(conv_rewards)) if conv_rewards else 0.0
            conventional_ann_len          = float(np.mean(conv_lengths)) if conv_lengths else 0.0
            conventional_ann_success_rate = float(np.mean(np.asarray(conv_rewards) >= reward_threshold) * 100.0) if conv_rewards else 0.0
            conventional_ann_wall_ms_avg  = float(np.mean(conv_wall_ms)) if conv_wall_ms else 0.0

            logger.record("comparison/conventional_ann/reward",         conventional_ann_reward)
            logger.record("comparison/conventional_ann/success_rate",   conventional_ann_success_rate)
            logger.record("comparison/conventional_ann/episode_length", conventional_ann_len)
            logger.record("comparison/conventional_ann/wall_clock_ms",  conventional_ann_wall_ms_avg)
            logger.record("comparison/delta/snn_minus_conventional_ann_reward",
                          float(zero_shot_reward - conventional_ann_reward))
            logger.record("comparison/delta/converted_ann_minus_conventional_ann_reward",
                          float(converted_ann_reward - conventional_ann_reward))

            conventional_ann_metrics = {
                "conventional_ann_reward":                           conventional_ann_reward,
                "conventional_ann_success_rate":                     conventional_ann_success_rate,
                "conventional_ann_episode_length":                   conventional_ann_len,
                "conventional_ann_wall_clock_ms":                    conventional_ann_wall_ms_avg,
                "delta_snn_minus_conventional_ann_reward":           float(zero_shot_reward - conventional_ann_reward),
                "delta_converted_ann_minus_conventional_ann_reward": float(converted_ann_reward - conventional_ann_reward),
            }
            comparison_metrics.update(conventional_ann_metrics)
        except Exception as e:
            logger_mod.warning(f"Conventional ANN comparison failed: {e}")
    elif comparison_cfg.get("conventional_ann_checkpoint"):
        logger_mod.warning("Conventional ANN checkpoint not found: %s",
                           comparison_cfg["conventional_ann_checkpoint"])

    if save_ckpt_flag:
        save_checkpoint(
            agent=agent_snn, optimizer=None, logger=logger,
            episode=0, num_timesteps=logger.num_timesteps,
            save_dir=ckpt_dir, filename="checkpoint_post_conversion_zeroshot.pt",
            mean_reward=zero_shot_reward, config=config, phase="snn_zero_shot",
            energy_metrics={"total": cumulative_energy, "total_dynamic": cumulative_dynamic_energy},
            extra_data={"zero_shot_success_rate": zs_success_rate, **comparison_metrics},
        )
        logger_mod.info("Saved zero-shot checkpoint.")

    logger.dump()

    if record_zero_shot_video:
        try:
            logger_mod.info("Recording Zero-Shot SNN video...")
            record_best_agent(
                agent_snn, env_id=env_id,
                video_root=os.path.join(log_dir, "videos"),
                exp_name=f"{config.get('run_name', 'ann2snn')}_zeroshot",
                seed=seed, sticky_action=sticky_cfg.get("eval", True), max_steps=500,
                env_kwargs=env_cfg.get("kwargs"),
                partial_obs=env_cfg.get("partial_obs"),
                frame_stack=env_cfg.get("frame_stack"),
                frame_stack_flatten=env_cfg.get("frame_stack_flatten", True),
            )
        except Exception as e:
            logger_mod.warning(f"Zero-shot video recording failed: {e}")

    # ============================================================
    # Phase 4: SNN Fine-tuning
    # ============================================================
    snn_finetune_cfg = config.get("snn_finetune", {})
    test_rewards_ft  = []

    if snn_finetune_cfg.get("enabled", False) and not zs_is_solved:
        logger_mod.info("--- Phase 4: SNN Fine-tuning ---")

        ft_lr          = snn_finetune_cfg.get("lr", 1e-4)
        optimizer_ft   = torch.optim.Adam(agent_snn.parameters(), lr=ft_lr)
        ft_start_update      = 1
        best_ft_rolling_avg  = -float("inf")
        has_logged_ft_solution = False

        if os.path.exists(ckpt_dir):
            ft_ckpts = glob.glob(os.path.join(ckpt_dir, "checkpoint_finetuned_*.pt"))
            if ft_ckpts:
                latest_ft_ckpt = max(ft_ckpts, key=os.path.getctime)
                logger_mod.info(f"Found finetune checkpoint: {latest_ft_ckpt}. Attempting resume...")
                try:
                    data = load_checkpoint(
                        latest_ft_ckpt, agent=agent_snn, optimizer=optimizer_ft,
                        logger=logger, map_location=device,
                    )
                    ft_start_update        = data["episode"] + 1
                    logger.num_timesteps   = data["num_timesteps"]
                    logger.iteration       = ft_start_update + 10000

                    if "energy_metrics" in data and data["energy_metrics"]:
                        cumulative_energy         = data["energy_metrics"].get("total", 0.0)
                        cumulative_dynamic_energy = data["energy_metrics"].get("total_dynamic", 0.0)
                    if "best_rolling_avg" in data:
                        best_ft_rolling_avg = data["best_rolling_avg"]

                    vec_state = data.get("vecnorm_state") or {}
                    if env_cfg.get("vec_normalize", False) and vec_state:
                        _apply_rms_state(env_train.obs_rms, vec_state.get("obs_rms"))
                        _apply_rms_state(env_train.ret_rms, vec_state.get("ret_rms"))
                        env_eval.obs_rms  = env_train.obs_rms
                        agent_snn.obs_rms = env_train.obs_rms
                        agent_snn.ret_rms = env_train.ret_rms

                    logger_mod.info(f"Resumed finetune from update {ft_start_update} (Energy: {cumulative_energy:.2f} J)")
                except Exception as e:
                    logger_mod.error(f"Failed to load finetune checkpoint: {e}")

        ft_trainer = CoreTrainer(
            agent=agent_snn, env_train=env_train, env_eval=env_eval,
            optimizer=optimizer_ft, logger=logger, config=config, hooks=hooks,
        )

        ft_updates = snn_finetune_cfg.get("total_updates", 50)
        # window_size computed once before the loop — n_eval_episodes_cfg and
        # solved_window_eval_episodes don't change, so recalculating inside
        # the eval block every step was wasted work.
        ft_window_size = window_size

        logger_mod.info(f"Starting finetune loop from update {ft_start_update} to {ft_updates}")
        if ft_start_update > ft_updates:
            logger_mod.info("Finetune checkpoint already reached total_updates; skipping.")

        update = max(0, ft_start_update - 1)
        for update in range(ft_start_update, ft_updates + 1):
            logger.iteration = update + 10000

            # ---- A. Train ----
            ft_train_meter, ft_train_t0 = energy_hook.start_span()
            tr_ft, _                    = ft_trainer.train_episode()
            ft_train_stats              = energy_hook.stop_span(ft_train_meter, ft_train_t0)

            ft_train_joules         = float(ft_train_stats.get("total_joules", 0.0))
            ft_train_duration       = float(ft_train_stats.get("duration_seconds", 0.0))
            ft_train_dynamic_joules = _compute_dynamic_joules(
                ft_train_joules, ft_train_duration, idle_power_watts
            )
            cumulative_energy         += ft_train_joules
            cumulative_dynamic_energy += ft_train_dynamic_joules

            snn_algo_latency = get_last_latency(agent_snn.actor)

            logger.record("post_conversion_ft/train_reward",            tr_ft)
            logger.record("energy/train_full_update",                   ft_train_joules)
            logger.record("energy/train_full_update_dynamic",           ft_train_dynamic_joules)
            if snn_algo_latency > 0:
                logger.record("post_conversion_ft/train_latency", snn_algo_latency)

            # ---- B. Periodic Eval ----
            if update % eval_interval == 0 or update == ft_updates:
                ft_eval_meter, ft_eval_t0 = energy_hook.start_span()
                te_ft, success_rate_ft, avg_len_ft, success_count_ft, n_eval_ep_ft = ft_trainer.evaluate()
                ft_eval_stats = energy_hook.stop_span(ft_eval_meter, ft_eval_t0)

                ft_eval_joules         = float(ft_eval_stats.get("total_joules", 0.0))
                ft_eval_duration       = float(ft_eval_stats.get("duration_seconds", 0.0))
                ft_eval_dynamic_joules = _compute_dynamic_joules(
                    ft_eval_joules, ft_eval_duration, idle_power_watts
                )
                cumulative_energy         += ft_eval_joules
                cumulative_dynamic_energy += ft_eval_dynamic_joules

                logger.record("energy/eval_update",         ft_eval_joules)
                logger.record("energy/eval_update_dynamic", ft_eval_dynamic_joules)

                test_rewards_ft.append(te_ft)
                if len(test_rewards_ft) >= ft_window_size:
                    rolling_avg_ft = np.mean(test_rewards_ft[-ft_window_size:])
                else:
                    rolling_avg_ft = np.mean(test_rewards_ft)

                is_solved_ft = (
                    (rolling_avg_ft >= reward_threshold)
                    and (success_rate_ft >= success_rate_threshold)
                )

                # Canonical Phase-4 eval keys — one name per concept.
                logger.record("post_conversion_ft/rolling_reward",  float(rolling_avg_ft))
                logger.record("post_conversion_ft/current_reward",  float(te_ft))
                logger.record("post_conversion_ft/episode_length",  float(avg_len_ft))
                logger.record("post_conversion_ft/success_rate",    float(success_rate_ft))
                logger.record("post_conversion_ft/success_count",   float(success_count_ft))
                logger.record("post_conversion_ft/n_eval_episodes", float(n_eval_ep_ft))
                if is_solved_ft and not has_logged_ft_solution:
                    logger.record("post_conversion_ft/is_solved_at_update", float(update))
                logger.record_episode(
                    reward=te_ft, length=avg_len_ft,
                    success=min(100.0, success_rate_ft), source="snn_finetune",
                )

                if save_ckpt_flag and (update % ckpt_interval == 0):
                    save_checkpoint(
                        agent=agent_snn, optimizer=optimizer_ft, logger=logger,
                        episode=update, num_timesteps=logger.num_timesteps,
                        save_dir=ckpt_dir, filename=f"checkpoint_finetuned_ep{update:05d}.pt",
                        mean_reward=te_ft, config=config, phase="snn_finetune",
                        energy_metrics={"total": cumulative_energy, "total_dynamic": cumulative_dynamic_energy},
                        extra_data={"best_rolling_avg": best_ft_rolling_avg},
                    )

                if rolling_avg_ft > best_ft_rolling_avg:
                    best_ft_rolling_avg = rolling_avg_ft
                    if save_ckpt_flag:
                        save_checkpoint(
                            agent=agent_snn, optimizer=optimizer_ft, logger=logger,
                            episode=update, num_timesteps=logger.num_timesteps,
                            save_dir=ckpt_dir, mean_reward=te_ft, config=config,
                            suffix="finetuned_best", phase="snn_finetune",
                            energy_metrics={"total": cumulative_energy, "total_dynamic": cumulative_dynamic_energy},
                            extra_data={"best_rolling_avg": best_ft_rolling_avg},
                        )
                        logger_mod.info(f"New best SNN model saved! MA: {best_ft_rolling_avg:.2f}")

                if is_solved_ft and not has_logged_ft_solution:
                    logger_mod.info(
                        f"✅ SNN fine-tuning solved at update {update} "
                        f"(MA: {rolling_avg_ft:.2f})! Continuing for stability..."
                    )
                    has_logged_ft_solution = True
                    if record_finetune_solution_video:
                        try:
                            record_best_agent(
                                agent_snn, env_id=env_id,
                                video_root=os.path.join(log_dir, "videos"),
                                exp_name=f"{config.get('run_name', 'ann2snn')}_finetuned",
                                seed=seed, sticky_action=sticky_cfg.get("eval", True),
                                max_steps=500, env_kwargs=env_cfg.get("kwargs"),
                                partial_obs=env_cfg.get("partial_obs"),
                                frame_stack=env_cfg.get("frame_stack"),
                                frame_stack_flatten=env_cfg.get("frame_stack_flatten", True),
                            )
                        except Exception as e:
                            logger_mod.warning(f"Fine-tuned video recording failed: {e}")

            # Write cumulative totals once per update.
            logger.record("energy/total",         cumulative_energy)
            logger.record("energy/total_dynamic", cumulative_dynamic_energy)

            if update % 10 == 0:
                logger.dump()

    elif snn_finetune_cfg.get("enabled", False):
        logger_mod.info(
            "Skipping SNN fine-tuning: zero-shot reward %.2f >= threshold %.2f.",
            zero_shot_reward, reward_threshold,
        )

    # ============================================================
    # Post-Conversion Validation Data Collection
    # ============================================================
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

        collected_logits         = []
        collected_output_spikes  = []
        collected_actor_logits   = []
        collected_actor_spikes   = []
        collected_values         = []
        collected_step_rewards   = []
        collected_episode_index  = []
        collected_step_in_episode = []
        collected_activations    = {"layer_0": [], "layer_1": [], "output": []}

        def _budget_reached() -> bool:
            step_cap    = (val_n_steps    is not None) and (step_count         >= val_n_steps)
            episode_cap = (val_n_episodes is not None) and (episodes_completed >= val_n_episodes)
            return step_cap or episode_cap

        while not _budget_reached():
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                if obs_t.ndim == 1:
                    obs_t = obs_t.unsqueeze(0)

                actor_ref = agent_snn.actor if hasattr(agent_snn, "actor") else agent_snn
                if hasattr(actor_ref, "forward_T"):
                    logits, _, acts = actor_ref.forward_T(obs_t, return_activations=True)
                    if logits is not None:
                        logits_vec = logits.detach().cpu().reshape(logits.shape[0], -1)[0]
                        collected_actor_logits.append(logits_vec.numpy())
                        collected_logits.append(float(logits.mean().item()))
                    for k, v in acts.items():
                        if k in collected_activations:
                            collected_activations[k].append(v.mean().item())
                    if "output" in acts:
                        collected_output_spikes.append(float(acts["output"].mean().item()))
                    if "output_per_action" in acts:
                        spike_vec = acts["output_per_action"].detach().cpu().reshape(
                            acts["output_per_action"].shape[0], -1
                        )[0]
                        collected_actor_spikes.append(spike_vec.numpy())
                else:
                    logits = first_output(actor_ref(obs_t))
                    collected_logits.append(float(logits.mean().item()))

                v = first_output(agent_snn.critic(obs_t))
                collected_values.append(float(v.mean().item()))

                if val_deterministic:
                    action = int(torch.argmax(logits, dim=-1).item())
                else:
                    action = int(torch.distributions.Categorical(logits=logits).sample().item())

                reward_val = 0.0
                done_flag  = False
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
                collected_step_rewards.append(reward_val)
                collected_episode_index.append(episodes_completed)
                collected_step_in_episode.append(ep_len)

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
            "rewards":          np.array(collected_step_rewards),
            "episode_index":    np.array(collected_episode_index),
            "step_in_episode":  np.array(collected_step_in_episode),
            "activations":      {k: np.array(v) for k, v in collected_activations.items()},
            "output_logits":    np.array(collected_logits),
            "output_spikes":    np.array(collected_output_spikes),
            "critic_values":    np.array(collected_values),
        }
        episode_metrics_out = {
            "returns":       np.array(episode_returns),
            "lengths":       np.array(episode_lengths),
            "num_completed": int(episodes_completed),
        }

        # First-episode critic value trace for intra-episode value dynamics
        critic_values_single_episode = [
            float(v) for v, ep_idx in zip(collected_values, collected_episode_index)
            if int(ep_idx) == 0
        ]

        validation_data = {
            "step_traces":                   step_traces,
            "episode_metrics":               episode_metrics_out,
            "critic_values_single_episode":  np.array(critic_values_single_episode),
            "activations":                   step_traces["activations"],
            "output_logits":                 step_traces["output_logits"],
            "output_spikes":                 step_traces["output_spikes"],
            "intra_episode_values":          step_traces["critic_values"],
            "metadata": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "seed":          int(seed),
                "checkpoint":    ckpt_dir,
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
        if collected_actor_logits:
            validation_data["actor_output_potentials"] = np.asarray(collected_actor_logits, dtype=float).T
        if collected_actor_spikes:
            validation_data["actor_output_spikes"] = np.asarray(collected_actor_spikes, dtype=float).T
        if ann_actor_out_np is not None and snn_actor_out_np is not None:
            validation_data["ann_actor_outputs"] = ann_actor_out_np
            validation_data["snn_actor_outputs"] = snn_actor_out_np
        if ann_critic_out_np is not None and snn_critic_out_np is not None:
            validation_data["ann_critic_outputs"] = ann_critic_out_np
            validation_data["snn_critic_outputs"] = snn_critic_out_np

        if episode_returns:
            logger.record("post_eval/episode_return_mean", float(np.mean(episode_returns)))
            logger.record("post_eval/episode_length_mean", float(np.mean(episode_lengths)))
        logger.record("post_eval/steps_collected",    float(step_count))
        logger.record("post_eval/episodes_completed", float(episodes_completed))

    except Exception as e:
        logger_mod.warning(f"Validation data collection failed: {e}")

    # --- Return ---
    ret = {
        "agent":              agent_snn,
        "logger":             logger,
        "zero_shot_reward":   zero_shot_reward,
        "comparison_metrics": comparison_metrics,
        "validation_metrics": verification_metrics,
        "train_rewards":      train_rewards + (test_rewards_ft if test_rewards_ft else []),
        "test_rewards":       test_rewards + test_rewards_ft,
        "validation_data":    validation_data,
    }
    if test_rewards_ft:
        ret["finetune_mean_reward"] = test_rewards_ft[-1]

    return ret