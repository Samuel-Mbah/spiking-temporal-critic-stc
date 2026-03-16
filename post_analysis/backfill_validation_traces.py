#!/usr/bin/env python3
"""
Backfill validation traces from saved checkpoints (no retraining).

This script regenerates `validation_data.npz` for existing seed runs so
multi-seed intra-episode and timing micro-dynamics plots can be produced.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

# --- Path setup ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.training.agents import make_agent, resolve_cartpole_types
from src.training.envs import make_envs, set_global_seeds
from src.training.evaluate import get_last_latency
from src.utils.checkpoint import load_checkpoint
from src.utils.torch_utils import first_output


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def register_custom_envs() -> None:
    modules = [
        "src.envs.t_maze",
    ]
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:
            logger.debug(f"Custom env module import skipped ({module}): {exc}")


def infer_seed_from_dir(seed_dir: Path) -> Optional[int]:
    m = re.search(r"seed_(\d+)", seed_dir.name)
    if not m:
        return None
    return int(m.group(1))


def choose_checkpoint(seed_dir: Path, checkpoint_name: str) -> Optional[Path]:
    ckpt_dir = seed_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    preferred = ckpt_dir / checkpoint_name
    if preferred.exists():
        return preferred
    all_ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return all_ckpts[0] if all_ckpts else None


def _save_validation_traces(log_dir: Path, validation_data: Dict[str, Any]) -> Optional[Path]:
    payload: Dict[str, np.ndarray] = {}
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
        return None

    out_path = log_dir / "validation_data.npz"
    np.savez_compressed(out_path, **payload)
    return out_path


@torch.no_grad()
def collect_validation_data(agent: torch.nn.Module, env_eval, config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    env_cfg = config.get("env", {}) or {}
    post_eval_cfg = config.get("post_eval", config.get("ppo", {}).get("post_eval", {})) or {}

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

    def _budget_reached() -> bool:
        step_cap = (val_n_steps is not None) and (step_count >= val_n_steps)
        episode_cap = (val_n_episodes is not None) and (episodes_completed >= val_n_episodes)
        return step_cap or episode_cap

    while not _budget_reached():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        value_raw_t = None
        val_tau = None
        val = 0.0
        critic_ref = agent.critic if hasattr(agent, "critic") else None
        actor_ref = agent.actor if hasattr(agent, "actor") else agent

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

        lat = float(val_tau) if val_tau is not None else get_last_latency(critic_ref)
        if lat <= 0:
            lat = get_last_latency(actor_ref)
        collected_values.append(val)
        collected_latencies.append(lat)

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
        obs_next, reward, terminated, truncated, _ = env_eval.step([act_int])
        reward_val = float(reward[0])
        done_flag = bool(terminated[0] or truncated[0])
        obs = obs_next[0]

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

    critic_values_single_episode = []
    for v, ep_idx in zip(collected_values, collected_episode_index):
        if int(ep_idx) == 0:
            critic_values_single_episode.append(float(v))
        elif int(ep_idx) > 0 and critic_values_single_episode:
            break

    step_traces = {
        "critic_values": np.array(collected_values),
        "critic_timings": np.array(collected_latencies),
        "rewards": np.array(collected_step_rewards),
        "episode_index": np.array(collected_episode_index),
        "step_in_episode": np.array(collected_step_in_episode),
    }
    episode_metrics = {
        "returns": np.array(episode_returns),
        "lengths": np.array(episode_lengths),
        "num_completed": int(episodes_completed),
    }

    return {
        "step_traces": step_traces,
        "episode_metrics": episode_metrics,
        "critic_values": step_traces["critic_values"],
        "critic_values_single_episode": np.array(critic_values_single_episode),
        "critic_timings": step_traces["critic_timings"],
        "intra_episode_values": np.array(collected_values),
    }


def backfill_seed(seed_dir: Path, config: Dict[str, Any], checkpoint_name: str, overwrite: bool, env_active: Optional[bool], device_override: Optional[str]) -> str:
    out_file = seed_dir / "validation_data.npz"
    if out_file.exists() and not overwrite:
        return f"skip: exists ({out_file})"

    ckpt = choose_checkpoint(seed_dir, checkpoint_name=checkpoint_name)
    if ckpt is None:
        return "skip: no checkpoint found"

    seed_cfg = copy.deepcopy(config)
    seed_num = infer_seed_from_dir(seed_dir)
    if seed_num is not None:
        seed_cfg["env_seed"] = int(seed_num)

    if env_active is not None:
        seed_cfg.setdefault("env", {}).setdefault("kwargs", {})["active"] = bool(env_active)

    if device_override:
        seed_cfg["device"] = str(device_override)

    # Deterministic seed control for reproducible validation traces.
    seed_control = seed_cfg.get("seed_control", {}) or {}
    set_global_seeds(
        int(seed_cfg.get("env_seed", 42)),
        deterministic_torch=bool(seed_control.get("deterministic_torch", True)),
        cudnn_benchmark=bool(seed_control.get("cudnn_benchmark", False)),
    )

    env_cfg = seed_cfg.get("env", {}) or {}
    env_train, env_eval = make_envs(
        seed=int(seed_cfg.get("env_seed", 42)),
        env_id=env_cfg.get("id", "CartPole-v1"),
        n_envs=int(env_cfg.get("n_envs", 8)),
        env_kwargs=env_cfg.get("kwargs", {}),
        partial_obs=env_cfg.get("partial_obs"),
        frame_stack=env_cfg.get("frame_stack"),
        frame_stack_flatten=env_cfg.get("frame_stack_flatten", True),
    )

    try:
        device = torch.device(seed_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        model_cfg = seed_cfg.get("model", {}) or {}
        ppo_cfg = seed_cfg.get("ppo", {}) or {}
        snn_cfg = seed_cfg.get("snn", {}) or {}
        mode = model_cfg.get("mode", "snn_actor_ann_critic")
        actor_type, critic_type = resolve_cartpole_types(mode)
        agent = make_agent(
            actor_type=actor_type,
            critic_type=critic_type,
            hidden_dim=model_cfg.get("hidden_dim", 64),
            in_dim=model_cfg.get("in_features", 4),
            act_dim=env_train.single_action_space.n if hasattr(env_train, "single_action_space") else env_train.action_space.n,
            gamma=ppo_cfg.get("gamma", 0.99),
            critic_informs_actor=bool(model_cfg.get("critic_informs_actor", False)),
            detach_critic_for_actor=bool(model_cfg.get("detach_critic_for_actor", True)),
            normalize_critic_for_actor=bool(model_cfg.get("normalize_critic_for_actor", True)),
            critic_actor_value_clip=model_cfg.get("critic_actor_value_clip", 5.0),
            critic_actor_norm_momentum=float(model_cfg.get("critic_actor_norm_momentum", 0.01)),
            **snn_cfg,
        )
        agent = agent.to(device)
        agent.eval()

        load_checkpoint(str(ckpt), agent=agent, optimizer=None, logger=None, map_location=device)
        validation_data = collect_validation_data(agent=agent, env_eval=env_eval, config=seed_cfg, device=device)
        saved = _save_validation_traces(seed_dir, validation_data)
        if saved is None:
            return "error: no trace payload produced"
        return f"ok: {saved.name} from {ckpt.name}"
    finally:
        try:
            env_train.close()
        except Exception:
            pass
        try:
            env_eval.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser("Backfill validation_data.npz from checkpoints")
    parser.add_argument("--experiment-root", required=True, type=str, help="Root directory containing seed_* folders")
    parser.add_argument("--config", required=True, type=str, help="Config YAML used for this experiment")
    parser.add_argument("--seed-glob", default="seed_*", type=str, help="Glob under experiment root")
    parser.add_argument("--checkpoint-name", default="checkpoint_best.pt", type=str, help="Preferred checkpoint filename")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing validation_data.npz")
    parser.add_argument("--max-seeds", type=int, default=0, help="Optional limit (0 = all)")
    parser.add_argument("--env-active", type=str, default="auto", choices=["auto", "true", "false"], help="Override env.kwargs.active")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g., cpu, cuda)")
    args = parser.parse_args()

    register_custom_envs()

    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    exp_root = Path(args.experiment_root)
    seed_dirs = sorted(p for p in exp_root.glob(args.seed_glob) if p.is_dir())
    if args.max_seeds and args.max_seeds > 0:
        seed_dirs = seed_dirs[: args.max_seeds]

    if not seed_dirs:
        logger.error(f"No seed directories found: {exp_root / args.seed_glob}")
        return 1

    env_active_override: Optional[bool] = None
    if args.env_active == "true":
        env_active_override = True
    elif args.env_active == "false":
        env_active_override = False
    else:
        root_l = str(exp_root).lower()
        if "tmaze_active" in root_l:
            env_active_override = True
        elif "tmaze_passive" in root_l:
            env_active_override = False

    ok = 0
    skip = 0
    err = 0
    for seed_dir in seed_dirs:
        try:
            result = backfill_seed(
                seed_dir=seed_dir,
                config=config,
                checkpoint_name=args.checkpoint_name,
                overwrite=bool(args.overwrite),
                env_active=env_active_override,
                device_override=args.device,
            )
            if result.startswith("ok:"):
                ok += 1
                logger.info(f"{seed_dir.name}: {result}")
            elif result.startswith("skip:"):
                skip += 1
                logger.info(f"{seed_dir.name}: {result}")
            else:
                err += 1
                logger.error(f"{seed_dir.name}: {result}")
        except Exception as exc:
            err += 1
            logger.exception(f"{seed_dir.name}: failed with exception: {exc}")

    logger.info(f"Backfill summary | ok={ok} skip={skip} err={err}")
    return 0 if err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
