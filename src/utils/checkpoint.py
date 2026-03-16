"""
Utilities for saving and loading checkpoints during training.
"""
from __future__ import annotations

import os
import torch
import tempfile
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import numpy as np

# Create a dedicated system logger for this module's text output
system_logger = logging.getLogger(__name__)

@dataclass
class Checkpoint:
    episode: int
    num_timesteps: int
    actor_state: Dict[str, Any]
    critic_state: Dict[str, Any]
    optimizer_state: Optional[Dict[str, Any]] = None
    logger_state: Optional[Any] = None 
    mean_reward: Optional[float] = None
    config: Optional[Dict[str, Any]] = None
    phase: Optional[str] = None
    energy_metrics: Optional[Dict[str, float]] = None
    vecnorm_state: Optional[Dict[str, Dict[str, Any]]] = None


def _serialize_rms(rms) -> Optional[Dict[str, Any]]:
    if rms is None:
        return None
    return {
        "mean": np.asarray(rms.mean).tolist(),
        "var": np.asarray(rms.var).tolist(),
        "count": float(rms.count),
    }


def save_checkpoint(
    *,
    agent,
    optimizer,
    logger,
    episode: int,
    num_timesteps: int,
    save_dir: str,
    mean_reward: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    phase: Optional[str] = None,
    energy_metrics: Optional[Dict[str, float]] = None,
    filename: Optional[str] = None,
    suffix: Optional[str] = None,       # <--- Added
    extra_data: Optional[Dict] = None,  # <--- Added
) -> str:
    """
    Atomically save a training checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. Filename Logic with Suffix Support ---
    if filename is None:
        if suffix:
            # If suffix is "best", filename becomes "checkpoint_best.pt"
            filename = f"checkpoint_{suffix}.pt"
        elif mean_reward is not None:
            filename = f"checkpoint_ep{episode:05d}_R{mean_reward:.2f}.pt"
        else:
            filename = f"checkpoint_ep{episode:05d}.pt"
            
    final_path = os.path.join(save_dir, filename)

    # --- Capture Logger State ---
    logger_state = None
    if hasattr(logger, "buffers"):
        logger_state = dict(logger.buffers)
    elif hasattr(logger, "episode_buffer"):
        logger_state = logger.episode_buffer

    # Create the structured Checkpoint object
    vecnorm_state = None
    if hasattr(agent, "obs_rms") or hasattr(agent, "ret_rms"):
        vecnorm_state = {
            "obs_rms": _serialize_rms(getattr(agent, "obs_rms", None)),
            "ret_rms": _serialize_rms(getattr(agent, "ret_rms", None)),
        }

    ckpt = Checkpoint(
        episode=episode,
        num_timesteps=num_timesteps,
        actor_state=agent.actor.state_dict(),
        critic_state=agent.critic.state_dict(),
        optimizer_state=optimizer.state_dict() if optimizer else None,
        logger_state=logger_state,
        mean_reward=mean_reward,
        config=config,
        phase=phase,
        energy_metrics=energy_metrics,
        vecnorm_state=vecnorm_state,
    )

    temp_name = None
    try:
        # Convert dataclass to dict
        ckpt_dict = asdict(ckpt)
        
        # --- 2. Inject Extra Data (Safe Update) ---
        # This adds 'best_rolling_avg' to the dict without breaking the Checkpoint dataclass
        if extra_data:
            ckpt_dict.update(extra_data)
        
        with tempfile.NamedTemporaryFile(dir=save_dir, delete=False, suffix=".tmp") as tmp:
            torch.save(ckpt_dict, tmp)
            temp_name = tmp.name

        os.replace(temp_name, final_path)
        system_logger.debug(f"Saved checkpoint: {final_path}")
        return final_path

    except Exception as e:
        system_logger.error(f"Failed to save checkpoint to {final_path}: {e}")
        if temp_name and os.path.exists(temp_name):
            os.remove(temp_name)
        return ""


def load_checkpoint(
    checkpoint_path: str,
    *,
    agent,
    optimizer=None,
    logger=None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """
    Load checkpoint safely across devices. 
    Robust to legacy naming conventions (e.g. actor_state vs actor_state_dict).
    """
    system_logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # weights_only=False required for complex objects
    data = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    if not isinstance(data, dict) and hasattr(data, "__dict__"):
        data = data.__dict__

    # --- Robust Loading (New vs Legacy Keys) ---
    
    # 1. Actor
    actor_st = data.get("actor_state") or data.get("actor_state_dict")
    if actor_st is not None:
        agent.actor.load_state_dict(actor_st)
    else:
        system_logger.warning("Checkpoint missing 'actor_state'. Skipping actor load.")

    # 2. Critic
    critic_st = data.get("critic_state") or data.get("critic_state_dict")
    if critic_st is not None:
        agent.critic.load_state_dict(critic_st)
    else:
        system_logger.warning("Checkpoint missing 'critic_state'. Skipping critic load.")

    # 3. Optimizer
    if optimizer:
        opt_st = data.get("optimizer_state") or data.get("optimizer_state_dict")
        if opt_st is not None:
            optimizer.load_state_dict(opt_st)

    # 4. Logger
    if logger:
        state = data.get("logger_state") or data.get("logger_buffer")
        if state is not None:
            if hasattr(logger, "buffers") and isinstance(state, dict):
                # We can extend the existing defaultdict with the loaded dict data
                for k, v in state.items():
                    if k in logger.buffers:
                        logger.buffers[k].extend(v)
            elif hasattr(logger, "episode_buffer"):
                logger.episode_buffer = state

    return {
        "episode": data.get("episode", 0),
        "num_timesteps": data.get("num_timesteps", 0),
        "mean_reward": data.get("mean_reward"),
        "config": data.get("config"),
        "energy_metrics": data.get("energy_metrics"),
        "vecnorm_state": data.get("vecnorm_state"),
    }
