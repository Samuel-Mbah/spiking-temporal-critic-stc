import os
from datetime import datetime
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from typing import Optional

from src.training.evaluate import get_last_spike_count, get_model_device
from src.training.envs import apply_obs_wrappers


def make_recording_env(
    env_id: str,
    video_dir: str,
    exp_name: str,
    seed: int,
    *,
    env_kwargs: dict | None = None,
    partial_obs: dict | None = None,
    frame_stack: int | None = None,
    frame_stack_flatten: bool = True,
) -> gym.Env:
    """Creates an environment wrapped for video recording."""
    unique_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_exp_name = f"{exp_name}_seed{seed}_{unique_suffix}"
    out_dir = os.path.join(video_dir, unique_exp_name)
    os.makedirs(out_dir, exist_ok=True)

    make_kwargs = dict(env_kwargs or {})
    make_kwargs["render_mode"] = "rgb_array"
    env = gym.make(env_id, **make_kwargs)
    env = apply_obs_wrappers(env, partial_obs=partial_obs, frame_stack=frame_stack, frame_stack_flatten=frame_stack_flatten)
    
    # Record first episode only
    env = RecordVideo(
        env,
        video_folder=out_dir,
        episode_trigger=lambda ep: ep == 0, 
        name_prefix=unique_exp_name
    )
    
    # env.reset(seed=seed)
    return env, out_dir

@torch.no_grad()
def record_best_agent(
    agent,
    *,
    env_id: str = "CartPole-v1",
    video_root: str = "videos",
    exp_name: str = "default_exp",
    max_steps: int = 500,
    sticky_action: bool = True,
    seed: int = 42,
    **kwargs,
):
    """
    Records a single episode of the agent acting in the environment.
    Applies sticky actions if specified.
    """
    agent.eval()
    device = get_model_device(agent)

    env, out_dir = make_recording_env(
        env_id,
        video_root,
        exp_name,
        seed,
        env_kwargs=kwargs.get("env_kwargs"),
        partial_obs=kwargs.get("partial_obs"),
        frame_stack=kwargs.get("frame_stack"),
        frame_stack_flatten=kwargs.get("frame_stack_flatten", True),
    )
    obs, _ = env.reset(seed=seed)
    
    total_reward = 0.0
    prev_action: Optional[int] = None
    
    try:
        for _ in range(max_steps):
            # 1. Manual Normalization (if agent expects it)
            if hasattr(agent, "obs_rms"):
                rms = agent.obs_rms
                obs = np.clip((obs - rms.mean) / np.sqrt(rms.var + 1e-8), -10.0, 10.0)

            # 2. Forward Pass
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            # Prefer deterministic policy action
            if hasattr(agent, "act"):
                action, _, _ = agent.act(obs_t, deterministic=True)
                action = int(action.item())
            else:
                logits, _ = agent(obs_t)
                action = int(torch.argmax(logits, dim=-1).item())

            # 3. Sticky Action Override (match evaluate() behavior)
            allow_sticky = (
                sticky_action
                and hasattr(agent, "actor")
                and (hasattr(agent.actor, "last_spike_count") or hasattr(agent.actor, "get_spike_stats"))
            )
            if allow_sticky:
                if get_last_spike_count(agent.actor) == 0.0 and prev_action is not None:
                    action = prev_action

            # 4. Step
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            prev_action = action

            if terminated or truncated:
                break
    finally:
        env.close()
    print(f"🎥 Video saved to {out_dir} | Reward: {total_reward}")
