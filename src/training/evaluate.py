import torch
import numpy as np
import time
from typing import Tuple, Dict, Any, Union, Optional

# --------------------------- Helper for evaluate function ------------------------------------------------#
def _to_float(x) -> float:
    """Robust float conversion for tensors and scalars."""
    if isinstance(x, (int, float)):
        return float(x)
    if torch.is_tensor(x):
        return float(x.detach().item())
    try:
        return float(x)
    except Exception:
        return 0.0


def get_last_spike_count(module) -> float:
    """
    Safely retrieve spike count from SNN modules.
    Handles wrappers (Actor) and methods vs attributes.
    """
    if module is None:
        return 0.0

    # Try direct access (Attribute or Method)
    if hasattr(module, "last_spike_count"):
        val = module.last_spike_count
        if callable(val):
            return float(val())
        return float(val)

    # Try unwrapping 'backbone'
    if hasattr(module, "backbone"):
        return get_last_spike_count(module.backbone)

    # 3. Try unwrapping 'actor'
    if hasattr(module, "actor"):
        return get_last_spike_count(module.actor)

    return 0.0


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Return the device of the first parameter."""
    return next(model.parameters()).device


def get_last_latency(module) -> float:
    """Safely retrieve mean latency from SNN modules."""
    if module is None: return 0.0

    # Try direct access
    if hasattr(module, "last_latency"):
        val = module.last_latency
        return float(val() if callable(val) else val)

    # Unwrap Actor/Backbone
    if hasattr(module, "actor"): return get_last_latency(module.actor)
    if hasattr(module, "backbone"): return get_last_latency(module.backbone)

    return 0.0

def get_spike_stats_safe(module):
    """Safely retrieve cumulative spike stats from SNN modules, handling wrappers."""
    if module is None:
        return {"total_spikes": 0.0, "total_timesteps": 0.0, "firing_rate": 0.0, "sparsity": 0.0}

    # 1. Direct match
    if hasattr(module, "get_spike_stats"):
        try:
            stats = module.get_spike_stats() or {}
            return {
                "total_spikes": float(stats.get("total_spikes", 0.0) or 0.0),
                "total_timesteps": float(stats.get("total_timesteps", 0.0) or 0.0),
                "firing_rate": float(stats.get("firing_rate", 0.0) or 0.0),
                "sparsity": float(stats.get("sparsity", 0.0) or 0.0),
            }
        except Exception:
            pass

    # 2. Generic SNNBlock-style fallback (works for timing/spike critics too)
    block_names = ("block1", "block2", "block_out")
    has_any_block = any(hasattr(module, b) for b in block_names)
    if has_any_block:
        total_spikes = 0.0
        total_timesteps = 0.0
        for b in block_names:
            blk = getattr(module, b, None)
            if blk is None:
                continue
            sp = getattr(blk, "total_spikes", 0.0)
            ts = getattr(blk, "total_timesteps", 0.0)
            if torch.is_tensor(sp):
                sp = sp.detach().cpu().item()
            if torch.is_tensor(ts):
                ts = ts.detach().cpu().item()
            total_spikes += float(sp or 0.0)
            total_timesteps += float(ts or 0.0)
        fr = (total_spikes / total_timesteps) if total_timesteps > 0 else 0.0
        return {
            "total_spikes": total_spikes,
            "total_timesteps": total_timesteps,
            "firing_rate": fr,
            "sparsity": 1.0 - fr,
        }

    # 3. Unwrap common wrappers
    if hasattr(module, "actor"):
        return get_spike_stats_safe(module.actor)
    if hasattr(module, "critic"):
        return get_spike_stats_safe(module.critic)
    if hasattr(module, "backbone"):
        return get_spike_stats_safe(module.backbone)

    return {"total_spikes": 0.0, "total_timesteps": 0.0, "firing_rate": 0.0, "sparsity": 0.0}

# --------------------------- End of helper ---------------------------------------------------------------- #

@torch.no_grad()
def evaluate(
    env,
    agent,
    *,
    sticky_action: bool = True,
    return_metrics: bool = False,
    seed: Optional[int] = None,
) -> Union[Tuple[float, int], Tuple[float, int, Dict[str, float]]]:
    """
    Evaluate an agent for a single episode.
    
    Args:
        env: The gymnasium environment.
        agent: The policy/agent model.
        sticky_action: Whether to reuse previous action if no spikes occur (SNNs).
        return_metrics: If True, returns (reward, steps, metrics_dict).
    
    Returns:
        (reward, steps) OR (reward, steps, metrics_dict)
    """
    agent.eval()
    device = get_model_device(agent)

    # Detect if env is vectorized
    is_vector_env = hasattr(env, "num_envs")

    if seed is None:
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(seed=seed)
    done = False
    episode_return = 0.0
    prev_action: int | None = None
    steps = 0

    # Tracking for SNNs + wall-clock latency
    total_latency = 0.0
    total_spikes = 0.0
    total_actor_spikes = 0.0
    total_critic_spikes = 0.0
    total_wall_clock = 0.0
    # Episode-local cumulative baselines (avoid counting pre-existing training totals).
    prev_actor_spikes = float(get_spike_stats_safe(getattr(agent, "actor", None)).get("total_spikes", 0.0))
    prev_critic_spikes = float(get_spike_stats_safe(getattr(agent, "critic", None)).get("total_spikes", 0.0))
    
    while not done:
        step_start = time.perf_counter()
        # Prepare observation
        if is_vector_env:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        logits, _ = agent(obs_t)

        # Deterministic policy
        action = int(torch.argmax(logits, dim=-1).item())

        # Sticky actions only for spiking actors
        allow_sticky = (
            sticky_action
            and hasattr(agent, "actor")
            and (hasattr(agent.actor, "last_spike_count") or hasattr(agent.actor, "get_spike_stats"))
        )
        actor_stats = get_spike_stats_safe(getattr(agent, "actor", None))
        critic_stats = get_spike_stats_safe(getattr(agent, "critic", None))
        actor_cum = float(actor_stats.get("total_spikes", 0.0))
        critic_cum = float(critic_stats.get("total_spikes", 0.0))
        if actor_cum < prev_actor_spikes:
            actor_step_spikes = actor_cum
        else:
            actor_step_spikes = actor_cum - prev_actor_spikes
        if critic_cum < prev_critic_spikes:
            critic_step_spikes = critic_cum
        else:
            critic_step_spikes = critic_cum - prev_critic_spikes
        prev_actor_spikes = actor_cum
        prev_critic_spikes = critic_cum
        spike_count = actor_step_spikes if allow_sticky else 0.0
        
        # Track metrics if requested
        if return_metrics:
            total_actor_spikes += actor_step_spikes
            total_critic_spikes += critic_step_spikes
            total_spikes += actor_step_spikes + critic_step_spikes
            total_latency += get_last_latency(agent.actor) if hasattr(agent, "actor") else 0.0
            total_wall_clock += time.perf_counter() - step_start

        if allow_sticky:
            if spike_count == 0.0 and prev_action is not None:
                action = prev_action

        # Step the environment
        if is_vector_env:
            obs, reward, terminated, truncated, _ = env.step([action])
            reward = reward[0]
            done = terminated[0] or truncated[0]
        else:
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        episode_return += reward
        steps += 1
        prev_action = action

    if return_metrics:
        metrics = {
            "eval/latency": total_latency / steps if steps > 0 else 0.0,
            "eval/spikes": total_spikes,
            "eval/spikes_actor": total_actor_spikes,
            "eval/spikes_critic": total_critic_spikes,
            "eval/spikes_per_step": total_spikes / steps if steps > 0 else 0.0,
            "eval/spikes_actor_per_step": total_actor_spikes / steps if steps > 0 else 0.0,
            "eval/spikes_critic_per_step": total_critic_spikes / steps if steps > 0 else 0.0,
            "eval/wall_clock_ms": (total_wall_clock / steps) * 1000.0 if steps > 0 else 0.0,
        }
        return float(episode_return), steps, metrics

    return float(episode_return), steps

@torch.no_grad()
def gather_observations(
    env,
    policy_net: torch.nn.Module,
    *,
    episodes: int = 5,
    stochastic_actions: bool = False,
    obs_noise_std: float = 0.0,
    max_steps_per_episode: int = 0,
) -> torch.Tensor:
    """
    Run the policy in the environment and collect visited observations.
    Supports both standard and vectorized environments.
    """
    device = get_model_device(policy_net)
    policy_net.eval()

    is_vector_env = hasattr(env, "num_envs")
    observations = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0

        while not done:
            if is_vector_env:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            # Optional observation perturbation to improve calibration diversity.
            if obs_noise_std > 0.0:
                obs_t = obs_t + torch.randn_like(obs_t) * float(obs_noise_std)

            # Use policy to get action
            if hasattr(policy_net, "act"):
                if stochastic_actions:
                    try:
                        action, _, _ = policy_net.act(obs_t, deterministic=False)
                    except TypeError:
                        action, _, _ = policy_net.act(obs_t)
                else:
                    try:
                        action, _, _ = policy_net.act(obs_t, deterministic=True)
                    except TypeError:
                        action, _, _ = policy_net.act(obs_t)
            else:
                logits = policy_net(obs_t)
                if isinstance(logits, tuple): logits = logits[0]
                if stochastic_actions:
                    action = torch.distributions.Categorical(logits=logits).sample()
                else:
                    action = torch.argmax(logits, dim=-1)

            # Store observation(s)
            if is_vector_env:
                for i in range(obs_t.shape[0]):
                    observations.append(obs_t[i].detach().cpu())
            else:
                observations.append(obs_t.detach().cpu().squeeze(0))
            
            # Step
            if is_vector_env:
                if torch.is_tensor(action):
                    action_arr = action.detach().cpu().view(-1).numpy().astype(int).tolist()
                elif isinstance(action, (list, tuple, np.ndarray)):
                    action_arr = np.asarray(action).reshape(-1).astype(int).tolist()
                else:
                    action_arr = [int(action)]
                if len(action_arr) == 1 and int(getattr(env, "num_envs", 1)) > 1:
                    action_arr = action_arr * int(getattr(env, "num_envs", 1))
                obs, _, terminated, truncated, _ = env.step(action_arr)
                done = bool(np.all(np.asarray(terminated) | np.asarray(truncated)))
            else:
                if torch.is_tensor(action):
                    act = int(action.detach().cpu().view(-1)[0].item())
                else:
                    act = int(action)
                obs, _, terminated, truncated, _ = env.step(act)
                done = terminated or truncated
            step_count += 1
            if max_steps_per_episode and step_count >= int(max_steps_per_episode):
                done = True

    return torch.stack(observations, dim=0)

@torch.no_grad()
def evaluate_snn(
        env,
        agent,
        *,
        sticky_action: bool = True,
):
    """
    Evaluates an SNN agent and tracks spike/latency metrics per step.
    Returns: (reward, length, metrics_dict)
    """
    agent.eval()
    device = get_model_device(agent)

    is_vector_env = hasattr(env, "num_envs")

    obs, _ = env.reset()
    done = False

    total_reward = 0.0
    steps = 0
    prev_action = None

    # Accumulators
    ep_spikes = 0.0
    ep_actor_spikes = 0.0
    ep_critic_spikes = 0.0
    ep_latency = 0.0
    ep_critic_latency = 0.0
    ep_sparsity = 0.0 
    zero_spike_steps = 0

    # Episode-local cumulative baselines (avoid counting pre-existing training totals).
    prev_actor_spikes = float(get_spike_stats_safe(getattr(agent, "actor", None)).get("total_spikes", 0.0))
    prev_critic_spikes = float(get_spike_stats_safe(getattr(agent, "critic", None)).get("total_spikes", 0.0))

    while not done:
        if is_vector_env:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        # Forward pass
        logits, _ = agent(obs_t)
        action = int(torch.argmax(logits, dim=-1).item())

        # Sticky action logic + explicit actor/critic spike channels.
        actor_stats = get_spike_stats_safe(getattr(agent, "actor", None))
        critic_stats = get_spike_stats_safe(getattr(agent, "critic", None))
        actor_cum = float(actor_stats.get("total_spikes", 0.0))
        critic_cum = float(critic_stats.get("total_spikes", 0.0))
        if actor_cum < prev_actor_spikes:
            actor_step_spikes = actor_cum
        else:
            actor_step_spikes = actor_cum - prev_actor_spikes
        if critic_cum < prev_critic_spikes:
            critic_step_spikes = critic_cum
        else:
            critic_step_spikes = critic_cum - prev_critic_spikes
        prev_actor_spikes = actor_cum
        prev_critic_spikes = critic_cum
        spike_count = actor_step_spikes

        if spike_count == 0:
            zero_spike_steps += 1

        if sticky_action and spike_count == 0 and prev_action is not None:
            action = prev_action

        # Collect SNN Metrics
        ep_actor_spikes += actor_step_spikes
        ep_critic_spikes += critic_step_spikes
        ep_spikes += actor_step_spikes + critic_step_spikes
        ep_latency += get_last_latency(agent.actor)
        ep_critic_latency += get_last_latency(agent.critic)
        
        stats = get_spike_stats_safe(agent.actor)
        ep_sparsity += stats.get("sparsity", 0.0)

        # Step
        if is_vector_env:
            obs, reward, terminated, truncated, _ = env.step([action])
            reward = reward[0]
            done = terminated[0] or truncated[0]
        else:
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        total_reward += reward
        steps += 1
        prev_action = action

    metrics = {
        "total_spikes": ep_spikes,
        "actor_spikes": ep_actor_spikes,
        "critic_spikes": ep_critic_spikes,
        "sparsity": ep_sparsity / steps if steps > 0 else 0.0, 
        "mean_latency": ep_latency / steps if steps > 0 else 0.0,
        "critic_mean_latency": ep_critic_latency / steps if steps > 0 else 0.0,
        "avg_spikes_per_step": ep_spikes / steps if steps > 0 else 0.0,
        "avg_actor_spikes_per_step": ep_actor_spikes / steps if steps > 0 else 0.0,
        "avg_critic_spikes_per_step": ep_critic_spikes / steps if steps > 0 else 0.0,
        "no_spike_rate": zero_spike_steps / steps if steps > 0 else 0.0
    }

    return float(total_reward), steps, metrics
