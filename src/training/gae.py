"""Rollout collection and Generalised Advantage Estimation (GAE).

Implements sticky-action support for zero-spike SNN timesteps.
"""
import torch
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
from typing import Tuple, Dict

# ----------------- Merged Helper Functions ----------------- #

def agent_forward(agent, obs: torch.Tensor):
    """Modular agent forward with automatic squeezing of value."""
    out = agent(obs)
    if isinstance(out, (tuple, list)):
        logits = out[0]
        value = out[1] if len(out) > 1 else None
    else:
        logits = out
        value = None

    if value is None:
        raise RuntimeError("Agent must return value estimates for PPO.")

    return logits, value.squeeze(-1)

def apply_sticky_action(action, prev_action, actor, enabled, step):
    """Vectorized sticky action logic for SNNs."""
    if not enabled or step == 0 or not hasattr(actor, "last_spike_count"):
        return action
    try:
        sc = actor.last_spike_count()
        if torch.is_tensor(sc):
            # If spikes are 0, use the action from the previous timestep
            zero_spike = (sc == 0)
            action[zero_spike] = prev_action[zero_spike]
    except Exception:
        pass
    return action

def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """Vectorized GAE computation."""
    T = rewards.size(0)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns

# ----------------- Main Merged Rollout Function ----------------- #

@torch.no_grad()
def collect_rollout(
    env,
    agent,
    *,
    n_steps: int,
    gamma: float = 0.99,
    lam: float = 0.95,
    sticky_action: bool = False,
    no_action_cfg: Dict = None,
) -> Tuple:
    device = next(agent.parameters()).device
    n_envs = env.num_envs
    agent.train() # SNNs might need train mode for membrane potential tracking
    
    # Pre-allocate Buffers (Optimization from snippet 2)
    obs_buf = torch.zeros((n_steps, n_envs) + env.single_observation_space.shape, device=device)
    act_buf = torch.zeros((n_steps, n_envs), dtype=torch.long, device=device)
    logp_buf = torch.zeros((n_steps, n_envs), device=device)
    rew_buf = torch.zeros((n_steps, n_envs), device=device)
    done_buf = torch.zeros((n_steps, n_envs), device=device)
    val_buf = torch.zeros((n_steps, n_envs), device=device)

    # State Tracking
    obs, _ = env.reset() if not hasattr(env, "_last_obs") else (env._last_obs, {})
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    prev_action = torch.zeros(n_envs, dtype=torch.long, device=device)
    
    # Reward Tracking (Robust logic from snippet 1)
    raw_ep_rewards = []
    done_count = 0
    episode_returns = np.zeros(n_envs, dtype=np.float32)
    raw_episode_returns = np.zeros(n_envs, dtype=np.float32)

    for t in range(n_steps):
        obs_buf[t] = obs_t
        
        # 1. Action Selection
        logits, value = agent_forward(agent, obs_t)
        dist = D.Categorical(logits=logits)
        action = dist.sample()
        
        # 2. SNN Sticky Logic
        action = apply_sticky_action(action, prev_action, agent.actor, sticky_action, t)
        logp = dist.log_prob(action)

        # 3. Env Step
        next_obs, reward, terminated, truncated, infos = env.step(action.cpu().numpy())
        done = terminated | truncated
        episode_returns += np.asarray(reward, dtype=np.float32)
        raw_reward = infos.get("raw_reward", reward)
        raw_episode_returns += np.asarray(raw_reward, dtype=np.float32)

        # 4. Storage
        act_buf[t] = action
        logp_buf[t] = logp
        rew_buf[t] = torch.as_tensor(reward, device=device)
        done_buf[t] = torch.as_tensor(done, dtype=torch.float32, device=device)
        val_buf[t] = value

        # 5. Robust Info Extraction (Handles Gymnasium VectorEnv structures)
        used_info = np.zeros(n_envs, dtype=bool)
        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                if done[i] and info and "episode" in info:
                    raw_ep_rewards.append(float(raw_episode_returns[i]))
                    used_info[i] = True
        elif "episode" in infos:
            ep = infos["episode"]
            if isinstance(ep, (list, np.ndarray)):
                for i, item in enumerate(ep):
                    if done[i] and isinstance(item, dict) and "r" in item:
                        raw_ep_rewards.append(float(raw_episode_returns[i]))
                        used_info[i] = True
            elif isinstance(ep, dict) and "r" in ep:
                raw_ep_rewards.append(float(raw_episode_returns[done][0] if np.any(done) else 0.0))
                used_info[done] = True

        # Fallback if no RecordEpisodeStatistics wrapper is present
        if np.any(done):
            done_count += int(np.sum(done))
            for i in np.where(done)[0]:
                if not used_info[i]:
                    raw_ep_rewards.append(float(raw_episode_returns[i]))
            episode_returns[done] = 0.0
            raw_episode_returns[done] = 0.0

        obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
        prev_action = action

    # Store for next rollout
    env._last_obs = next_obs

    # 6. GAE Calculation
    _, last_val = agent_forward(agent, obs_t)
    advantages, returns = compute_gae(rew_buf, val_buf, done_buf, last_val, gamma, lam)

    # 7. Flatten and Normalize
    def flat(x): return x.flatten(0, 1)

    b_adv = flat(advantages)
    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

    raw_mean = np.mean(raw_ep_rewards) if raw_ep_rewards else np.nan

    return (
        flat(obs_buf), 
        flat(act_buf), 
        flat(logp_buf), 
        b_adv, 
        flat(returns), 
        flat(val_buf), 
        raw_mean, 
        len(raw_ep_rewards), 
        done_count
    )
