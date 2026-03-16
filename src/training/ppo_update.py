import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.distributions as D
from typing import Dict, Any, Optional

def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Standard score normalization."""
    return (x - x.mean()) / (x.std(unbiased=False) + eps)

def ppo_policy_loss(logp: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor, clip_eps: float):
    """Calculates PPO Clipped Surrogate Loss."""
    ratio = torch.exp(logp - logp_old)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    return -torch.min(unclipped, clipped).mean(), ratio

def spike_regulariser(agent) -> float:
    """Aggregates regularization terms from SNN modules."""
    reg = 0.0
    for module in (agent.actor, agent.critic):
        if hasattr(module, "regulariser"):
            r = module.regulariser()
            if isinstance(r, torch.Tensor):
                r = r.mean()
            reg += r
    return reg


def _grad_l2_norm(module: torch.nn.Module) -> float:
    sq_sum = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq_sum += float(torch.sum(g * g).item())
    return float(np.sqrt(max(sq_sum, 0.0)))


def _param_grad_l2_norm(module: torch.nn.Module, param_name: str) -> float:
    p = dict(module.named_parameters()).get(param_name)
    if p is None or p.grad is None:
        return 0.0
    g = p.grad.detach()
    return float(torch.sqrt(torch.sum(g * g)).item())

def update_policy(
    agent,
    states: torch.Tensor,
    actions: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    values_old: torch.Tensor,
    *,
    n_epochs: int = 10,
    batch_size: int = 256,
    shuffle_minibatches: bool = True,
    clip_eps: float = 0.2,
    clip_range_vf: Optional[float] = None,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    lambda_spike: float = 1e-3,
    target_kl: Optional[float] = None,
    kl_coef: float = 0.0,
    logger=None,
    scale_returns_to_value: bool = False,
    normalize_value_targets: bool = False,
    **kwargs, 
) -> Dict[str, float]:
    """
    Performs PPO Update Epochs.
    Includes Value Function Clipping and Spike Regularization.
    """
    device = next(agent.parameters()).device
    agent.train()

    tensors = [states, actions, logp_old, advantages, returns, values_old]
    sizes = [t.size(0) for t in tensors]
    min_size = min(sizes)
    max_size = max(sizes)
    if min_size == 0:
        raise ValueError("Empty rollout buffer: one or more tensors have size 0.")
    if min_size != max_size:
        if logger is not None:
            try:
                logger.record("debug/ppo_size_mismatch", float(max_size - min_size))
            except Exception:
                pass
        states = states[:min_size]
        actions = actions[:min_size]
        logp_old = logp_old[:min_size]
        advantages = advantages[:min_size]
        returns = returns[:min_size]
        values_old = values_old[:min_size]

    # Critical for training stability when Value predictions are large (e.g. 500)
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset = TensorDataset(
        states.to(device),
        actions.to(device),
        logp_old.to(device),
        advantages.to(device),
        returns.to(device),
        values_old.to(device)
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_minibatches)

    # Metric Accumulators
    metrics = {
        "policy_loss": [], "value_loss": [], "entropy": [],
        "kl": [], "clip_frac": [], "kl_penalty": [],
        "ratio_mean": [], "ratio_std": [],
        "grad_actor_total": [], "grad_critic_total": [],
        "grad_critic_block1_linear_weight": [],
        "grad_critic_block2_linear_weight": [],
        "grad_critic_block_out_linear_weight": [],
        "grad_critic_value_mapper_weight": [],
        "explained_variance": 0.0,
    }
    
    # For Explained Variance
    y_pred_all = []
    y_true_all = []

    updates = 0
    epochs_ran = 0
    early_stop_kl = False

    returns_detached = returns.detach()
    values_old_detached = values_old.detach()
    returns_stats = {
        "returns_mean": float(returns_detached.mean().item()),
        "returns_std": float(returns_detached.std(unbiased=False).item()),
        "returns_min": float(returns_detached.min().item()),
        "returns_max": float(returns_detached.max().item()),
        "values_old_mean": float(values_old_detached.mean().item()),
        "values_old_std": float(values_old_detached.std(unbiased=False).item()),
        "values_old_min": float(values_old_detached.min().item()),
        "values_old_max": float(values_old_detached.max().item()),
    }

    for epoch in range(n_epochs):
        epochs_ran += 1
        for obs, act, lp_old, adv, ret, v_old in loader:
            logits, value = agent(obs)
            value = value.squeeze(-1)

            # Collect predictions for EV calc (first epoch only for speed)
            if epoch == 0:
                y_pred_all.append(value.detach().cpu().numpy())
                y_true_all.append(ret.detach().cpu().numpy())

            # Policy Loss
            dist = D.Categorical(logits=logits)
            logp = dist.log_prob(act)
            entropy = dist.entropy().mean()
            
            pol_loss, ratio = ppo_policy_loss(logp, lp_old, adv, clip_eps)

            # Optionally scale returns to match value prediction scale
            if scale_returns_to_value:
                r_mean = ret.mean()
                r_std = ret.std(unbiased=False)
                v_mean = v_old.mean()
                v_std = v_old.std(unbiased=False)
                if r_std > 1e-8:
                    ret = (ret - r_mean) / r_std
                else:
                    ret = ret - r_mean
                if v_std > 1e-8:
                    ret = ret * v_std + v_mean
                else:
                    ret = ret + v_mean

            # Value Loss (with optional clipping)
            if normalize_value_targets:
                ret_mean = ret.mean()
                ret_std = ret.std(unbiased=False) + 1e-8
                ret_for_loss = (ret - ret_mean) / ret_std
                value_for_loss = (value - ret_mean) / ret_std
                v_old_for_loss = (v_old - ret_mean) / ret_std
            else:
                ret_for_loss = ret
                value_for_loss = value
                v_old_for_loss = v_old

            if clip_range_vf is None:
                val_loss = F.mse_loss(value_for_loss, ret_for_loss)
            else:
                v_clipped = v_old_for_loss + torch.clamp(value_for_loss - v_old_for_loss, -clip_range_vf, clip_range_vf)
                vf_losses1 = (value_for_loss - ret_for_loss) ** 2
                vf_losses2 = (v_clipped - ret_for_loss) ** 2
                val_loss = 0.5 * torch.mean(torch.max(vf_losses1, vf_losses2))
            
            # SNN Regularization
            reg = spike_regulariser(agent)

            approx_kl = torch.mean(lp_old - logp)
            kl_pen = torch.clamp(approx_kl - float(target_kl), min=0.0) if (target_kl and kl_coef > 0.0) else torch.zeros((), device=approx_kl.device)

            total_loss = (
                pol_loss
                + value_coef * val_loss
                - entropy_coef * entropy
                + lambda_spike * reg
                + kl_coef * kl_pen
            )

            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            # Logging Stats
            with torch.no_grad():
                clip_frac = (torch.abs(ratio - 1) > clip_eps).float().mean()

            metrics["policy_loss"].append(pol_loss.item())
            metrics["value_loss"].append(val_loss.item())
            metrics["entropy"].append(entropy.item())
            metrics["kl"].append(approx_kl.item())
            metrics["clip_frac"].append(clip_frac.item())
            metrics["kl_penalty"].append(float(kl_pen.item()))
            metrics["ratio_mean"].append(float(ratio.mean().item()))
            metrics["ratio_std"].append(float(ratio.std(unbiased=False).item()))

            metrics["grad_actor_total"].append(_grad_l2_norm(agent.actor))
            metrics["grad_critic_total"].append(_grad_l2_norm(agent.critic))
            metrics["grad_critic_block1_linear_weight"].append(
                _param_grad_l2_norm(agent.critic, "block1.linear.weight")
            )
            metrics["grad_critic_block2_linear_weight"].append(
                _param_grad_l2_norm(agent.critic, "block2.linear.weight")
            )
            metrics["grad_critic_block_out_linear_weight"].append(
                _param_grad_l2_norm(agent.critic, "block_out.linear.weight")
            )
            metrics["grad_critic_value_mapper_weight"].append(
                _param_grad_l2_norm(agent.critic, "value_mapper.weight")
            )

            updates += 1

        # Early Stopping (KL Divergence)
        if target_kl and np.mean(metrics["kl"][-len(loader):]) > 1.5 * target_kl:
            early_stop_kl = True
            break

    # Finalize Metrics
    if y_pred_all:
        y_pred_np = np.concatenate(y_pred_all)
        y_true_np = np.concatenate(y_true_all)
        var_y = np.var(y_true_np)
        if np.isclose(var_y, 0):
            metrics["explained_variance"] = np.nan
        else:
            metrics["explained_variance"] = 1 - np.var(y_true_np - y_pred_np) / var_y

    final_metrics = {
        "policy_loss": float(np.mean(metrics["policy_loss"])),
        "value_loss": float(np.mean(metrics["value_loss"])),
        "entropy": float(np.mean(metrics["entropy"])),
        "approx_kl": float(np.mean(metrics["kl"])),
        "clip_fraction": float(np.mean(metrics["clip_frac"])),
        "ratio_mean": float(np.mean(metrics["ratio_mean"])),
        "ratio_std": float(np.mean(metrics["ratio_std"])),
        "kl_penalty": float(np.mean(metrics["kl_penalty"])) if metrics["kl_penalty"] else 0.0,
        "grad_actor_total": float(np.mean(metrics["grad_actor_total"])) if metrics["grad_actor_total"] else 0.0,
        "grad_critic_total": float(np.mean(metrics["grad_critic_total"])) if metrics["grad_critic_total"] else 0.0,
        "grad_critic_block1_linear_weight": float(np.mean(metrics["grad_critic_block1_linear_weight"])) if metrics["grad_critic_block1_linear_weight"] else 0.0,
        "grad_critic_block2_linear_weight": float(np.mean(metrics["grad_critic_block2_linear_weight"])) if metrics["grad_critic_block2_linear_weight"] else 0.0,
        "grad_critic_block_out_linear_weight": float(np.mean(metrics["grad_critic_block_out_linear_weight"])) if metrics["grad_critic_block_out_linear_weight"] else 0.0,
        "grad_critic_value_mapper_weight": float(np.mean(metrics["grad_critic_value_mapper_weight"])) if metrics["grad_critic_value_mapper_weight"] else 0.0,
        "explained_variance": float(metrics["explained_variance"]),
        "ppo_epochs_ran": float(epochs_ran),
        "ppo_early_stop_kl": float(1.0 if early_stop_kl else 0.0),
        "n_updates": updates
    }
    final_metrics.update(returns_stats)

    if y_pred_all:
        final_metrics["values_pred_mean"] = float(np.mean(y_pred_np))
        final_metrics["values_pred_std"] = float(np.std(y_pred_np))
        final_metrics["values_pred_min"] = float(np.min(y_pred_np))
        final_metrics["values_pred_max"] = float(np.max(y_pred_np))

    if logger:
        for k, v in final_metrics.items():
            logger.record(f"train/{k}", v)
        logger.increment_updates()

    return final_metrics
