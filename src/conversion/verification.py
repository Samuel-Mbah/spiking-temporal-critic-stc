"""
Verification utilities for ANN-to-SNN conversion.

Provides metrics to quantify the fidelity of the conversion process:
1. Argmax Agreement: Do the ANN and SNN choose the same action?
2. Centered MSE: Do the relative activation magnitudes match?
3. Pearson Correlation: Are the output dynamics linearly correlated?
"""
from __future__ import annotations
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Union

# Configure module logger
logger = logging.getLogger(__name__)


def standardize_tensor(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """
    Standardizes a tensor (Z-score normalization) along a specific dimension.
    
    Used to compare the *dynamics* of ANN vs SNN outputs, ignoring 
    constant scaling factors (like rate_scale) or bias shifts.
    
    Args:
        x: Input tensor.
        dim: Dimension to normalize over (usually -1 for logits).
        eps: Epsilon for numerical stability.
    """
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    return (x - mean) / (std + eps)


@torch.no_grad()
def verify_actor_conversion(
    ann_actor: nn.Module,
    snn_actor: nn.Module,
    obs_tensor: torch.Tensor,
) -> Dict[str, Any]:
    """
    Verifies ANN->SNN actor fidelity.

    Metrics:
        - argmax_agreement: Frequency of matching actions.
        - mse_centered: MSE between Z-scored logits (scale-invariant).
        - mean_pearson: Average Pearson correlation across action dimensions.

    Args:
        ann_actor: The source Artificial Neural Network.
        snn_actor: The target Spiking Neural Network.
        obs_tensor: Batch of observations [Batch, ObsDim].

    Returns:
        Dictionary of scalar metrics.
    """
    ann_actor.eval()
    snn_actor.eval()

    device = next(ann_actor.parameters()).device
    obs = obs_tensor.to(device)

    # 1. Forward Pass
    ann_logits = ann_actor(obs)
    snn_logits = snn_actor(obs)

    # 2. Standardized MSE (Focus on relative dynamics)
    ann_norm = standardize_tensor(ann_logits, dim=-1)
    snn_norm = standardize_tensor(snn_logits, dim=-1)
    
    mse = F.mse_loss(snn_norm, ann_norm).item()

    # 3. Argmax Agreement (Functional correctness)
    ann_actions = ann_logits.argmax(dim=-1)
    snn_actions = snn_logits.argmax(dim=-1)
    agreement = (ann_actions == snn_actions).float().mean().item()

    # 4. Pearson Correlation (Linearity)
    ann_np = ann_norm.cpu().numpy()
    snn_np = snn_norm.cpu().numpy()

    correlations = []
    num_actions = ann_np.shape[1]

    for a in range(num_actions):
        # Handle constant outputs (std ~ 0) which cause NaN correlation
        std_ann = np.std(ann_np[:, a])
        std_snn = np.std(snn_np[:, a])
        
        if std_ann < 1e-8 or std_snn < 1e-8:
            # If both are constant (likely 0), they match perfectly.
            # If only one is constant, they are uncorrelated.
            corr = 1.0 if (std_ann < 1e-8 and std_snn < 1e-8) else 0.0
        else:
            corr = float(np.corrcoef(ann_np[:, a], snn_np[:, a])[0, 1])
            
        correlations.append(corr)

    return {
        "mse_centered": mse,
        "argmax_agreement": agreement,
        "mean_pearson": float(np.mean(correlations)),
        "per_action_pearson": correlations,
        "num_samples": obs.shape[0],
    }


@torch.no_grad()
def verify_critic_conversion(
    ann_critic: nn.Module,
    snn_critic: nn.Module,
    obs_tensor: torch.Tensor,
) -> Dict[str, Any]:
    """
    Verifies ANN->SNN critic fidelity.

    Since critics output a single scalar (Value), we compare the correlation
    of the value predictions across the batch.

    Args:
        ann_critic: Source ANN Value function.
        snn_critic: Target SNN Value function.
        obs_tensor: Batch of observations.

    Returns:
        Dictionary containing MSE and Pearson correlation.
    """
    ann_critic.eval()
    snn_critic.eval()

    device = next(ann_critic.parameters()).device
    obs = obs_tensor.to(device)

    # 1. Forward Pass (Values are [Batch, 1])
    ann_val = ann_critic(obs).squeeze(-1)
    snn_val = snn_critic(obs).squeeze(-1)

    # 2. Standardized MSE (Batch-wise normalization)
    # Note: For critic, we normalize across the BATCH dimension (dim=0)
    # because the output dimension is 1.
    ann_norm = standardize_tensor(ann_val, dim=0)
    snn_norm = standardize_tensor(snn_val, dim=0)

    mse = F.mse_loss(snn_norm, ann_norm).item()

    # 3. Pearson Correlation
    ann_np = ann_val.cpu().numpy()
    snn_np = snn_val.cpu().numpy()

    std_ann = np.std(ann_np)
    std_snn = np.std(snn_np)

    if std_ann < 1e-8 or std_snn < 1e-8:
        corr = 1.0 if (std_ann < 1e-8 and std_snn < 1e-8) else 0.0
    else:
        corr = float(np.corrcoef(ann_np, snn_np)[0, 1])

    return {
        "mse_centered": mse,
        "mean_pearson": corr,
        "num_samples": obs.shape[0]
    }