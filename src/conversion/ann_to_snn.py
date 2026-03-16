"""
Conversion utilities for ANN-to-SNN actor conversion in CartPole experiments.
Provides functions to calibrate and convert trained ANN actors to SNN actors.

Standard Reference: 
    Diehl, P. U., et al. "Fast-classifying, high-accuracy spiking deep networks through weight normalization." 
    2015 International Joint Conference on Neural Networks (IJCNN).
"""

import logging
import torch
import torch.nn as nn
from typing import Sequence, Tuple, List, Dict, Optional, Any

from src.models.ann import BackboneNetwork
from src.models.snn_spikeactor import SNNSpikeActor
from src.models.snn_spikevaluecritic import SNNSpikeValueCritic
from src.models.ActorCritic import ActorCritic
from src.conversion.scales import pick_scales
from src.conversion.calibration import estimate_ann_percentiles
from src.training.evaluate import evaluate

# Configure Module Logger
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Layer Extraction
# ---------------------------------------------------------------------
def extract_ann_linear_layers(
        ann: nn.Module,
        *,
        num_hidden: int = 2,
) -> Tuple[Sequence[nn.Linear], nn.Linear]:
    """
    Extracts ANN Linear layers in a canonical way, handling various architectural definitions.
    
    Args:
        ann: The ANN backbone module.
        num_hidden: Expected number of hidden layers.
        
    Returns:
        (hidden_layers, output_layer)
    """
    # Case 1: Modern BackboneNetwork (using nn.ModuleList)
    if hasattr(ann, "layers") and isinstance(ann.layers, nn.ModuleList):
        if len(ann.layers) < num_hidden + 1:
            raise ValueError(f"ANN backbone must contain at least {num_hidden + 1} Linear layers.")
        
        # Slicing: [0..N-1] are hidden, [N] is output
        hidden = [layer for layer in ann.layers[:num_hidden] if isinstance(layer, nn.Linear)]
        out = ann.layers[num_hidden]
        
        if not isinstance(out, nn.Linear):
             raise TypeError(f"Expected output layer to be nn.Linear, got {type(out)}")
             
        return hidden, out

    # Case 2: Legacy / Explicit attributes (fc1, fc2, fc_out)
    # This supports older saved checkpoints or different model definitions
    if all(hasattr(ann, k) for k in ("fc1", "fc2", "fc_out")):
        hidden = [ann.fc1, ann.fc2]
        out = ann.fc_out
        return hidden[:num_hidden], out

    raise TypeError(f"Unsupported ANN backbone structure: {type(ann)}. "
                    "Expected 'layers' (ModuleList) or 'fc1/fc2/fc_out' attributes.")


# ---------------------------------------------------------------------
# Weight Transfer (Data-Driven Normalization)
# ---------------------------------------------------------------------
@torch.no_grad()
def transfer_linear_weights(
        ann_hidden: List[nn.Linear],
        ann_out: nn.Linear,
        snn_hidden: List[nn.Linear],
        snn_out: nn.Linear,
        scales: Dict[str, float],
) -> None:
    """
    Copies and scales ANN Linear weights into SNN Linear layers.
    
    Implements Data-Driven Normalization:
       W_snn = W_ann * (lambda_current / lambda_previous)
       b_snn = b_ann * lambda_current
       
    Args:
        ann_hidden: List of source ANN hidden layers.
        ann_out: Source ANN output layer.
        snn_hidden: List of target SNN hidden layers.
        snn_out: Target SNN output layer.
        scales: Dictionary mapping layer names to scale factors (V_th / percentile).
    """
    if len(ann_hidden) != len(snn_hidden):
        raise ValueError(f"Mismatch in hidden layer count: ANN={len(ann_hidden)}, SNN={len(snn_hidden)}")

    # --- Robust Scale Lookup Helper ---
    def get_scale(layer_idx: int, default: float = 1.0) -> float:
        # Priority list of naming conventions
        candidates = [
            f"backbone.layers.{layer_idx}",  # Wrapped Actor
            f"layers.{layer_idx}",           # Raw Backbone
            f"layer_{layer_idx}",
            f"fc{layer_idx + 1}"
        ]
        
        for k in candidates:
            if k in scales:
                return float(scales[k])
        
        # Fallback for input layer
        if layer_idx == -1 and "input" in scales:
            return float(scales["input"])
            
        return default

    # 1. Input Scale
    # Usually corresponds to the 99th percentile of the observation space.
    # Crucial for models trained with VecNormalize.
    prev_scale = get_scale(-1, default=1.0)
    logger.debug(f"[Conversion] Input Scale: {prev_scale:.4f}")

    # --- Hidden Layers ---
    for i, (ann_l, snn_l) in enumerate(zip(ann_hidden, snn_hidden)):
        curr_scale = get_scale(i)

        logger.debug(f"[Conversion] Layer {i} | Scaling Factor: {curr_scale:.4f} / {prev_scale:.4f}")

        # Scale weights: W_snn = W_ann * (scale_curr / scale_prev)
        weight_factor = curr_scale / prev_scale

        snn_l.weight.copy_(ann_l.weight * weight_factor)
        
        if ann_l.bias is not None:
            # Bias maps directly to current potential: b_snn = b_ann * scale_curr
            snn_l.bias.copy_(ann_l.bias * curr_scale)

        # Update previous scale for next layer
        prev_scale = curr_scale

    # --- Output Layer ---
    # Try finding explicit output scale, otherwise default to 1.0 (logits)
    out_candidates = ["policy_head", "value_head", "output", "fc_out", "out"]
    out_scale = 1.0
    for k in out_candidates:
        if k in scales:
            out_scale = float(scales[k])
            break

    logger.debug(f"[Conversion] Output Layer | Scaling Factor: {out_scale:.4f} / {prev_scale:.4f}")

    weight_factor = out_scale / prev_scale
    snn_out.weight.copy_(ann_out.weight * weight_factor)
    
    if ann_out.bias is not None:
        snn_out.bias.copy_(ann_out.bias * out_scale)


# ---------------------------------------------------------------------
# Conversion Orchestration
# ---------------------------------------------------------------------
@torch.no_grad()
def snn_from_ann(
        ann: BackboneNetwork,
        *,
        actor_T: int = 32,
        beta: float = 0.95,
        V_th: float = 1.0,
        scales: Optional[Dict[str, float]] = None,
        poisson_encode: bool = False,
        rate_scale: float = 1.0,
        logit_temp: float = 1.0,
        center_logits: bool = False,
) -> SNNSpikeActor:
    """
    Constructs and populates an SNNSpikeActor from an ANN Backbone.
    """
    hidden, out = extract_ann_linear_layers(ann, num_hidden=2)
    scales = scales or {}

    # Create SNN Actor Architecture
    snn_actor = SNNSpikeActor(
        in_dim=hidden[0].in_features,
        hid_dim=hidden[0].out_features,
        out_dim=out.out_features,
        beta=beta,
        V_th=V_th,
        T=actor_T,
        poisson_encode=poisson_encode,
        rate_scale=rate_scale,
        logit_temp=logit_temp,
        center_logits=center_logits,
    )

    # Transfer Weights
    transfer_linear_weights(
        ann_hidden=list(hidden),
        ann_out=out,
        snn_hidden=[
            snn_actor.block1.linear,
            snn_actor.block2.linear,
        ],
        snn_out=snn_actor.block_out.linear,
        scales=scales,
    )

    return snn_actor


@torch.no_grad()
def snn_critic_from_ann(
        ann: BackboneNetwork,
        *,
        T: int = 32,
        beta: float = 0.95,
        V_th: float = 1.0,
        scales: Optional[Dict[str, float]] = None,
        poisson_encode: bool = False,
        rate_scale: float = 1.0,
) -> SNNSpikeValueCritic:
    """
    Constructs and populates an SNNSpikeValueCritic from an ANN Backbone.
    """
    hidden, out = extract_ann_linear_layers(ann, num_hidden=2)
    if out.out_features != 1:
        raise ValueError(f"ANN critic must have out_features=1, got {out.out_features}")

    scales = scales or {}

    snn_critic = SNNSpikeValueCritic(
        in_dim=hidden[0].in_features,
        hid_dim=hidden[0].out_features,
        beta=beta,
        V_th=V_th,
        T=T,
        poisson_encode=poisson_encode,
        rate_scale=rate_scale,
    )

    transfer_linear_weights(
        ann_hidden=list(hidden),
        ann_out=out,
        snn_hidden=[
            snn_critic.block1.linear,
            snn_critic.block2.linear,
        ],
        snn_out=snn_critic.block_out.linear,
        scales=scales,
    )

    return snn_critic


# ---------------------------------------------------------------------
# Full Pipeline Wrapper
# ---------------------------------------------------------------------
def convert_actor_to_snn(
        ann_agent: Any,
        env: Any,
        *,
        T: int = 100,
        beta: float = 0.95,
        V_th: float = 1.0,
        poisson_encode: bool = False,
        rate_scale: float = 1.0,
        logit_temp: float = 1.0,
        center_logits: bool = False,
        calibration_episodes: int = 32
) -> Tuple[ActorCritic, float, Dict[str, float]]:
    """
    High-level utility to Calibrate and Convert an ANN Actor to SNN.
    
    Steps:
    1. Collects calibration observations from the environment.
    2. Estimates activation percentiles.
    3. Calculates optimal scaling factors.
    4. Converts the model and runs a zero-shot evaluation.
    
    Returns:
        (Converted SNN Agent, Zero-Shot Reward, Scales Dict)
    """
    device = next(ann_agent.actor.parameters()).device

    # --- 1. Calibration Data Collection ---
    logger.info(f"Collecting calibration data ({calibration_episodes} episodes)...")
    obs_buf = []
    for _ in range(calibration_episodes):
        out = env.reset()
        obs = out[0] if isinstance(out, (tuple, list)) else out
        obs_buf.append(torch.as_tensor(obs, dtype=torch.float32))
    
    if not obs_buf:
        raise ValueError("Environment reset returned no observations for calibration.")
        
    obs_tensor = torch.stack(obs_buf).to(device)

    # --- 2. Estimate Percentiles ---
    if not isinstance(ann_agent.actor, nn.Module):
        raise TypeError("ann_agent.actor must be an nn.Module for calibration")

    percentiles = estimate_ann_percentiles(ann_agent.actor, obs_tensor, percentile=0.999)

    # Add input percentile (Crucial for robust input scaling)
    input_p99 = torch.quantile(obs_tensor.abs(), 0.999).item()
    percentiles["input"] = input_p99

    # Calculate scales WITHOUT canonicalization (preserving 'layers.0' keys)
    scales = pick_scales(percentiles, V_th=V_th)

    logger.info(f"Calibration complete. Input P99: {input_p99:.4f}")
    logger.debug(f"Calculated Scales: {scales}")

    # --- 3. Conversion ---
    # Unwrap backbone if necessary (handling Actor wrapper)
    if hasattr(ann_agent.actor, "backbone"):
        ann_backbone = ann_agent.actor.backbone
    else:
        ann_backbone = ann_agent.actor

    snn_actor = snn_from_ann(
        ann_backbone,
        actor_T=T,
        beta=beta,
        V_th=V_th,
        scales=scales,
        poisson_encode=poisson_encode,
        rate_scale=rate_scale,
        logit_temp=logit_temp,
        center_logits=center_logits,
    )

    # Re-wrap in ActorCritic (Hybrid: SNN Actor + ANN Critic)
    # This maintains the interface expected by evaluation/training loops
    agent_snn = ActorCritic(snn_actor, ann_agent.critic).to(device)

    # --- 4. Zero-shot Evaluation ---
    logger.info("Running zero-shot evaluation...")
    zero_shot_reward = evaluate(env, agent_snn)
    logger.info(f"Zero-shot Reward: {zero_shot_reward:.2f}")

    return agent_snn, zero_shot_reward, scales




