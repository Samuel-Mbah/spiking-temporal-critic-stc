"""
Calibration utilities for ANN-to-SNN conversion.

Provides functions to:
1. Collect observation statistics from the environment.
2. Estimate layer-wise activation statistics (percentiles) from an ANN
   to determine optimal V_th scaling factors.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Callable, Union

# Configure module-level logger
logger = logging.getLogger(__name__)

def collect_calibration_observations(
    env,
    num_samples: int = 128,
    preprocess_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Collects a batch of initial observations via repeated environment resets.
    
    Args:
        env: Gymnasium environment (can be VectorEnv).
        num_samples: Number of observations to collect.
        preprocess_fn: Optional function to process observations (e.g., normalization).

    Returns:
        Tensor of shape (num_samples, *obs_shape).
    """
    obs_list = []
    samples_collected = 0

    # Handle VectorEnvs (which return batches) vs Single Envs
    is_vector_env = hasattr(env, "num_envs")
    
    while samples_collected < num_samples:
        # Reset environment to get initial states
        reset_out = env.reset()
        
        # Unpack gym (obs, info) tuple if present
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        
        # Convert to tensor
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32)

        if preprocess_fn is not None:
            obs = preprocess_fn(obs)

        # Handle Batch Dimension
        if is_vector_env:
            # obs shape: [n_envs, obs_dim]
            # Flatten batch into list
            for i in range(obs.shape[0]):
                obs_list.append(obs[i])
                samples_collected += 1
                if samples_collected >= num_samples:
                    break
        else:
            # obs shape: [obs_dim]
            obs_list.append(obs)
            samples_collected += 1

    # Stack results: [num_samples, obs_dim]
    return torch.stack(obs_list[:num_samples], dim=0)


class ActivationCollector:
    """
    Forward hook handler to collect absolute pre-activation outputs.
    """

    def __init__(self):
        self.buffers: Dict[str, List[torch.Tensor]] = {}

    def hook(self, name: str):
        """Creates a hook function for a specific layer name."""
        def _hook_fn(module, inputs, output):
            # Ensure we only track tensors (skip tuples/None)
            if not isinstance(output, torch.Tensor):
                return
            
            # Record absolute values flattened
            # Detach to save memory (no gradients needed for calibration)
            self.buffers.setdefault(name, []).append(
                output.detach().abs().flatten()
            )
        return _hook_fn

    def get_flat(self, name: str) -> Optional[torch.Tensor]:
        """Concatenates all collected buffers for a layer into one flat tensor."""
        vals = self.buffers.get(name)
        if not vals:
            return None
        return torch.cat(vals, dim=0)


@torch.no_grad()
def estimate_ann_percentiles(
    model: nn.Module,
    obs_tensor: torch.Tensor,
    *,
    percentile: float = 99.9,
    device: Optional[torch.device] = None,
    linear_only: bool = True,
) -> Dict[str, float]:
    """
    Estimates activation percentiles for ANN layers to guide SNN thresholds.

    Args:
        model: The ANN model (Actor or Critic).
        obs_tensor: Calibration data [Batch, ObsDim].
        percentile: Percentile to estimate (e.g., 99.9 for robust max).
        device: Calculation device. Defaults to model's device.
        linear_only: If True, only hooks nn.Linear layers.

    Returns:
        Dictionary mapping layer names to their activation percentile value.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Prepare model
    model = model.to(device)
    was_training = model.training
    model.eval()

    obs = obs_tensor.to(device)
    collector = ActivationCollector()
    hooks = []

    try:
        # 1. Register Hooks
        for name, module in model.named_modules():
            if linear_only and not isinstance(module, nn.Linear):
                continue
            
            # Skip container modules if they don't do computation themselves
            if linear_only and len(list(module.children())) > 0:
                continue

            h = module.register_forward_hook(collector.hook(name))
            hooks.append(h)
            logger.debug(f"Registered calibration hook on: {name}")

        # 2. Forward Pass (Inference)
        _ = model(obs)

    finally:
        # 3. Cleanup Hooks (Guaranteed)
        for h in hooks:
            h.remove()
        model.train(was_training)

    # 4. Compute Statistics
    percentiles: Dict[str, float] = {}
    
    for name in collector.buffers:
        flat_activations = collector.get_flat(name)
        
        if flat_activations is None or flat_activations.numel() == 0:
            logger.warning(f"Layer {name} produced no activations during calibration.")
            percentiles[name] = 1.0 # Default fallback
            continue

        # Compute Quantile
        # Note: quantile requires float32/64
        val = torch.quantile(flat_activations.float(), percentile / 100.0)
        percentiles[name] = float(val.item())

    return percentiles