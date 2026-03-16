"""
Scaling factor computation for ANN-to-SNN conversion.

This module implements "Data-Driven Normalization" logic. It maps the continuous
activation magnitude of an ANN to the discrete firing threshold of an SNN.

Heuristic:
    W_snn = W_ann * scale_factor
    scale_factor = (safety_margin * V_th) / activation_percentile

Reference:
    Diehl, P. U., et al. "Fast-classifying, high-accuracy spiking deep networks 
    through weight normalization." IJCNN 2015.
"""
from __future__ import annotations
import logging
from typing import Dict, Optional, Any

# Configure module logger
logger = logging.getLogger(__name__)

# Default Constants
DEFAULT_VTH = 1.0
DEFAULT_SAFETY = 0.99
MIN_SCALE_FLOOR = 1e-9


def compute_base_scale(
    percentile: float,
    *,
    V_th: float,
    safety: float,
    min_scale: float,
) -> float:
    """
    Computes the base scaling factor from a single activation statistic.

    Args:
        percentile: The max (or p99) activation value observed in the ANN.
        V_th: The target firing threshold of the SNN neuron.
        safety: A fractional margin (<1.0) to prevent over-saturation.
        min_scale: A numerical floor to prevent division by zero or underflow.

    Returns:
        float: The multiplicative scale factor.
    """
    # Handle dead neurons (zero activation) gracefully
    if percentile <= 0.0:
        return 1.0

    # Logic: If max_act is 10.0 and V_th is 1.0, we must scale weights by 0.1
    # so that the input 10.0 becomes 1.0 (threshold).
    scale = (safety * V_th) / percentile
    
    return max(min_scale, float(scale))


def pick_scales(
    percentiles: Dict[str, float],
    *,
    V_th: float = DEFAULT_VTH,
    safety: float = DEFAULT_SAFETY,
    min_scale: float = MIN_SCALE_FLOOR,
    per_layer_factors: Optional[Dict[str, float]] = None,
    strict: bool = False,
) -> Dict[str, float]:
    """
    Converts activation statistics into layer-wise weight scaling factors.

    Args:
        percentiles: Dictionary mapping layer names to their activation statistic (e.g., 99th percentile).
        V_th: The firing threshold of the target SNN (default: 1.0).
        safety: Safety margin to reduce firing rates (default: 0.99).
        min_scale: Numerical stability floor.
        per_layer_factors: Optional manual multipliers for specific layers (e.g., {'fc1': 0.5}).
        strict: If True, raises ValueError for missing or invalid percentiles.

    Returns:
        Dict[str, float]: A dictionary of scaling factors ready for weight transformation.
    """
    per_layer_factors = per_layer_factors or {}
    scales: Dict[str, float] = {}

    logger.info(f"Computing scales with V_th={V_th}, safety={safety}")

    for name, p in percentiles.items():
        # Validate data quality
        if p is None or p <= 0.0:
            if strict:
                raise ValueError(f"Invalid percentile for layer '{name}': {p}")
            
            logger.warning(f"Layer '{name}' has invalid activation ({p}). Defaulting scale to 1.0.")
            base = 1.0
        else:
            base = compute_base_scale(
                p,
                V_th=V_th,
                safety=safety,
                min_scale=min_scale,
            )

        # Apply manual per-layer tweaks (e.g., to dampen early layers)
        factor = float(per_layer_factors.get(name, 1.0))
        final_scale = max(min_scale, base * factor)
        
        scales[name] = final_scale
        logger.debug(f"Layer '{name}': p={p:.4f} -> scale={final_scale:.4f} (factor={factor})")

    return scales


def default_layer_factors() -> Dict[str, float]:
    """
    Returns default heuristic multipliers for standard MLP topologies.

    These factors dampen early layers to prevent "firing rate explosion" 
    in deep networks, preserving dynamic range for later layers.
    """
    return {
        "fc1": 0.3,
        "fc2": 0.3,
        # Output layer is usually left at 1.0 to preserve logits magnitude
    }