
"""
Utilities for interacting with Weights & Biases (WandB).

Designed to be:
- Non-blocking: Failure to log should not crash the training loop.
- Optional: Code runs fine even if `wandb` is not installed.
- Robust: Handles file existence checks and serialization automatically.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Literal
import wandb

# Configure Logger
logger = logging.getLogger(__name__)


def is_active() -> bool:
    """Returns True if WandB is installed and a run is currently active."""
    return wandb is not None and getattr(wandb, "run", None) is not None


def init_wandb(
    config: Dict[str, Any],
    project: str,
    entity: Optional[str] = None,
    name: Optional[str] = None,
    group: Optional[str] = None,
    tags: Optional[List[str]] = None,
    mode: Literal["online", "offline", "disabled", "shared"] = "online",
) -> None:
    """
    Safely initializes a WandB run.
    
    Args:
        config: Experiment configuration dict.
        project: W&B project name.
        entity: W&B user/org name.
        name: Display name for this run.
        group: Group name for organizing repeats/sweeps.
        tags: List of tags for filtering.
        mode: 'online', 'offline', or 'disabled'.
    """
    if wandb is None:
        logger.warning("WandB is not installed. Metrics will not be synced.")
        return

    # 1. Sanitize Config (Ensure JSON serializability)
    # Converts types like <class '...'> or Path() objects to strings to prevent crashes.
    clean_config = {}
    for k, v in config.items():
        if isinstance(v, (int, float, str, bool, list, dict, type(None))):
            clean_config[k] = v
        else:
            clean_config[k] = str(v)

    # 2. Initialize
    try:
        wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            tags=tags,
            config=clean_config,
            mode=mode,
            reinit=True,
        )
        assert wandb.run is not None
        logger.info(f"WandB initialized: {name} (ID: {wandb.run.id})")
    except Exception as e:
        logger.error(f"Failed to initialize WandB: {e}")


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Logs scalar metrics to the active run.
    """
    if not is_active():
        return

    try:
        wandb.log(metrics, step=step)
    except Exception as e:
        logger.warning(f"Failed to log metrics: {e}")


def safe_log_image(
    path: Union[str, Path],
    key: str,
    step: Optional[int] = None,
    caption: Optional[str] = None
) -> bool:
    """
    Safely uploads an image file to W&B.

    Args:
        path: File path to the image.
        key: The metric key (e.g. "plots/training_curve").
        step: Global step to associate with the image.
        caption: Optional text caption.

    Returns:
        bool: True if queued successfully, False otherwise.
    """
    if not is_active():
        return False

    file_path = Path(path)
    
    # 1. Verification
    if not file_path.exists():
        logger.warning(f"Image not found at {file_path}. Skipping upload for '{key}'.")
        return False

    # 2. Upload
    try:
        # wandb.Image handles formatting automatically
        image = wandb.Image(str(file_path), caption=caption)
        wandb.log({key: image}, step=step)
        return True
    except Exception as e:
        logger.error(f"Failed to upload image '{key}': {e}")
        return False


def finish_wandb() -> None:
    """Cleanly closes the W&B run."""
    if is_active():
        wandb.finish()























