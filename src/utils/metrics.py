"""

Post-hoc utilities for loading, aggregating, and exporting PPO metrics.

KEY MAP — trainer canonical names (all stale keys updated here):
  eval/episode_reward_ma100       → eval/rolling_reward
  eval/ep_len                     → eval/episode_length
  energy/inference                → energy/eval_update
  energy/inference_dynamic        → energy/eval_update_dynamic
  energy/train_rollout            → energy/train_full_update   (rollout-only was removed)
  energy/train_rollout_dynamic    → energy/train_full_update_dynamic
  post_conversion/inference_energy→ post_conversion/zs_energy
  post_conversion_ft/energy/inference → energy/eval_update
  post_conversion/solved_success_rate → post_conversion/zero_shot_success_rate
  post_conversion_ft/eval_reward  → post_conversion_ft/current_reward
  eval/spikes                     → spikes/eval_total
  eval/spikes_actor               → spikes/eval_actor
  eval/spikes_critic              → spikes/eval_critic
  eval/spikes_per_step            → eval/spikes_actor_per_step
"""

from __future__ import annotations

import os
import json
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Union, Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Metric key variants (canonical key → list of fallback aliases)
METRIC_VARIANTS = {
    "train_rewards": ["train/rollout_reward", "train/reward"],
    "test_rewards": ["eval/rolling_reward", "eval/current_reward"],
    "episode_lengths": ["post_eval/episode_length_mean"],
    "train_rollout_energy": ["energy/train_full_update"],
    "train_rollout_dynamic_energy": ["energy/train_full_update_dynamic"],
    "train_full_update_energy": ["energy/train_full_update"],
    "train_full_update_dynamic_energy": ["energy/train_full_update_dynamic"],
    "total_energy": ["energy/total"],
    "total_dynamic_energy": ["energy/total_dynamic"],
    "inference_energy": ["energy/eval_update"],
    "inference_dynamic_energy": ["energy/eval_update_dynamic"],
    "idle_power_watts": ["energy/idle_power_watts"],
    "latency_ms": ["latency/mean_ms"],
    "spike_timing_steps": ["latency/spike_timing_steps"],
    "actor_spike_timing_steps": ["latency/actor_spike_timing_steps"],
    "critic_spike_timing_steps": ["latency/critic_spike_timing_steps"],
    "eval_success_rate": ["eval/success_rate"],
    "eval_success_count": ["eval/success_count"],
    "eval_n_eval_episodes": ["eval/n_eval_episodes"],
    "eval_episode_length": ["eval/episode_length"],
    "eval_spikes_total": ["spikes/eval_total"],
    "eval_spikes_actor": ["spikes/eval_actor"],
    "eval_spikes_critic": ["spikes/eval_critic"],
    "zs_reward": ["post_conversion/zero_shot_reward"],
    "zs_energy": ["post_conversion/zs_energy"],
    "zs_latency": ["post_conversion/mean_latency"],
    "zs_spikes": ["post_conversion/total_spikes"],
}

# ---------------------------------------------------------------------
# IO utilities
# ---------------------------------------------------------------------

def load_training_data(log_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training_metrics.csv and per_episode_metrics.csv defensively.
    Returns: (training_metrics, per_episode_metrics)
    """
    log_path = Path(log_dir)

    metrics_file     = log_path / "training_metrics.csv"
    per_episode_file = log_path / "per_episode_metrics.csv"

    try:
        training = pd.read_csv(metrics_file) if metrics_file.exists() else pd.DataFrame()
    except Exception:
        logger.exception("Failed to load training_metrics.csv")
        training = pd.DataFrame()

    try:
        per_episode = (
            pd.read_csv(per_episode_file, on_bad_lines="skip")
            if per_episode_file.exists()
            else pd.DataFrame()
        )
    except Exception:
        logger.exception("Failed to load per_episode_metrics.csv")
        per_episode = pd.DataFrame()

    if not per_episode.empty:
        if "train_reward" not in per_episode.columns:
            if "train_rollout_reward" in per_episode.columns:
                per_episode["train_reward"] = per_episode["train_rollout_reward"]
            elif "ann_reward" in per_episode.columns:
                per_episode["train_reward"] = per_episode["ann_reward"]

        if "test_reward" not in per_episode.columns:
            if "eval_episode_reward" in per_episode.columns:
                per_episode["test_reward"] = per_episode["eval_episode_reward"]

    return training, per_episode


def atomic_save_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Writes a DataFrame to CSV atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode='w', suffix='.tmp') as tmp:
            df.to_csv(tmp.name, index=False)
            tmp_name = tmp.name
        shutil.move(tmp_name, str(path))
    except Exception as e:
        logger.error(f"Failed to save CSV to {path}: {e}")
        if 'tmp_name' in locals() and Path(tmp_name).exists():
            Path(tmp_name).unlink()


# ---------------------------------------------------------------------
# Math & Alignment Helpers
# ---------------------------------------------------------------------

def cumulative(x: Sequence[float]) -> np.ndarray:
    return np.cumsum(np.asarray(x, dtype=float))


def rolling_success_rate(
    rewards: Sequence[float],
    threshold: float = 475.0,
    window: int = 50,
) -> np.ndarray:
    r    = np.asarray(rewards, dtype=float)
    hits = (r >= threshold).astype(float)
    out  = np.empty_like(hits)
    for i in range(len(hits)):
        lo = max(0, i - window + 1)
        out[i] = hits[lo : i + 1].mean() * 100.0
    return out


def pad_to_len(
    arr: Union[Sequence[float], np.ndarray],
    L: int,
    fill: float = np.nan,
) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    out = np.full(L, fill, dtype=float)
    out[: min(len(arr), L)] = arr[:L]
    return out


def align_sparse_data(sparse_data: list, n: int) -> np.ndarray:
    out = np.full(n, np.nan, dtype=float)
    if not sparse_data:
        return out
    k = len(sparse_data)
    if k == 1:
        out[-1] = sparse_data[0]
    elif k > 0:
        indices = np.linspace(0, n - 1, k, dtype=int)
        for i, idx in enumerate(indices):
            out[idx] = sparse_data[i]
    return out


def calculate_cumulative_steps(per_episode_data: pd.DataFrame) -> np.ndarray:
    if 'total_timesteps' in per_episode_data.columns:
        return per_episode_data['total_timesteps'].values
    elif 'episode_length_steps' in per_episode_data.columns:
        return np.cumsum(np.array(per_episode_data['episode_length_steps'].values, dtype=float))
    else:
        return np.cumsum(np.array(per_episode_data['test_reward'].values, dtype=float))


def compute_updates_to_solve(
    df: pd.DataFrame,
    reward_threshold: float,
    reward_col: str = "test_reward",
    steps_col: str = "total_timesteps",
    update_col: str = "update",
) -> Optional[Dict[str, float]]:
    """Return the first training update at which the rolling eval reward exceeds the threshold.

    Works purely post-hoc on the per_episode_metrics CSV produced by the trainer.
    ``reward_col`` is sparse (only populated at eval intervals); rows with NaN are skipped.

    Returns:
        Dict with keys ``update`` and ``total_steps``, or ``None`` if never solved.
    """
    eval_rows = df[df[reward_col].notna()].copy()
    if eval_rows.empty:
        return None
    solved = eval_rows[eval_rows[reward_col] >= reward_threshold]
    if solved.empty:
        return None
    first = solved.iloc[0]
    result: Dict[str, float] = {"update": float(first[update_col])}
    if steps_col in df.columns:
        result["total_steps"] = float(first[steps_col])
    return result


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


# ---------------------------------------------------------------------
# CSV Export / Calculation
# ---------------------------------------------------------------------

def _extract_metric(logger_obj, key: str) -> List[float]:
    if hasattr(logger_obj, "history"):
        events = logger_obj.history.get(key, [])
        return [e.value for e in events]
    if hasattr(logger_obj, "_metric_series"):
        return logger_obj._metric_series(key)
    return []


def _extract_metric_from_raw_json(out_dir: str, key: str) -> List[float]:
    raw_path = Path(out_dir) / "metrics_raw.json"
    if not raw_path.exists():
        return []
    try:
        with open(raw_path, "r") as f:
            data = json.load(f)
        events = data.get(key, [])
        if not events:
            return []
        if isinstance(events[0], dict):
            return [float(e.get("value", np.nan)) for e in events]
        return [float(v) for v in events]
    except Exception:
        return []


def _extract_steps_from_raw_json(out_dir: str, key: str) -> List[float]:
    raw_path = Path(out_dir) / "metrics_raw.json"
    if not raw_path.exists():
        return []
    try:
        with open(raw_path, "r") as f:
            data = json.load(f)
        events = data.get(key, [])
        if not events or not isinstance(events[0], dict):
            return []
        return [float(e.get("step", np.nan)) for e in events]
    except Exception:
        return []


def _resolve_metric(
    logger_obj: Any,
    key_variants: List[str],
    out_dir: Optional[str] = None,
    existing_df: Optional[pd.DataFrame] = None,
) -> List[float]:
    """
    Resolve a metric from multiple sources in priority order.

    Args:
        logger_obj: PPO logger object (or None)
        key_variants: List of key names to try in order
        out_dir: Directory containing metrics_raw.json (fallback)
        existing_df: Existing CSV DataFrame (final fallback)

    Returns:
        List of metric values, or empty list if not found
    """
    # Try logger object first
    if logger_obj:
        for key in key_variants:
            result = _extract_metric(logger_obj, key)
            if result:
                return result

    # Try raw JSON
    if out_dir:
        for key in key_variants:
            result = _extract_metric_from_raw_json(out_dir, key)
            if result:
                return result

    # Try existing CSV
    if existing_df is not None:
        for key in key_variants:
            if key in existing_df.columns:
                result = existing_df[key].dropna().tolist()
                if result:
                    return result

    return []


def _align_values_to_train_timeline(
    values: List[float],
    sparse_steps: List[float],
    train_steps: List[float],
    n: int,
) -> np.ndarray:
    out = np.full(n, np.nan, dtype=float)
    if not values:
        return out
    if len(values) == n:
        return pad_to_len(values, n)
    if len(sparse_steps) == len(values) and len(train_steps) == n:
        timeline     = np.asarray(train_steps, dtype=float)
        step_to_idx  = {int(round(float(s))): i for i, s in enumerate(timeline)}
        for step, val in zip(sparse_steps, values):
            if step is None:
                continue
            step_f = float(step)
            if np.isnan(step_f):
                continue
            idx = step_to_idx.get(int(round(step_f)))
            if idx is None:
                idx = int(np.argmin(np.abs(timeline - step_f)))
            out[idx] = float(val)
        return out
    return pad_to_len(align_sparse_data(values, n), n)


def calculate_and_save_metrics_csv(
    result: dict,
    out_dir: str,
    env_name: str = "CartPole-v1",
    reward_threshold: float = 475.0,
    timed_eval_episodes: int = 15,
    quick: bool = False,
) -> Dict[str, Any]:
    """
    Calculates metrics from a result dict and saves them to CSVs.
    Preserves energy/latency data from existing logger CSVs via intelligent merge.
    """
    out_dir = ensure_dir(out_dir)
    logger_log = logging.getLogger(__name__)
    logger_obj = result.get("logger")

    train_rewards = result.get("train_rewards", [])
    test_rewards = result.get("test_rewards", [])
    episode_lengths = result.get("episode_lengths", [])
    train_steps = []
    done_counts = []
    episode_counts = []
    eval_steps = []
    test_source_key = None

    spike_train = result.get("spike_counts_train", [])
    spike_eval = result.get("spike_counts_eval", [])

    # Load existing CSV for metric recovery
    csv_path = os.path.join(out_dir, "per_episode_metrics.csv")
    existing_df = pd.DataFrame()
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
        except Exception:
            pass

    # Metric dict: maps local var name → (key_variants, extract_fn_choice)
    metrics_to_resolve = {
        "train_rollout_energy": ["energy/train_full_update"],
        "train_rollout_dynamic_energy": ["energy/train_full_update_dynamic"],
        "train_full_update_energy": ["energy/train_full_update"],
        "train_full_update_dynamic_energy": ["energy/train_full_update_dynamic"],
        "total_energy": ["energy/total"],
        "total_dynamic_energy": ["energy/total_dynamic"],
        "inference_energy": ["energy/eval_update"],
        "inference_dynamic_energy": ["energy/eval_update_dynamic"],
        "idle_power_watts": ["energy/idle_power_watts"],
        "latency_ms": ["latency/mean_ms"],
        "spike_timing_steps": ["latency/spike_timing_steps"],
        "actor_spike_timing_steps": ["latency/actor_spike_timing_steps"],
        "critic_spike_timing_steps": ["latency/critic_spike_timing_steps"],
        "eval_success_rate": ["eval/success_rate"],
        "eval_success_count": ["eval/success_count"],
        "eval_n_eval_episodes": ["eval/n_eval_episodes"],
        "eval_episode_length": ["eval/episode_length"],
        "eval_spikes_total": ["spikes/eval_total"],
        "eval_spikes_actor": ["spikes/eval_actor"],
        "eval_spikes_critic": ["spikes/eval_critic"],
        "zs_reward": ["post_conversion/zero_shot_reward"],
        "zs_energy": ["post_conversion/zs_energy"],
        "zs_latency": ["post_conversion/mean_latency"],
        "zs_spikes": ["post_conversion/total_spikes"],
    }

    resolved = {}
    for var_name, key_variants in metrics_to_resolve.items():
        resolved[var_name] = _resolve_metric(logger_obj, key_variants, out_dir, existing_df)

    # Unpack resolved metrics into local variables
    train_rollout_energy = resolved["train_rollout_energy"]
    train_rollout_dynamic_energy = resolved["train_rollout_dynamic_energy"]
    train_full_update_energy = resolved["train_full_update_energy"]
    train_full_update_dynamic_energy = resolved["train_full_update_dynamic_energy"]
    total_energy = resolved["total_energy"]
    total_dynamic_energy = resolved["total_dynamic_energy"]
    inference_energy = resolved["inference_energy"]
    inference_dynamic_energy = resolved["inference_dynamic_energy"]
    idle_power_watts = resolved["idle_power_watts"]
    latency_ms = resolved["latency_ms"]
    spike_timing_steps = resolved["spike_timing_steps"]
    actor_spike_timing_steps = resolved["actor_spike_timing_steps"]
    critic_spike_timing_steps = resolved["critic_spike_timing_steps"]
    eval_success_rate = resolved["eval_success_rate"]
    eval_success_count = resolved["eval_success_count"]
    eval_n_eval_episodes = resolved["eval_n_eval_episodes"]
    eval_episode_length = resolved["eval_episode_length"]
    eval_spikes_total = resolved["eval_spikes_total"]
    eval_spikes_actor = resolved["eval_spikes_actor"]
    eval_spikes_critic = resolved["eval_spikes_critic"]
    zs_reward = resolved["zs_reward"]
    zs_energy = resolved["zs_energy"]
    zs_latency = resolved["zs_latency"]
    zs_spikes = resolved["zs_spikes"]

    spike_total = []
    spike_actor = []
    spike_critic = []
    spikes_per_step = []
    spikes_actor_per_step = []
    spikes_critic_per_step = []
    firing_rate = []
    critic_eval_spike_timing_steps = []
    eval_critic_tau_mean = []
    eval_critic_tau_std = []
    eval_critic_value_mean = []
    eval_critic_value_std = []
    eval_wall_clock_ms = []
    eval_spike_timing_steps = []
    zs_sparsity = []
    eval_spikes_per_step = []
    eval_spikes_actor_per_step = []
    eval_spikes_critic_per_step = []
    ft_success_rate = []
    ft_success_count = []
    ft_n_eval_episodes = []
    zs_solved_success_rate = []
    ft_train_reward = []
    ft_eval_reward = []
    ft_energy = []
    ft_latency = []

    if logger_obj:
        if not train_rewards:
            train_rewards = _resolve_metric(logger_obj, ["train/rollout_reward", "train/reward"])
        if not test_rewards:
            test_rewards = _extract_metric(logger_obj, "eval/rolling_reward")
            if test_rewards:
                test_source_key = "eval/rolling_reward"
            else:
                test_rewards = _extract_metric(logger_obj, "eval/current_reward")
                if test_rewards:
                    test_source_key = "eval/current_reward"
            if not test_rewards and hasattr(logger_obj, "_last_eval_rewards"):
                test_rewards = list(logger_obj._last_eval_rewards)

        if not spike_train:
            spike_train = _extract_metric(logger_obj, "spikes/train_count")
        if not spike_eval:
            spike_eval = _extract_metric(logger_obj, "spikes/eval_count")
        if not spike_total:
            spike_total = _extract_metric(logger_obj, "spikes/eval_total") or _extract_metric(logger_obj, "spike_count_total")
        if not spike_actor:
            spike_actor = _extract_metric(logger_obj, "spikes/eval_actor")
        if not spike_critic:
            spike_critic = _extract_metric(logger_obj, "spikes/eval_critic")
        if not spikes_per_step:
            spikes_per_step = _extract_metric(logger_obj, "eval/spikes_actor_per_step")
        if not spikes_actor_per_step:
            spikes_actor_per_step = _extract_metric(logger_obj, "eval/spikes_actor_per_step")
        if not spikes_critic_per_step:
            spikes_critic_per_step = _extract_metric(logger_obj, "eval/spikes_critic_per_step")
        if not firing_rate:
            firing_rate = _extract_metric(logger_obj, "spikes/eval_sparsity")

        zs_sparsity = _extract_metric(logger_obj, "post_conversion/sparsity") or []
        zs_solved_success_rate = _extract_metric(logger_obj, "post_conversion/zero_shot_success_rate") or []
        ft_train_reward = _extract_metric(logger_obj, "post_conversion_ft/train_reward") or []
        ft_eval_reward = _extract_metric(logger_obj, "post_conversion_ft/current_reward") or []
        ft_energy = _extract_metric(logger_obj, "energy/eval_update") or []
        ft_latency = _extract_metric(logger_obj, "post_conversion_ft/train_latency") or []
        ft_success_rate = _extract_metric(logger_obj, "post_conversion_ft/success_rate") or []
        ft_success_count = _extract_metric(logger_obj, "post_conversion_ft/success_count") or []
        ft_n_eval_episodes = _extract_metric(logger_obj, "post_conversion_ft/n_eval_episodes") or []
        critic_eval_spike_timing_steps = _extract_metric(logger_obj, "latency/critic_eval_spike_timing_steps") or []
        eval_wall_clock_ms = _extract_metric(logger_obj, "latency/eval_wall_clock_ms") or []
        eval_spike_timing_steps = _extract_metric(logger_obj, "latency/eval_spike_timing_steps") or []
        eval_critic_tau_mean = _extract_metric(logger_obj, "eval/critic_tau_mean") or []
        eval_critic_tau_std = _extract_metric(logger_obj, "eval/critic_tau_std") or []
        eval_critic_value_mean = _extract_metric(logger_obj, "eval/critic_value_mean") or []
        eval_critic_value_std = _extract_metric(logger_obj, "eval/critic_value_std") or []
        eval_spikes_per_step = _extract_metric(logger_obj, "eval/spikes_actor_per_step") or []
        eval_spikes_actor_per_step = _extract_metric(logger_obj, "eval/spikes_actor_per_step") or []
        eval_spikes_critic_per_step = _extract_metric(logger_obj, "eval/spikes_critic_per_step") or []

        if hasattr(logger_obj, "history"):
            step_events = logger_obj.history.get("train/rollout_reward", [])
            train_steps = [e.step for e in step_events]
            done_events = logger_obj.history.get("train/rollout_done_count", [])
            done_counts = [e.value for e in done_events]
            episode_events = logger_obj.history.get("train/rollout_episode_count", [])
            episode_counts = [e.value for e in episode_events]
            if test_source_key:
                eval_events = logger_obj.history.get(test_source_key, [])
                eval_steps = [e.step for e in eval_events]

    if not train_rewards:
        train_rewards = _resolve_metric(logger_obj, ["train/rollout_reward", "train/reward"], out_dir, existing_df)
    if not test_rewards:
        test_rewards = _resolve_metric(logger_obj, ["eval/rolling_reward", "eval/current_reward"], out_dir, existing_df)
        if not test_source_key:
            test_source_key = "eval/rolling_reward" if test_rewards else "eval/current_reward"
    if not train_steps:
        train_steps = _extract_steps_from_raw_json(out_dir, "train/rollout_reward")
    if not eval_steps and test_source_key:
        eval_steps = _extract_steps_from_raw_json(out_dir, test_source_key)
    if not done_counts:
        done_counts = _extract_metric_from_raw_json(out_dir, "train/rollout_done_count")
    if not episode_counts:
        episode_counts = _extract_metric_from_raw_json(out_dir, "train/rollout_episode_count")
    if not episode_lengths:
        episode_lengths = _extract_metric_from_raw_json(out_dir, "post_eval/episode_length_mean")
    if not spike_train:
        spike_train = _extract_metric_from_raw_json(out_dir, "spikes/train_count")
    if not spike_eval:
        spike_eval = _extract_metric_from_raw_json(out_dir, "spikes/eval_count")
    if not spike_total:
        spike_total = _extract_metric_from_raw_json(out_dir, "spikes/eval_total") or _extract_metric_from_raw_json(out_dir, "spike_count_total")
    if not spike_actor:
        spike_actor = _extract_metric_from_raw_json(out_dir, "spikes/eval_actor")
    if not spike_critic:
        spike_critic = _extract_metric_from_raw_json(out_dir, "spikes/eval_critic")
    if not spikes_actor_per_step:
        spikes_actor_per_step = _extract_metric_from_raw_json(out_dir, "eval/spikes_actor_per_step")
    if not spikes_critic_per_step:
        spikes_critic_per_step = _extract_metric_from_raw_json(out_dir, "eval/spikes_critic_per_step")

    if not train_rewards and not test_rewards:
        if not existing_df.empty and "train_reward" in existing_df.columns:
            train_rewards = existing_df["train_reward"].dropna().tolist()
            if "test_reward" in existing_df.columns:
                test_rewards = existing_df["test_reward"].dropna().tolist()
            logger_log.info("No new rewards in memory; reusing existing per_episode_metrics.csv data.")
        else:
            logger_log.warning("No rewards found in result dict or CSV. Skipping generation.")
            return {}

    # Alignment
    n = len(train_rewards)

    test_rewards_padded = _align_values_to_train_timeline(test_rewards, eval_steps, train_steps, n)
    eval_episode_length_padded = _align_values_to_train_timeline(eval_episode_length, eval_steps, train_steps, n)

    if np.all(np.isnan(test_rewards_padded)):
        logger.warning(
            "All eval rewards are NaN — substituting training rewards in metrics CSV. "
            "Check that eval logging is running correctly."
        )
        test_rewards_padded = pad_to_len(train_rewards, n)

    cum_train = cumulative(train_rewards)
    sr        = rolling_success_rate(test_rewards_padded, threshold=reward_threshold, window=50)

    if len(episode_lengths) < n:
        pad_val        = np.mean(episode_lengths) if episode_lengths else np.nan
        episode_lengths = pad_to_len(episode_lengths, n, fill=pad_val)
    elif len(episode_lengths) > n:
        episode_lengths = episode_lengths[:n]

    train_reward_unscaled = _extract_metric(logger_obj, "train/rollout_reward_raw") if logger_obj else []
    use_train_steps = len(train_steps) == n
    if use_train_steps:
        total_timesteps = np.asarray(train_steps, dtype=float)
        step_deltas     = np.diff(np.concatenate(([0.0], total_timesteps)))
        episode_length_steps = np.full(n, np.nan, dtype=float)
    else:
        episode_length_steps = episode_lengths
        total_timesteps      = np.cumsum(episode_length_steps)
        step_deltas          = np.asarray(episode_length_steps, dtype=float)

    completion_counts = []
    if len(episode_counts) == n:
        completion_counts = episode_counts
    elif len(done_counts) == n:
        completion_counts = done_counts

    if completion_counts and use_train_steps:
        done_arr = np.asarray(completion_counts, dtype=float)
        avg_episode_len = np.divide(
            step_deltas, done_arr,
            out=np.full(n, np.nan),
            where=done_arr > 0,
        )
        episode_length_steps = np.where(done_arr > 0, avg_episode_len, np.nan)
    elif use_train_steps and len(episode_lengths) == n:
        episode_length_steps = np.asarray(episode_lengths, dtype=float)

    df = pd.DataFrame({
        "update":                         np.arange(1, n + 1),
        "train_reward":                   train_rewards,
        "test_reward":                    test_rewards_padded,
        "success_hit":                    (test_rewards_padded >= reward_threshold).astype(int),
        "total_cumulative_train_reward":  cum_train,
        "episode_length_steps":           episode_length_steps,
    })
    if train_reward_unscaled:
        df["train_reward_unscaled"] = pad_to_len(train_reward_unscaled, n)
    if done_counts:
        df["train_rollout_done_count"] = pad_to_len(done_counts, n)
    if episode_counts:
        df["train_rollout_episode_count"] = pad_to_len(episode_counts, n)
    if use_train_steps:
        df["train_rollout_steps"] = step_deltas

    # Energy columns
    if train_rollout_energy:
        df["train_rollout_energy"] = pad_to_len(train_rollout_energy, n)
    if train_rollout_dynamic_energy:
        df["train_rollout_dynamic_energy"] = pad_to_len(train_rollout_dynamic_energy, n)
    if train_full_update_energy:
        df["train_full_update_energy"] = pad_to_len(train_full_update_energy, n)
    if train_full_update_dynamic_energy:
        df["train_full_update_dynamic_energy"] = pad_to_len(train_full_update_dynamic_energy, n)
    if total_energy:
        df["total_energy"] = pad_to_len(total_energy, n)
    if total_dynamic_energy:
        df["total_dynamic_energy"] = pad_to_len(total_dynamic_energy, n)
    if inference_energy:
        df["inference_energy"] = _align_values_to_train_timeline(inference_energy, eval_steps, train_steps, n)
    if inference_dynamic_energy:
        df["inference_dynamic_energy"] = _align_values_to_train_timeline(inference_dynamic_energy, eval_steps, train_steps, n)
    if idle_power_watts:
        df["energy_idle_power_watts"] = pad_to_len(idle_power_watts, n)
    if latency_ms:
        df["latency_mean_ms"] = pad_to_len(latency_ms, n)
    if spike_timing_steps:
        df["latency/spike_timing_steps"] = pad_to_len(spike_timing_steps, n)
    if actor_spike_timing_steps:
        df["latency/actor_spike_timing_steps"] = pad_to_len(actor_spike_timing_steps, n)
    if critic_spike_timing_steps:
        df["latency/critic_spike_timing_steps"] = pad_to_len(critic_spike_timing_steps, n)
    if critic_eval_spike_timing_steps:
        df["latency/critic_eval_spike_timing_steps"] = align_sparse_data(critic_eval_spike_timing_steps, n)
    if eval_wall_clock_ms:
        df["latency/eval_wall_clock_ms"] = align_sparse_data(eval_wall_clock_ms, n)
    if eval_spike_timing_steps:
        df["latency/eval_spike_timing_steps"] = align_sparse_data(eval_spike_timing_steps, n)
    if eval_critic_tau_mean:
        df["eval/critic_tau_mean"] = align_sparse_data(eval_critic_tau_mean, n)
    if eval_critic_tau_std:
        df["eval/critic_tau_std"] = align_sparse_data(eval_critic_tau_std, n)
    if eval_critic_value_mean:
        df["eval/critic_value_mean"] = align_sparse_data(eval_critic_value_mean, n)
    if eval_critic_value_std:
        df["eval/critic_value_std"] = align_sparse_data(eval_critic_value_std, n)
    if eval_success_rate:
        df["eval/success_rate"] = _align_values_to_train_timeline(eval_success_rate, eval_steps, train_steps, n)
    if eval_success_count:
        df["eval/success_count"] = _align_values_to_train_timeline(eval_success_count, eval_steps, train_steps, n)
    if eval_n_eval_episodes:
        df["eval/n_eval_episodes"] = _align_values_to_train_timeline(eval_n_eval_episodes, eval_steps, train_steps, n)
    if eval_spikes_total:
        df["eval/spikes"] = _align_values_to_train_timeline(eval_spikes_total, eval_steps, train_steps, n)
    if eval_spikes_actor:
        df["eval/spikes_actor"] = _align_values_to_train_timeline(eval_spikes_actor, eval_steps, train_steps, n)
    if eval_spikes_critic:
        df["eval/spikes_critic"] = _align_values_to_train_timeline(eval_spikes_critic, eval_steps, train_steps, n)
    if eval_spikes_per_step:
        df["eval/spikes_per_step"] = _align_values_to_train_timeline(eval_spikes_per_step, eval_steps, train_steps, n)
    if eval_spikes_actor_per_step:
        df["eval/spikes_actor_per_step"] = _align_values_to_train_timeline(eval_spikes_actor_per_step, eval_steps, train_steps, n)
    if eval_spikes_critic_per_step:
        df["eval/spikes_critic_per_step"] = _align_values_to_train_timeline(eval_spikes_critic_per_step, eval_steps, train_steps, n)
    if np.any(~np.isnan(eval_episode_length_padded)):
        df["eval_episode_length"] = eval_episode_length_padded

    def _align_tail(values: list, length: int) -> np.ndarray:
        out = np.full(length, np.nan, dtype=float)
        if not values:
            return out
        k = min(len(values), length)
        out[length - k:] = np.asarray(values[-k:], dtype=float)
        return out

    if zs_reward:
        df["post_conversion/zero_shot_reward"] = align_sparse_data(zs_reward, n)
    if zs_energy:
        # FIX: column name reflects new trainer key
        df["post_conversion/zs_energy"] = align_sparse_data(zs_energy, n)
    if zs_latency:
        df["post_conversion/mean_latency"] = align_sparse_data(zs_latency, n)
    if zs_spikes:
        df["post_conversion/total_spikes"] = align_sparse_data(zs_spikes, n)
    if zs_sparsity:
        df["post_conversion/sparsity"] = align_sparse_data(zs_sparsity, n)
    if zs_solved_success_rate:
        df["post_conversion/zero_shot_success_rate"] = align_sparse_data(zs_solved_success_rate, n)
    if ft_train_reward:
        df["post_conversion_ft/train_reward"] = _align_tail(ft_train_reward, n)
    if ft_eval_reward:
        df["post_conversion_ft/current_reward"] = _align_tail(ft_eval_reward, n)
    if ft_energy:
        df["post_conversion_ft/energy/eval_update"] = _align_tail(ft_energy, n)
    if ft_latency:
        df["post_conversion_ft/train_latency"] = _align_tail(ft_latency, n)
    if ft_success_rate:
        df["post_conversion_ft/success_rate"] = _align_tail(ft_success_rate, n)
    if ft_success_count:
        df["post_conversion_ft/success_count"] = _align_tail(ft_success_count, n)
    if ft_n_eval_episodes:
        df["post_conversion_ft/n_eval_episodes"] = _align_tail(ft_n_eval_episodes, n)

    if spike_train:
        df["spike_count_train"] = pad_to_len(spike_train, n)
    if spike_eval:
        df["spike_count_eval"] = pad_to_len(align_sparse_data(spike_eval, n), n)
    if spike_total:
        df["spike_count_total"] = pad_to_len(spike_total, n)
    if spike_actor:
        df["spikes/actor"] = pad_to_len(spike_actor, n)
    if spike_critic:
        df["spikes/critic"] = pad_to_len(spike_critic, n)
    if spikes_per_step:
        df["spikes/per_step"] = pad_to_len(spikes_per_step, n)
    if spikes_actor_per_step:
        df["spikes/actor_per_step"] = pad_to_len(spikes_actor_per_step, n)
    if spikes_critic_per_step:
        df["spikes/critic_per_step"] = pad_to_len(spikes_critic_per_step, n)
    if firing_rate:
        df["spikes/firing_rate"] = pad_to_len(firing_rate, n)

    df["total_timesteps"] = total_timesteps

    atomic_save_csv(df, csv_path)
    logger_log.info(f"Saved merged metrics to {csv_path}")

    return {
        "train_rewards": train_rewards,
        "test_rewards":  test_rewards,
        "sr":            sr,
        "cum_train":     cum_train,
    }