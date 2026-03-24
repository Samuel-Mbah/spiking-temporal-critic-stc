"""
logger.py

Research-grade PPO logger with semantic buffering and phase-aware CSV alignment.
"""

from __future__ import annotations

import json
import time
import shutil
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from numbers import Number

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricEvent:
    """Immutable record of a single metric point."""
    key: str
    value: float
    step: int
    iteration: int
    timestamp: float
    phase: str


class PPOLogger:
    """
    Semantic, deterministic PPO logger.
    """

    def __init__(self, log_dir: str = "logs", window: int = 100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.window = window

        self.history: Dict[str, List[MetricEvent]] = defaultdict(list)
        self.buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.step_info: List[Dict[str, Any]] = []

        self.num_timesteps = 0
        self.iteration = 0
        self.n_updates = 0
        self.start_time = time.time()
        self.current_phase = "train"

        print(f"📊 PPOLogger initialized at: {self.log_dir.resolve()}")

    # -----------------------------------------------------------------
    # State Management
    # -----------------------------------------------------------------

    def set_phase(self, phase_name: str):
        self.current_phase = phase_name

    def increment_updates(self):
        self.n_updates += 1

    # -----------------------------------------------------------------
    # Recording
    # -----------------------------------------------------------------

    @staticmethod
    def _to_scalar(x: Any) -> float:
        if isinstance(x, Number):
            return float(x)
        if hasattr(x, "item"):
            return float(x.item())
        if isinstance(x, (list, np.ndarray)):
            return float(np.mean(x))
        try:
            return float(x)
        except (ValueError, TypeError):
            return np.nan

    def record(self, key: str, value: Any, exclude_from_console: bool = False):
        val = self._to_scalar(value)

        event = MetricEvent(
            key=key,
            value=val,
            step=self.num_timesteps,
            iteration=self.iteration,
            timestamp=time.time(),
            phase=self.current_phase,
        )
        self.history[key].append(event)

        if not exclude_from_console:
            self.buffers[key].append(val)

    def record_episode(
        self,
        reward: float,
        length: int,
        success: bool = False,
        source: str = "train",
    ):
        """
        Special handler for episode boundaries.

        Keys written:
          {source}/current_reward  — most recent single-episode reward
          {source}/episode_length  — episode step count
          {source}/success_rate    — success flag / percentage
        """
        self.record(f"{source}/current_reward", reward)
        self.record(f"{source}/episode_length", length)
        self.record(f"{source}/success_rate",   float(success))

    def record_step_info(self, **kwargs):
        """Log metadata for the current update (FPS, time, etc)."""
        info = dict(kwargs)
        info['timestamp'] = time.time()
        info['step'] = self.num_timesteps
        self.step_info.append(info)

    # -----------------------------------------------------------------
    # Data Retrieval Helpers
    # -----------------------------------------------------------------

    def _get_mean(self, key: str) -> float:
        buf = self.buffers.get(key)
        return float(np.mean(buf)) if buf else np.nan

    def _get_last(self, key: str) -> float:
        buf = self.buffers.get(key)
        return float(buf[-1]) if buf else np.nan

    def _metric_series(self, key: str) -> List[float]:
        """Compatibility helper for metrics.py to fetch full history."""
        return [e.value for e in self.history.get(key, [])]

    # -----------------------------------------------------------------
    # Output / Visualization
    # -----------------------------------------------------------------

    def dump(self):
        """Prints a research-style summary table to console and flushes data to disk."""
        elapsed = time.time() - self.start_time
        time_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed / 60:.1f}m"

        last_info = self.step_info[-1] if self.step_info else {}

        sections = {
            "Status": [
                ("Updates",   self.n_updates,               "raw"),
                ("Timesteps", self.num_timesteps,            "raw"),
                ("Time",      time_str,                      "raw"),
                ("FPS",       last_info.get("fps", np.nan),  "raw"),
            ],
            "Training": [
                ("Policy Loss",   "train/policy_loss",        "last"),
                ("Value Loss",    "train/value_loss",         "last"),
                ("Explained Var", "train/explained_variance", "last"),
                ("Entropy",       "train/entropy",            "last"),
                ("Approx KL",     "train/approx_kl",          "last"),
                ("Clip Frac",     "train/clip_fraction",      "last"),
                ("LR",            "train/learning_rate",      "last"),
            ],
            "Performance": [
                ("Train Reward",   "train/rollout_reward", "mean"),
                ("Eval (Current)", "eval/current_reward",  "last"),
                ("Eval (Rolling)", "eval/rolling_reward",  "last"),
                ("Success Rate",   "eval/success_rate",    "last"),
                ("Ep Length",      "eval/episode_length",  "last"),
            ],
            "SNN / Spikes": [
                ("Total Spikes (Eval)", "spikes/eval_total",        "mean"),
                ("Sparsity",           "spikes/eval_sparsity",      "mean"),
                ("No-Spike Rate",      "spikes/eval_no_spike_rate", "mean"),
                ("Mean Latency",       "latency/mean_ms",           "mean"),
            ],
            "Zero-Shot": [
                ("ZS Reward", "post_conversion/zero_shot_reward", "last"),
                ("ZS Energy", "post_conversion/zs_energy",        "last"),
            ],
        }

        print("=" * 48)
        for section, items in sections.items():
            valid_items = []
            for label, key_or_val, mode in items:
                val = np.nan
                if mode == "raw":
                    val = key_or_val
                elif mode == "mean":
                    val = self._get_mean(key_or_val)
                elif mode == "last":
                    val = self._get_last(key_or_val)

                if mode == "raw" or (isinstance(val, Number) and not np.isnan(val)):
                    valid_items.append((label, val))

            if not valid_items:
                continue

            for label, val in valid_items:
                if isinstance(val, float):
                    print(f"| {label:<20} | {val:>18.4f} |")
                else:
                    print(f"| {label:<20} | {val:>18} |")
            print("-" * 48)
        print("=" * 48)

        self.save()

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self):
        """Saves raw metrics and aligned CSVs safely."""
        self._save_raw_metrics()
        self.save_per_episode_metrics()

    def _save_raw_metrics(self):
        """Save full event history as JSON (atomic write)."""
        export_data = {
            k: [asdict(e) for e in events]
            for k, events in self.history.items()
        }
        target_file = self.log_dir / "metrics_raw.json"
        temp_file   = self.log_dir / "metrics_raw.tmp"

        try:
            with open(temp_file, "w") as f:
                json.dump(export_data, f)
            shutil.move(str(temp_file), str(target_file))
        except Exception as e:
            print(f"Warning: Failed to save raw metrics: {e}")

    def save_per_episode_metrics(self):
        """
        Exports a 'Learning Curve' CSV where ANN and SNN phases are aligned.
        """
        ann_map = {
            "ann_reward":   "train/rollout_reward",
            "train_reward": "train/rollout_reward",
            "ann_energy":   "energy/train_full_update",
            "ann_timesteps": "total_timesteps_ann",
        }

        snn_map = {
            "snn_ft_reward":  "post_conversion_ft/train_reward",
            "snn_ft_energy":  "energy/eval_update",
            "snn_ft_latency": "post_conversion_ft/train_latency",
            "snn_timesteps":  "total_timesteps_snn",
        }

        zs_map = {
            "zs_reward":   "post_conversion/zero_shot_reward",
            "zs_energy":   "post_conversion/zs_energy",
            "zs_latency":  "post_conversion/mean_latency",
            "zs_success":  "post_conversion/zero_shot_success_rate",
        }

        def _fetch_phase_df(mapping):
            data = {}
            for col, key in mapping.items():
                if key in self.history:
                    data[col] = [e.value for e in self.history[key]]
            if not data:
                return pd.DataFrame()
            return pd.DataFrame({k: pd.Series(v) for k, v in data.items()})

        df_ann = _fetch_phase_df(ann_map)
        df_snn = _fetch_phase_df(snn_map)
        df_zs  = _fetch_phase_df(zs_map)

        # Prefer trainer-side rolling_reward over record_episode output for test_reward
        eval_key = (
            "eval/rolling_reward"
            if "eval/rolling_reward" in self.history
            else "eval/current_reward"
        )
        eval_events = self.history.get(eval_key, [])
        if eval_events:
            df_eval = pd.DataFrame({
                "test_reward": [e.value for e in eval_events],
                "eval_step":   [e.step  for e in eval_events],
            })
        else:
            df_eval = pd.DataFrame()

        dfs_to_concat = [d for d in [df_ann, df_snn, df_zs, df_eval] if not d.empty]

        if not dfs_to_concat:
            return

        df_final = pd.concat(dfs_to_concat, axis=1)
        df_final.index.name = "relative_update"
        df_final.reset_index(inplace=True)

        target_csv = self.log_dir / "per_episode_metrics.csv"
        temp_csv   = self.log_dir / "per_episode_metrics.tmp"

        try:
            df_final.to_csv(temp_csv, index=False)
            shutil.move(str(temp_csv), str(target_csv))
        except Exception as e:
            print(f"Warning: Failed to save aligned metrics: {e}")