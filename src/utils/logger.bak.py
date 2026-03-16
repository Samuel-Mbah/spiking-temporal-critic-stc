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

        # --- Storage ---
        self.history: Dict[str, List[MetricEvent]] = defaultdict(list)
        self.buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.step_info: List[Dict[str, Any]] = []

        # --- State ---
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
            phase=self.current_phase
        )
        self.history[key].append(event)

        if not exclude_from_console:
            self.buffers[key].append(val)

    def record_episode(self, reward: float, length: int, success: bool = False, source: str = "train"):
        """
        Special handler for episode boundaries.
        Args:
            source: 'train' or 'eval' (matches baseline_trainer.py)
        """
        # Map source to prefix if necessary, or use directly
        prefix = source 
        self.record(f"{prefix}/reward", reward)
        self.record(f"{prefix}/ep_len", length)
        self.record(f"{prefix}/success_rate", float(success))

    def record_step_info(self, **kwargs):
        """
        Log metadata for the current update (FPS, time, etc).
        Accepts arbitrary kwargs (updates, timesteps, fps).
        """
        info = kwargs
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
                ("Updates", self.n_updates, 'raw'),
                ("Timesteps", self.num_timesteps, 'raw'),
                ("Time", time_str, 'raw'),
                ("FPS", last_info.get("fps", np.nan), 'raw'),
            ],
            "Training": [
                ("Policy Loss", "train/policy_loss", 'last'),
                ("Value Loss", "train/value_loss", 'last'),
                ("Explained Var", "train/explained_variance", 'last'),
                ("Entropy", "train/entropy", 'last'),
                ("Approx KL", "train/approx_kl", 'last'),
                ("Clip Frac", "train/clip_fraction", 'last'),
                ("LR", "train/lr", 'last'),
            ],
            "Performance": [
                ("Train Reward", "train/reward", 'mean'),
                ("Eval (Current)", "eval/reward", 'last'),
                ("Eval (100-ep)", "eval/mean_100ep", 'last'),
                ("Success Rate", "train/success_rate", 'mean'),
                ("Ep Length", "train/ep_len", 'mean'),
            ],
            "SNN / Spikes": [
                ("Total Spikes (Rollout)", "spikes/total", 'mean'),
                ("Spikes/Step", "spikes/per_step", 'mean'),
                ("Firing Rate", "spikes/firing_rate", 'mean'),
                ("Sparsity (%)", "spikes/sparsity", 'mean'),
                ("Mean Latency", "latency/mean_ms", 'mean'),
            ],
             # Phase 3: Post-Conversion / Zero-Shot
            "Zero-Shot": [
                 ("ZS Reward", "post_conversion/zero_shot_reward", 'last'),
                 ("ZS Energy", "post_conversion/inference_energy", 'last'),
            ]
        }

        print("=" * 48)
        for section, items in sections.items():
            valid_items = []
            for label, key_or_val, mode in items:
                val = np.nan
                if mode == 'raw':
                    val = key_or_val
                elif mode == 'mean':
                    val = self._get_mean(key_or_val)
                elif mode == 'last':
                    val = self._get_last(key_or_val)

                if mode == 'raw' or (isinstance(val, Number) and not np.isnan(val)):
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
        temp_file = self.log_dir / "metrics_raw.tmp"

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
        # Mapping definitions (Column Name -> Internal Key)
        ann_map = {
            "ann_reward": "train/rollout_reward", # Mapped via trainer
            "train_reward": "train/reward",       # Standard log
            "ann_energy": "energy/train_rollout",
            "ann_timesteps": "total_timesteps_ann"
        }

        snn_map = {
            "snn_ft_reward": "post_conversion_ft/train_reward",
            "snn_ft_energy": "post_conversion_ft/energy/inference",
            "snn_ft_latency": "post_conversion_ft/train_latency",
            "snn_timesteps": "total_timesteps_snn"
        }
        
        zs_map = {
            "zs_reward": "post_conversion/zero_shot_reward",
            "zs_energy": "post_conversion/inference_energy",
            "zs_latency": "post_conversion/mean_latency",
        }

        # Helper: Fetch Phase DataFrame
        def _fetch_phase_df(mapping):
            data = {}
            for col, key in mapping.items():
                if key in self.history:
                    values = [e.value for e in self.history[key]]
                    data[col] = values
            
            if not data:
                return pd.DataFrame()
            return pd.DataFrame({k: pd.Series(v) for k, v in data.items()})

        # Create Phase DFs
        df_ann = _fetch_phase_df(ann_map)
        df_snn = _fetch_phase_df(snn_map)
        df_zs = _fetch_phase_df(zs_map)

        # Handle Eval (Global)
        eval_events = self.history.get("eval/reward", [])
        if eval_events:
            df_eval = pd.DataFrame({
                "test_reward": [e.value for e in eval_events], 
                "eval_step": [e.step for e in eval_events]
            })
        else:
            df_eval = pd.DataFrame()

        # Merge Side-by-Side
        dfs_to_concat = [d for d in [df_ann, df_snn, df_zs, df_eval] if not d.empty]
        
        if not dfs_to_concat:
            return

        df_final = pd.concat(dfs_to_concat, axis=1)
        df_final.index.name = "relative_update"
        df_final.reset_index(inplace=True)

        # Atomic Save
        target_csv = self.log_dir / "per_episode_metrics.csv"
        temp_csv = self.log_dir / "per_episode_metrics.tmp"
        
        try:
            df_final.to_csv(temp_csv, index=False)
            shutil.move(str(temp_csv), str(target_csv))
        except Exception as e:
            print(f"Warning: Failed to save aligned metrics: {e}")













# """
# logger.py

# Research-grade PPO logger with explicit semantic buffers.
# Designed for ANN, ANN→SNN, and SNN surrogate-gradient experiments.
# """

# from __future__ import annotations

# import os
# import json
# import time
# from dataclasses import dataclass
# from collections import defaultdict, deque
# from typing import Optional

# import numpy as np
# import pandas as pd


# # ---------------------------------------------------------------------
# # Typed metric event
# # ---------------------------------------------------------------------

# @dataclass
# class MetricEvent:
#     key: str
#     value: float
#     step: int
#     iteration: int
#     timestamp: float


# # ---------------------------------------------------------------------
# # PPO Logger
# # ---------------------------------------------------------------------

# class PPOLogger:
#     """
#     Semantic, deterministic PPO logger.
#     """

#     def __init__(self, log_dir: str = "logs", window: int = 100):
#         self.log_dir = log_dir
#         os.makedirs(log_dir, exist_ok=True)

#         self.window = window

#         # -------- raw metric history --------
#         self._metrics: dict[str, list[MetricEvent]] = defaultdict(list)

#         # -------- semantic buffers (rolling) --------
#         self._last_train_rewards = deque(maxlen=window)
#         self._last_eval_rewards = deque(maxlen=window)
#         self._last_episode_lengths = deque(maxlen=window)
#         self._last_success_rates = deque(maxlen=window)

#         self._last_policy_loss = deque(maxlen=window)
#         self._last_value_loss = deque(maxlen=window)
#         self._last_entropy = deque(maxlen=window)
#         self._last_explained_variance = deque(maxlen=window)
#         self._last_approx_kl = deque(maxlen=window)
#         self._last_clip_frac = deque(maxlen=window)

#         self._last_total_spikes = deque(maxlen=window)
#         self._last_sparsity = deque(maxlen=window)
#         self._last_mean_latency = deque(maxlen=window)
#         self._last_no_spike_rate = deque(maxlen=window)

#         # step info (for pretty printing)
#         self._steps = []

#         # counters
#         self.num_timesteps = 0
#         self.iteration = 0
#         self._n_updates = 0
#         self.start_time = time.time()

#         print(f"📊 PPOLogger → {os.path.abspath(log_dir)}")

#     # -----------------------------------------------------------------
#     # helpers
#     # -----------------------------------------------------------------

#     @staticmethod
#     def _scalar(x) -> float:
#         if isinstance(x, (float, int)):
#             return float(x)
#         if hasattr(x, "item"):
#             return float(x.item())
#         return float(np.mean(x))

#     def _mean(self, buf):
#         return float(np.nanmean(buf)) if buf else np.nan

#     def _last(self, buf):
#         return float(buf[-1]) if buf else np.nan

#     def _metric_series(self, key: str):
#         return [e.value for e in self._metrics.get(key, [])]

#     # -----------------------------------------------------------------
#     # recording
#     # -----------------------------------------------------------------

#     def record(self, key: str, value):
#         val = self._scalar(value)

#         routing = {
#             "train/policy_loss": self._last_policy_loss,
#             "train/value_loss": self._last_value_loss,
#             "train/entropy": self._last_entropy,
#             "train/explained_variance": self._last_explained_variance,
#             "train/approx_kl": self._last_approx_kl,
#             "train/clip_fraction": self._last_clip_frac,
#             "spikes/total": self._last_total_spikes,
#             "spikes/sparsity": self._last_sparsity,
#             "latency/mean_ms": self._last_mean_latency,
#             "spikes/no_spike_rate": self._last_no_spike_rate,
#         }

#         if key in routing:
#             routing[key].append(val)

#         self._metrics[key].append(
#             MetricEvent(
#                 key=key,
#                 value=val,
#                 step=self.num_timesteps,
#                 iteration=self.iteration,
#                 timestamp=time.time(),
#             )
#         )

#     def record_episode(self, reward, length, *, success=False, source="train"):
#         if source == "train":
#             self._last_train_rewards.append(float(reward))
#         else:
#             self._last_eval_rewards.append(float(reward))

#         self._last_episode_lengths.append(int(length))
#         self._last_success_rates.append(float(success))

#     def record_step_info(self, **kwargs):
#         self._steps.append(kwargs)

#     # -----------------------------------------------------------------
#     # display
#     # -----------------------------------------------------------------

#     def dump(self):
#         t = []

#         # --- Helper to retrieve latest raw value for custom keys ---
#         def _get_raw_last(key, default=np.nan):
#             # Check if key exists and has at least one event
#             if key in self._metrics and self._metrics[key]:
#                 return self._metrics[key][-1].value
#             return default

#         # --- evaluation (authoritative performance metric) ---
#         current_eval = self._last(self._last_eval_rewards) # The most recent batch
#         rolling_eval = self._mean(self._last_eval_rewards) # The 100-ep average
        
#         t.append(("Eval Reward (Current)", current_eval))
#         t.append(("Eval Reward (100-ep)", rolling_eval))
#         # t.append(("Episode Return (Eval)", self._mean(self._last_eval_rewards)))
        
#         t.append(("Episode Length", self._mean(self._last_episode_lengths)))
#         t.append(("Success Rate (%)", self._mean(self._last_success_rates)))

#         # --- step info ---
#         last = self._steps[-1] if self._steps else {}
#         t.append(("Updates", last.get("updates", 0)))
#         t.append(("Total Timesteps", last.get("timesteps", 0)))
#         t.append(("FPS", last.get("fps", np.nan)))

#         # --- time formatting ---
#         elapsed = time.time() - self.start_time
#         if elapsed < 60:
#             elapsed_str = f"{elapsed:.1f} sec"
#         else:
#             elapsed_str = f"{elapsed / 60:.1f} min"
#         t.append(("Time Elapsed", elapsed_str))

#         # --- optimisation stats ---
#         t.append(("Policy Loss", self._last(self._last_policy_loss)))
#         t.append(("Value Loss", self._last(self._last_value_loss)))
#         t.append(("Policy Entropy", self._last(self._last_entropy)))
#         t.append(("Explained Variance", self._last(self._last_explained_variance)))

#         kl = self._last(self._last_approx_kl)
#         cf = self._last(self._last_clip_frac)
#         ratio = kl / cf if cf and not np.isnan(cf) else np.nan

#         t.append(("Approx KL", kl))
#         t.append(("Clip Fraction", cf))
#         t.append(("KL / ClipFrac", ratio))

#         # --- Standard Spikes (Training/Fine-tuning) ---
#         # Only show if standard buffers are populated (avoids printing NaNs if you cleared them)
#         if self._last_total_spikes:
#             t.append(("Total Spikes", self._last(self._last_total_spikes)))
#             t.append(("Sparsity (%)", self._last(self._last_sparsity)))
#             t.append(("Mean Spike Latency", self._last(self._last_mean_latency)))
#             t.append(("No-Spike Rate", self._last(self._last_no_spike_rate)))

#         # --- Phase 3: Post-Conversion / Zero-Shot (Custom Keys) ---
#         # This section ONLY appears if you have recorded 'post_conversion' metrics
#         if "post_conversion/zero_shot_reward" in self._metrics:
#             t.append(("-" * 20, "-" * 10)) # Visual Separator
#             t.append(("Zero-Shot Reward", _get_raw_last("post_conversion/zero_shot_reward")))
#             t.append(("Inference Energy (J)", _get_raw_last("post_conversion/inference_energy")))
#             t.append(("Cumulative Energy (J)", _get_raw_last("post_conversion/energy/total")))
            
#             t.append(("PC Total Spikes", _get_raw_last("post_conversion/total_spikes")))
#             t.append(("PC Sparsity (%)", _get_raw_last("post_conversion/sparsity")))
#             t.append(("PC Mean Latency", _get_raw_last("post_conversion/mean_latency")))
#             t.append(("PC No-Spike Rate", _get_raw_last("post_conversion/no_spike_rate")))

#         # --- pretty print ---
#         print("=" * 56)
#         for k, v in t:
#             if isinstance(v, str) and "-" in v: # Handle separator
#                  print(f"| {k:<28}   {v:>12} |")
#             elif isinstance(v, float):
#                 print(f"| {k:<28} | {v:>12.4f} |")
#             else:
#                 print(f"| {k:<28} | {v:>12} |")
#         print("=" * 56)

#         self._save()
#         self.save_per_episode_metrics()

#     # -----------------------------------------------------------------
#     # persistence
#     # -----------------------------------------------------------------

#     def _save(self):
#         rows = []
#         for events in self._metrics.values():
#             for e in events:
#                 rows.append(vars(e))

#         if rows:
#             pd.DataFrame(rows).to_csv(
#                 os.path.join(self.log_dir, "training_metrics.csv"),
#                 index=False,
#             )

#         latest = {k: v[-1].value for k, v in self._metrics.items() if v}
#         with open(os.path.join(self.log_dir, "latest_metrics.json"), "w") as f:
#             json.dump(latest, f, indent=2)

   
#     def save_per_episode_metrics(self):
#         """
#         Export metrics separating phases into aligned columns:
#         1. ANN Phase (Dense, starts at row 0)
#         2. SNN Fine-tuning Phase (Dense, starts at row 0)
#         3. Zero-Shot (Sparse)
#         """
#         # --- 1. Fetch Series ---
#         # ANN Phase
#         rollout_rewards = self._metric_series("train/rollout_reward")
#         train_rollout_energy = self._metric_series("energy/train_rollout")
        
#         # SNN Fine-tuning Phase
#         ft_rewards = self._metric_series("post_conversion_ft/train_reward")
#         ft_energy = self._metric_series("post_conversion_ft/energy/inference")
#         ft_latency = self._metric_series("post_conversion_ft/train_latency")
        
#         # Zero-Shot (Single points, typically)
#         zs_reward = self._metric_series("post_conversion/zero_shot_reward")
#         zs_energy = self._metric_series("post_conversion/inference_energy")
#         zs_latency = self._metric_series("post_conversion/mean_latency")
        
#         # Global/Shared
#         total_energy = self._metric_series("energy/total")
        
#         # Timesteps (Crucial for alignment)
#         rollout_steps = [e.step for e in self._metrics.get("train/rollout_reward", [])]
#         ft_steps = [e.step for e in self._metrics.get("post_conversion_ft/train_reward", [])]
        
#         # Eval Metrics
#         eval_rewards = list(self._last_eval_rewards)
#         eval_ma = self._metric_series("eval/episode_reward_ma100")
        
#         # SNN Metrics
#         spikes = self._metric_series("spikes/zero_shot")
#         if not spikes: spikes = self._metric_series("spikes/total")
#         latency = self._metric_series("latency/mean_ms")

#         # --- 2. Alignment Logic ---
#         # We calculate 'n' based on the longest phase to ensure the CSV holds all data.
#         # Note: We align phases "side-by-side" (Row 0 = Update 1 of ANN *and* Update 1 of SNN).
#         n = max(
#             len(rollout_rewards),
#             len(ft_rewards), 
#             len(eval_rewards),
#             len(total_energy)
#         )
        
#         if n == 0: return

#         # Helper to pad dense data (truncate if too long, pad with NaN if short)
#         def pad_dense(data, size):
#             if len(data) > size:
#                 return data[:size]
#             return data + [np.nan] * (size - len(data))

#         # Helper to align sparse data (distribute evenly across rows)
#         def align_sparse(sparse_data, size):
#             col = [np.nan] * size
#             k = len(sparse_data)
#             if k > 0:
#                 if k == 1: col[-1] = sparse_data[0]
#                 else:
#                     indices = np.linspace(0, size - 1, k, dtype=int)
#                     for i, idx in enumerate(indices): col[idx] = sparse_data[i]
#             return col

#         # --- 3. Construct DataFrame ---
#         df = pd.DataFrame({
#             "update": np.arange(1, n + 1),
            
#             # --- Time Axes (Separate columns ensure correct plotting) ---
#             "total_timesteps_ann": pad_dense(rollout_steps, n),
#             "total_timesteps_snn": pad_dense(ft_steps, n),
#             "time/total_timesteps": pad_dense(rollout_steps + ft_steps, n), # Fallback/Legacy
            
#             # --- ANN Metrics ---
#             "train/rollout_reward": pad_dense(rollout_rewards, n),
#             "train_rollout_energy": pad_dense(train_rollout_energy, n),
#             "energy/inference": align_sparse(self._metric_series("energy/inference"), n),
            
#             # --- SNN Post-Conversion (Fine-Tuning) ---
#             "post_conversion_ft/train_reward": pad_dense(ft_rewards, n),
#             "post_conversion_ft/eval_reward": align_sparse(self._metric_series("post_conversion_ft/eval_reward"), n),
#             "post_conversion_ft/energy/inference": align_sparse(ft_energy, n),
#             "post_conversion_ft/train_latency": pad_dense(ft_latency, n),
            
#             # --- Zero-Shot ---
#             "post_conversion/zero_shot_reward": align_sparse(zs_reward, n),
#             "post_conversion/inference_energy": align_sparse(zs_energy, n),
#             "post_conversion/mean_latency": align_sparse(zs_latency, n),
            
#             # --- Global/Eval ---
#             "total_energy": pad_dense(total_energy, n),
#             "eval_episode_reward": align_sparse(eval_rewards, n),
#             "eval_ma100": pad_dense(eval_ma, n),
#             "spike_count_total": align_sparse(spikes, n),
#             "latency_mean_ms": pad_dense(latency, n),
#         })

#         df.to_csv(os.path.join(self.log_dir, "per_episode_metrics.csv"), index=False)
        
#     # -----------------------------------------------------------------
#     # counters
#     # -----------------------------------------------------------------

#     def increment_updates(self):
#         self._n_updates += 1
#         self.iteration += 1
