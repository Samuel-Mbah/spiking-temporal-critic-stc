# """
# metrics.py

# Post-hoc utilities for loading, aggregating, and exporting PPO metrics.
# """

# from __future__ import annotations

# import os
# import time
# from pathlib import Path
# from typing import Dict, List, Tuple, Sequence, Union, Any

# import numpy as np
# import pandas as pd
# import logging

# # ---------------------------------------------------------------------
# # IO utilities
# # ---------------------------------------------------------------------

# def load_training_data(log_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Load training_metrics.csv and per_episode_metrics.csv defensively.
#     """
#     log_path = Path(log_dir)

#     metrics_file = log_path / "training_metrics.csv"
#     per_episode_file = log_path / "per_episode_metrics.csv"

#     try:
#         training = pd.read_csv(metrics_file) if metrics_file.exists() else pd.DataFrame()
#     except Exception:
#         logging.exception("Failed to load training_metrics.csv")
#         training = pd.DataFrame()

#     try:
#         per_episode = (
#             pd.read_csv(per_episode_file, on_bad_lines="skip")
#             if per_episode_file.exists()
#             else pd.DataFrame()
#         )
#     except Exception:
#         logging.exception("Failed to load per_episode_metrics.csv")
#         per_episode = pd.DataFrame()

#     # Standardize column names for plotting compatibility
#     if not per_episode.empty:
#         # Map logger's specific energy columns to generic ones if needed
#         if "train_rollout_reward" in per_episode.columns:
#             per_episode["train_reward"] = per_episode["train_rollout_reward"]
#         if "eval_episode_reward" in per_episode.columns:
#             per_episode["test_reward"] = per_episode["eval_episode_reward"]
    
#     return training, per_episode


# # ---------------------------------------------------------------------
# # Numeric helpers
# # ---------------------------------------------------------------------

# def cumulative(x: Sequence[float]) -> np.ndarray:
#     arr = np.asarray(x, dtype=float)
#     return np.cumsum(arr) 

# def rolling_success_rate(
#     rewards: Sequence[float],
#     threshold: float = 475.0,
#     window: int = 50,
# ) -> np.ndarray:
#     r = np.asarray(rewards, dtype=float)
#     hits = (r >= threshold).astype(float)
#     out = np.empty_like(hits)

#     for i in range(len(hits)):
#         lo = max(0, i - window + 1)
#         out[i] = hits[lo : i + 1].mean() * 100.0

#     return out

# def pad_to_len(
#     arr: Union[Sequence[float], np.ndarray],
#     L: int,
#     fill: float = np.nan,
# ) -> np.ndarray:
#     arr = np.asarray(arr, dtype=float)
#     out = np.full(L, fill, dtype=float)
#     out[: min(len(arr), L)] = arr[:L]
#     return out

# def align_sparse_data(sparse_data: list, n: int) -> np.ndarray:
#     """Aligns sparse evaluation data (like inference energy) to dense steps."""
#     out = np.full(n, np.nan, dtype=float)
#     if not sparse_data:
#         return out
    
#     # If we have data, distribute it evenly or place at end if size mismatch
#     k = len(sparse_data)
#     if k == 1:
#         out[-1] = sparse_data[0]
#     elif k > 0:
#         indices = np.linspace(0, n - 1, k, dtype=int)
#         for i, idx in enumerate(indices):
#             out[idx] = sparse_data[i]
#     return out

# def calculate_cumulative_steps(per_episode_data: pd.DataFrame) -> np.ndarray:
#     """Calculate cumulative environment steps from per-episode data."""
#     if 'total_timesteps' in per_episode_data.columns:
#          return per_episode_data['total_timesteps'].values
#     elif 'episode_length_steps' in per_episode_data.columns:
#         episode_lengths = np.array(per_episode_data['episode_length_steps'].values, dtype=float)
#         return np.cumsum(episode_lengths)
#     else:
#         # Fallback: assume episode length equals reward (for CartPole)
#         episode_lengths = np.array(per_episode_data['test_reward'].values, dtype=float)
#         return np.cumsum(episode_lengths)

# def ensure_dir(p): 
#     os.makedirs(p, exist_ok=True)
#     return p

# # ---------------------------------------------------------------------
# # CSV Export / Calculation
# # ---------------------------------------------------------------------

# def calculate_and_save_metrics_csv(
#     result: dict,
#     out_dir: str,
#     env_name: str = "CartPole-v1",
#     reward_threshold: float = 475.0,
#     timed_eval_episodes: int = 15,
#     quick: bool = False,
# ) -> Dict[str, Any]:
#     """
#     Calculates metrics from a result dict and saves them to CSVs.
#     INTELLIGENT MERGE: Preserves energy/latency data from existing logger CSVs.
#     """
#     out_dir = ensure_dir(out_dir)
#     logger_log = logging.getLogger(__name__)

#     # 1. Extract Data from Result
#     train_rewards = result.get("train_rewards", [])
#     test_rewards = result.get("test_rewards", [])
#     episode_lengths = result.get("episode_lengths", [])

#     # Try to get spike counts from result dict (populated by surrogate_trainer)
#     spike_train = result.get("spike_counts_train", [])
#     spike_eval = result.get("spike_counts_eval", [])
    
#     spike_total = []
    
#     # If list is empty, try to pull from logger object
#     logger_obj = result.get("logger")
    
#     # Prepare Extraction for Energy/Latency
#     train_rollout_energy = []
#     total_energy = []
#     inference_energy = []
#     latency_ms = []

#     if logger_obj:
#         # Pull rewards if missing
#         if not train_rewards:
#             train_rewards = logger_obj._metric_series("train/rollout_reward")
#         if not test_rewards:
#             test_rewards = list(logger_obj._last_eval_rewards)
            
#         # Pull Energy & Latency (The Critical Fix)
#         train_rollout_energy = logger_obj._metric_series("energy/train_rollout")
#         total_energy = logger_obj._metric_series("energy/total")
#         inference_energy = logger_obj._metric_series("energy/inference")
#         latency_ms = logger_obj._metric_series("latency/mean_ms")
        
#         # Pull Spikes from Logger if missing in result dict
#         if not spike_train:
#             spike_train = logger_obj._metric_series("spikes/train_count")
#         if not spike_eval:
#             # Eval spikes might be sparse, similar to eval rewards
#             spike_eval = logger_obj._metric_series("spikes/eval_count")

#     # If still empty, check if we can load from existing CSV (Persistence)
#     csv_path = os.path.join(out_dir, "per_episode_metrics.csv")
#     existing_df = pd.DataFrame()
#     if os.path.exists(csv_path):
#         try:
#             existing_df = pd.read_csv(csv_path)
#             # Recover energy if not found in memory
#             if not train_rollout_energy and "train_rollout_energy" in existing_df:
#                 train_rollout_energy = existing_df["train_rollout_energy"].dropna().tolist()
#             if not total_energy and "total_energy" in existing_df:
#                 total_energy = existing_df["total_energy"].dropna().tolist()
#             if not inference_energy and "inference_energy" in existing_df:
#                 inference_energy = existing_df["inference_energy"].dropna().tolist()
#             # Recover spikes if not found
#             if not spike_train and "spike_count_train" in existing_df:
#                 spike_train = existing_df["spike_count_train"].dropna().tolist()
#             if not spike_eval and "spike_count_eval" in existing_df:
#                 spike_eval = existing_df["spike_count_eval"].dropna().tolist()
#             if "spike_count_total" in existing_df:
#                 spike_total = existing_df["spike_count_total"].dropna().tolist()
#         except Exception:
#             pass

#     if not train_rewards and not test_rewards:
#         logger_log.warning("No rewards found in result dict or CSV. Skipping generation.")
#         return {}

#     # 2. Alignment
#     n = len(train_rewards)
    
#     # Pad Test Rewards (Sparse to Dense)
#     # If we have fewer test points than train points, pad/align them
#     test_rewards_padded = pad_to_len(align_sparse_data(test_rewards, n), n)
#     # If we have missing test rewards (NaN), fill forward or copy train (fallback)
#     if np.all(np.isnan(test_rewards_padded)):
#         test_rewards_padded = pad_to_len(train_rewards, n)

#     # 3. Calculations
#     cum_train = cumulative(train_rewards)
#     sr = rolling_success_rate(test_rewards_padded, threshold=reward_threshold, window=50)

#     # Ensure episode_lengths matches n
#     if len(episode_lengths) < n:
#         pad_val = np.mean(episode_lengths) if episode_lengths else 4096.0 
#         episode_lengths = pad_to_len(episode_lengths, n, fill=pad_val)
#     elif len(episode_lengths) > n:
#         episode_lengths = episode_lengths[:n]

#     # 4. Construct DataFrame
#     df = pd.DataFrame({
#         "update": np.arange(1, n + 1),
#         "train_reward": train_rewards,
#         "test_reward": test_rewards_padded,
#         "success_hit": (test_rewards_padded >= reward_threshold).astype(int),
#         "total_cumulative_train_reward": cum_train,
#         "episode_length_steps": episode_lengths,
#     })

#     # Add Energy/Latency Columns (if available)
#     if train_rollout_energy:
#         df["train_rollout_energy"] = pad_to_len(train_rollout_energy, n)
#     if total_energy:
#         df["total_energy"] = pad_to_len(total_energy, n)
#     if inference_energy:
#         # Inference is sparse, use align helper
#         df["inference_energy"] = align_sparse_data(inference_energy, n)
#     if latency_ms:
#         df["latency_mean_ms"] = pad_to_len(latency_ms, n)

#     # --- ADDED: Add Spike Columns to DataFrame ---
#     if spike_train:
#         df["spike_count_train"] = pad_to_len(spike_train, n)
    
#     if spike_eval:
#          # Align eval spikes just like eval rewards
#         spike_eval_padded = align_sparse_data(spike_eval, n)
#         df["spike_count_eval"] = pad_to_len(spike_eval_padded, n)
        
#     if spike_total:
#         df["spike_count_total"] = pad_to_len(spike_total, n)
    
#     # Total Timesteps Calculation
#     df["total_timesteps"] = df["episode_length_steps"].cumsum()

#     # 5. Save
#     df.to_csv(csv_path, index=False)
#     logger_log.info(f"Saved merged metrics to {csv_path}")

#     return {
#         "train_rewards": train_rewards,
#         "test_rewards": test_rewards,
#         "sr": sr,
#         "cum_train": cum_train
#     }



# """
# metrics.py

# Post-hoc utilities for loading, aggregating, and exporting PPO metrics.
# """

# from __future__ import annotations

# import os
# import json
# import shutil
# import tempfile
# import logging
# from pathlib import Path
# from typing import Dict, List, Tuple, Sequence, Union, Any, Optional

# import numpy as np
# import pandas as pd

# logger = logging.getLogger(__name__)

# # ---------------------------------------------------------------------
# # IO utilities
# # ---------------------------------------------------------------------

# def load_training_data(log_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Load training_metrics.csv and per_episode_metrics.csv defensively.
#     Returns: (training_metrics, per_episode_metrics)
#     """
#     log_path = Path(log_dir)

#     metrics_file = log_path / "training_metrics.csv"
#     per_episode_file = log_path / "per_episode_metrics.csv"

#     try:
#         training = pd.read_csv(metrics_file) if metrics_file.exists() else pd.DataFrame()
#     except Exception:
#         logger.exception("Failed to load training_metrics.csv")
#         training = pd.DataFrame()

#     try:
#         per_episode = (
#             pd.read_csv(per_episode_file, on_bad_lines="skip")
#             if per_episode_file.exists()
#             else pd.DataFrame()
#         )
#     except Exception:
#         logger.exception("Failed to load per_episode_metrics.csv")
#         per_episode = pd.DataFrame()

#     # Standardize column names for plotting compatibility
#     if not per_episode.empty:
#         # Standardize Reward Columns
#         if "train_reward" not in per_episode.columns:
#             if "train_rollout_reward" in per_episode.columns:
#                 per_episode["train_reward"] = per_episode["train_rollout_reward"]
#             elif "ann_reward" in per_episode.columns:
#                 per_episode["train_reward"] = per_episode["ann_reward"]
        
#         if "test_reward" not in per_episode.columns:
#             if "eval_episode_reward" in per_episode.columns:
#                 per_episode["test_reward"] = per_episode["eval_episode_reward"]
    
#     return training, per_episode


# def atomic_save_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
#     """
#     Writes a DataFrame to CSV atomically.
#     Useful if you generate new derived datasets and want to save them safely.
#     """
#     path = Path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)
    
#     try:
#         # Write to temp file in the same directory
#         with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode='w', suffix='.tmp') as tmp:
#             df.to_csv(tmp.name, index=False)
#             tmp_name = tmp.name
        
#         # Atomic rename
#         shutil.move(tmp_name, str(path))
#     except Exception as e:
#         logger.error(f"Failed to save CSV to {path}: {e}")
#         if 'tmp_name' in locals() and Path(tmp_name).exists():
#             Path(tmp_name).unlink()


# # ---------------------------------------------------------------------
# # Analysis Helpers
# # ---------------------------------------------------------------------

# def split_phases(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Splits the side-by-side DataFrame into separate ANN and SNN/FT phases
#     for independent analysis or plotting.
    
#     Returns:
#         (df_ann, df_snn)
#     """
#     # Filter columns by prefix (standardized in logger.py)
#     ann_cols = [c for c in df.columns if c.startswith("ann_") or c in ["relative_update", "eval_ma100"]]
#     snn_cols = [c for c in df.columns if c.startswith("snn_") or c in ["relative_update", "eval_ma100"]]
    
#     df_ann = df[ann_cols].dropna(subset=["ann_reward"]) if "ann_reward" in df.columns else pd.DataFrame()
#     df_snn = df[snn_cols].dropna(subset=["snn_ft_reward"]) if "snn_ft_reward" in df.columns else pd.DataFrame()
    
#     return df_ann, df_snn


# def interpolate_sparse_metrics(df: pd.DataFrame, target_col: str) -> pd.Series:
#     """
#     Linearly interpolates sparse data (like Eval Reward) to match the
#     dense training steps. Useful for plotting smooth comparison lines.
#     """
#     if target_col not in df.columns:
#         return pd.Series(dtype=float)
        
#     return df[target_col].interpolate(method='linear', limit_direction='both')


# # ---------------------------------------------------------------------
# # Reporting
# # ---------------------------------------------------------------------

# def generate_summary_report(
#     log_dir: Union[str, Path], 
#     threshold: float = 475.0
# ) -> Dict[str, float]:
#     """
#     Generates a high-level summary JSON for the experiment.
#     Useful for creating LaTeX tables or comparing multiple runs.
#     """
#     # Use the local load_training_data which returns a tuple (training, per_episode)
#     _, df = load_training_data(log_dir)
    
#     if df.empty:
#         return {}

#     report = {}

#     # --- ANN Stats ---
#     # Handle various column naming conventions
#     ann_key = next((k for k in ["ann_reward", "train_reward", "train_rollout_reward"] if k in df.columns), None)
    
#     if ann_key:
#         ann_rewards = df[ann_key].dropna()
#         report["ann_max_reward"] = float(ann_rewards.max())
#         report["ann_final_ma100"] = float(ann_rewards.tail(100).mean())
        
#         # Convergence step (first time crossing threshold)
#         hits = df.loc[df[ann_key] >= threshold]
        
#         # Try to find a timestep column
#         ts_key = next((k for k in ["total_timesteps_ann", "total_timesteps", "time/total_timesteps"] if k in df.columns), None)
#         if hits.empty:
#             report["ann_convergence_step"] = -1
#         else:
#             report["ann_convergence_step"] = int(hits.iloc[0][ts_key]) if ts_key else int(hits.index[0])

#     # --- SNN FT Stats ---
#     if "snn_ft_reward" in df.columns:
#         snn_rewards = df["snn_ft_reward"].dropna()
#         report["snn_max_reward"] = float(snn_rewards.max())
#         report["snn_final_ma100"] = float(snn_rewards.tail(100).mean())

#     # --- Efficiency Stats ---
#     if "spike_count" in df.columns:
#         report["mean_spikes"] = float(df["spike_count"].mean())
#     elif "spike_count_total" in df.columns:
#         report["mean_spikes"] = float(df["spike_count_total"].mean())

#     # --- Save Report ---
#     save_path = Path(log_dir) / "summary_report.json"
#     try:
#         with open(save_path, "w") as f:
#             json.dump(report, f, indent=4)
#     except Exception as e:
#         logger.error(f"Failed to save summary report: {e}")
        
#     return report


# # ---------------------------------------------------------------------
# # Math & Alignment Helpers
# # ---------------------------------------------------------------------

# def cumulative(x: Sequence[float]) -> np.ndarray:
#     arr = np.asarray(x, dtype=float)
#     return np.cumsum(arr) 

# def rolling_success_rate(
#     rewards: Sequence[float],
#     threshold: float = 475.0,
#     window: int = 50,
# ) -> np.ndarray:
#     r = np.asarray(rewards, dtype=float)
#     hits = (r >= threshold).astype(float)
#     out = np.empty_like(hits)

#     for i in range(len(hits)):
#         lo = max(0, i - window + 1)
#         out[i] = hits[lo : i + 1].mean() * 100.0

#     return out

# def pad_to_len(
#     arr: Union[Sequence[float], np.ndarray],
#     L: int,
#     fill: float = np.nan,
# ) -> np.ndarray:
#     arr = np.asarray(arr, dtype=float)
#     out = np.full(L, fill, dtype=float)
#     out[: min(len(arr), L)] = arr[:L]
#     return out

# def align_sparse_data(sparse_data: list, n: int) -> np.ndarray:
#     """Aligns sparse evaluation data (like inference energy) to dense steps."""
#     out = np.full(n, np.nan, dtype=float)
#     if not sparse_data:
#         return out
    
#     # If we have data, distribute it evenly or place at end if size mismatch
#     k = len(sparse_data)
#     if k == 1:
#         out[-1] = sparse_data[0]
#     elif k > 0:
#         indices = np.linspace(0, n - 1, k, dtype=int)
#         for i, idx in enumerate(indices):
#             out[idx] = sparse_data[i]
#     return out

# def calculate_cumulative_steps(per_episode_data: pd.DataFrame) -> np.ndarray:
#     """Calculate cumulative environment steps from per-episode data."""
#     if 'total_timesteps' in per_episode_data.columns:
#          return per_episode_data['total_timesteps'].values
#     elif 'episode_length_steps' in per_episode_data.columns:
#         episode_lengths = np.array(per_episode_data['episode_length_steps'].values, dtype=float)
#         return np.cumsum(episode_lengths)
#     else:
#         # Fallback: assume episode length equals reward (for CartPole)
#         episode_lengths = np.array(per_episode_data['test_reward'].values, dtype=float)
#         return np.cumsum(episode_lengths)

# def ensure_dir(p): 
#     os.makedirs(p, exist_ok=True)
#     return p


# # ---------------------------------------------------------------------
# # CSV Export / Calculation
# # ---------------------------------------------------------------------

# def _extract_metric(logger_obj, key: str) -> List[float]:
#     """Helper to extract metrics compatible with both Old and New Logger structures."""
#     # 1. New Logger (self.history = dict of MetricEvents)
#     if hasattr(logger_obj, "history"):
#         events = logger_obj.history.get(key, [])
#         return [e.value for e in events]
    
#     # 2. Old Logger (self._metrics = dict of MetricEvents or simple list)
#     if hasattr(logger_obj, "_metric_series"):
#         return logger_obj._metric_series(key)
        
#     return []

# def calculate_and_save_metrics_csv(
#     result: dict,
#     out_dir: str,
#     env_name: str = "CartPole-v1",
#     reward_threshold: float = 475.0,
#     timed_eval_episodes: int = 15,
#     quick: bool = False,
# ) -> Dict[str, Any]:
#     """
#     Calculates metrics from a result dict and saves them to CSVs.
#     INTELLIGENT MERGE: Preserves energy/latency data from existing logger CSVs.
#     """
#     out_dir = ensure_dir(out_dir)
#     logger_log = logging.getLogger(__name__)

#     # 1. Extract Data from Result
#     train_rewards = result.get("train_rewards", [])
#     test_rewards = result.get("test_rewards", [])
#     episode_lengths = result.get("episode_lengths", [])

#     # Try to get spike counts from result dict (populated by surrogate_trainer)
#     spike_train = result.get("spike_counts_train", [])
#     spike_eval = result.get("spike_counts_eval", [])
#     spike_total = []
    
#     # If list is empty, try to pull from logger object
#     logger_obj = result.get("logger")
    
#     # Prepare Extraction for Energy/Latency
#     train_rollout_energy = []
#     total_energy = []
#     inference_energy = []
#     latency_ms = []

#     if logger_obj:
#         # Pull rewards if missing
#         if not train_rewards:
#             # Check both legacy and new keys
#             train_rewards = _extract_metric(logger_obj, "train/rollout_reward") or _extract_metric(logger_obj, "train/reward")
#         if not test_rewards:
#             # Check both legacy and new keys
#             test_rewards = _extract_metric(logger_obj, "eval/episode_reward_ma100") or _extract_metric(logger_obj, "eval/reward")
#             if not test_rewards and hasattr(logger_obj, "_last_eval_rewards"):
#                 test_rewards = list(logger_obj._last_eval_rewards)
            
#         # Pull Energy & Latency (The Critical Fix)
#         train_rollout_energy = _extract_metric(logger_obj, "energy/train_rollout")
#         total_energy = _extract_metric(logger_obj, "energy/total")
#         inference_energy = _extract_metric(logger_obj, "energy/inference")
#         latency_ms = _extract_metric(logger_obj, "latency/mean_ms")
        
#         # Pull Spikes from Logger if missing in result dict
#         if not spike_train:
#             spike_train = _extract_metric(logger_obj, "spikes/train_count")
#         if not spike_eval:
#             # Eval spikes might be sparse, similar to eval rewards
#             spike_eval = _extract_metric(logger_obj, "spikes/eval_count")

#     # If still empty, check if we can load from existing CSV (Persistence)
#     csv_path = os.path.join(out_dir, "per_episode_metrics.csv")
#     existing_df = pd.DataFrame()
#     if os.path.exists(csv_path):
#         try:
#             existing_df = pd.read_csv(csv_path)
#             # Recover energy if not found in memory
#             if not train_rollout_energy and "train_rollout_energy" in existing_df:
#                 train_rollout_energy = existing_df["train_rollout_energy"].dropna().tolist()
#             if not total_energy and "total_energy" in existing_df:
#                 total_energy = existing_df["total_energy"].dropna().tolist()
#             if not inference_energy and "inference_energy" in existing_df:
#                 inference_energy = existing_df["inference_energy"].dropna().tolist()
#             # Recover spikes if not found
#             if not spike_train and "spike_count_train" in existing_df:
#                 spike_train = existing_df["spike_count_train"].dropna().tolist()
#             if not spike_eval and "spike_count_eval" in existing_df:
#                 spike_eval = existing_df["spike_count_eval"].dropna().tolist()
#             if "spike_count_total" in existing_df:
#                 spike_total = existing_df["spike_count_total"].dropna().tolist()
#         except Exception:
#             pass

#     if not train_rewards and not test_rewards:
#         logger_log.warning("No rewards found in result dict or CSV. Skipping generation.")
#         return {}

#     # 2. Alignment
#     n = len(train_rewards)
    
#     # Pad Test Rewards (Sparse to Dense)
#     # If we have fewer test points than train points, pad/align them
#     test_rewards_padded = pad_to_len(align_sparse_data(test_rewards, n), n)
#     # If we have missing test rewards (NaN), fill forward or copy train (fallback)
#     if np.all(np.isnan(test_rewards_padded)):
#         test_rewards_padded = pad_to_len(train_rewards, n)

#     # 3. Calculations
#     cum_train = cumulative(train_rewards)
#     sr = rolling_success_rate(test_rewards_padded, threshold=reward_threshold, window=50)

#     # Ensure episode_lengths matches n
#     if len(episode_lengths) < n:
#         pad_val = np.mean(episode_lengths) if episode_lengths else 4096.0 
#         episode_lengths = pad_to_len(episode_lengths, n, fill=pad_val)
#     elif len(episode_lengths) > n:
#         episode_lengths = episode_lengths[:n]

#     # 4. Construct DataFrame
#     df = pd.DataFrame({
#         "update": np.arange(1, n + 1),
#         "train_reward": train_rewards,
#         "test_reward": test_rewards_padded,
#         "success_hit": (test_rewards_padded >= reward_threshold).astype(int),
#         "total_cumulative_train_reward": cum_train,
#         "episode_length_steps": episode_lengths,
#     })

#     # Add Energy/Latency Columns (if available)
#     if train_rollout_energy:
#         df["train_rollout_energy"] = pad_to_len(train_rollout_energy, n)
#     if total_energy:
#         df["total_energy"] = pad_to_len(total_energy, n)
#     if inference_energy:
#         # Inference is sparse, use align helper
#         df["inference_energy"] = align_sparse_data(inference_energy, n)
#     if latency_ms:
#         df["latency_mean_ms"] = pad_to_len(latency_ms, n)

#     # --- ADDED: Add Spike Columns to DataFrame ---
#     if spike_train:
#         df["spike_count_train"] = pad_to_len(spike_train, n)
    
#     if spike_eval:
#          # Align eval spikes just like eval rewards
#         spike_eval_padded = align_sparse_data(spike_eval, n)
#         df["spike_count_eval"] = pad_to_len(spike_eval_padded, n)
        
#     if spike_total:
#         df["spike_count_total"] = pad_to_len(spike_total, n)
    
#     # Total Timesteps Calculation
#     df["total_timesteps"] = df["episode_length_steps"].cumsum()

#     # 5. Save
#     atomic_save_csv(df, csv_path)
#     logger_log.info(f"Saved merged metrics to {csv_path}")

#     return {
#         "train_rewards": train_rewards,
#         "test_rewards": test_rewards,
#         "sr": sr,
#         "cum_train": cum_train
#     }









"""
metrics.py

Post-hoc utilities for loading, aggregating, and exporting PPO metrics.
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

# ---------------------------------------------------------------------
# IO utilities
# ---------------------------------------------------------------------

def load_training_data(log_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training_metrics.csv and per_episode_metrics.csv defensively.
    Returns: (training_metrics, per_episode_metrics)
    """
    log_path = Path(log_dir)

    metrics_file = log_path / "training_metrics.csv"
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

    # Standardize column names for plotting compatibility
    if not per_episode.empty:
        # Standardize Reward Columns
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
    """
    Writes a DataFrame to CSV atomically.
    Useful if you generate new derived datasets and want to save them safely.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Write to temp file in the same directory
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode='w', suffix='.tmp') as tmp:
            df.to_csv(tmp.name, index=False)
            tmp_name = tmp.name
        
        # Atomic rename
        shutil.move(tmp_name, str(path))
    except Exception as e:
        logger.error(f"Failed to save CSV to {path}: {e}")
        if 'tmp_name' in locals() and Path(tmp_name).exists():
            Path(tmp_name).unlink()


# ---------------------------------------------------------------------
# Analysis Helpers
# ---------------------------------------------------------------------

def split_phases(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the side-by-side DataFrame into separate ANN and SNN/FT phases
    for independent analysis or plotting.
    
    Returns:
        (df_ann, df_snn)
    """
    # Filter columns by prefix (standardized in logger.py)
    ann_cols = [c for c in df.columns if c.startswith("ann_") or c in ["relative_update", "eval_ma100"]]
    snn_cols = [c for c in df.columns if c.startswith("snn_") or c in ["relative_update", "eval_ma100"]]
    
    df_ann = df[ann_cols].dropna(subset=["ann_reward"]) if "ann_reward" in df.columns else pd.DataFrame()
    df_snn = df[snn_cols].dropna(subset=["snn_ft_reward"]) if "snn_ft_reward" in df.columns else pd.DataFrame()
    
    return df_ann, df_snn


def interpolate_sparse_metrics(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Linearly interpolates sparse data (like Eval Reward) to match the
    dense training steps. Useful for plotting smooth comparison lines.
    """
    if target_col not in df.columns:
        return pd.Series(dtype=float)
        
    return df[target_col].interpolate(method='linear', limit_direction='both')


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def generate_summary_report(
    log_dir: Union[str, Path], 
    threshold: float = 475.0
) -> Dict[str, float]:
    """
    Generates a high-level summary JSON for the experiment.
    Useful for creating LaTeX tables or comparing multiple runs.
    """
    # Use the local load_training_data which returns a tuple (training, per_episode)
    _, df = load_training_data(log_dir)
    
    if df.empty:
        return {}

    report = {}

    # --- ANN Stats ---
    # Handle various column naming conventions
    ann_key = next((k for k in ["ann_reward", "train_reward", "train_rollout_reward"] if k in df.columns), None)
    
    if ann_key:
        ann_rewards = df[ann_key].dropna()
        report["ann_max_reward"] = float(ann_rewards.max())
        report["ann_final_ma100"] = float(ann_rewards.tail(100).mean())
        
        # Convergence step (first time crossing threshold)
        hits = df.loc[df[ann_key] >= threshold]
        
        # Try to find a timestep column
        ts_key = next((k for k in ["total_timesteps_ann", "total_timesteps", "time/total_timesteps"] if k in df.columns), None)
        if hits.empty:
            report["ann_convergence_step"] = -1
        else:
            report["ann_convergence_step"] = int(hits.iloc[0][ts_key]) if ts_key else int(hits.index[0])

    # --- SNN FT Stats ---
    if "snn_ft_reward" in df.columns:
        snn_rewards = df["snn_ft_reward"].dropna()
        report["snn_max_reward"] = float(snn_rewards.max())
        report["snn_final_ma100"] = float(snn_rewards.tail(100).mean())

    # --- Efficiency Stats ---
    if "spike_count" in df.columns:
        report["mean_spikes"] = float(df["spike_count"].mean())
    elif "spike_count_total" in df.columns:
        report["mean_spikes"] = float(df["spike_count_total"].mean())

    # --- Save Report ---
    save_path = Path(log_dir) / "summary_report.json"
    try:
        with open(save_path, "w") as f:
            json.dump(report, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save summary report: {e}")
        
    return report


# ---------------------------------------------------------------------
# Math & Alignment Helpers
# ---------------------------------------------------------------------

def cumulative(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.cumsum(arr) 

def rolling_success_rate(
    rewards: Sequence[float],
    threshold: float = 475.0,
    window: int = 50,
) -> np.ndarray:
    r = np.asarray(rewards, dtype=float)
    hits = (r >= threshold).astype(float)
    out = np.empty_like(hits)

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
    """Aligns sparse evaluation data (like inference energy) to dense steps."""
    out = np.full(n, np.nan, dtype=float)
    if not sparse_data:
        return out
    
    # If we have data, distribute it evenly or place at end if size mismatch
    k = len(sparse_data)
    if k == 1:
        out[-1] = sparse_data[0]
    elif k > 0:
        indices = np.linspace(0, n - 1, k, dtype=int)
        for i, idx in enumerate(indices):
            out[idx] = sparse_data[i]
    return out

def calculate_cumulative_steps(per_episode_data: pd.DataFrame) -> np.ndarray:
    """Calculate cumulative environment steps from per-episode data."""
    if 'total_timesteps' in per_episode_data.columns:
         return per_episode_data['total_timesteps'].values
    elif 'episode_length_steps' in per_episode_data.columns:
        episode_lengths = np.array(per_episode_data['episode_length_steps'].values, dtype=float)
        return np.cumsum(episode_lengths)
    else:
        # Fallback: assume episode length equals reward (for CartPole)
        episode_lengths = np.array(per_episode_data['test_reward'].values, dtype=float)
        return np.cumsum(episode_lengths)

def ensure_dir(p): 
    os.makedirs(p, exist_ok=True)
    return p


# ---------------------------------------------------------------------
# CSV Export / Calculation
# ---------------------------------------------------------------------

def _extract_metric(logger_obj, key: str) -> List[float]:
    """Helper to extract metrics compatible with both Old and New Logger structures."""
    # 1. New Logger (self.history = dict of MetricEvents)
    if hasattr(logger_obj, "history"):
        events = logger_obj.history.get(key, [])
        return [e.value for e in events]
    
    # 2. Old Logger (self._metrics = dict of MetricEvents or simple list)
    if hasattr(logger_obj, "_metric_series"):
        return logger_obj._metric_series(key)
        
    return []

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
    INTELLIGENT MERGE: Preserves energy/latency data from existing logger CSVs.
    """
    out_dir = ensure_dir(out_dir)
    logger_log = logging.getLogger(__name__)

    # 1. Extract Data from Result
    train_rewards = result.get("train_rewards", [])
    test_rewards = result.get("test_rewards", [])
    episode_lengths = result.get("episode_lengths", [])
    train_steps = []
    done_counts = []
    episode_counts = []

    # Try to get spike counts from result dict (populated by surrogate_trainer)
    spike_train = result.get("spike_counts_train", [])
    spike_eval = result.get("spike_counts_eval", [])
    spike_total = []
    spikes_per_step = []
    firing_rate = []
    
    # If list is empty, try to pull from logger object
    logger_obj = result.get("logger")
    
    # Prepare Extraction for Energy/Latency
    train_rollout_energy = []
    total_energy = []
    inference_energy = []
    latency_ms = []
    spike_timing_steps = []
    eval_wall_clock_ms = []
    eval_spike_timing_steps = []
    zs_reward = []
    zs_energy = []
    zs_latency = []
    zs_spikes = []
    zs_sparsity = []
    eval_success_rate = []
    eval_success_count = []
    eval_n_eval_episodes = []
    ft_success_rate = []
    ft_success_count = []
    ft_n_eval_episodes = []
    zs_solved_success_rate = []
    ft_train_reward = []
    ft_eval_reward = []
    ft_energy = []
    ft_latency = []

    if logger_obj:
        # Pull rewards if missing
        if not train_rewards:
            # Check both legacy and new keys
            train_rewards = _extract_metric(logger_obj, "train/rollout_reward") or _extract_metric(logger_obj, "train/reward")
        if not test_rewards:
            # Check both legacy and new keys
            test_rewards = _extract_metric(logger_obj, "eval/episode_reward_ma100") or _extract_metric(logger_obj, "eval/reward")
            if not test_rewards and hasattr(logger_obj, "_last_eval_rewards"):
                test_rewards = list(logger_obj._last_eval_rewards)
            
        # Pull Energy & Latency (The Critical Fix)
        train_rollout_energy = _extract_metric(logger_obj, "energy/train_rollout")
        total_energy = _extract_metric(logger_obj, "energy/total")
        inference_energy = _extract_metric(logger_obj, "energy/inference")
        latency_ms = _extract_metric(logger_obj, "latency/mean_ms")
        spike_timing_steps = _extract_metric(logger_obj, "latency/spike_timing_steps")
        actor_spike_timing_steps = _extract_metric(logger_obj, "latency/actor_spike_timing_steps")
        critic_spike_timing_steps = _extract_metric(logger_obj, "latency/critic_spike_timing_steps")
        critic_eval_spike_timing_steps = _extract_metric(logger_obj, "latency/critic_eval_spike_timing_steps")
        eval_wall_clock_ms = _extract_metric(logger_obj, "latency/eval_wall_clock_ms")
        eval_spike_timing_steps = _extract_metric(logger_obj, "latency/eval_spike_timing_steps")
        eval_critic_tau_mean = _extract_metric(logger_obj, "eval/critic_tau_mean")
        eval_critic_tau_std = _extract_metric(logger_obj, "eval/critic_tau_std")
        eval_critic_value_mean = _extract_metric(logger_obj, "eval/critic_value_mean")
        eval_critic_value_std = _extract_metric(logger_obj, "eval/critic_value_std")
        eval_success_rate = _extract_metric(logger_obj, "eval/success_rate")
        eval_success_count = _extract_metric(logger_obj, "eval/success_count")
        eval_n_eval_episodes = _extract_metric(logger_obj, "eval/n_eval_episodes")

        # Post-Conversion / SNN metrics (for plot_snn_phase)
        zs_reward = _extract_metric(logger_obj, "post_conversion/zero_shot_reward")
        zs_energy = _extract_metric(logger_obj, "post_conversion/inference_energy")
        zs_latency = _extract_metric(logger_obj, "post_conversion/mean_latency")
        zs_spikes = _extract_metric(logger_obj, "post_conversion/total_spikes")
        zs_sparsity = _extract_metric(logger_obj, "post_conversion/sparsity")
        zs_solved_success_rate = _extract_metric(logger_obj, "post_conversion/solved_success_rate")
        ft_train_reward = _extract_metric(logger_obj, "post_conversion_ft/train_reward")
        ft_eval_reward = _extract_metric(logger_obj, "post_conversion_ft/eval_reward")
        ft_energy = _extract_metric(logger_obj, "post_conversion_ft/energy/inference")
        ft_latency = _extract_metric(logger_obj, "post_conversion_ft/train_latency")
        ft_success_rate = _extract_metric(logger_obj, "post_conversion_ft/success_rate")
        ft_success_count = _extract_metric(logger_obj, "post_conversion_ft/success_count")
        ft_n_eval_episodes = _extract_metric(logger_obj, "post_conversion_ft/n_eval_episodes")
        
        # Pull Spikes from Logger if missing in result dict
        if not spike_train:
            spike_train = _extract_metric(logger_obj, "spikes/train_count")
        if not spike_eval:
            # Eval spikes might be sparse, similar to eval rewards
            spike_eval = _extract_metric(logger_obj, "spikes/eval_count")
        
        # --- FIX: Explicitly extract spike_total ---
        if not spike_total:
             spike_total = _extract_metric(logger_obj, "spikes/total") or _extract_metric(logger_obj, "spike_count_total")
        if not spikes_per_step:
            spikes_per_step = _extract_metric(logger_obj, "spikes/per_step")
        if not firing_rate:
            firing_rate = _extract_metric(logger_obj, "spikes/firing_rate")

        # Prefer step-aligned x-axis from logger history if available.
        if hasattr(logger_obj, "history"):
            step_events = logger_obj.history.get("train/rollout_reward", [])
            train_steps = [e.step for e in step_events]
            done_events = logger_obj.history.get("train/rollout_done_count", [])
            done_counts = [e.value for e in done_events]
            episode_events = logger_obj.history.get("train/rollout_episode_count", [])
            episode_counts = [e.value for e in episode_events]

    # If still empty, check if we can load from existing CSV (Persistence)
    csv_path = os.path.join(out_dir, "per_episode_metrics.csv")
    existing_df = pd.DataFrame()
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            # Recover energy if not found in memory
            if not train_rollout_energy and "train_rollout_energy" in existing_df:
                train_rollout_energy = existing_df["train_rollout_energy"].dropna().tolist()
            if not total_energy and "total_energy" in existing_df:
                total_energy = existing_df["total_energy"].dropna().tolist()
            if not inference_energy and "inference_energy" in existing_df:
                inference_energy = existing_df["inference_energy"].dropna().tolist()
            # Recover spikes if not found
            if not spike_train and "spike_count_train" in existing_df:
                spike_train = existing_df["spike_count_train"].dropna().tolist()
            if not spike_eval and "spike_count_eval" in existing_df:
                spike_eval = existing_df["spike_count_eval"].dropna().tolist()
            if "spike_count_total" in existing_df:
                spike_total = existing_df["spike_count_total"].dropna().tolist()
        except Exception:
            pass

    if not train_rewards and not test_rewards:
        logger_log.warning("No rewards found in result dict or CSV. Skipping generation.")
        return {}

    # 2. Alignment
    n = len(train_rewards)
    
    # Pad Test Rewards (Sparse to Dense)
    # If we have fewer test points than train points, pad/align them
    test_rewards_padded = pad_to_len(align_sparse_data(test_rewards, n), n)
    # If we have missing test rewards (NaN), fill forward or copy train (fallback)
    if np.all(np.isnan(test_rewards_padded)):
        test_rewards_padded = pad_to_len(train_rewards, n)

    # 3. Calculations
    cum_train = cumulative(train_rewards)
    sr = rolling_success_rate(test_rewards_padded, threshold=reward_threshold, window=50)

    # Ensure episode_lengths matches n
    if len(episode_lengths) < n:
        pad_val = np.mean(episode_lengths) if episode_lengths else 4096.0 
        episode_lengths = pad_to_len(episode_lengths, n, fill=pad_val)
    elif len(episode_lengths) > n:
        episode_lengths = episode_lengths[:n]

    train_reward_unscaled = _extract_metric(logger_obj, "train/rollout_reward_raw")
    use_train_steps = len(train_steps) == n
    if use_train_steps:
        total_timesteps = np.asarray(train_steps, dtype=float)
        episode_length_steps = np.diff(np.concatenate(([0.0], total_timesteps)))
    else:
        episode_length_steps = episode_lengths
        total_timesteps = np.cumsum(episode_length_steps)
    if len(done_counts) == n and use_train_steps:
        done_arr = np.asarray(done_counts, dtype=float)
        step_deltas = np.diff(np.concatenate(([0.0], total_timesteps)))
        avg_episode_len = np.divide(
            step_deltas,
            done_arr,
            out=np.full(n, np.nan),
            where=done_arr > 0,
        )

    # 4. Construct DataFrame
    df = pd.DataFrame({
        "update": np.arange(1, n + 1),
        "train_reward": train_rewards,
        "test_reward": test_rewards_padded,
        "success_hit": (test_rewards_padded >= reward_threshold).astype(int),
        "total_cumulative_train_reward": cum_train,
        "episode_length_steps": episode_length_steps,
    })
    if train_reward_unscaled:
        df["train_reward_unscaled"] = pad_to_len(train_reward_unscaled, n)
    if done_counts:
        df["train_rollout_done_count"] = pad_to_len(done_counts, n)
    if episode_counts:
        df["train_rollout_episode_count"] = pad_to_len(episode_counts, n)

    # Add Energy/Latency Columns (if available)
    if train_rollout_energy:
        df["train_rollout_energy"] = pad_to_len(train_rollout_energy, n)
    if total_energy:
        df["total_energy"] = pad_to_len(total_energy, n)
    if inference_energy:
        # Inference is sparse, use align helper
        df["inference_energy"] = align_sparse_data(inference_energy, n)
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
        df["eval/success_rate"] = align_sparse_data(eval_success_rate, n)
    if eval_success_count:
        df["eval/success_count"] = align_sparse_data(eval_success_count, n)
    if eval_n_eval_episodes:
        df["eval/n_eval_episodes"] = align_sparse_data(eval_n_eval_episodes, n)

    def _align_tail(values: list, length: int) -> np.ndarray:
        out = np.full(length, np.nan, dtype=float)
        if not values:
            return out
        k = min(len(values), length)
        out[length - k:] = np.asarray(values[-k:], dtype=float)
        return out

    # Post-Conversion / SNN Columns (aligned to tail for plotting)
    if zs_reward:
        df["post_conversion/zero_shot_reward"] = align_sparse_data(zs_reward, n)
    if zs_energy:
        df["post_conversion/inference_energy"] = align_sparse_data(zs_energy, n)
    if zs_latency:
        df["post_conversion/mean_latency"] = align_sparse_data(zs_latency, n)
    if zs_spikes:
        df["post_conversion/total_spikes"] = align_sparse_data(zs_spikes, n)
    if zs_sparsity:
        df["post_conversion/sparsity"] = align_sparse_data(zs_sparsity, n)
    if zs_solved_success_rate:
        df["post_conversion/solved_success_rate"] = align_sparse_data(zs_solved_success_rate, n)
    if ft_train_reward:
        df["post_conversion_ft/train_reward"] = _align_tail(ft_train_reward, n)
    if ft_eval_reward:
        df["post_conversion_ft/eval_reward"] = _align_tail(ft_eval_reward, n)
    if ft_energy:
        df["post_conversion_ft/energy/inference"] = _align_tail(ft_energy, n)
    if ft_latency:
        df["post_conversion_ft/train_latency"] = _align_tail(ft_latency, n)
    if ft_success_rate:
        df["post_conversion_ft/success_rate"] = _align_tail(ft_success_rate, n)
    if ft_success_count:
        df["post_conversion_ft/success_count"] = _align_tail(ft_success_count, n)
    if ft_n_eval_episodes:
        df["post_conversion_ft/n_eval_episodes"] = _align_tail(ft_n_eval_episodes, n)

    # --- ADDED: Add Spike Columns to DataFrame ---
    if spike_train:
        df["spike_count_train"] = pad_to_len(spike_train, n)
    
    if spike_eval:
         # Align eval spikes just like eval rewards
        spike_eval_padded = align_sparse_data(spike_eval, n)
        df["spike_count_eval"] = pad_to_len(spike_eval_padded, n)
        
    if spike_total:
        df["spike_count_total"] = pad_to_len(spike_total, n)
    if spikes_per_step:
        df["spikes/per_step"] = pad_to_len(spikes_per_step, n)
    if firing_rate:
        df["spikes/firing_rate"] = pad_to_len(firing_rate, n)
    
    # Total Timesteps Calculation
    df["total_timesteps"] = total_timesteps

    # 5. Save
    atomic_save_csv(df, csv_path)
    logger_log.info(f"Saved merged metrics to {csv_path}")

    return {
        "train_rewards": train_rewards,
        "test_rewards": test_rewards,
        "sr": sr,
        "cum_train": cum_train
    }
