"""
High-level reporting and dashboard generation for PPO experiments.

This module acts as the orchestration layer for visualization. It:
  1. Loads raw metric CSVs.
  2. Infers experiment context (ANN vs SNN, Hyperparameters).
  3. Declaratively executes a suite of standard plots.
  4. Injects metadata subtitles into plots for publication readiness.
"""
from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any, Sequence, Union, Tuple

import numpy as np
import pandas as pd

from src.utils.metrics import load_training_data
from src.utils import plotting as plotting_mod
from src.utils.plotting import (
    plot_success_rate_vs_steps,
    plot_energy_vs_steps,
    plot_spikes_vs_steps,
    plot_latency_vs_steps,
    plot_energy_vs_spikes,
    plot_reward_vs_spikes,
    plot_latency_vs_spikes,
    plot_conversion_validation,
    plot_train_rollout_vs_steps,
    plot_eval_return_vs_steps,
    plot_intra_episode_values,
    plot_actor_readout_validation,
    plot_eval_checkpoint_value_trend,
    plot_timing_critic_dynamics,
    plot_timing_critic_macro_dynamics,
    plot_output_readout_validation,
    plot_snn_phase,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Experiment Inference & Metadata
# =============================================================================

def _extract_numeric_events(raw: Dict[str, Any], key: str) -> pd.DataFrame:
    events = raw.get(key, [])
    if not events:
        return pd.DataFrame(columns=["step", "iteration", "value"])
    if isinstance(events[0], dict):
        rows = []
        for e in events:
            rows.append(
                {
                    "step": e.get("step"),
                    "iteration": e.get("iteration"),
                    "value": e.get("value"),
                }
            )
        out = pd.DataFrame(rows)
    else:
        out = pd.DataFrame(
            {
                "step": [None] * len(events),
                "iteration": [None] * len(events),
                "value": events,
            }
        )
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["step"] = pd.to_numeric(out["step"], errors="coerce")
    out["iteration"] = pd.to_numeric(out["iteration"], errors="coerce")
    return out


def _hydrate_alias_columns(per_episode: pd.DataFrame, log_dir: Path) -> pd.DataFrame:
    """
    Fill plotting-critical columns from metrics_raw.json or deterministic aliases.
    Keeps existing values intact and only fills missing/empty columns.
    """
    if per_episode.empty:
        return per_episode

    df = per_episode.copy()
    raw_path = log_dir / "metrics_raw.json"
    raw = {}
    if raw_path.exists():
        try:
            with open(raw_path, "r") as f:
                raw = json.load(f)
        except Exception:
            logger.warning("Failed to read metrics_raw.json for alias hydration.")

    def _try_align(events: pd.DataFrame, align_col: str, df_col: str) -> Optional[pd.Series]:
        """Try to align event values by matching a column."""
        if align_col not in events.columns or not events[align_col].notna().any():
            return None
        if df_col not in df.columns:
            return None

        mapping = (
            events.dropna(subset=[align_col, "value"])
            .drop_duplicates(subset=[align_col], keep="last")
            .set_index(align_col)["value"]
            .to_dict()
        )
        series = pd.to_numeric(df[df_col], errors="coerce")
        return series.map(mapping)

    def fill_from_key(target_col: str, key: str):
        events = _extract_numeric_events(raw, key)
        if events.empty:
            return

        existing = df[target_col] if target_col in df.columns else pd.Series(np.nan, index=df.index)
        filled = existing.copy()

        # Strategy 1: direct length alignment
        if len(events) == len(df):
            vals = events["value"].reset_index(drop=True)
            filled = filled.reset_index(drop=True).where(~filled.reset_index(drop=True).isna(), vals)
            filled.index = df.index
        else:
            # Strategy 2: align by update/iteration
            aligned = _try_align(events, "iteration", "update")
            if aligned is not None:
                filled = filled.where(~filled.isna(), aligned)
            else:
                # Strategy 3: align by total_timesteps/step
                aligned = _try_align(events, "step", "total_timesteps")
                if aligned is not None:
                    filled = filled.where(~filled.isna(), aligned)

        if target_col in df.columns or not filled.isna().any():
            df[target_col] = filled

    aliases = {
        "spikes/per_step": ["spikes/per_step"],
        "spikes/firing_rate": ["spikes/firing_rate"],
        "spikes/total": ["spikes/total", "spike_count_total"],
        "spikes/eval_total": ["spikes/eval_total", "eval/spikes"],
        "eval/spikes": ["eval/spikes", "spikes/eval_total"],
        "eval/spikes_per_step": ["eval/spikes_per_step"],
        "spikes/actor_per_step": ["spikes/actor_per_step", "spikes/actor"],
        "spikes/critic_per_step": ["spikes/critic_per_step", "spikes/critic"],
        "latency/spike_timing_steps": ["latency/spike_timing_steps"],
        "latency/actor_spike_timing_steps": ["latency/actor_spike_timing_steps"],
        "latency/critic_spike_timing_steps": ["latency/critic_spike_timing_steps"],
        "latency/mean_ms": ["latency/mean_ms", "latency_mean_ms"],
        "latency_mean_ms": ["latency_mean_ms", "latency/mean_ms"],
        "eval/success_rate": ["eval/success_rate", "success_rate"],
        "eval/reward": ["eval/current_reward", "eval/rolling_reward", "test_reward"],
    }
    for target, sources in aliases.items():
        missing = target not in df.columns or df[target].isna().all()
        if not missing:
            continue
        # Prefer existing dataframe columns before raw JSON keys.
        filled = False
        for src in sources:
            if src in df.columns and df[src].notna().any():
                df[target] = pd.to_numeric(df[src], errors="coerce")
                filled = True
                break
        if filled:
            continue
        for src in sources:
            if src in raw:
                fill_from_key(target, src)
                if target in df.columns and df[target].notna().any():
                    break

    # Deterministic fallback for per-step spikes when only totals are present.
    if ("spikes/per_step" not in df.columns or df["spikes/per_step"].isna().all()):
        if "spike_count_total" in df.columns:
            denom_col = "episode_length_steps" if "episode_length_steps" in df.columns else None
            if denom_col:
                denom = pd.to_numeric(df[denom_col], errors="coerce").replace(0, np.nan)
                df["spikes/per_step"] = pd.to_numeric(df["spike_count_total"], errors="coerce") / denom
            else:
                df["spikes/per_step"] = pd.to_numeric(df["spike_count_total"], errors="coerce")

    return df


def _slice_post_conversion_phase(per_episode: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows at/after the zero-shot conversion marker for ANN2SNN dashboards.
    """
    if per_episode.empty:
        return per_episode

    marker_cols = [
        "post_conversion/zero_shot_reward",
        "post_conversion/inference_energy",
        "post_conversion/total_spikes",
        "post_conversion/mean_latency",
    ]
    available = [c for c in marker_cols if c in per_episode.columns]
    if not available:
        return per_episode

    marker_mask = pd.Series(False, index=per_episode.index)
    for c in available:
        marker_mask = marker_mask | pd.to_numeric(per_episode[c], errors="coerce").notna()

    if not marker_mask.any():
        return per_episode

    first_idx = marker_mask[marker_mask].index[0]
    # Use iloc boundary to preserve monotonic row order even when index is non-consecutive.
    start_pos = int(per_episode.index.get_loc(first_idx))
    return per_episode.iloc[start_pos:].copy()


def _concat_seeds_for_plotting(per_episode_list: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate seed dataframes and tag with a seed id for pooled analyses."""
    frames: List[pd.DataFrame] = []
    for sid, seed_df in enumerate(per_episode_list):
        if not isinstance(seed_df, pd.DataFrame) or seed_df.empty:
            continue
        tmp = seed_df.copy().reset_index(drop=True)
        tmp["__seed_id"] = int(sid)
        frames.append(tmp)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def _as_1d_float_array(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    return arr.reshape(-1).astype(float)


def _as_1d_int_array(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.size == 0:
        return np.asarray([], dtype=int)
    return arr.reshape(-1).astype(int)


def _load_validation_trace_file(log_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load compact validation traces saved per seed for post-hoc multi-seed dashboards.
    """
    trace_path = log_dir / "validation_data.npz"
    if not trace_path.exists():
        return {}
    try:
        out: Dict[str, np.ndarray] = {}
        with np.load(trace_path, allow_pickle=False) as data:
            for key in data.files:
                out[key] = np.asarray(data[key]).reshape(-1)
        return out
    except Exception:
        logger.warning(f"Failed to load validation traces from {trace_path}")
        return {}


def _looks_like_multiseed_trace(x: Any) -> bool:
    if not isinstance(x, (list, tuple)) or len(x) == 0:
        return False
    first = x[0]
    return isinstance(first, (list, tuple, np.ndarray, pd.Series))


def _merge_multiseed_validation_data(
    validation_data: Optional[Dict[str, Any]],
    log_paths: Sequence[Path],
) -> Optional[Dict[str, Any]]:
    """
    For multi-seed dashboards, gather per-seed validation traces (if available) and
    return an aggregated payload consumable by plotting functions.
    """
    if len(log_paths) <= 1:
        return validation_data

    # Already aggregated by caller: keep as-is.
    if isinstance(validation_data, dict):
        if _looks_like_multiseed_trace(validation_data.get("critic_timings")):
            return validation_data
        if _looks_like_multiseed_trace(validation_data.get("critic_values_single_episode")):
            return validation_data
        if _looks_like_multiseed_trace(validation_data.get("intra_episode_values")):
            return validation_data

    per_seed_validation: List[Dict[str, np.ndarray]] = []
    for log_path in log_paths:
        traces = _load_validation_trace_file(log_path)
        if traces:
            per_seed_validation.append(traces)

    # If files are absent, fallback to any provided single-seed payload.
    if len(per_seed_validation) < 2:
        return validation_data

    merged: Dict[str, Any] = dict(validation_data or {})

    # Intra-episode profiles: prefer deterministic single-episode trace when present.
    for source_key, profile_mode in (
        ("critic_values_single_episode", "multi_seed_single_episode"),
        ("intra_episode_values", "multi_seed_profile"),
        ("critic_values", "multi_seed_profile"),
    ):
        traces = [
            _as_1d_float_array(seed[source_key])
            for seed in per_seed_validation
            if source_key in seed and np.asarray(seed[source_key]).size > 0
        ]
        traces = [t for t in traces if t.size > 0]
        if len(traces) >= 2:
            if source_key == "critic_values_single_episode":
                merged["critic_values_single_episode"] = traces
            else:
                merged["intra_episode_values"] = traces
            merged["intra_episode_values_profile"] = profile_mode
            break

    # Timing micro traces: keep seed alignment for tau/value/episode index.
    timing_seeds: List[np.ndarray] = []
    value_seeds: List[np.ndarray] = []
    episode_idx_seeds: List[np.ndarray] = []
    for seed in per_seed_validation:
        if "critic_timings" not in seed or "critic_values" not in seed:
            continue
        tau = _as_1d_float_array(seed["critic_timings"])
        val = _as_1d_float_array(seed["critic_values"])
        n = min(tau.size, val.size)
        if n == 0:
            continue
        timing_seeds.append(tau[:n])
        value_seeds.append(val[:n])
        if "step_traces_episode_index" in seed:
            epi = _as_1d_int_array(seed["step_traces_episode_index"])
            if epi.size >= n:
                episode_idx_seeds.append(epi[:n])
            else:
                episode_idx_seeds.append(np.asarray([], dtype=int))
        else:
            episode_idx_seeds.append(np.asarray([], dtype=int))

    if len(timing_seeds) >= 2:
        merged["critic_timings"] = timing_seeds
        merged["critic_values"] = value_seeds
        if all(ep.size > 0 for ep in episode_idx_seeds):
            merged.setdefault("step_traces", {})
            if isinstance(merged["step_traces"], dict):
                merged["step_traces"]["episode_index"] = episode_idx_seeds

    return merged if merged else None


def _has_spike_activity(per_episode_data: pd.DataFrame) -> bool:
    spike_cols = (
        "spike_count_total",
        "spike_count_train",
        "spike_count_eval",
        "post_conversion/total_spikes",
        "spikes/total",
        "spikes/per_step",
        "spikes_total",
        "eval/spikes",
    )
    for col in spike_cols:
        if col in per_episode_data.columns:
            series = per_episode_data[col].fillna(0)
            if (series.abs().sum() > 0):
                return True
    return False

def infer_experiment_type(per_episode_data: pd.DataFrame, log_dir: Optional[Path] = None) -> str:
    """
    Heuristically infers the experiment type (ANN vs SNN).
    Checks DataFrame columns first, then falls back to metrics_raw.json if available.
    """
    # 1. Check CSV Columns
    spike_cols = {
        "spike_count_total",
        "spike_count_train",
        "spike_count_eval",
        "post_conversion/total_spikes",
        "spikes/total",
        "spikes_total",
        "eval/spikes"
    }
    
    present_cols = list(spike_cols.intersection(per_episode_data.columns))
    if present_cols:
        total_activity = per_episode_data[present_cols].fillna(0).sum().sum()
        if total_activity > 0:
            return "SNN"

    # 2. Fallback: Check metrics_raw.json (in case CSV dropped the columns)
    if log_dir:
        json_path = log_dir / "metrics_raw.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                    # Check if any spike key exists AND has non-zero activity
                    for k in spike_cols:
                        if k in data:
                            # data[k] is a list of event dicts (from PPOLogger)
                            events = data[k]
                            if not events:
                                continue
                                
                            # Calculate total spikes recorded
                            # Handle both list of dicts (new logger) and list of floats (legacy)
                            if isinstance(events[0], dict):
                                total = sum(e.get("value", 0) for e in events)
                            else:
                                total = sum(events)
                                
                            if total > 0:
                                return "SNN"
                            
            except Exception:
                pass

    if "total_energy" in per_episode_data.columns:
        return "ANN (Instrumented)"
        
    return "ANN"


def extract_experiment_metadata(
    per_episode_data: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Extracts key hyperparameters for plot subtitles.
    """
    meta: Dict[str, str] = {}

    if config:
        meta["env"] = str(config.get("env", {}).get("id", config.get("env_id", "Unknown")))
        meta["seed"] = str(config.get("env_seed", "?"))

        snn_cfg = config.get("snn", {})
        if snn_cfg:
            if "T" in snn_cfg: meta["T"] = str(snn_cfg["T"])
            if "beta" in snn_cfg: meta["β"] = str(snn_cfg["beta"])

        mode = config.get("model", {}).get("mode", "unknown")
        meta["mode"] = mode

    if "critic_time_steps" in per_episode_data.columns:
        meta["mode"] = "Timing Critic"

    return meta


def format_subtitle(exp_type: str, meta: Dict[str, str]) -> str:
    parts = [f"Type: {exp_type}"]
    if "mode" in meta: parts.append(f"Mode: {meta['mode']}")
    if "T" in meta: parts.append(f"T={meta['T']}")
    if "β" in meta: parts.append(f"β={meta['β']}")
    if "seed" in meta: parts.append(f"Seed={meta['seed']}")
    return " | ".join(parts)


# =============================================================================
# Plot Specification Strategy
# =============================================================================

@dataclass
class PlotSpec:
    name: str
    filename: str
    plot_fn: Callable
    condition: Optional[Callable[[pd.DataFrame], bool]] = None
    kwargs_fn: Optional[Callable[[pd.DataFrame], dict]] = None


STANDARD_PLOTS: List[PlotSpec] = [
    PlotSpec(
        name="Training Dynamics",
        filename="plot_01_train_rollout_vs_steps.png",
        plot_fn=plot_train_rollout_vs_steps,
        condition=lambda df: "train_reward" in df.columns,
    ),
    PlotSpec(
        name="Evaluation Return",
        filename="plot_02_eval_return_vs_steps.png",
        plot_fn=plot_eval_return_vs_steps,
        condition=lambda df: "test_reward" in df.columns,
    ),
    PlotSpec(
        name="Success Rate",
        filename="plot_02_success_rate_vs_steps.png",
        plot_fn=plot_success_rate_vs_steps,
        condition=lambda df: (
            "test_reward" in df.columns
            or "eval/success_rate" in df.columns
            or "success_rate" in df.columns
            or "eval_success_rate" in df.columns
        ),
    ),
    PlotSpec(
        name="Energy Consumption",
        filename="plot_03_energy_vs_steps.png",
        plot_fn=plot_energy_vs_steps,
        condition=lambda df: (
            "total_energy" in df.columns
            or "total_dynamic_energy" in df.columns
            or "inference_energy" in df.columns
            or "inference_dynamic_energy" in df.columns
            or "train_full_update_energy" in df.columns
            or "train_full_update_dynamic_energy" in df.columns
            or "train_rollout_energy" in df.columns
        ),
    ),
    PlotSpec(
        name="Spike Activity",
        filename="plot_04_spikes_vs_steps.png",
        plot_fn=plot_spikes_vs_steps,
        condition=_has_spike_activity,
        kwargs_fn=lambda df: {
            "spike_col": (
                "spikes/per_step" if "spikes/per_step" in df.columns else
                "spikes/firing_rate" if "spikes/firing_rate" in df.columns else
                "spikes/total" if "spikes/total" in df.columns else
                "spike_count_total"
            )
        }
    ),
    PlotSpec(
        name="Latency",
        filename="plot_05_latency_vs_steps.png",
        plot_fn=plot_latency_vs_steps,
        condition=lambda df: any(
            c in df.columns
            for c in (
                "latency_mean_ms",
                "latency/mean_ms",
                "latency_ms",
                "eval/latency",
                "latency/spike_timing_steps",
                "latency/eval_wall_clock_ms",
                "latency/eval_spike_timing_steps",
            )
        ),
    ),
    PlotSpec(
        name="Energy vs Spikes",
        filename="plot_06_energy_vs_spikes.png",
        plot_fn=plot_energy_vs_spikes,
        condition=lambda df: _has_spike_activity(df)
        and any(
            c in df.columns
            for c in (
                "inference_dynamic_energy",
                "inference_energy",
                "train_full_update_dynamic_energy",
                "train_full_update_energy",
                "train_rollout_dynamic_energy",
                "train_rollout_energy",
                "total_dynamic_energy",
                "total_energy",
            )
        ),
    ),
    PlotSpec(
        name="Reward vs Spikes",
        filename="plot_07_reward_vs_spikes.png",
        plot_fn=plot_reward_vs_spikes,
        condition=lambda df: _has_spike_activity(df)
        and any(c in df.columns for c in ("test_reward", "train_reward")),
    ),
    PlotSpec(
        name="Latency vs Spikes",
        filename="plot_08_latency_vs_spikes.png",
        plot_fn=plot_latency_vs_spikes,
        condition=lambda df: _has_spike_activity(df)
        and any(c in df.columns for c in ("latency_mean_ms", "latency/mean_ms", "latency/spike_timing_steps")),
        kwargs_fn=lambda df: {
            "component": "Actor",
        },
    ),
]

ANN2SNN_ACTOR_CONVERSION_PLOTS: List[PlotSpec] = [
    PlotSpec(
        name="Spike Activity",
        filename="plot_04_spikes_vs_steps.png",
        plot_fn=plot_spikes_vs_steps,
        condition=_has_spike_activity,
    ),
    PlotSpec(
        name="Latency",
        filename="plot_05_latency_vs_steps.png",
        plot_fn=plot_latency_vs_steps,
        condition=lambda df: any(
            c in df.columns
            for c in (
                "latency_mean_ms",
                "latency/mean_ms",
                "latency/spike_timing_steps",
                "latency/actor_spike_timing_steps",
                "latency/critic_spike_timing_steps",
            )
        ),
    ),
    PlotSpec(
        name="Energy vs Spikes",
        filename="plot_06_energy_vs_spikes.png",
        plot_fn=plot_energy_vs_spikes,
        condition=lambda df: _has_spike_activity(df)
        and any(
            c in df.columns
            for c in (
                "inference_dynamic_energy",
                "inference_energy",
                "train_full_update_dynamic_energy",
                "train_full_update_energy",
                "train_rollout_dynamic_energy",
                "train_rollout_energy",
                "total_dynamic_energy",
                "total_energy",
            )
        ),
    ),
    PlotSpec(
        name="Reward vs Spikes",
        filename="plot_07_reward_vs_spikes.png",
        plot_fn=plot_reward_vs_spikes,
        condition=lambda df: _has_spike_activity(df)
        and any(c in df.columns for c in ("test_reward", "train_reward")),
    ),
    PlotSpec(
        name="Latency vs Spikes",
        filename="plot_08_latency_vs_spikes.png",
        plot_fn=plot_latency_vs_spikes,
        condition=lambda df: _has_spike_activity(df)
        and any(c in df.columns for c in ("latency/actor_spike_timing_steps", "latency/spike_timing_steps", "latency/mean_ms", "latency_mean_ms")),
        kwargs_fn=lambda df: {"component": "Actor"},
    ),
]

ANN2SNN_BOTH_CONVERSION_PLOTS: List[PlotSpec] = ANN2SNN_ACTOR_CONVERSION_PLOTS + [
    PlotSpec(
        name="Latency vs Spikes (Critic)",
        filename="plot_08b_latency_vs_spikes_critic.png",
        plot_fn=plot_latency_vs_spikes,
        condition=lambda df: _has_spike_activity(df)
        and any(
            c in df.columns
            for c in (
                "latency/critic_spike_timing_steps",
                "latency/mean_ms",
                "latency_mean_ms",
            )
        ),
        kwargs_fn=lambda df: {"component": "Critic"},
    ),
]


# =============================================================================
# Plotting Engine
# =============================================================================

def run_plot_specs(
    specs: List[PlotSpec],
    per_episode_data: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    output_dir: Path,
    title_prefix: str,
    threshold: float,
    subtitle: str,
    config: Optional[Dict[str, Any]] = None,
):
    df_ref = per_episode_data[0] if isinstance(per_episode_data, Sequence) and not isinstance(per_episode_data, pd.DataFrame) else per_episode_data
    is_multiseed = isinstance(per_episode_data, Sequence) and not isinstance(per_episode_data, pd.DataFrame)
    multiseed_capable = {plot_eval_return_vs_steps, plot_success_rate_vs_steps, plot_spikes_vs_steps}
    pooled_multiseed_capable = {plot_latency_vs_steps, plot_energy_vs_spikes, plot_reward_vs_spikes, plot_latency_vs_spikes}
    pooled_df = _concat_seeds_for_plotting(per_episode_data) if is_multiseed else pd.DataFrame()
    for spec in specs:
        if spec.condition and not spec.condition(df_ref):
            continue

        kwargs = {
            "title": f"{title_prefix} – {spec.name}",
            "subtitle": subtitle,
        }
        if "config" in spec.plot_fn.__code__.co_varnames:
            kwargs["config"] = config
        if "exp_name" in spec.plot_fn.__code__.co_varnames and config:
            kwargs["exp_name"] = str(
                config.get("model", {}).get(
                    "mode",
                    config.get("run_name", "experiment"),
                )
            )
        if "env_name" in spec.plot_fn.__code__.co_varnames and config:
            kwargs["env_name"] = str(
                config.get("env", {}).get(
                    "id",
                    config.get("env_id", "Unknown Env"),
                )
            )
        if spec.name == "Success Rate" and config:
            eval_eps = config.get("ppo", {}).get("eval_episodes")
            if eval_eps is not None:
                kwargs["eval_window"] = int(eval_eps)

        if "threshold" in spec.plot_fn.__code__.co_varnames:
            kwargs["threshold"] = threshold

        if spec.kwargs_fn:
            kwargs.update(spec.kwargs_fn(df_ref))

        save_path = output_dir / spec.filename
        if is_multiseed and spec.plot_fn in multiseed_capable:
            plot_data = per_episode_data
            kwargs["multi_seed"] = True
            kwargs["num_seeds"] = len(per_episode_data)
        elif is_multiseed and spec.plot_fn in pooled_multiseed_capable and not pooled_df.empty:
            plot_data = pooled_df
            kwargs["multi_seed"] = True
            kwargs["num_seeds"] = len(per_episode_data)
        else:
            plot_data = df_ref
        try:
            spec.plot_fn(plot_data, str(save_path), **kwargs)
            logger.debug(f"Generated: {save_path.name}")
        except Exception as e:
            logger.error(f"Failed to generate plot '{spec.name}': {e}")
            logger.exception("Plot generation traceback:")


def _export_energy_audit_table(
    per_episode_data: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Export an audit-oriented energy summary table for thesis reporting."""
    if isinstance(per_episode_data, Sequence) and not isinstance(per_episode_data, pd.DataFrame):
        if not per_episode_data:
            return
        df = per_episode_data[0].copy()
    else:
        df = per_episode_data.copy()  # type: ignore[assignment]
    if df.empty:
        return

    def _add_metric(rows: List[Dict[str, Any]], metric_id: str, source_col: str, unit: str, normalization: str):
        if source_col not in df.columns:
            return
        s = pd.to_numeric(df[source_col], errors="coerce").dropna()
        if s.empty:
            return
        rows.append(
            {
                "metric_id": metric_id,
                "source_col": source_col,
                "unit": unit,
                "normalization": normalization,
                "count": int(s.shape[0]),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=0)),
                "min": float(s.min()),
                "max": float(s.max()),
                "final": float(s.iloc[-1]),
            }
        )

    rows: List[Dict[str, Any]] = []
    _add_metric(rows, "idle_power", "energy_idle_power_watts", "W", "none")
    _add_metric(rows, "train_rollout_energy_raw", "train_rollout_energy", "J", "per-update")
    _add_metric(rows, "train_rollout_energy_dynamic", "train_rollout_dynamic_energy", "J", "per-update")
    _add_metric(rows, "train_full_update_energy_raw", "train_full_update_energy", "J", "per-update")
    _add_metric(rows, "train_full_update_energy_dynamic", "train_full_update_dynamic_energy", "J", "per-update")
    _add_metric(rows, "inference_energy_raw", "inference_energy", "J", "per-eval-window")
    _add_metric(rows, "inference_energy_dynamic", "inference_dynamic_energy", "J", "per-eval-window")
    _add_metric(rows, "total_energy_raw", "total_energy", "J", "cumulative")
    _add_metric(rows, "total_energy_dynamic", "total_dynamic_energy", "J", "cumulative")

    # Symmetric denominator policy for train/inference:
    # both are normalized by environment steps (J / environment-step).
    train_denom = None
    train_norm_label = None
    if "train_rollout_steps" in df.columns:
        train_denom = pd.to_numeric(df["train_rollout_steps"], errors="coerce").replace(0, np.nan)
        train_norm_label = "train_rollout_steps"
    elif "episode_length_steps" in df.columns:
        train_denom = pd.to_numeric(df["episode_length_steps"], errors="coerce").replace(0, np.nan)
        train_norm_label = "episode_length_steps"

    infer_denom = None
    infer_norm_label = None
    if "eval/n_eval_episodes" in df.columns and "eval_episode_length" in df.columns:
        n_eval = pd.to_numeric(df["eval/n_eval_episodes"], errors="coerce")
        eval_len = pd.to_numeric(df["eval_episode_length"], errors="coerce")
        infer_denom = (n_eval * eval_len).replace(0, np.nan)
        infer_norm_label = "eval_n_eval_episodes * eval_episode_length"
    elif "episode_length_steps" in df.columns:
        infer_denom = pd.to_numeric(df["episode_length_steps"], errors="coerce").replace(0, np.nan)
        infer_norm_label = "episode_length_steps"

    train_jps_raw_col = None
    train_jps_dyn_col = None
    infer_jps_raw_col = None
    infer_jps_dyn_col = None

    if train_denom is not None and "train_rollout_energy" in df.columns:
        df["energy/train_joules_per_step_raw_audit"] = pd.to_numeric(df["train_rollout_energy"], errors="coerce") / train_denom
        train_jps_raw_col = "energy/train_joules_per_step_raw_audit"
        _add_metric(rows, "train_energy_per_step_raw", train_jps_raw_col, "J/step", str(train_norm_label))
    if train_denom is not None and "train_rollout_dynamic_energy" in df.columns:
        df["energy/train_joules_per_step_dynamic_audit"] = pd.to_numeric(df["train_rollout_dynamic_energy"], errors="coerce") / train_denom
        train_jps_dyn_col = "energy/train_joules_per_step_dynamic_audit"
        _add_metric(rows, "train_energy_per_step_dynamic", train_jps_dyn_col, "J/step", str(train_norm_label))

    if infer_denom is not None and "inference_energy" in df.columns:
        df["energy/inference_joules_per_step_raw_audit"] = pd.to_numeric(df["inference_energy"], errors="coerce") / infer_denom
        infer_jps_raw_col = "energy/inference_joules_per_step_raw_audit"
        _add_metric(rows, "inference_energy_per_step_raw", infer_jps_raw_col, "J/step", str(infer_norm_label))
    if infer_denom is not None and "inference_dynamic_energy" in df.columns:
        df["energy/inference_joules_per_step_dynamic_audit"] = pd.to_numeric(df["inference_dynamic_energy"], errors="coerce") / infer_denom
        infer_jps_dyn_col = "energy/inference_joules_per_step_dynamic_audit"
        _add_metric(rows, "inference_energy_per_step_dynamic", infer_jps_dyn_col, "J/step", str(infer_norm_label))

    # Backward-compatible metric IDs (prefer dynamic if present, else raw).
    if train_jps_dyn_col is not None:
        _add_metric(rows, "train_energy_per_step", train_jps_dyn_col, "J/step", str(train_norm_label))
    elif train_jps_raw_col is not None:
        _add_metric(rows, "train_energy_per_step", train_jps_raw_col, "J/step", str(train_norm_label))

    if infer_jps_dyn_col is not None:
        _add_metric(rows, "inference_energy_per_step", infer_jps_dyn_col, "J/step", str(infer_norm_label))
    elif infer_jps_raw_col is not None:
        _add_metric(rows, "inference_energy_per_step", infer_jps_raw_col, "J/step", str(infer_norm_label))

    # Inference/Train ratios (same scope when possible).
    if train_jps_raw_col is not None and infer_jps_raw_col is not None:
        denom = pd.to_numeric(df[train_jps_raw_col], errors="coerce").replace(0, np.nan)
        df["energy/inference_train_ratio_raw_audit"] = pd.to_numeric(df[infer_jps_raw_col], errors="coerce") / denom
        _add_metric(rows, "inference_train_ratio_raw", "energy/inference_train_ratio_raw_audit", "ratio", "raw_scope")
    if train_jps_dyn_col is not None and infer_jps_dyn_col is not None:
        denom = pd.to_numeric(df[train_jps_dyn_col], errors="coerce").replace(0, np.nan)
        df["energy/inference_train_ratio_dynamic_audit"] = pd.to_numeric(df[infer_jps_dyn_col], errors="coerce") / denom
        _add_metric(rows, "inference_train_ratio_dynamic", "energy/inference_train_ratio_dynamic_audit", "ratio", "dynamic_scope")
    if "energy/inference_train_ratio_dynamic_audit" in df.columns:
        _add_metric(rows, "inference_train_ratio", "energy/inference_train_ratio_dynamic_audit", "ratio", "dynamic_scope")
    elif "energy/inference_train_ratio_raw_audit" in df.columns:
        _add_metric(rows, "inference_train_ratio", "energy/inference_train_ratio_raw_audit", "ratio", "raw_scope")

    if rows:
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(output_dir / "table_energy_audit_summary.csv", index=False)

    methodology = {
        "measurement_scope": "GPU-side software telemetry from NVML sampling",
        "raw_energy_definition": "Integrated GPU power over measurement window",
        "dynamic_energy_definition": "max(0, total_joules - idle_power_watts * duration_seconds)",
        "training_window": "Full PPO update (rollout + optimization), with rollout-only logged separately",
        "primary_energy_scope_for_reporting": "dynamic (raw reported as appendix/audit)",
        "symmetric_normalization_target": "J / environment-step for both train and inference",
        "train_normalization": str(train_norm_label) if train_norm_label else "not_available",
        "inference_normalization": str(infer_norm_label) if infer_norm_label else "not_available",
        "execution_mode_guidance": "Use same device, precision, and similar batching/parallelism for train-vs-inference comparisons",
        "notes": "Values are not whole-system wall power (CPU/DRAM/PSU excluded).",
    }
    if config:
        methodology["env_id"] = str(config.get("env", {}).get("id", config.get("env_id", "Unknown")))
        methodology["mode"] = str(config.get("model", {}).get("mode", "unknown"))
        methodology["device"] = str(
            config.get("training", {}).get("device", config.get("device", "unknown"))
        )
        methodology["precision"] = str(
            config.get("training", {}).get("precision", config.get("precision", "unknown"))
        )
        methodology["parallel_envs"] = str(
            config.get("env", {}).get("n_envs", config.get("training", {}).get("n_envs", "unknown"))
        )
    pd.DataFrame([methodology]).to_csv(output_dir / "table_energy_methodology.csv", index=False)


# =============================================================================
# Public API
# =============================================================================

def create_training_dashboard(
    log_dir: Union[str, Sequence[str]],
    output_dir: str,
    title_prefix: str = "PPO Training",
    threshold: float = 475.0,
    validation_data: Optional[Dict[str, Any]] = None,
    dashboard_mode: str = "standard",
    config: Optional[Dict[str, Any]] = None,
):
    log_paths = [Path(p) for p in log_dir] if isinstance(log_dir, Sequence) and not isinstance(log_dir, str) else [Path(log_dir)]
    out_path = Path(output_dir)
    
    logger.info(f"Generating dashboard for: {log_paths[0].name}")

    per_episode_list: List[pd.DataFrame] = []
    for p in log_paths:
        _, per_episode = load_training_data(str(p))
        per_episode = _hydrate_alias_columns(per_episode, p)
        if not per_episode.empty:
            per_episode_list.append(per_episode)

    if not per_episode_list:
        logger.warning(f"No per-episode data found in {log_paths[0]}. Aborting dashboard.")
        return

    out_path.mkdir(parents=True, exist_ok=True)

    # Pass log_path to use JSON fallback if CSV is incomplete
    exp_type = infer_experiment_type(per_episode_list[0], log_dir=log_paths[0])
    meta = extract_experiment_metadata(per_episode_list[0], config)
    if len(per_episode_list) > 1:
        meta["seed"] = f"multi ({len(per_episode_list)})"
    subtitle = format_subtitle(exp_type, meta)
    if len(per_episode_list) == 1:
        subtitle = f"{subtitle} | Single seed"

    if dashboard_mode in {"ann2snn_actor_conversion", "ann2snn_both_conversion"}:
        subtitle = f"Phase 2/3 Conversion Analysis (ANN baseline omitted) | {subtitle}"

    logger.info(f"Experiment Type: {exp_type}")
    logger.info(f"Plot Subtitle:   {subtitle}")

    per_episode_data = per_episode_list[0] if len(per_episode_list) == 1 else per_episode_list

    # ANN2SNN dashboards should report SNN-phase dynamics (post-conversion) by default.
    if dashboard_mode in {"ann2snn_actor_conversion", "ann2snn_both_conversion"}:
        if isinstance(per_episode_data, pd.DataFrame):
            trimmed = _slice_post_conversion_phase(per_episode_data)
            if not trimmed.empty:
                per_episode_data = trimmed
        else:
            trimmed_list = [_slice_post_conversion_phase(df) for df in per_episode_data]
            trimmed_list = [df for df in trimmed_list if not df.empty]
            if trimmed_list:
                per_episode_data = trimmed_list

    if dashboard_mode in {"ann2snn_actor_conversion", "ann2snn_both_conversion"}:
        _exp_name = "ann2snn_both" if dashboard_mode == "ann2snn_both_conversion" else "ann2snn_actor"
        _env_name = config.get("env", {}).get("id", "CartPole") if config else "CartPole"
        try:
            plot_snn_phase(str(log_paths[0]), str(out_path), exp_name=_exp_name, env_name=_env_name)
        except Exception as e:
            logger.warning(f"plot_snn_phase failed: {e}")

    if dashboard_mode == "ann2snn_actor_conversion":
        specs = ANN2SNN_ACTOR_CONVERSION_PLOTS
    elif dashboard_mode == "ann2snn_both_conversion":
        specs = ANN2SNN_BOTH_CONVERSION_PLOTS
    else:
        specs = STANDARD_PLOTS
    run_plot_specs(
        specs,
        per_episode_data,
        out_path,
        title_prefix,
        threshold,
        subtitle,
        config=config,
    )
    try:
        _export_energy_audit_table(per_episode_data, out_path, config=config)
    except Exception as e:
        logger.warning(f"Failed to export energy audit tables: {e}")

    # Timing-critic macro dynamics from per-episode series (single-seed and multi-seed).
    if dashboard_mode not in {"ann2snn_actor_conversion", "ann2snn_both_conversion"}:
        if isinstance(per_episode_data, pd.DataFrame):
            has_value_mean = "eval/critic_value_mean" in per_episode_data.columns
            has_tau_mean = "eval/critic_tau_mean" in per_episode_data.columns
        else:
            has_value_mean = any("eval/critic_value_mean" in s.columns for s in per_episode_data)
            has_tau_mean = any("eval/critic_tau_mean" in s.columns for s in per_episode_data)
        if has_value_mean:
            plot_eval_checkpoint_value_trend(
                per_episode_data,
                str(out_path / "plot_11_eval_checkpoint_critic_values.png"),
                exp_name=config.get("model", {}).get("mode", "ann_baseline") if config else "ann_baseline",
                env_name=config.get("env", {}).get("id", "Unknown Env") if config else "Unknown Env",
                title="Critic Value Trend Across Evaluation Checkpoints",
            )
        if has_tau_mean and has_value_mean:
            plot_timing_critic_macro_dynamics(
                per_episode_data,
                str(out_path / "plot_10_timing_critic_macro_dynamics.png"),
                config=config,
                tau_col="eval/critic_tau_mean",
                val_col="eval/critic_value_mean",
                title_prefix="Timing Critic Macro Dynamics",
            )

    validation_data = _merge_multiseed_validation_data(validation_data, log_paths)
    if validation_data:
        # Critic timing correlation plot
        if "critic_timings" in validation_data and "critic_values" in validation_data:
            plot_timing_critic_correlation = getattr(plotting_mod, "plot_timing_critic_correlation", None)
            if callable(plot_timing_critic_correlation):
                plot_timing_critic_correlation(
                    validation_data["critic_timings"],
                    validation_data["critic_values"],
                    str(out_path / "validation_critic_timing.png"),
                    subtitle=subtitle
                )
            else:
                logger.warning(
                    "Skipping timing-critic correlation plot: "
                    "plot_timing_critic_correlation is unavailable."
                )

        # Conversion accuracy plot
        if "ann_critic_outputs" in validation_data and "snn_critic_outputs" in validation_data:
            save_name = "validation_conversion_critic.png" if dashboard_mode in {"ann2snn_actor_conversion", "ann2snn_both_conversion"} else "validation_conversion_accuracy.png"
            plot_conversion_validation(
                validation_data["ann_critic_outputs"],
                validation_data["snn_critic_outputs"],
                str(out_path / save_name),
                exp_name=config.get("model", {}).get("mode", "ann2snn_both") if config else "ann2snn_both",
                env_name=config.get("env", {}).get("id", "Unknown Env") if config else "Unknown Env",
                component="Critic",
            )
        if "ann_actor_outputs" in validation_data and "snn_actor_outputs" in validation_data:
            plot_conversion_validation(
                validation_data["ann_actor_outputs"],
                validation_data["snn_actor_outputs"],
                str(out_path / "validation_conversion_actor.png"),
                exp_name="ann2snn_actor",
                env_name=config.get("env", {}).get("id", "Unknown Env") if config else "Unknown Env",
                component="Actor",
            )
            
        # Activation counts plot
        if "activations" in validation_data:
            acts = validation_data["activations"]
            # Check if valid data exists
            if any(len(v) > 0 for v in acts.values()):
                plot_activation_counts = getattr(plotting_mod, "plot_activation_counts", None)
                if callable(plot_activation_counts):
                    plot_activation_counts(
                        acts,
                        temporal=True,
                        title="Layer Activation Activity (Validation)",
                        save_path=str(out_path / "validation_activations.png")
                    )
                else:
                    logger.warning("Skipping activation-count plot: plot_activation_counts is unavailable.")
        mode_name = str(config.get("model", {}).get("mode", "")) if config else ""
        # Output-readout validation is only meaningful when critic/output head is spiking.
        has_spiking_output_readout = (
            ("ann_critic" not in mode_name)
            and (
                "snn_critic" in mode_name
                or "timing_critic" in mode_name
                or "ann2snn_both" in mode_name
            )
        )
        if has_spiking_output_readout:
            # Prefer exact internal decision-window traces when available.
            if "actor_decision_potentials" in validation_data and "actor_decision_spikes" in validation_data:
                plot_output_readout_validation(
                    validation_data["actor_decision_potentials"],
                    validation_data["actor_decision_spikes"],
                    str(out_path / "validation_output_readout.png"),
                    title="Validation: Output Readout Dynamics (Membrane Potential vs. Spikes)",
                    exp_name=mode_name,
                )
            elif "output_logits" in validation_data and "output_spikes" in validation_data:
                plot_output_readout_validation(
                    validation_data["output_logits"],
                    validation_data["output_spikes"],
                    str(out_path / "validation_output_readout.png"),
                    title="Validation: Output Readout Dynamics (Membrane Potential vs. Spikes)",
                    exp_name=mode_name,
                )
        # Actor readout: prefer exact per-decision internal tau trace when available.
        if "actor_decision_potentials" in validation_data and "actor_decision_spikes" in validation_data:
            plot_actor_readout_validation(
                validation_data["actor_decision_potentials"],
                validation_data["actor_decision_spikes"],
                str(out_path / "validation_actor_readout.png"),
                exp_name=mode_name,
                title="Actor Output Neuron Dynamics (Validation)",
            )
        # Backward-compatible fallback for older logs.
        elif mode_name == "snn_actor_ann_critic" and "output_logits" in validation_data and "output_spikes" in validation_data:
            plot_actor_readout_validation(
                validation_data["output_logits"],
                validation_data["output_spikes"],
                str(out_path / "validation_actor_readout.png"),
                exp_name=mode_name,
                title="Actor Output Neuron Dynamics (Validation)",
            )
        elif "actor_output_potentials" in validation_data and "actor_output_spikes" in validation_data:
            plot_actor_readout_validation(
                validation_data["actor_output_potentials"],
                validation_data["actor_output_spikes"],
                str(out_path / "validation_actor_readout.png"),
                exp_name=config.get("model", {}).get("mode", "snn_actor_ann_critic") if config else "snn_actor_ann_critic",
                title="Actor Output Neuron Dynamics (Validation)",
            )
        if "critic_values_single_episode" in validation_data:
            critic_single = validation_data["critic_values_single_episode"]
            is_multiseed_single = _looks_like_multiseed_trace(critic_single)
            plot_intra_episode_values(
                critic_single,
                str(out_path / "plot_09_intra_episode_values.png"),
                exp_name=config.get("model", {}).get("mode", "ann_baseline") if config else "ann_baseline",
                config=config,
                env_name=config.get("env", {}).get("id", "Unknown Env") if config else "Unknown Env",
                title="Intra-Episode Critic Value Dynamics",
                profile_desc=(
                    "Multi-seed aligned deterministic post-eval trace (mean ±95% CI)"
                    if is_multiseed_single
                    else "Deterministic post-eval single episode trace (x-axis: step within episode)"
                ),
            )
        elif "intra_episode_values" in validation_data:
            profile_mode = str(validation_data.get("intra_episode_values_profile", "single_episode"))
            profile_desc = (
                "Mean profile across completed post-eval episodes"
                if profile_mode == "mean_post_eval"
                else (
                    "Multi-seed aligned profile (mean ±95% CI)"
                    if profile_mode.startswith("multi_seed")
                    else "Single episode evaluation"
                )
            )
            plot_intra_episode_values(
                validation_data["intra_episode_values"],
                str(out_path / "plot_09_intra_episode_values.png"),
                exp_name=config.get("model", {}).get("mode", "ann_baseline") if config else "ann_baseline",
                config=config,
                env_name=config.get("env", {}).get("id", "Unknown Env") if config else "Unknown Env",
                title="Intra-Episode Value Dynamics",
                profile_desc=profile_desc,
            )
        elif "critic_values" in validation_data:
            plot_intra_episode_values(
                validation_data["critic_values"],
                str(out_path / "plot_09_intra_episode_values.png"),
                exp_name=config.get("model", {}).get("mode", "ann_baseline") if config else "ann_baseline",
                config=config,
                env_name=config.get("env", {}).get("id", "Unknown Env") if config else "Unknown Env",
                title="Intra-Episode Value Dynamics",
            )
        if "critic_timings" in validation_data and "critic_values" in validation_data:
            plot_scope = "all_steps"
            display_actor_steps = None
            if config:
                try:
                    plot_scope = str(
                        config.get("reporting", {}).get(
                            "timing_critic_plot_scope",
                            config.get("report", {}).get("timing_critic_plot_scope", "all_steps"),
                        )
                    ).strip().lower()
                    if plot_scope not in {"all_steps", "first_episode"}:
                        plot_scope = "all_steps"

                    # Optional explicit cap; when omitted, uses full selected scope length.
                    reporting_override = config.get("reporting", {}).get(
                        "timing_critic_plot_actor_steps",
                        config.get("report", {}).get("timing_critic_plot_actor_steps", None),
                    )
                    if reporting_override is not None:
                        display_actor_steps = int(reporting_override)
                except (TypeError, ValueError):
                    plot_scope = "all_steps"
                    display_actor_steps = None

            raw_timing = validation_data["critic_timings"]
            raw_values = validation_data["critic_values"]
            multi_seed_timing = _looks_like_multiseed_trace(raw_timing) and _looks_like_multiseed_trace(raw_values)
            if multi_seed_timing:
                step_traces = validation_data.get("step_traces", {})
                ep_idx_raw = step_traces.get("episode_index", []) if isinstance(step_traces, dict) else []
                has_ep_idx_per_seed = _looks_like_multiseed_trace(ep_idx_raw)

                timing_seeds: List[np.ndarray] = []
                value_seeds: List[np.ndarray] = []
                for sid, (tau_seed_raw, val_seed_raw) in enumerate(zip(raw_timing, raw_values)):
                    tau_seed = _as_1d_float_array(tau_seed_raw)
                    val_seed = _as_1d_float_array(val_seed_raw)
                    n_seed = min(tau_seed.size, val_seed.size)
                    if n_seed == 0:
                        continue
                    tau_seed = tau_seed[:n_seed]
                    val_seed = val_seed[:n_seed]
                    if plot_scope == "first_episode" and has_ep_idx_per_seed and sid < len(ep_idx_raw):
                        ep_seed = _as_1d_int_array(ep_idx_raw[sid])
                        if ep_seed.size >= n_seed:
                            ep_seed = ep_seed[:n_seed]
                            first_mask = ep_seed == int(ep_seed[0])
                            tau_seed = tau_seed[first_mask]
                            val_seed = val_seed[first_mask]
                    n_seed = min(tau_seed.size, val_seed.size)
                    if n_seed == 0:
                        continue
                    timing_seeds.append(tau_seed[:n_seed])
                    value_seeds.append(val_seed[:n_seed])

                if timing_seeds and value_seeds:
                    if display_actor_steps is None:
                        display_actor_steps = int(min(min(len(a) for a in timing_seeds), min(len(a) for a in value_seeds)))
                    display_actor_steps = max(1, display_actor_steps)
                    timing_series = timing_seeds
                    value_series = value_seeds
                else:
                    timing_series = _as_1d_float_array(raw_timing)
                    value_series = _as_1d_float_array(raw_values)
            else:
                timing_series = _as_1d_float_array(raw_timing)
                value_series = _as_1d_float_array(raw_values)
                step_traces = validation_data.get("step_traces", {})
                ep_idx = np.asarray(step_traces.get("episode_index", []), dtype=int).reshape(-1) if isinstance(step_traces, dict) else np.asarray([], dtype=int)
                if (
                    plot_scope == "first_episode"
                    and ep_idx.size
                    and timing_series.size == ep_idx.size
                    and value_series.size == ep_idx.size
                ):
                    first_ep_mask = ep_idx == int(ep_idx[0])
                    timing_series = timing_series[first_ep_mask]
                    value_series = value_series[first_ep_mask]
                if display_actor_steps is None:
                    display_actor_steps = int(min(len(timing_series), len(value_series)))
                display_actor_steps = max(1, display_actor_steps)

            if display_actor_steps is None:
                display_actor_steps = 1

            plot_timing_critic_dynamics(
                timing_series,
                value_series,
                str(out_path / "validation_timing_critic_dynamics.png"),
                config=config,
                actor_steps=display_actor_steps,
                max_actor_steps=display_actor_steps,
                max_annotations=min(display_actor_steps, 16),
                title="Timing Critic Internal Dynamics (Validation)",
            )
        if dashboard_mode in {"ann2snn_actor_conversion", "ann2snn_both_conversion"} and "comparison_metrics" in validation_data:
            try:
                appendix_path = out_path / "appendix_ann_baseline_reference.json"
                with open(appendix_path, "w") as f:
                    json.dump(validation_data["comparison_metrics"], f, indent=2)
            except Exception:
                logger.warning("Failed to write ANN baseline appendix reference file.")
            
    logger.info(f"Dashboard complete. Output: {out_path}")
