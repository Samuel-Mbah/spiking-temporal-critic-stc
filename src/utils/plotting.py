"""
plotting.py

Research-grade visualization utilities for RL + NeuroAI experiments.
Generates publication-ready plots for training dynamics, energy efficiency, and SNN conversion.
"""
from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from typing import Optional, Tuple, Dict, Union, Iterable, List, Sequence

from sklearn.metrics import mean_squared_error, r2_score

from scipy.stats import linregress

from src.utils.metrics import calculate_cumulative_steps

# --- Style Configuration ---
plt.style.use("default")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "font.family": "serif",  # Professional look
})

# ---------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------

def _ensure_numpy(x: Union[torch.Tensor, np.ndarray, pd.Series, list]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, pd.Series):
        return x.values
    return np.asarray(x)

def _rolling_mean(x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Computes a backward rolling mean; returns mean, std, and window used."""
    if len(x) == 0:
        return np.array([]), np.array([]), 1

    # Dynamic Window: If data is shorter than window, shrink window to 20% of data
    real_window = max(1, int(len(x) * 0.2)) if len(x) < window else window
    real_window = max(1, real_window)
    
    # Backward rolling (center=False)
    roll = pd.Series(x).rolling(window=real_window, min_periods=1, center=False)
    mean = roll.mean().to_numpy()
    std = roll.std().fillna(0).to_numpy()
    
    return mean, std, real_window


def _get_steps(df: pd.DataFrame, prefer_cols: Iterable[str]) -> np.ndarray:
    for col in prefer_cols:
        if col in df.columns:
            return _ensure_numpy(df[col])
    try:
        return _ensure_numpy(calculate_cumulative_steps(df))
    except Exception:
        return np.arange(len(df))

def _format_steps_axis(ax: plt.Axes):
    """Formats x-axis with M (millions) or K (thousands) suffix."""
    def human_format(num, pos):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '%.1f%s' % (num, ['', 'K', 'M', 'G'][magnitude]) if magnitude else f"{int(num)}"

    ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    ax.set_xlabel("Environment Steps", fontweight="bold")

def _xlabel_from_cols(dfs: List[pd.DataFrame], prefer_cols: Iterable[str]) -> str:
    cols = set()
    for df in dfs:
        cols.update(df.columns)
    if "total_timesteps" in cols:
        return "Environment Steps"
    if "update" in cols:
        return "Training Updates"
    if "episode" in cols:
        return "Episodes"
    if "time" in cols:
        return "Time"
    if "cumulative" in cols:
        return "Cumulative Steps"
    return "Cumulative Steps"

def _set_titles(ax: plt.Axes, title: str, subtitle: Optional[str] = None):
    def _latex_safe_text(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        # Matplotlib + LaTeX requires escaping special symbols in plain text.
        if bool(plt.rcParams.get("text.usetex", False)):
            return s.replace("&", r"\&").replace("%", r"\%")
        return s

    safe_title = _latex_safe_text(title)
    safe_subtitle = _latex_safe_text(subtitle)

    ax.set_title(safe_title, fontsize=16, fontweight="bold", pad=18)
    if safe_subtitle:
        ax.text(
            0.5,
            1.01,
            safe_subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="gray",
        )

def _resolve_reward_threshold(
    env_name: str,
    config: Optional[Dict] = None,
    threshold: Optional[float] = None,
    fallback: Optional[float] = None,
) -> Optional[float]:
    if threshold is not None:
        return float(threshold)
    if config is not None:
        cfg_thr = config.get("ppo", {}).get("reward_threshold")
        if cfg_thr is not None:
            return float(cfg_thr)
    env_lower = env_name.lower()
    if "cartpole" in env_lower:
        return 475.0
    if "tmaze" in env_lower or "t_maze" in env_lower or "t-maze" in env_lower:
        return 0.95
    return fallback

def _savefig(fig: Figure, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Tight layout not applied.*",
            category=UserWarning,
        )
        fig.tight_layout(pad=2.0)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _placeholder_plot(message: str, save_path: str):
    """Generates an empty plot with a message if data is missing."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="gray")
    _savefig(fig, save_path)

def _select_first_available(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def _as_numeric_1d(values) -> np.ndarray:
    """Coerce values to 1D numeric numpy array with NaNs for invalid entries."""
    if isinstance(values, pd.DataFrame):
        if values.shape[1] == 0:
            return np.array([], dtype=float)
        values = values.iloc[:, 0]
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr[:, 0]
    return pd.to_numeric(pd.Series(arr).reset_index(drop=True), errors="coerce").to_numpy(dtype=float)


def _as_seed_dfs(df_or_dfs: Union[pd.DataFrame, Sequence[pd.DataFrame]]) -> List[pd.DataFrame]:
    """Normalize a DataFrame or sequence of DataFrames into a cleaned seed list."""
    if isinstance(df_or_dfs, pd.DataFrame):
        seeds = [df_or_dfs]
    elif isinstance(df_or_dfs, Sequence):
        seeds = [d for d in df_or_dfs if isinstance(d, pd.DataFrame)]
    else:
        seeds = []
    return [d.copy() for d in seeds if d is not None and not d.empty]


def _aggregate_seed_stat(
    seeds: List[pd.DataFrame],
    metric_col: str,
    prefer_steps: Sequence[str] = ("total_timesteps", "update"),
) -> pd.DataFrame:
    """
    Aggregate a metric across seeds on shared step coordinates.
    Returns columns: step, mean, lo, hi, n.
    """
    rows = []
    for sid, seed_df in enumerate(seeds):
        if metric_col not in seed_df.columns:
            continue
        vals = _as_numeric_1d(seed_df[metric_col])
        steps = _as_numeric_1d(_get_steps(seed_df, prefer_steps))
        n = min(len(vals), len(steps))
        if n == 0:
            continue
        local = pd.DataFrame({"seed": sid, "step": steps[:n], "value": vals[:n]}).dropna(subset=["step", "value"])
        local = local[np.isfinite(local["step"]) & np.isfinite(local["value"])]
        local = local.sort_values("step").drop_duplicates(subset=["step"], keep="last")
        if not local.empty:
            rows.append(local)
    if not rows:
        return pd.DataFrame(columns=["step", "mean", "lo", "hi", "n"])

    all_df = pd.concat(rows, ignore_index=True)
    agg = all_df.groupby("step", as_index=False).agg(
        mean=("value", "mean"),
        lo=("value", lambda a: np.percentile(a, 2.5)),
        hi=("value", lambda a: np.percentile(a, 97.5)),
        n=("value", "count"),
    )
    agg = agg[np.isfinite(agg["step"]) & np.isfinite(agg["mean"]) & np.isfinite(agg["lo"]) & np.isfinite(agg["hi"])]
    return agg.sort_values("step").reset_index(drop=True)


def _resolve_spike_activity(df: pd.DataFrame, candidate_cols: List[str]) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    Resolve spike activity with robust aliases.
    Returns (series_in_spikes_per_step, source_col_name).
    """
    x_col = _select_first_available(
        df,
        candidate_cols + ["spikes/per_step", "spikes/firing_rate", "spikes/total", "spike_count_total"],
    )
    if not x_col:
        return None, None

    spikes = pd.to_numeric(df[x_col], errors="coerce")
    # Convert totals to per-step when possible.
    if x_col in ("spikes/total", "spike_count_total"):
        denom_col = "episode_length_steps" if "episode_length_steps" in df.columns else ("episode_length" if "episode_length" in df.columns else None)
        if denom_col is not None:
            denom = pd.to_numeric(df[denom_col], errors="coerce").replace(0, np.nan)
            spikes = spikes / denom
    return spikes, x_col


def _safe_legend(ax: plt.Axes, *args, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    visible = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith("_")]
    if not visible:
        return
    handles, labels = zip(*visible)
    ax.legend(handles, labels, *args, **kwargs)


def _experiment_color(exp_name: str, default: str = "tab:blue") -> str:
    styles = {
        "ann_baseline": "tab:blue",
        "snn_actor_ann_critic": "tab:green",
        "ann2snn_actor": "tab:orange",
        "ann2snn_both": "tab:red",
        "snn_actor_snn_timing_critic": "tab:purple",
    }
    return styles.get(exp_name, default)


def _plot_constant_spike_fallback(
    data: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    subtitle: str,
    save_path: str,
    note: str,
):
    """
    Replace degenerate x-vs-y spike correlation panels when spike variance is zero.
    """
    if data.empty or y_col not in data.columns:
        _placeholder_plot("Missing data for fallback panel", save_path)
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    y = pd.to_numeric(data[y_col], errors="coerce").dropna().to_numpy()
    if y.size == 0:
        _placeholder_plot("Missing data for fallback panel", save_path)
        return
    x = np.arange(len(y), dtype=int)
    ax.plot(x, y, color="tab:blue", lw=1.8, alpha=0.85, label="Observed Values")

    mean_y = float(np.mean(y))
    ax.axhline(mean_y, color="tab:red", ls="--", lw=1.7, alpha=0.8, label=f"Mean = {mean_y:.4g}")
    ax.text(
        0.02,
        0.96,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    ax.set_xlabel("Sample Index", fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    _set_titles(ax, title, subtitle=subtitle)
    ax.grid(True, linestyle=":", alpha=0.6)
    _safe_legend(ax, loc="best", frameon=True)
    _savefig(fig, save_path)


def _select_readout_trace(
    potentials: np.ndarray,
    spikes: np.ndarray,
    action_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize readout inputs to a single [tau] trace.
    Supports [tau], [actions, tau], [tau, actions], and higher-rank tensors.
    """
    pot = _ensure_numpy(potentials)
    spk = _ensure_numpy(spikes)

    if pot.size == 0 or spk.size == 0:
        return np.array([]), np.array([])

    # Align dimensions conservatively by slicing to overlapping shape.
    if pot.shape != spk.shape:
        min_rank = min(pot.ndim, spk.ndim)
        if min_rank == 0:
            return np.array([]), np.array([])
        common_shape = tuple(min(pot.shape[-min_rank + i], spk.shape[-min_rank + i]) for i in range(min_rank))
        if common_shape:
            pot = pot.reshape((-1,) + pot.shape[-min_rank:]) if pot.ndim > min_rank else pot
            spk = spk.reshape((-1,) + spk.shape[-min_rank:]) if spk.ndim > min_rank else spk
            pot = pot[(0,) * (pot.ndim - min_rank)] if pot.ndim > min_rank else pot
            spk = spk[(0,) * (spk.ndim - min_rank)] if spk.ndim > min_rank else spk
            slices = tuple(slice(0, d) for d in common_shape)
            pot = pot[slices]
            spk = spk[slices]

    # Reduce to <=2D while preserving temporal structure.
    while pot.ndim > 2:
        pot = pot[0]
    while spk.ndim > 2:
        spk = spk[0]

    if pot.ndim == 1 or spk.ndim == 1:
        tau = min(pot.size, spk.size)
        return pot.reshape(-1)[:tau], spk.reshape(-1)[:tau]

    # 2D case: prefer [actions, tau] orientation (tau is usually larger axis).
    if pot.shape[1] < pot.shape[0]:
        pot = pot.T
        spk = spk.T

    num_actions, tau = pot.shape
    if num_actions == 0 or tau == 0:
        return np.array([]), np.array([])

    if action_index is None:
        spike_counts = np.sum(spk > 0, axis=1)
        action_index = int(np.argmax(spike_counts))
    action_index = max(0, min(int(action_index), num_actions - 1))

    return pot[action_index], spk[action_index]


# --------------------------------------------------------------------- training rollout reward --------------------------------------------------------------------------------------------------------------------------------

def plot_train_rollout_vs_steps(
    df: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    save_path: str,
    exp_name: str = "ann_baseline",
    config: Optional[Dict] = None,
    env_name: Optional[str] = None,
    window: int = 50,
    threshold: Optional[float] = None,
    title: str = "Training Dynamics",
    **kwargs
):
    """
    Enhanced training-dynamics plot.
    Supports single-seed or multi-seed (mean ±95% CI) inputs.
    """
    seeds = _as_seed_dfs(df)
    if not seeds:
        _placeholder_plot("No training rollout data", save_path)
        return

    metric_col = "train_reward_unscaled" if any("train_reward_unscaled" in s.columns for s in seeds) else (
        "train_reward" if any("train_reward" in s.columns for s in seeds) else None
    )
    if metric_col is None:
        _placeholder_plot("No training rollout data", save_path)
        return

    # Dynamic configuration
    env_name = env_name or (config.get("env", {}).get("id", "Unknown Env") if config else "Unknown Env")
    
    env_lower = env_name.lower()
    solved_threshold = _resolve_reward_threshold(
        env_name=env_name,
        config=config,
        threshold=threshold,
        fallback=None,
    )
    if "cartpole" in env_lower:
        expected_max = 500.0
    elif "tmaze" in env_lower or "t_maze" in env_lower or "t-maze" in env_lower:
        expected_max = 1.0
    else:
        maxima = []
        for seed_df in seeds:
            if metric_col in seed_df.columns:
                vals = pd.to_numeric(seed_df[metric_col], errors="coerce").dropna().to_numpy()
                if vals.size > 0:
                    maxima.append(float(np.max(vals)))
        expected_max = max(maxima) if maxima else 1.0

    color_rw = _experiment_color(exp_name)
    multi_seed = len(seeds) > 1
    fig, ax1 = plt.subplots(figsize=(12, 7))

    if multi_seed:
        agg = _aggregate_seed_stat(seeds, metric_col, prefer_steps=("total_timesteps", "update"))
        if agg.empty:
            _placeholder_plot("No valid training rollout data", save_path)
            return
        smooth_reward, _, used_window = _rolling_mean(agg["mean"].to_numpy(), window)
        steps = agg["step"].to_numpy()
        raw_reward = agg["mean"].to_numpy()
        ax1.plot(agg["step"], agg["mean"], alpha=0.25, color=color_rw, lw=1.1, label="Raw Mean Reward")
        ax1.plot(agg["step"], smooth_reward, lw=2.6, color=color_rw, label=f"Smoothed Mean Reward (w={used_window})")
        ax1.fill_between(agg["step"], agg["lo"], agg["hi"], color=color_rw, alpha=0.15, label="95% CI")
    else:
        df_single = seeds[0]
        df_valid = df_single.dropna(subset=[metric_col])
        if df_valid.empty:
            _placeholder_plot("No training rollout data", save_path)
            return
        steps = _get_steps(df_valid, ["total_timesteps", "update"])
        raw_reward = _ensure_numpy(df_valid[metric_col])
        smooth_reward, std_reward, used_window = _rolling_mean(raw_reward, window)
        ax1.plot(steps, raw_reward, alpha=0.15, color=color_rw, lw=1, label="Raw Reward")
        ax1.plot(steps, smooth_reward, lw=2.5, color=color_rw, label=f"Smoothed Reward (w={used_window})")
        ax1.fill_between(
            steps,
            smooth_reward - std_reward,
            smooth_reward + std_reward,
            color=color_rw,
            alpha=0.15,
            label="Policy Stability (±1 Std)",
        )

    # 2. Annotate Peak Performance
    max_idx = int(np.argmax(smooth_reward))
    ax1.plot(steps[max_idx], smooth_reward[max_idx], marker='*', color='gold', markersize=12, markeredgecolor='black', zorder=10)
    peak_label = "Peak Mean" if multi_seed else "Peak"
    ax1.annotate(f'{peak_label}: {smooth_reward[max_idx]:.2f}', 
                 xy=(steps[max_idx], smooth_reward[max_idx]),
                 xytext=(-30, 15), textcoords='offset points', fontweight='bold', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gold", lw=1.5))

    if solved_threshold is not None:
        ax1.axhline(solved_threshold, ls="--", lw=2, color="red", alpha=0.8, label="Solved Threshold")

    ax1.set_xlabel("Environment Steps", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Rollout Reward", fontsize=12, fontweight="bold", color=color_rw)
    ax1.tick_params(axis='y', labelcolor=color_rw)
    ax1.grid(True, linestyle=":", alpha=0.7)
    
    actual_max = max(expected_max, np.max(raw_reward))
    ax1.set_ylim(min(0, np.min(raw_reward)), actual_max + (actual_max * 0.15))

    # 3. Secondary Y-Axis for Episode Length (if data exists)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = [], []

    ep_col = None
    for c in ("episode_length", "episode_length_steps"):
        if any(c in s.columns for s in seeds):
            ep_col = c
            break
    if ep_col is not None:
        ax2 = ax1.twinx()
        color_el = "tab:orange"
        if multi_seed:
            ep_agg = _aggregate_seed_stat(seeds, ep_col, prefer_steps=("total_timesteps", "update"))
            if not ep_agg.empty:
                smooth_ep_len, _, _ = _rolling_mean(ep_agg["mean"].to_numpy(), window)
                ax2.plot(ep_agg["step"], smooth_ep_len, lw=2.5, ls="-.", color=color_el, label="Smoothed Mean Episode Length")
                ax2.fill_between(ep_agg["step"], ep_agg["lo"], ep_agg["hi"], color=color_el, alpha=0.10, label="Episode Length 95% CI")
                ax2.set_ylim(0, max(1.0, float(np.nanmax(ep_agg["hi"])) * 1.1))
        else:
            df_single = seeds[0]
            if ep_col in df_single.columns:
                valid_ep = df_single.dropna(subset=[ep_col])
                if not valid_ep.empty:
                    ep_steps = _get_steps(valid_ep, ["total_timesteps", "update"])
                    raw_ep_len = _ensure_numpy(valid_ep[ep_col])
                    smooth_ep_len, _, _ = _rolling_mean(raw_ep_len, window)
                    ax2.plot(ep_steps, smooth_ep_len, lw=2.5, ls="-.", color=color_el, label="Smoothed Avg Episode Length")
                    ax2.set_ylim(0, max(1.0, float(np.nanmax(smooth_ep_len)) * 1.1))
        ax2.set_ylabel("Avg Episode Length (steps)", fontsize=12, fontweight="bold", color=color_el)
        ax2.tick_params(axis='y', labelcolor=color_el)
        lines_2, labels_2 = ax2.get_legend_handles_labels()

    # Combine Legends
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right", framealpha=0.9, fontsize=10)
    
    _format_steps_axis(ax1)
    seed_label = "Multi-seed Mean ±95% CI" if multi_seed else "Single Seed"
    _set_titles(ax1, f"{title} - {env_name}", subtitle=f"{exp_name} ({seed_label})")
    
    _savefig(fig, save_path)

# --------------------------------------------------------------------- evaluation performance --------------------------------------------------------------------------------------------------------------------------------
def plot_eval_return_vs_steps(
    df: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    save_path: str,
    exp_name: str = "ann_baseline",
    config: Optional[Dict] = None,
    env_name: Optional[str] = None,
    threshold: Optional[float] = None,
    title: str = "Evaluation Performance",
    **kwargs
):
    """
    Enhanced single-seed plot for Evaluation Performance.
    Shows sparse evaluation returns, peak performance annotation, and optionally eval episode length.
    """
    def _as_numeric_1d(values) -> np.ndarray:
        # Guard against duplicate-column selection returning DataFrame.
        if isinstance(values, pd.DataFrame):
            if values.shape[1] == 0:
                return np.array([], dtype=float)
            values = values.iloc[:, 0]
        arr = np.asarray(values)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim > 1:
            arr = arr[:, 0]
        return pd.to_numeric(pd.Series(arr).reset_index(drop=True), errors="coerce").to_numpy(dtype=float)

    if isinstance(df, Sequence) and not isinstance(df, pd.DataFrame):
        seeds = [d for d in df if isinstance(d, pd.DataFrame) and not d.empty]
        if not seeds:
            _placeholder_plot("No evaluation data", save_path)
            return

        env_name = env_name or (config.get("env", {}).get("id", "Unknown Env") if config else "Unknown Env")
        solved_threshold = _resolve_reward_threshold(env_name=env_name, config=config, threshold=threshold, fallback=None)
        color_rw = _experiment_color(exp_name)

        rows = []
        for sid, d in enumerate(seeds):
            metric_col = "test_reward" if "test_reward" in d.columns else ("eval_reward" if "eval_reward" in d.columns else None)
            if not metric_col:
                continue
            dv = d.dropna(subset=[metric_col]).copy()
            if dv.empty:
                continue
            steps = _as_numeric_1d(_get_steps(dv, ["total_timesteps", "update"]))
            rewards = _as_numeric_1d(dv[metric_col])
            n_local = min(len(steps), len(rewards))
            if n_local == 0:
                continue
            local = pd.DataFrame({"seed": sid, "step": steps[:n_local], "reward": rewards[:n_local]}).dropna(subset=["step", "reward"])
            local = local[np.isfinite(local["step"]) & np.isfinite(local["reward"])]
            local = local.sort_values("step").drop_duplicates(subset=["step"], keep="last")
            if not local.empty:
                rows.append(local)

        if not rows:
            _placeholder_plot("No valid evaluation points", save_path)
            return

        all_df = pd.concat(rows, ignore_index=True)
        agg = all_df.groupby("step", as_index=False).agg(
            mean=("reward", "mean"),
            lo=("reward", lambda a: np.percentile(a, 2.5)),
            hi=("reward", lambda a: np.percentile(a, 97.5)),
        )
        agg = agg[np.isfinite(agg["step"]) & np.isfinite(agg["mean"]) & np.isfinite(agg["lo"]) & np.isfinite(agg["hi"])]
        if agg.empty:
            _placeholder_plot("No valid evaluation points", save_path)
            return
        smooth, _, used_window = _rolling_mean(agg["mean"].to_numpy(), window=10)

        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax1.plot(agg["step"], agg["mean"], lw=1.2, alpha=0.25, color=color_rw, label="Raw Mean Eval Return")
        ax1.plot(agg["step"], smooth, "o-", lw=2.5, markersize=5, color=color_rw, label=f"Smoothed Mean (w={used_window})")
        ax1.fill_between(agg["step"], agg["lo"], agg["hi"], color=color_rw, alpha=0.15, label="95% CI")
        if solved_threshold is not None:
            ax1.axhline(solved_threshold, ls="--", lw=2, color="red", alpha=0.8, label="Solved Threshold")
        max_idx = int(np.argmax(smooth))
        ax1.plot(agg["step"].iloc[max_idx], smooth[max_idx], marker='*', color='gold', markersize=13, markeredgecolor='black', zorder=10)
        ax1.annotate(f'Peak Mean: {smooth[max_idx]:.2f}', xy=(agg["step"].iloc[max_idx], smooth[max_idx]), xytext=(-40, 15),
                     textcoords='offset points', fontweight='bold', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gold", lw=1.5))
        ax1.set_ylabel("Episode Return", fontsize=12, fontweight="bold", color=color_rw)
        ax1.tick_params(axis='y', labelcolor=color_rw)
        ax1.grid(True, linestyle=":", alpha=0.7)
        _format_steps_axis(ax1)
        ax1.legend(loc="lower right", framealpha=0.9, fontsize=10)
        _set_titles(ax1, f"{title} - {env_name}", subtitle=f"{exp_name} (Multi-seed Mean ±95% CI)")
        _savefig(fig, save_path)
        return

    assert isinstance(df, pd.DataFrame)
    # 1. Determine Evaluation Metric Column
    if "test_reward" in df.columns:
        metric_col = "test_reward"
    elif "eval_reward" in df.columns:
        metric_col = "eval_reward"
    else:
        _placeholder_plot("No evaluation data", save_path)
        return

    df_valid = df.dropna(subset=[metric_col])
    eval_reward_arr = _as_numeric_1d(df_valid[metric_col])
    steps_arr = _as_numeric_1d(_get_steps(df_valid, ["total_timesteps", "update"]))
    n_points = min(len(eval_reward_arr), len(steps_arr), len(df_valid))
    if n_points == 0:
        _placeholder_plot("No valid evaluation points", save_path)
        return
    eval_reward_arr = eval_reward_arr[:n_points]
    steps_arr = steps_arr[:n_points]
    df_valid = df_valid.iloc[:n_points].copy()
    finite_mask = np.isfinite(eval_reward_arr) & np.isfinite(steps_arr)
    df_valid = df_valid.iloc[finite_mask].copy()
    if df_valid.empty:
        _placeholder_plot("No valid evaluation points", save_path)
        return

    steps = steps_arr[finite_mask]
    eval_reward = eval_reward_arr[finite_mask]

    # 2. Dynamic Environment Configuration
    env_name = env_name or (config.get("env", {}).get("id", "Unknown Env") if config else "Unknown Env")
    
    env_lower = env_name.lower()
    solved_threshold = _resolve_reward_threshold(
        env_name=env_name,
        config=config,
        threshold=threshold,
        fallback=None,
    )
    if "cartpole" in env_lower:
        expected_max = 500.0
    elif "tmaze" in env_lower or "t_maze" in env_lower or "t-maze" in env_lower:
        expected_max = 1.0
    else:
        expected_max = np.max(eval_reward)

    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Keep experiment identity color consistent with training-rollout plot.
    color_rw = _experiment_color(exp_name)

    # 3. Plot Evaluation Reward with Explicit Markers
    ax1.plot(steps, eval_reward, "o-", lw=2.5, markersize=7, color=color_rw, label="Evaluation Return")

    # 4. Annotate Peak Performance
    max_idx = np.argmax(eval_reward)
    ax1.plot(steps[max_idx], eval_reward[max_idx], marker='*', color='gold', markersize=15, markeredgecolor='black', zorder=10)
    ax1.annotate(f'Peak Eval: {eval_reward[max_idx]:.2f}', 
                 xy=(steps[max_idx], eval_reward[max_idx]),
                 xytext=(-40, 15), textcoords='offset points', fontweight='bold', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gold", lw=1.5))

    if solved_threshold is not None:
        ax1.axhline(solved_threshold, ls="--", lw=2, color="red", alpha=0.8, label="Solved Threshold")

    # Formatting axes
    ax1.set_xlabel("Environment Steps", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Episode Return", fontsize=12, fontweight="bold", color=color_rw)
    ax1.tick_params(axis='y', labelcolor=color_rw)
    ax1.grid(True, linestyle=":", alpha=0.7)
    
    actual_max = max(expected_max, float(np.max(eval_reward)))
    y_min = float(min(0.0, np.min(eval_reward)))
    y_max = float(actual_max + (actual_max * 0.15))
    if not np.isfinite(y_min):
        y_min = 0.0
    if not np.isfinite(y_max):
        y_max = y_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    ax1.set_ylim(y_min, y_max)

    # 5. Secondary Y-Axis for Evaluation Episode Length
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = [], []
    
    # Check common column names for eval episode length
    ep_len_col = None
    for col in ["test_episode_length", "eval_episode_length", "test_ep_len", "eval_ep_len"]:
        if col in df_valid.columns:
            ep_len_col = col
            break
            
    if ep_len_col:
        raw_eval_ep_len = _as_numeric_1d(df_valid[ep_len_col])
        n_ep = min(len(raw_eval_ep_len), len(steps))
        if n_ep == 0:
            raw_eval_ep_len = np.array([], dtype=float)
        else:
            raw_eval_ep_len = raw_eval_ep_len[:n_ep]
            step_ep = steps[:n_ep]
        finite_ep = np.isfinite(raw_eval_ep_len)
        if np.any(finite_ep):
            ax2 = ax1.twinx()
            # Match training-rollout episode-length style and label semantics.
            color_el = "tab:orange"
            ax2.plot(step_ep[finite_ep], raw_eval_ep_len[finite_ep], "s-.", lw=2.0, markersize=5, color=color_el, alpha=0.8, label="Avg Episode Length")
            ax2.set_ylabel("Avg Episode Length (steps)", fontsize=12, fontweight="bold", color=color_el)
            ax2.tick_params(axis='y', labelcolor=color_el)
            ep_max = float(np.max(raw_eval_ep_len[finite_ep]))
            ax2.set_ylim(0.0, max(1.0, ep_max * 1.15))
            
            lines_2, labels_2 = ax2.get_legend_handles_labels()

    # Combine Legends & Formatting
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right", framealpha=0.9, fontsize=10)
    
    xlabel_str = _xlabel_from_cols([df_valid], ["total_timesteps", "update"])
    if xlabel_str == "Environment Steps":
        _format_steps_axis(ax1)
    else:
        ax1.set_xlabel(xlabel_str, fontsize=12, fontweight="bold")

    _set_titles(ax1, f"{title} - {env_name}", subtitle=f"{exp_name} (Single Seed)")
    
    _savefig(fig, save_path)
    

# --------------------------------------------------------------------- success rate --------------------------------------------------------------------------------------------------------------------------------
def plot_success_rate_vs_steps(
    df: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    save_path: str,
    exp_name: str = "ann_baseline",
    config: Optional[Dict] = None,
    env_name: Optional[str] = None,
    window: int = 10,
    threshold: Optional[float] = None,
    title: str = "Success Rate",
    **kwargs
):
    """
    Industry-standard single-seed Success Rate plot.
    Uses a backward rolling window and dynamic environment thresholds.
    """
    if isinstance(df, Sequence) and not isinstance(df, pd.DataFrame):
        seeds = [d for d in df if isinstance(d, pd.DataFrame) and not d.empty]
        if not seeds:
            _placeholder_plot("No evaluation data for success rate", save_path)
            return

        env_name = env_name or (config.get("env", {}).get("id", "Unknown Env") if config else "Unknown Env")
        solved_threshold = _resolve_reward_threshold(
            env_name=env_name,
            config=config,
            threshold=threshold if threshold is not None else kwargs.get("threshold"),
            fallback=0.0,
        )

        rows = []
        for sid, d in enumerate(seeds):
            success_col = None
            for col in ("eval/success_rate", "success_rate", "eval_success_rate"):
                if col in d.columns:
                    success_col = col
                    break
            if success_col is not None:
                dv = d.dropna(subset=[success_col]).copy()
                if dv.empty:
                    continue
                steps = pd.to_numeric(pd.Series(_get_steps(dv, ["total_timesteps", "update"])), errors="coerce")
                hits = pd.to_numeric(dv[success_col], errors="coerce")
                if hits.notna().any() and float(np.nanmax(hits.to_numpy())) <= 1.0 + 1e-8:
                    hits = hits * 100.0
            else:
                if "test_reward" not in d.columns:
                    continue
                dv = d.dropna(subset=["test_reward"]).copy()
                if dv.empty:
                    continue
                steps = pd.to_numeric(pd.Series(_get_steps(dv, ["total_timesteps", "update"])), errors="coerce")
                rewards = pd.to_numeric(dv["test_reward"], errors="coerce")
                hits = (rewards >= solved_threshold).astype(float) * 100.0

            local = pd.DataFrame({"seed": sid, "step": steps, "success": hits}).dropna(subset=["step", "success"])
            local = local.sort_values("step").drop_duplicates(subset=["step"], keep="last")
            if not local.empty:
                rows.append(local)

        if not rows:
            _placeholder_plot("No evaluation data for success rate", save_path)
            return

        all_df = pd.concat(rows, ignore_index=True)
        agg = all_df.groupby("step", as_index=False).agg(
            mean=("success", "mean"),
            lo=("success", lambda a: np.percentile(a, 2.5)),
            hi=("success", lambda a: np.percentile(a, 97.5)),
        )
        smooth, _, used_window = _rolling_mean(agg["mean"].to_numpy(), window=max(3, int(window)))

        fig, ax = plt.subplots(figsize=(10, 6))
        color = _experiment_color(exp_name)
        ax.plot(agg["step"], agg["mean"], lw=1.2, color=color, alpha=0.25, label="Raw Mean Success")
        ax.plot(agg["step"], smooth, lw=2.5, color=color, label=f"Smoothed Mean (w={used_window})")
        ax.fill_between(agg["step"], agg["lo"], agg["hi"], color=color, alpha=0.12, label="95% CI")
        max_idx = int(np.argmax(smooth))
        ax.plot(agg["step"].iloc[max_idx], smooth[max_idx], marker='*', color='gold', markersize=12, markeredgecolor='black', zorder=10)
        peak_txt = f'Peak Mean: {smooth[max_idx]:.1f}%'
        if bool(plt.rcParams.get("text.usetex", False)):
            peak_txt = peak_txt.replace("%", r"\%")
        ax.annotate(peak_txt, xy=(agg["step"].iloc[max_idx], smooth[max_idx]), xytext=(-30, 15),
                    textcoords='offset points', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gold", lw=1.5))
        ax.set_ylim(-5, 105)
        _format_steps_axis(ax)
        success_ylabel = "Success Rate (%)"
        if bool(plt.rcParams.get("text.usetex", False)):
            success_ylabel = success_ylabel.replace("%", r"\%")
        ax.set_ylabel(success_ylabel, fontweight="bold")
        _set_titles(ax, f"{title} - {env_name}", subtitle="Multi-seed Mean ±95% CI (Eval Checkpoints)")
        ax.legend(frameon=True, loc="lower right")
        _savefig(fig, save_path)
        return

    assert isinstance(df, pd.DataFrame)
    # Prefer explicitly logged success-rate metrics when available.
    success_col = None
    for col in ("eval/success_rate", "success_rate", "eval_success_rate"):
        if col in df.columns:
            success_col = col
            break

    if success_col is not None:
        df_valid = df.dropna(subset=[success_col])
        if df_valid.empty:
            return
    else:
        if "test_reward" not in df.columns:
            _placeholder_plot("No evaluation data for success rate", save_path)
            return
        df_valid = df.dropna(subset=["test_reward"])
        if df_valid.empty:
            return

    # --- Dynamic Environment Configuration ---
    env_name = env_name or (config.get("env", {}).get("id", "Unknown Env") if config else "Unknown Env")
    solved_threshold = _resolve_reward_threshold(
        env_name=env_name,
        config=config,
        threshold=threshold if threshold is not None else kwargs.get("threshold"),
        fallback=0.0,
    )

    # 1. Build success-rate series.
    steps = _get_steps(df_valid, ["total_timesteps", "update"])
    if success_col is not None:
        hits = _ensure_numpy(df_valid[success_col]).astype(float)
        # Handle either [0,1] or [0,100] conventions.
        if np.nanmax(hits) <= 1.0 + 1e-8:
            hits = hits * 100.0
        success_source = success_col
    else:
        # Fallback: infer success from evaluation returns and solved threshold.
        rewards = _ensure_numpy(df_valid["test_reward"])
        hits = (rewards >= solved_threshold).astype(float) * 100.0
        success_source = f"test_reward >= {solved_threshold:g}"

    # 2. Prefer raw eval checkpoints from metrics_raw.json when available,
    # then deduplicate by step/iteration to remove logger duplicates.
    series_df = pd.DataFrame({"step": steps, "success": hits}).dropna()
    log_dir = (config or {}).get("log_dir")
    if log_dir:
        raw_path = os.path.join(log_dir, "metrics_raw.json")
        if os.path.exists(raw_path):
            try:
                import json
                with open(raw_path, "r") as f:
                    raw = json.load(f)
                events = raw.get("eval/success_rate", [])
                if isinstance(events, list) and len(events) > 0 and isinstance(events[0], dict):
                    raw_df = pd.DataFrame(
                        {
                            "iteration": [e.get("iteration") for e in events],
                            "step": [e.get("step") for e in events],
                            "success": [e.get("value") for e in events],
                        }
                    )
                    raw_df["step"] = pd.to_numeric(raw_df["step"], errors="coerce")
                    raw_df["iteration"] = pd.to_numeric(raw_df["iteration"], errors="coerce")
                    raw_df["success"] = pd.to_numeric(raw_df["success"], errors="coerce")
                    raw_df = raw_df.dropna(subset=["step", "success"])
                    raw_df = raw_df.sort_values(["step", "iteration"]).drop_duplicates(
                        subset=["step"], keep="last"
                    )
                    if not raw_df.empty:
                        series_df = raw_df[["step", "success"]]
                        success_source = "metrics_raw.json::eval/success_rate"
            except Exception:
                pass

    # Final safety dedupe/sort for any data source.
    series_df = series_df.sort_values("step").drop_duplicates(subset=["step"], keep="last")
    steps = series_df["step"].to_numpy()
    success_raw = series_df["success"].to_numpy()
    if success_raw.size == 0:
        _placeholder_plot("No evaluation data for success rate", save_path)
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Keep experiment identity color (SNN actor + ANN critic should be green).
    color = _experiment_color(exp_name)

    # 3. Plot raw checkpoint success with area fill.
    ax.plot(steps, success_raw, lw=2.5, color=color, label="Success Rate (raw checkpoints)")
    ax.fill_between(steps, 0, success_raw, color=color, alpha=0.1)

    # 4. Annotate Peak Success
    max_idx = np.argmax(success_raw)
    ax.plot(steps[max_idx], success_raw[max_idx], marker='*', color='gold', 
            markersize=12, markeredgecolor='black', zorder=10)
    
    peak_txt = f'Peak: {success_raw[max_idx]:.1f}%'
    if bool(plt.rcParams.get("text.usetex", False)):
        peak_txt = peak_txt.replace("%", r"\%")
    ax.annotate(peak_txt, 
                 xy=(steps[max_idx], success_raw[max_idx]),
                 xytext=(-30, 15), textcoords='offset points', fontsize=10, fontweight='bold', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gold", lw=1.5))

    # Formatting
    ax.set_ylim(-5, 105)
    xlabel = _xlabel_from_cols([df_valid], ["total_timesteps", "update"])
    if xlabel == "Environment Steps":
        _format_steps_axis(ax)
    else:
        ax.set_xlabel(xlabel, fontweight="bold")
        
    success_ylabel = "Success Rate (%)"
    if bool(plt.rcParams.get("text.usetex", False)):
        success_ylabel = success_ylabel.replace("%", r"\%")
    ax.set_ylabel(success_ylabel, fontweight="bold")
    _set_titles(
        ax,
        f"{title} - {env_name}",
        subtitle="Single Seed, Raw Eval Checkpoints",
    )
    ax.legend(frameon=True, loc="lower right")
    
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------- energy efficiency --------------------------------------------------------------------------------------------------------------------------------
def plot_energy_vs_steps(
    df: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    save_path: str,
    exp_name: str = "ann_baseline",
    config: Optional[Dict] = None,
    env_name: Optional[str] = None,
    window: int = 50,
    title: str = "Energy Consumption & Efficiency",
    **kwargs
):
    """
    Industry-standard energy plot.
    Supports single-seed or multi-seed (mean ±95% CI).
    """
    seeds = _as_seed_dfs(df)
    if not seeds:
        _placeholder_plot("No energy data found in DataFrame", save_path)
        return

    def _prepare_seed_energy(seed_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        df_local = seed_df.copy()
        if "total_timesteps" in df_local.columns:
            step_delta = pd.to_numeric(df_local["total_timesteps"], errors="coerce").diff().fillna(
                pd.to_numeric(df_local["total_timesteps"], errors="coerce")
            )
            step_delta = step_delta.replace(0, np.nan)
        else:
            step_delta = pd.Series(np.nan, index=df_local.index)

        inference_col = None
        if "inference_dynamic_energy" in df_local.columns:
            dyn = pd.to_numeric(df_local["inference_dynamic_energy"], errors="coerce")
            if dyn.notna().any() and float(dyn.fillna(0.0).abs().sum()) > 0.0:
                inference_col = "inference_dynamic_energy"
        if inference_col is None and "inference_energy" in df_local.columns:
            inf = pd.to_numeric(df_local["inference_energy"], errors="coerce")
            if inf.notna().any():
                inference_col = "inference_energy"
        if inference_col is None and "inference_dynamic_energy" in df_local.columns:
            inference_col = "inference_dynamic_energy"

        train_col = (
            "train_full_update_dynamic_energy" if "train_full_update_dynamic_energy" in df_local.columns else
            "train_full_update_energy" if "train_full_update_energy" in df_local.columns else
            "train_rollout_dynamic_energy" if "train_rollout_dynamic_energy" in df_local.columns else
            "train_rollout_energy" if "train_rollout_energy" in df_local.columns else
            None
        )
        if inference_col is not None:
            metric_col = "inference_energy_per_step"
            eval_count_col = _select_first_available(df_local, ["eval/n_eval_episodes", "n_eval_episodes"])
            eval_len_col = _select_first_available(df_local, ["eval_episode_length", "eval/ep_len", "test_episode_length", "eval_episode_len"])
            if eval_count_col and eval_len_col:
                denom = pd.to_numeric(df_local[eval_count_col], errors="coerce") * pd.to_numeric(df_local[eval_len_col], errors="coerce")
                denom = denom.replace(0, np.nan)
                df_local[metric_col] = pd.to_numeric(df_local[inference_col], errors="coerce") / denom
            else:
                ep_len_col = "episode_length_steps" if "episode_length_steps" in df_local.columns else ("episode_length" if "episode_length" in df_local.columns else None)
                if ep_len_col in df_local.columns:
                    denom = pd.to_numeric(df_local[ep_len_col], errors="coerce").replace(0, np.nan)
                    df_local[metric_col] = pd.to_numeric(df_local[inference_col], errors="coerce") / denom
                else:
                    df_local[metric_col] = pd.to_numeric(df_local[inference_col], errors="coerce") / step_delta
            is_dynamic_metric = inference_col.endswith("_dynamic_energy")
            y_label_local = "Dynamic Inference Energy (J/step)" if is_dynamic_metric else "Inference Energy (J/step)"
        elif train_col is not None:
            metric_col = "train_energy_per_step"
            raw_energy = pd.to_numeric(df_local[train_col], errors="coerce")
            df_local[metric_col] = raw_energy / step_delta
            is_dynamic_metric = train_col.endswith("_dynamic_energy")
            y_label_local = "Dynamic Training Energy (J/step)" if is_dynamic_metric else "Training Energy (J/step)"
        else:
            return None

        energy_source = (
            "total_dynamic_energy" if "total_dynamic_energy" in df_local.columns else
            "total_energy" if "total_energy" in df_local.columns else
            train_col if train_col in df_local.columns else
            inference_col
        )
        if energy_source is None or energy_source not in df_local.columns:
            return None

        steps = _as_numeric_1d(_get_steps(df_local, ["total_timesteps", "update"]))
        eff = _as_numeric_1d(df_local[metric_col])
        src = _as_numeric_1d(df_local[energy_source])
        n = min(len(steps), len(eff), len(src))
        if n == 0:
            return None
        out = pd.DataFrame({"step": steps[:n], "eff": eff[:n], "src_energy": src[:n]}).dropna(subset=["step", "eff"])
        out = out[np.isfinite(out["step"]) & np.isfinite(out["eff"])]
        if out.empty:
            return None
        out = out.sort_values("step").drop_duplicates(subset=["step"], keep="last")
        out["cum_kj"] = np.cumsum(pd.to_numeric(out["src_energy"], errors="coerce").fillna(0.0).to_numpy()) / 1000.0
        out.attrs["y_label"] = y_label_local
        return out

    prepared = [s for s in (_prepare_seed_energy(seed) for seed in seeds) if s is not None]
    if not prepared:
        _placeholder_plot("No energy data found in DataFrame", save_path)
        return

    y_label = str(prepared[0].attrs.get("y_label", "Energy (J/step)"))
    multi_seed = len(prepared) > 1

    # Dynamic configuration
    env_name = env_name or (config.get("env", {}).get("id", "Unknown Env") if config else "Unknown Env")
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    # Keep experiment identity color consistent across training/eval/success/intra plots.
    color_eff = _experiment_color(exp_name)

    # 2. Plot Efficiency (Left Axis)
    if multi_seed:
        eff_rows = []
        cum_rows = []
        for sid, s in enumerate(prepared):
            eff_raw = pd.to_numeric(s["eff"], errors="coerce")
            smooth, _, _ = _rolling_mean(eff_raw.to_numpy(), window)
            eff_rows.append(pd.DataFrame({"seed": sid, "step": s["step"], "raw": eff_raw, "smooth": smooth}))
            cum_rows.append(pd.DataFrame({"seed": sid, "step": s["step"], "cum_kj": pd.to_numeric(s["cum_kj"], errors="coerce")}))

        eff_df = pd.concat(eff_rows, ignore_index=True).dropna(subset=["step", "raw", "smooth"])
        eff_agg = eff_df.groupby("step", as_index=False).agg(
            raw_mean=("raw", "mean"),
            smooth_mean=("smooth", "mean"),
            smooth_lo=("smooth", lambda a: np.percentile(a, 2.5)),
            smooth_hi=("smooth", lambda a: np.percentile(a, 97.5)),
        )
        ax1.plot(eff_agg["step"], eff_agg["raw_mean"], alpha=0.22, color=color_eff, lw=1.0, label=f"Raw Mean {y_label}")
        ax1.plot(eff_agg["step"], eff_agg["smooth_mean"], lw=2.5, color=color_eff, label=f"Smoothed Mean {y_label}")
        ax1.fill_between(eff_agg["step"], eff_agg["smooth_lo"], eff_agg["smooth_hi"], color=color_eff, alpha=0.14, label="95% CI")

        final_step = float(eff_agg["step"].iloc[-1])
        final_eff = float(eff_agg["smooth_mean"].iloc[-1])
    else:
        single = prepared[0]
        steps = single["step"].to_numpy()
        raw_efficiency = pd.to_numeric(single["eff"], errors="coerce").to_numpy()
        smooth_eff, std_eff, used_window = _rolling_mean(raw_efficiency, window)
        ax1.plot(steps, raw_efficiency, alpha=0.15, color=color_eff, lw=1, label=y_label)
        ax1.plot(steps, smooth_eff, lw=2.5, color=color_eff, label=f"Smoothed {y_label} (w={used_window})")
        ax1.fill_between(steps, smooth_eff - std_eff, smooth_eff + std_eff, color=color_eff, alpha=0.15, label="Energy Variability (±1 Std)")
        final_step = float(steps[-1])
        final_eff = float(smooth_eff[-1])

    # 3. Annotate Final Efficiency
    ax1.plot(final_step, final_eff, marker='o', color=color_eff, markersize=8)
    ax1.annotate(
        f"Final: {final_eff:.4f} J/step",
        xy=(final_step, final_eff),
        xytext=(-150, -28),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=color_eff, lw=1.3, alpha=0.95),
        arrowprops=dict(arrowstyle="-|>", color=color_eff, lw=1.0, alpha=0.9),
    )

    ax1.set_xlabel("Environment Steps", fontsize=12, fontweight="bold")
    ax1.set_ylabel(y_label, fontsize=12, fontweight="bold", color=color_eff)
    ax1.tick_params(axis='y', labelcolor=color_eff)
    ax1.grid(True, linestyle=":", alpha=0.7)

    # 4. Plot Cumulative Energy (Right Axis)
    ax2 = ax1.twinx()
    color_total = "tab:red"
    if multi_seed:
        cum_df = pd.concat(cum_rows, ignore_index=True).dropna(subset=["step", "cum_kj"])
        cum_agg = cum_df.groupby("step", as_index=False).agg(
            mean=("cum_kj", "mean"),
            lo=("cum_kj", lambda a: np.percentile(a, 2.5)),
            hi=("cum_kj", lambda a: np.percentile(a, 97.5)),
        )
        ax2.plot(cum_agg["step"], cum_agg["mean"], lw=2, ls="--", color=color_total, label="Mean Cumulative Total Energy")
        ax2.fill_between(cum_agg["step"], cum_agg["lo"], cum_agg["hi"], color=color_total, alpha=0.12, label="Cumulative 95% CI")
        y_max = float(np.nanmax(cum_agg["hi"])) if not cum_agg.empty else 0.0
    else:
        cumulative_energy_kj = prepared[0]["cum_kj"].to_numpy(dtype=float)
        steps = prepared[0]["step"].to_numpy(dtype=float)
        ax2.plot(steps, cumulative_energy_kj, lw=2, ls="--", color=color_total, label="Cumulative Total Energy")
        y_max = float(np.nanmax(cumulative_energy_kj)) if len(cumulative_energy_kj) else 0.0
    ax2.set_ylabel("Total Energy Consumed (kJ)", fontsize=12, fontweight="bold", color=color_total)
    ax2.tick_params(axis='y', labelcolor=color_total)
    ax2.set_ylim(0.0, max(y_max * 1.2, 1e-9))

    # Combine Legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper left",
        ncol=2,
        fontsize=10,
        framealpha=0.92,
        borderaxespad=0.8,
    )

    _format_steps_axis(ax1)
    effective_title = title
    if y_label.startswith("Inference Energy"):
        effective_title = "Inference Energy and Cumulative Total"
    seed_label = "Multi-seed Mean ±95% CI" if multi_seed else "Single Seed"
    _set_titles(ax1, f"{effective_title} - {env_name}", subtitle=f"{exp_name} ({seed_label})")
    _savefig(fig, save_path)
    

# --------------------------------------------------------------------- spike activity --------------------------------------------------------------------------------------------------------------------------------    
def plot_spikes_vs_steps(
    df: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    save_path: str,
    exp_name: str = "snn_actor_snn_timing_critic",
    config: Optional[Dict] = None,
    **kwargs
):
    """
    Standardized spike-activity panel with explicit unit consistency:
    - Top: train/eval spike rate (Spikes / Env Step), raw + smoothed.
    - Bottom: cumulative spikes (count).
    Supports single-seed and multi-seed (mean +/- CI) inputs.
    """
    seeds: List[pd.DataFrame] = list(df) if isinstance(df, Sequence) and not isinstance(df, pd.DataFrame) else [df]  # type: ignore[list-item]
    seeds = [d.copy() for d in seeds if isinstance(d, pd.DataFrame) and not d.empty]
    if not seeds:
        _placeholder_plot("No spike data", save_path)
        return

    mode = str((config or {}).get("model", {}).get("mode", exp_name))
    window = int(kwargs.get("window", 50))

    component_specs = [
        ("actor", ["spikes/actor_per_step", "spikes/actor"]),
        ("critic", ["spikes/critic_per_step", "spikes/critic"]),
    ]
    total_rate_candidates = ["spikes/per_step", "spikes/firing_rate", "eval/spikes_per_step"]
    total_count_candidates = ["spike_count_total", "spikes/total", "post_conversion/total_spikes"]
    cumulative_total_candidates = ["spikes/cumulative_total"]

    colors = {"actor": "tab:red", "critic": "tab:orange", "total": "tab:purple"}
    labels = {"actor": "Actor", "critic": "Critic", "total": "Total"}

    def _series_from_seed(seed_df: pd.DataFrame, col_candidates: List[str]) -> pd.DataFrame:
        col = _select_first_available(seed_df, col_candidates)
        if not col:
            return pd.DataFrame(columns=["step", "value"])
        s = pd.to_numeric(seed_df[col], errors="coerce")
        steps = pd.to_numeric(pd.Series(_get_steps(seed_df, ["total_timesteps", "update"])), errors="coerce")
        out = pd.DataFrame({"step": steps, "value": s}).dropna(subset=["step", "value"])
        return out

    def _eval_rate_from_seed(seed_df: pd.DataFrame) -> pd.DataFrame:
        # Explicitly isolate eval-only points.
        steps = pd.to_numeric(pd.Series(_get_steps(seed_df, ["total_timesteps", "update"])), errors="coerce")
        eval_rate = None
        if "eval/spikes_per_step" in seed_df.columns:
            eval_rate = pd.to_numeric(seed_df["eval/spikes_per_step"], errors="coerce")
        elif "eval/spikes_actor_per_step" in seed_df.columns or "eval/spikes_critic_per_step" in seed_df.columns:
            actor_eval = pd.to_numeric(seed_df.get("eval/spikes_actor_per_step", 0.0), errors="coerce")
            critic_eval = pd.to_numeric(seed_df.get("eval/spikes_critic_per_step", 0.0), errors="coerce")
            eval_rate = actor_eval.fillna(0.0) + critic_eval.fillna(0.0)
        elif "eval/spikes" in seed_df.columns and "eval_episode_length" in seed_df.columns:
            denom = pd.to_numeric(seed_df["eval_episode_length"], errors="coerce").replace(0, np.nan)
            eval_rate = pd.to_numeric(seed_df["eval/spikes"], errors="coerce") / denom
        elif (
            "spikes/eval_total" in seed_df.columns
            and "eval_episode_length" in seed_df.columns
            and "eval/n_eval_episodes" in seed_df.columns
        ):
            denom = (
                pd.to_numeric(seed_df["eval_episode_length"], errors="coerce")
                * pd.to_numeric(seed_df["eval/n_eval_episodes"], errors="coerce")
            ).replace(0, np.nan)
            eval_rate = pd.to_numeric(seed_df["spikes/eval_total"], errors="coerce") / denom
        if eval_rate is None:
            return pd.DataFrame(columns=["step", "value"])
        out = pd.DataFrame({"step": steps, "value": eval_rate}).dropna(subset=["step", "value"])
        return out

    def _cumulative_from_seed(seed_df: pd.DataFrame, total_rate_df: pd.DataFrame) -> pd.DataFrame:
        cum_col = _select_first_available(seed_df, cumulative_total_candidates)
        if cum_col:
            s = pd.to_numeric(seed_df[cum_col], errors="coerce")
            steps = pd.to_numeric(pd.Series(_get_steps(seed_df, ["total_timesteps", "update"])), errors="coerce")
            out = pd.DataFrame({"step": steps, "value": s}).dropna(subset=["step", "value"])
            if not out.empty:
                return out
        count_col = _select_first_available(seed_df, total_count_candidates)
        if count_col:
            s = pd.to_numeric(seed_df[count_col], errors="coerce")
            steps = pd.to_numeric(pd.Series(_get_steps(seed_df, ["total_timesteps", "update"])), errors="coerce")
            out = pd.DataFrame({"step": steps, "value": s.cumsum()}).dropna(subset=["step", "value"])
            if not out.empty:
                return out
        if total_rate_df.empty:
            return pd.DataFrame(columns=["step", "value"])
        # Approximate cumulative count by integrating spikes/env-step over env-step delta.
        tr = total_rate_df.sort_values("step").copy()
        d_step = tr["step"].diff().fillna(tr["step"].iloc[0]).clip(lower=0)
        tr["value"] = (tr["value"] * d_step).cumsum()
        return tr[["step", "value"]]

    # Assemble per-seed component series.
    per_seed: Dict[str, List[pd.DataFrame]] = {"actor": [], "critic": [], "total": [], "eval_total": [], "cum_total": []}
    conversion_steps: List[float] = []
    for seed_df in seeds:
        # Conversion-only slices may retain original row indices; normalize to positional index
        # so boolean masks and derived step series align.
        seed_df = seed_df.reset_index(drop=True)
        actor_df = _series_from_seed(seed_df, component_specs[0][1])
        critic_df = _series_from_seed(seed_df, component_specs[1][1])
        total_df = _series_from_seed(seed_df, total_rate_candidates)
        if total_df.empty and (not actor_df.empty or not critic_df.empty):
            merged = pd.merge(actor_df, critic_df, on="step", how="outer", suffixes=("_a", "_c")).fillna(0.0)
            total_df = pd.DataFrame({"step": merged["step"], "value": merged["value_a"] + merged["value_c"]})

        eval_total_df = _eval_rate_from_seed(seed_df)
        cum_total_df = _cumulative_from_seed(seed_df, total_df)

        if not actor_df.empty:
            per_seed["actor"].append(actor_df.sort_values("step"))
        if not critic_df.empty:
            per_seed["critic"].append(critic_df.sort_values("step"))
        if not total_df.empty:
            per_seed["total"].append(total_df.sort_values("step"))
        if not eval_total_df.empty:
            per_seed["eval_total"].append(eval_total_df.sort_values("step"))
        if not cum_total_df.empty:
            per_seed["cum_total"].append(cum_total_df.sort_values("step"))

        for c in ("post_conversion/total_spikes", "post_conversion/zero_shot_reward", "post_conversion/inference_energy"):
            if c in seed_df.columns:
                mask = pd.to_numeric(seed_df[c], errors="coerce").notna()
                if mask.any():
                    step_series = pd.to_numeric(pd.Series(_get_steps(seed_df, ["total_timesteps", "update"])), errors="coerce")
                    conversion_steps.append(float(step_series[mask].iloc[0]))
                    break

    if not per_seed["total"] and not per_seed["actor"] and not per_seed["critic"]:
        _placeholder_plot("Missing Spike Activity Data", save_path)
        return

    def _component_signal(key: str) -> float:
        sig = 0.0
        for s in per_seed.get(key, []):
            y = pd.to_numeric(s["value"], errors="coerce").dropna().to_numpy()
            if y.size:
                sig = max(sig, float(np.nanmax(np.abs(y))))
        return sig

    def _plot_component(ax: plt.Axes, key: str):
        rows = []
        for sid, s in enumerate(per_seed[key]):
            x = pd.to_numeric(s["step"], errors="coerce")
            y = pd.to_numeric(s["value"], errors="coerce")
            smooth, _, _ = _rolling_mean(y.to_numpy(), window=window)
            rows.append(pd.DataFrame({"seed": sid, "step": x, "raw": y, "smooth": smooth}))
        if not rows:
            return None, pd.DataFrame()
        all_df = pd.concat(rows, ignore_index=True).dropna(subset=["step", "raw", "smooth"])
        agg = all_df.groupby("step", as_index=False).agg(
            raw_mean=("raw", "mean"),
            smooth_mean=("smooth", "mean"),
            smooth_lo=("smooth", lambda a: np.percentile(a, 2.5)),
            smooth_hi=("smooth", lambda a: np.percentile(a, 97.5)),
        )
        max_signal = float(np.nanmax(np.abs(agg["smooth_mean"].to_numpy()))) if not agg.empty else 0.0
        c = colors[key]
        ax.plot(agg["step"], agg["raw_mean"], lw=0.9, alpha=0.18, color=c, label="_nolegend_")
        ax.plot(agg["step"], agg["smooth_mean"], lw=2.3, color=c, label=f"{labels[key]} Smoothed")
        if len(rows) > 1:
            ax.fill_between(agg["step"], agg["smooth_lo"], agg["smooth_hi"], color=c, alpha=0.12, label=f"{labels[key]} 95% CI")
        return max_signal, agg[["step", "smooth_mean"]].copy()

    fig, (ax_rate, ax_cum) = plt.subplots(2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})

    # Plot components according to experiment structure.
    actor_signal = _component_signal("actor")
    critic_signal = _component_signal("critic")
    total_signal = max(_component_signal("total"), 1e-12)
    show_actor = actor_signal > 0
    show_critic = critic_signal > (0.01 * total_signal)  # hide near-zero critic traces for readability

    actor_curve = pd.DataFrame()
    critic_curve = pd.DataFrame()
    total_curve = pd.DataFrame()
    if show_actor and len(per_seed["actor"]) > 0:
        _, actor_curve = _plot_component(ax_rate, "actor")
    if show_critic and len(per_seed["critic"]) > 0:
        _, critic_curve = _plot_component(ax_rate, "critic")
    _, total_curve = _plot_component(ax_rate, "total")

    # Hide redundant total smoothed if it is effectively identical to actor/critic.
    hide_total_smoothed = False
    if not total_curve.empty:
        def _is_close(curve_a: pd.DataFrame, curve_b: pd.DataFrame, tol: float = 1e-6) -> bool:
            if curve_a.empty or curve_b.empty:
                return False
            merged = pd.merge(curve_a, curve_b, on="step", how="inner", suffixes=("_a", "_b"))
            if merged.empty:
                return False
            diff = np.abs(merged["smooth_mean_a"] - merged["smooth_mean_b"])
            scale = max(float(np.nanmax(np.abs(merged["smooth_mean_b"]))), 1.0)
            return float(np.nanmax(diff)) <= tol * scale
        if _is_close(total_curve, actor_curve) or _is_close(total_curve, critic_curve):
            hide_total_smoothed = True
    if hide_total_smoothed:
        for ln in list(ax_rate.lines):
            if ln.get_label() == "Total Smoothed":
                ln.remove()
    has_actor = show_actor
    has_critic = show_critic

    # Explicit eval points (separate train vs eval).
    if per_seed["eval_total"]:
        eval_rows = []
        for sid, s in enumerate(per_seed["eval_total"]):
            eval_rows.append(pd.DataFrame({"seed": sid, "step": s["step"], "value": s["value"]}))
        eval_df = pd.concat(eval_rows, ignore_index=True).dropna(subset=["step", "value"])
        eval_agg = eval_df.groupby("step", as_index=False).agg(value=("value", "mean"))
        ax_rate.plot(eval_agg["step"], eval_agg["value"], "o", ms=4, mfc="white", mec="black", mew=1.0, alpha=0.85, label="Eval Rate")

    if conversion_steps:
        conv_step = float(np.median(np.array(conversion_steps)))
        ax_rate.axvline(conv_step, color="tab:blue", ls="--", lw=1.5, alpha=0.85, label="Zero-shot Conversion")
        ax_cum.axvline(conv_step, color="tab:blue", ls="--", lw=1.5, alpha=0.85)

    # Cumulative panel (total activity only for consistency).
    if per_seed["cum_total"]:
        cum_rows = []
        for sid, s in enumerate(per_seed["cum_total"]):
            y = pd.to_numeric(s["value"], errors="coerce")
            smooth, _, _ = _rolling_mean(y.to_numpy(), window=max(5, window))
            cum_rows.append(pd.DataFrame({"seed": sid, "step": s["step"], "raw": y, "smooth": smooth}))
        cum_df = pd.concat(cum_rows, ignore_index=True).dropna(subset=["step", "raw", "smooth"])
        cum_agg = cum_df.groupby("step", as_index=False).agg(
            raw_mean=("raw", "mean"),
            smooth_mean=("smooth", "mean"),
            smooth_lo=("smooth", lambda a: np.percentile(a, 2.5)),
            smooth_hi=("smooth", lambda a: np.percentile(a, 97.5)),
        )
        ax_cum.plot(cum_agg["step"], cum_agg["raw_mean"], lw=0.9, color="tab:gray", alpha=0.22, label="_nolegend_")
        ax_cum.plot(cum_agg["step"], cum_agg["smooth_mean"], lw=2.0, color="black", label="Cumulative Smoothed")
        if len(per_seed["cum_total"]) > 1:
            ax_cum.fill_between(cum_agg["step"], cum_agg["smooth_lo"], cum_agg["smooth_hi"], color="black", alpha=0.1, label="Cumulative 95% CI")

    rate_note = "Normalization: spikes per environment step."
    if has_actor and has_critic:
        role_note = "Components: actor, critic, and total."
    elif has_actor:
        role_note = "Component: actor (+ total)."
    else:
        role_note = "Component: total activity stream."
    seed_note = "Multi-seed mean ±95% CI." if len(seeds) > 1 else "Single-seed raw + smoothed."

    ax_rate.set_ylabel("Spikes / Env Step", fontweight="bold")
    ax_cum.set_ylabel("Cumulative Spikes", fontweight="bold")
    ax_cum.set_xlabel("Environment Steps", fontweight="bold")
    _format_steps_axis(ax_cum)
    _safe_legend(ax_rate, loc="upper left", frameon=True)
    _safe_legend(ax_cum, loc="upper left", frameon=True)
    _set_titles(ax_rate, "Decomposed Spike Activity", subtitle=f"{exp_name} | {seed_note}")
    fig.text(0.012, 0.485, f"{rate_note} {role_note}", fontsize=9, color="dimgray", ha="left", va="bottom")

    _savefig(fig, save_path)


# --------------------------------------------------------------------- latency analysis --------------------------------------------------------------------------------------------------------------------------------
def plot_latency_vs_steps(
    df: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    save_path: str,
    exp_name: str = "snn_actor_snn_timing_critic",
    config: Optional[Dict] = None,
    env_name: Optional[str] = None,
    window: int = 50,
    title: str = "Decision Latency & Timing",
    **kwargs
):
    """
    Industry-standard latency plot.
    Handles decision-time (steps) for SNNs and wall-clock (ms) for ANNs.
    Supports single-seed or multi-seed (mean ±95% CI).
    """
    seeds = _as_seed_dfs(df)
    if not seeds:
        _placeholder_plot("No Latency Data Available", save_path)
        return
    multi_seed = len(seeds) > 1

    # 1. Identify Latency Type (Decision Time vs Wall-Clock)
    # Priority: Spike Timing (Internal steps) > Wall-Clock (ms)
    snn_cols = ["latency/actor_spike_timing_steps", "latency/critic_spike_timing_steps", "latency/spike_timing_steps"]
    ann_cols = ["latency_mean_ms", "latency/mean_ms", "latency/eval_wall_clock_ms"]
    all_cols = set()
    for seed in seeds:
        all_cols.update(seed.columns)
    active_snn_cols = [c for c in snn_cols if c in all_cols]
    active_ann_cols = [c for c in ann_cols if c in all_cols]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Standard Latency Styling
    styles = {
        'actor': {'color': 'tab:red', 'label': 'Actor Decision Time'},
        'critic': {'color': 'tab:orange', 'label': 'Critic Decision Time'},
        'general': {'color': 'tab:blue', 'label': 'Decision Latency'},
        'ann': {'color': 'tab:brown', 'label': 'ANN Wall-Clock'}
    }

    def _aggregate_col(col: str) -> pd.DataFrame:
        return _aggregate_seed_stat(seeds, col, prefer_steps=("total_timesteps", "update"))

    def _single_series(col: str) -> Tuple[np.ndarray, np.ndarray]:
        s0 = seeds[0]
        vals = _as_numeric_1d(s0[col]) if col in s0.columns else np.array([], dtype=float)
        steps = _as_numeric_1d(_get_steps(s0, ["total_timesteps", "update"]))
        n = min(len(vals), len(steps))
        if n == 0:
            return np.array([]), np.array([])
        mask = np.isfinite(vals[:n]) & np.isfinite(steps[:n])
        return steps[:n][mask], vals[:n][mask]

    # 2. Plotting Logic
    if active_snn_cols:
        ylabel = "Decision Latency (Internal Steps $\\tau$)"
        for col in active_snn_cols:
            # Determine sub-label
            label = styles['actor']['label'] if 'actor' in col else (styles['critic']['label'] if 'critic' in col else styles['general']['label'])
            color = styles['actor']['color'] if 'actor' in col else (styles['critic']['color'] if 'critic' in col else styles['general']['color'])

            if multi_seed:
                agg = _aggregate_col(col)
                if agg.empty:
                    continue
                ax.plot(agg["step"], agg["mean"], lw=2.5, color=color, label=f"{label} (Mean)")
                ax.fill_between(agg["step"], agg["lo"], agg["hi"], color=color, alpha=0.15, label=f"{label} 95% CI")
            else:
                steps, raw = _single_series(col)
                if raw.size == 0:
                    continue
                smooth, std, w = _rolling_mean(raw, window)
                ax.plot(steps, smooth, lw=2.5, color=color, label=f"{label} (w={w})")
                ax.fill_between(steps, smooth - std, smooth + std, color=color, alpha=0.15)
                min_idx = np.argmin(smooth)
                ax.plot(steps[min_idx], smooth[min_idx], marker='*', color='gold', markersize=10, markeredgecolor='black')

    elif active_ann_cols:
        ylabel = "Rollout Wall-Clock Latency (ms)"
        cols_to_plot = ["latency_mean_ms"] if "latency_mean_ms" in active_ann_cols else (
            ["latency/mean_ms"] if "latency/mean_ms" in active_ann_cols else ["latency/eval_wall_clock_ms"]
        )
        for col in cols_to_plot:
            if col in ("latency_mean_ms", "latency/mean_ms"):
                label = "ANN Rollout Latency"
                ls = "-"
            else:
                label = "ANN Eval Latency"
                ls = "--"
            if multi_seed:
                agg = _aggregate_col(col)
                if agg.empty:
                    continue
                ax.plot(agg["step"], agg["mean"], lw=2.5, ls=ls, color=styles['ann']['color'], label=f"{label} (Mean)")
                ax.fill_between(agg["step"], agg["lo"], agg["hi"], color=styles['ann']['color'], alpha=0.12, label=f"{label} 95% CI")
            else:
                steps, raw = _single_series(col)
                if raw.size == 0:
                    continue
                smooth, _, w = _rolling_mean(raw, window)
                ax.plot(steps, smooth, lw=2.5, ls=ls, color=styles['ann']['color'], label=f"{label} (w={w})")

    else:
        _placeholder_plot("No Latency Data Available", save_path)
        return

    # 3. Final Formatting
    ax.set_xlabel("Environment Steps", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.7)
    _format_steps_axis(ax)
    
    # Environment name for context
    env_name = env_name or (config.get("env", {}).get("id", "Unknown Env") if config else "Unknown Env")
    seed_label = "Multi-seed Mean ±95% CI" if multi_seed else "Single Seed"
    _set_titles(ax, f"{title} - {env_name}", subtitle=f"{exp_name} ({seed_label})")
    _safe_legend(ax, loc="upper right", frameon=True)
    
    _savefig(fig, save_path)


# --------------------------------------------------------------------- output readout validation --------------------------------------------------------------------------------------------------------------------------------
def plot_output_readout_validation(
    output_potential: np.ndarray,
    output_spikes: np.ndarray,
    save_path: str,
    threshold: float = 1.0,
    reset_val: float = 0.0,
    is_internal_window: bool = True, # New flag
    exp_name: str = "snn_actor_snn_timing_critic",
    title: str = "Critic Output Neuron Dynamics",
    **kwargs,
):
    """
    Plots critic membrane potential and spikes across internal timesteps (tau).
    """
    action_idx = kwargs.get("action_index")
    pot, spikes = _select_readout_trace(output_potential, output_spikes, action_index=action_idx)
    if pot.size == 0 or spikes.size == 0:
        _placeholder_plot("No output-readout data", save_path)
        return

    tau = min(pot.size, spikes.size)
    pot = pot[:tau]
    spikes = spikes[:tau]
    steps = np.arange(tau)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Potential and Threshold
    ax.plot(steps, pot, color="tab:blue", lw=2, label="Membrane Potential $U(\\tau)$")
    ax.axhline(y=threshold, color="tab:red", ls="--", alpha=0.8, label="Firing Threshold")

    # Spikes can be binary or rates; keep only clear spike events.
    spike_cutoff = 0.5 if np.nanmax(spikes) <= 1.0 else 0.0
    spike_indices = np.where(spikes > spike_cutoff)[0]
    if len(spike_indices) > 0:
        ax.scatter(
            spike_indices,
            [threshold] * len(spike_indices),
            color="tab:green",
            marker="^",
            s=80,
            label="Output Spike",
            zorder=4,
        )
        for idx in spike_indices:
            ax.axvline(x=idx, color="tab:green", alpha=0.18, ls=":")

    # Dynamic Labeling
    x_label = "Internal Simulation Step ($\\tau$)" if is_internal_window else "Environment Step ($T$)"
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel("Potential / Logits", fontsize=12, fontweight="bold")

    if title == "Critic Output Neuron Dynamics":
        title = "Validation: Output Readout Dynamics (Membrane Potential vs. Spikes)"
    _set_titles(ax, title, subtitle=f"Experiment: {exp_name}")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.5)

    _savefig(fig, save_path)
    
    
# ------------------------------------------------------------------------------ output readout validation (multiple actions- actor) --------------------------------------------------------------------------------------------------------------------------------
def plot_actor_readout_validation(
    output_potentials: np.ndarray, # Shape: [num_actions, tau]
    output_spikes: np.ndarray,     # Shape: [num_actions, tau]
    save_path: str,
    action_names: List[str] = None,
    threshold: float = 1.0,
    exp_name: str = "ann2snn_actor",
    title: str = "Actor Output Neuron Dynamics",
    **kwargs,
):
    """
    Actor output-readout validation in the same single-trace style as the
    code-generated output-readout panel (membrane potential + threshold + spikes).
    For multi-action actors, selects one action channel (default: most spikes).
    """
    action_idx = kwargs.get("action_index")
    pot, spikes = _select_readout_trace(output_potentials, output_spikes, action_index=action_idx)
    if pot.size == 0 or spikes.size == 0:
        _placeholder_plot("No actor output-readout data", save_path)
        return

    tau = min(pot.size, spikes.size)
    pot = pot[:tau]
    spikes = spikes[:tau]
    steps = np.arange(tau)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(steps, pot, color="tab:blue", lw=2, label="Membrane Potential $U(\\tau)$")
    ax.axhline(y=threshold, color="tab:red", ls="--", alpha=0.8, label="Firing Threshold")

    spike_idx = np.where(spikes > 0)[0]
    if spike_idx.size > 0:
        ax.scatter(spike_idx, np.full_like(spike_idx, threshold, dtype=float),
                   color="tab:green", marker="^", s=80, label="Output Spike", zorder=4)
        for idx in spike_idx:
            ax.axvline(x=idx, color="tab:green", alpha=0.18, ls=":")

    ax.set_xlabel("Internal Timestep ($\\tau$)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Potential / Logits", fontsize=12, fontweight="bold")
    _set_titles(ax, "Validation: Output Readout Dynamics (Membrane Potential vs. Spikes)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.5)

    _savefig(fig, save_path)


def plot_conversion_validation(
    ann_values, 
    snn_values, 
    save_path, 
    exp_name="ann2snn_both", 
    env_name="CartPole", # New param
    component="Actor",   # New param: 'Actor', 'Critic', or 'Both'
    **kwargs
):
    """
    Industry-standard Fidelity Plot with Domain and Component labels.
    Compares ANN vs SNN outputs with statistical metrics.
    """
    if ann_values is None or snn_values is None: return
    
    ann = _ensure_numpy(ann_values).ravel()
    snn = _ensure_numpy(snn_values).ravel()
    
    # Calculate Statistics
    raw_r2 = r2_score(ann, snn)
    raw_mse = mean_squared_error(ann, snn)

    # Centered/standardized metrics are more stable under scale/offset shifts.
    ann_std = float(np.std(ann))
    snn_std = float(np.std(snn))
    ann_norm = (ann - float(np.mean(ann))) / (ann_std + 1e-6)
    snn_norm = (snn - float(np.mean(snn))) / (snn_std + 1e-6)
    centered_mse = mean_squared_error(ann_norm, snn_norm)
    if ann_std < 1e-8 or snn_std < 1e-8:
        pearson_r = 1.0 if (ann_std < 1e-8 and snn_std < 1e-8) else 0.0
    else:
        pearson_r = float(np.corrcoef(ann, snn)[0, 1])
    
    # Identity line and axis policy
    env_lower = str(env_name).lower()
    unit_interval = bool(kwargs.get("unit_interval", ("tmaze" in env_lower)))
    if unit_interval:
        lims = [0.0, 1.0]
    else:
        mn, mx = min(ann.min(), snn.min()), max(ann.max(), snn.max())
        lims = [mn, mx]
    # Optional overlap-revealing jitter (for repeated points in low-diversity domains).
    ann_plot = ann.copy()
    snn_plot = snn.copy()
    reveal_overlap = bool(kwargs.get("reveal_overlap", True))
    if reveal_overlap and ann.size >= 4:
        pair = np.stack([ann, snn], axis=1)
        uniq = np.unique(np.round(pair, decimals=6), axis=0).shape[0]
        duplicate_ratio = 1.0 - (uniq / float(pair.shape[0]))
        if duplicate_ratio > 0.10:
            span = 1.0 if unit_interval else max(np.ptp(ann), np.ptp(snn), 1e-6)
            jitter_std = float(kwargs.get("jitter_std", 0.003 * span))
            if jitter_std > 0.0:
                rng = np.random.default_rng(0)  # deterministic
                noise = rng.normal(0.0, jitter_std, size=pair.shape)
                ann_plot = ann + noise[:, 0]
                snn_plot = snn + noise[:, 1]
                if unit_interval:
                    ann_plot = np.clip(ann_plot, 0.0, 1.0)
                    snn_plot = np.clip(snn_plot, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Color mapping based on error magnitude (computed on raw values, not jittered).
    errors = np.abs(ann - snn)
    scatter = ax.scatter(ann_plot, snn_plot, c=errors, cmap='viridis_r', s=11, alpha=0.52, edgecolors='none')

    ax.plot(lims, lims, "r--", lw=2, label="Ideal Fidelity ($y=x$)", zorder=5)
    
    # Left Box: Performance Metrics
    stats_text = f"Centered MSE: {centered_mse:.4f}\nPearson r: {pearson_r:.4f}\nRaw MSE: {raw_mse:.4f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # Keep raw R^2 as a secondary (smaller) indicator.
    ax.text(
        0.05,
        0.81,
        f"Raw $R^2$: {raw_r2:.4f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        color='dimgray',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.65, edgecolor='lightgray'),
    )

    # Right Box: Domain and Component Labels (The request)
    info_text = f"Domain: {env_name}\nComponent: {component.upper()}"
    ax.text(0.95, 0.05, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9, edgecolor='tab:blue'))

    # Formatting
    ax.set_xlabel("Original ANN Output", fontsize=12, fontweight="bold")
    ax.set_ylabel("Converted SNN Output", fontsize=12, fontweight="bold")
    
    _set_titles(ax, "ANN-SNN Conversion Fidelity", subtitle=f"Experiment: {exp_name}")
    
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_aspect('equal', 'box')
    if unit_interval:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
    
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Absolute Conversion Error', fontweight='bold')
    
    _savefig(fig, save_path)


# --------------------------------------------------------------------- energy vs spikes correlation --------------------------------------------------------------------------------------------------------------------------------
def plot_energy_vs_spikes(
    df: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    save_path: str,
    exp_name: str = "snn_actor_snn_timing_critic",
    env_name: str = "T-Maze",
    **kwargs
):
    """
    Industry-standard Efficiency Correlation plot.
    Calculates cost-per-spike and static energy overhead.
    """
    seeds = _as_seed_dfs(df)
    if not seeds:
        _placeholder_plot("Missing Spike or Energy Data", save_path)
        return

    rows = []
    for sid, seed_df in enumerate(seeds):
        spikes_series, _ = _resolve_spike_activity(
            seed_df,
            ["spikes/per_step", "spikes/total", "spike_count_total", "spikes/firing_rate"],
        )
        y_col = _select_first_available(
            seed_df,
            [
                "inference_dynamic_energy",
                "inference_energy",
                "train_full_update_dynamic_energy",
                "train_full_update_energy",
                "train_rollout_dynamic_energy",
                "train_rollout_energy",
                "total_dynamic_energy",
                "total_energy",
            ],
        )
        if spikes_series is None or not y_col:
            continue
        local = pd.DataFrame({"seed": sid, "x": pd.to_numeric(spikes_series, errors="coerce"), "y": pd.to_numeric(seed_df[y_col], errors="coerce")}).dropna()
        if not local.empty:
            rows.append(local)
    if not rows:
        _placeholder_plot("Missing Spike or Energy Data", save_path)
        return

    data = pd.concat(rows, ignore_index=True)
    x = data["x"].to_numpy()
    y = data["y"].to_numpy()

    can_fit = x.size >= 2 and np.ptp(x) > 0
    if not can_fit:
        _plot_constant_spike_fallback(
            data=data,
            y_col="y",
            y_label="Energy Consumption (Joules)",
            title="Energy Consumption (Constant Spike Activity)",
            subtitle=f"{exp_name} | {env_name}",
            save_path=save_path,
            note="Spike activity variance is zero; plotted energy distribution over samples.",
        )
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 3. Density Scatter (Color by energy magnitude)
    scatter = ax.scatter(x, y, c=y, cmap='magma', s=35, alpha=0.6, edgecolors='none', label="Pooled Samples")
    
    # 4. Regression Line (requires non-constant x)
    seed_slopes = []
    if can_fit:
        slope, intercept, r_val, _, _ = linregress(x, y)
        x_range = np.array([x.min(), x.max()])
        ax.plot(x_range, intercept + slope * x_range, 'r--', lw=2, label=f"Linear Fit ($R^2={r_val**2:.3f}$)")
        for sid, grp in data.groupby("seed"):
            gx, gy = grp["x"].to_numpy(), grp["y"].to_numpy()
            if gx.size >= 2 and np.ptp(gx) > 0:
                try:
                    seed_slopes.append(float(linregress(gx, gy).slope))
                except Exception:
                    pass
        # Slope is J/spike, intercept is static overhead J
        stats_text = (f"Cost/Spike: {slope*1000:.2f} mJ\n"
                      f"Static Overhead: {intercept*1000:.2f} mJ\n"
                      f"Pearson R: {r_val:.3f}")
        if len(seed_slopes) > 1:
            stats_text += f"\nSeed slope μ±σ: {np.mean(seed_slopes)*1000:.2f}±{np.std(seed_slopes)*1000:.2f} mJ/spike"
    else:
        stats_text = "Regression skipped:\nspike activity is constant."
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Formatting
    ax.set_xlabel("Spike Activity (Spikes/Step)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Energy Consumption (Joules)", fontsize=12, fontweight="bold")
    
    seed_note = f" | pooled {len(rows)} seed(s)" if len(rows) > 1 else ""
    _set_titles(ax, "Energy-Spike Efficiency Correlation", subtitle=f"{exp_name} | {env_name}{seed_note}")
    
    ax.grid(True, linestyle=":", alpha=0.6)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Energy Magnitude (J)', fontweight='bold')
    
    _safe_legend(ax, loc='lower right', frameon=True)
    _savefig(fig, save_path)


# --------------------------------------------------------------------- reward vs spikes correlation --------------------------------------------------------------------------------------------------------------------------------
def plot_reward_vs_spikes(
    df: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    save_path: str,
    exp_name: str = "snn_actor_snn_timing_critic",
    env_name: str = "CartPole",
    config: Optional[Dict] = None,
    threshold: Optional[float] = None,
    **kwargs
):
    """
    Industry-standard Reward-Sparsity Pareto Analysis.
    Visualizes the trade-off between neural activity and behavioral performance.
    """
    seeds = _as_seed_dfs(df)
    if not seeds:
        _placeholder_plot("Missing Spike or Reward Data", save_path)
        return

    rows = []
    for sid, seed_df in enumerate(seeds):
        spikes_series, _ = _resolve_spike_activity(
            seed_df,
            ["spikes/per_step", "spikes/total", "spike_count_total", "spikes/firing_rate"],
        )
        y_col = _select_first_available(seed_df, ["test_reward", "train_reward"])
        if spikes_series is None or not y_col:
            continue
        local = pd.DataFrame({"seed": sid, "x": pd.to_numeric(spikes_series, errors="coerce"), "y": pd.to_numeric(seed_df[y_col], errors="coerce")}).dropna()
        if not local.empty:
            rows.append(local)
    if not rows:
        _placeholder_plot("Missing Spike or Reward Data", save_path)
        return

    data = pd.concat(rows, ignore_index=True).sort_values(by="x")
    spikes = data["x"].to_numpy()
    reward = data["y"].to_numpy()

    # Determine Solved Threshold
    solved_threshold = _resolve_reward_threshold(
        env_name=env_name,
        config=config,
        threshold=threshold if threshold is not None else kwargs.get("threshold"),
        fallback=None,
    )

    if spikes.size >= 2 and np.ptp(spikes) == 0:
        _plot_constant_spike_fallback(
            data=pd.DataFrame({"y": reward}),
            y_col="y",
            y_label="Rollout Reward",
            title="Reward Dynamics (Constant Spike Activity)",
            subtitle=f"{exp_name} | {env_name}",
            save_path=save_path,
            note="Spike activity variance is zero; Pareto frontier is not identifiable.",
        )
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # 2. Density Scatter (Color by Reward)
    scatter = ax.scatter(
        spikes,
        reward,
        c=reward,
        cmap='viridis',
        s=58,
        alpha=0.9,
        edgecolors='black',
        linewidths=0.35,
        label="Evaluation Samples",
        zorder=12,
    )

    # 3. Efficiency Frontier (Moving average of reward across spike bins)
    smooth_reward = data["y"].rolling(window=max(1, len(data) // 10), min_periods=1, center=True).mean()
    ax.plot(spikes, smooth_reward, color='tab:red', lw=2.8, label="Efficiency Frontier", zorder=10)

    seed_slopes = []
    for sid, grp in data.groupby("seed"):
        gx, gy = grp["x"].to_numpy(), grp["y"].to_numpy()
        if gx.size >= 2 and np.ptp(gx) > 0:
            try:
                seed_slopes.append(float(linregress(gx, gy).slope))
            except Exception:
                pass

    # 4. Solved Threshold & Shading
    if solved_threshold is not None:
        ax.axhline(solved_threshold, color='red', linestyle='--', alpha=0.6, label=f"Solved ({solved_threshold})")
        ax.axhspan(
            solved_threshold,
            max(reward) * 1.1 if max(reward) > solved_threshold else solved_threshold * 1.1,
            color='green',
            alpha=0.05,
            label="Target Performance",
            zorder=0,
        )

        # 5. Sparsity Annotation
        solved_data = data[data["y"] >= solved_threshold]
        if not solved_data.empty:
            avg_spikes_solved = solved_data["x"].mean()
            x_mid = 0.5 * (float(np.min(spikes)) + float(np.max(spikes)))
            text_dx = -120 if avg_spikes_solved >= x_mid else 12
            text_ha = "right" if avg_spikes_solved >= x_mid else "left"
            ax.annotate(
                f"Sparsity at Solved:\n{avg_spikes_solved:.2f} Spikes/Step",
                xy=(avg_spikes_solved, solved_threshold),
                xytext=(text_dx, -34),
                textcoords="offset points",
                ha=text_ha,
                va="top",
                fontsize=10,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=1, alpha=0.92),
                fontweight="bold",
            )

    # Formatting
    ax.set_xlabel("Neural Activity (Spikes/Step)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rollout Reward", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    _set_titles(ax, "Reward-Sparsity Pareto Analysis", subtitle=f"{exp_name} | {env_name}")
    if len(seed_slopes) > 1:
        ax.text(
            0.03,
            0.96,
            f"Seed slope μ±σ: {np.mean(seed_slopes):.3f}±{np.std(seed_slopes):.3f} reward/spike",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
        )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Reward Level', fontweight='bold')
    
    ax.legend(loc='lower left', framealpha=0.9)
    _savefig(fig, save_path)
    
    
# --------------------------------------------------------------------- latency vs spikes correlation --------------------------------------------------------------------------------------------------------------------------------
def plot_latency_vs_spikes(
    df: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    save_path: str,
    exp_name: str = "snn_actor_snn_timing_critic",
    env_name: str = "CartPole",
    component: str = "Actor", # 'Actor' or 'Critic'
    **kwargs
):
    """
    Industry-standard plot for the Neural Speed-Cost Trade-off.
    Specially tuned to differentiate between Actor reaction time and Critic estimation time.
    """
    seeds = _as_seed_dfs(df)
    if not seeds:
        _placeholder_plot(f"Missing {component} Data", save_path)
        return

    rows = []
    y_col_selected = None
    for sid, seed_df in enumerate(seeds):
        if component.lower() == "actor":
            spikes_series, _ = _resolve_spike_activity(
                seed_df,
                ["spikes/actor_per_step", "spikes/per_step", "spikes/actor", "spike_count_total", "spikes/firing_rate"],
            )
            y_col = _select_first_available(seed_df, ["latency/actor_spike_timing_steps", "latency/spike_timing_steps", "latency/mean_ms", "latency_mean_ms", "latency/eval_wall_clock_ms"])
            color_map = 'Reds'
        else:
            spikes_series, _ = _resolve_spike_activity(seed_df, ["spikes/critic", "spikes/per_step", "spikes/firing_rate", "spike_count_total"])
            y_col = _select_first_available(seed_df, ["latency/critic_spike_timing_steps", "latency/spike_timing_steps", "latency/mean_ms", "latency_mean_ms", "latency/eval_wall_clock_ms"])
            color_map = 'Oranges'
        if spikes_series is None or not y_col:
            continue
        y_col_selected = y_col
        local = pd.DataFrame({"seed": sid, "x": pd.to_numeric(spikes_series, errors="coerce"), "y": pd.to_numeric(seed_df[y_col], errors="coerce")}).dropna()
        if not local.empty:
            rows.append(local)
    if not rows or y_col_selected is None:
        _placeholder_plot(f"Missing {component} Data", save_path)
        return

    data = pd.concat(rows, ignore_index=True)
    x, y = data["x"].to_numpy(), data["y"].to_numpy()

    can_fit = x.size >= 2 and np.ptp(x) > 0
    y_unit = "ms" if "ms" in y_col_selected else "$\\tau$"
    if not can_fit:
        _plot_constant_spike_fallback(
            data=pd.DataFrame({"y": y}),
            y_col="y",
            y_label=f"Decision Latency ({y_unit})",
            title=f"{component} Latency Dynamics (Constant Spike Activity)",
            subtitle=f"{exp_name} | {env_name}",
            save_path=save_path,
            note=f"{component} spike activity variance is zero; speed-cost slope is not identifiable.",
        )
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 3. Density Scatter (Red for Actor, Orange for Critic)
    scatter = ax.scatter(x, y, c=y, cmap=color_map, s=40, alpha=0.7, edgecolors='none')
    
    # 4. Correlation Line (requires non-constant x)
    seed_slopes = []
    if can_fit:
        slope, intercept, r_val, _, _ = linregress(x, y)
        x_range = np.array([x.min(), x.max()])
        ax.plot(x_range, intercept + slope * x_range, 'k--', lw=2, label=f"{component} Trend ($R^2={r_val**2:.3f}$)")
        for sid, grp in data.groupby("seed"):
            gx, gy = grp["x"].to_numpy(), grp["y"].to_numpy()
            if gx.size >= 2 and np.ptp(gx) > 0:
                try:
                    seed_slopes.append(float(linregress(gx, gy).slope))
                except Exception:
                    pass
        stats_text = (f"{component} Reactivity: {slope:.2f} {y_unit}/spike\n"
                      f"Min Latency: {intercept:.2f} {y_unit}\n"
                      f"Pearson R: {r_val:.3f}")
        if len(seed_slopes) > 1:
            stats_text += f"\nSeed slope μ±σ: {np.mean(seed_slopes):.2f}±{np.std(seed_slopes):.2f} {y_unit}/spike"
    else:
        stats_text = f"{component} trend skipped:\nconstant spike activity."

    # 5. Stats Annotation
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Formatting
    ax.set_xlabel(f"{component} Activity (Spikes/Step)", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Decision Latency ({y_unit})", fontsize=12, fontweight="bold")
    _set_titles(ax, f"{component} Speed-Cost Trade-off", subtitle=f"{exp_name} | {env_name}")
    
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.colorbar(scatter, ax=ax).set_label(f"Latency ({y_unit})", fontweight='bold')
    _safe_legend(ax, loc='lower right')
    
    _savefig(fig, save_path)
    

# --------------------------------------------------------------------- snn phase reward recovery --------------------------------------------------------------------------------------------------------------------------------
def plot_snn_phase(log_dir, save_dir=None, exp_name="ann2snn_both", env_name="CartPole"):
    """
    Industry-standard visualization for SNN Zero-Shot (Phase 3) and Fine-Tuning (Phase 4).
    Aligns time series to start from the Zero-Shot point (t=0) and tracks recovery.
    """
    # 1. Directory Setup
    if isinstance(log_dir, dict):
        log_dir = log_dir.get("log_dir")
    log_dir = os.fspath(log_dir)
    save_dir = save_dir or os.path.join(log_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(log_dir, "per_episode_metrics.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(log_dir, "progress.csv")
    if not os.path.exists(csv_path):
        print(f"[Plotting] Error: Metrics CSV not found at {log_dir}")
        return

    df = pd.read_csv(csv_path)
    
    # 2. Key Mapping (Ensure these match your logger)
    keys = {
        'ann_reward': 'train/rollout_reward',  
        'zs_reward': 'post_conversion/zero_shot_reward',
        'zs_energy': 'post_conversion/inference_energy',
        'zs_latency': 'post_conversion/mean_latency',
        'ft_train_reward': 'post_conversion_ft/train_reward',
        'ft_eval_reward': 'post_conversion_ft/eval_reward',
        'ft_energy': 'post_conversion_ft/energy/inference',
        'ft_latency': 'post_conversion_ft/train_latency',
        'time': 'time/total_timesteps_snn'
    }

    # 3. Data Extraction & Normalization
    # Calculate ANN Baseline (Target)
    ann_baseline = None
    if keys['ann_reward'] in df.columns:
        ann_data = df[df[keys['ann_reward']].notna()]
        if not ann_data.empty:
            tail_len = max(1, int(len(ann_data) * 0.05))
            ann_baseline = ann_data[keys['ann_reward']].iloc[-tail_len:].mean()

    # Extract Zero-Shot point
    zs_point = None
    start_step = 0
    if keys['zs_reward'] in df.columns:
        zs_rows = df[df[keys['zs_reward']].notna()]
        if not zs_rows.empty:
            row = zs_rows.iloc[-1]
            start_step = row.get(keys['time'], 0)
            zs_point = {
                'reward': row.get(keys['zs_reward']),
                'energy': row.get(keys['zs_energy']),
                'latency': row.get(keys['zs_latency'])
            }

    # Extract Fine-Tuning recovery data
    ft_df = df[df[keys['ft_train_reward']].notna()].copy() if keys['ft_train_reward'] in df.columns else pd.DataFrame()
    if not ft_df.empty:
        ft_df['rel_step'] = ft_df[keys['time']] - start_step
    elif zs_point is None:
        print("[Plotting] No SNN phase data available.")
        return

    # 4. Generate Professional 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'SNN Post-Conversion Dynamics: {exp_name}\nDomain: {env_name}', fontsize=20, fontweight='bold')

    # --- Plot A: Training Stability (The Path to Recovery) ---
    ax = axes[0, 0]
    if ann_baseline is not None:
        ax.axhline(ann_baseline, color='black', ls='--', alpha=0.5, label=f'ANN Target ({ann_baseline:.0f})')
    
    if not ft_df.empty:
        raw = ft_df[keys['ft_train_reward']]
        steps = ft_df['rel_step']
        ax.plot(steps, raw, color='tab:purple', alpha=0.15)
        ax.plot(steps, raw.rolling(20, min_periods=1).mean(), color='tab:purple', lw=2.5, label='FT Train (Smooth)')
    
    if zs_point:
        ax.plot(0, zs_point['reward'], '*', ms=15, color='gold', mec='k', label='Zero-Shot', zorder=10)
    
    ax.set_title("Training Reward Recovery", fontweight='bold')
    ax.set_ylabel("Reward")
    _safe_legend(ax, loc='lower right')

    # --- Plot B: Evaluation Performance (The Success Metric) ---
    ax = axes[0, 1]
    if ann_baseline is not None:
        ax.axhline(ann_baseline, color='black', ls='--', alpha=0.5)
    
    if not ft_df.empty and keys['ft_eval_reward'] in ft_df.columns:
        eval_data = ft_df.dropna(subset=[keys['ft_eval_reward']])
        ax.plot(eval_data['rel_step'], eval_data[keys['ft_eval_reward']], 'o-', color='tab:green', lw=2, label='Evaluation')
        
    if zs_point:
        ax.plot(0, zs_point['reward'], '*', ms=18, color='gold', mec='k', label='Zero-Shot', zorder=10)
        
    ax.set_title("Evaluation Performance Recovery", fontweight='bold')
    _safe_legend(ax)

    # --- Plot C: Inference Energy (The Hardware Benefit) ---
    ax = axes[1, 0]
    if not ft_df.empty and keys['ft_energy'] in ft_df.columns:
        ax.plot(ft_df['rel_step'], ft_df[keys['ft_energy']], '^-', color='tab:orange', markevery=max(1, len(ft_df)//10), label='Inference Energy')
    
    if zs_point and pd.notna(zs_point['energy']):
        ax.plot(0, zs_point['energy'], '*', ms=15, color='gold', mec='k', zorder=10)
        
    ax.set_title("Energy Efficiency Trend", fontweight='bold')
    ax.set_ylabel("Joules / Episode")
    _safe_legend(ax)

    # --- Plot D: Latency Recovery (The Speed Optimization) ---
    ax = axes[1, 1]
    if not ft_df.empty and keys['ft_latency'] in ft_df.columns:
        ax.plot(ft_df['rel_step'], ft_df[keys['ft_latency']], color='tab:red', lw=2, label='FT Decision Time')
    
    if zs_point and pd.notna(zs_point['latency']):
        ax.axhline(zs_point['latency'], color='tab:red', ls=':', alpha=0.5, label='Zero-Shot Latency')
        
    ax.set_title("Decision Latency Recovery", fontweight='bold')
    ax.set_ylabel("Steps ($\\tau$) or ms")
    _safe_legend(ax)

    # 5. Global Formatting
    def x_formatter(x, pos): return f'{x/1e3:.0f}K' if x >= 1e3 else f'{x:.0f}'
    for ax in axes.flat:
        ax.set_xlabel("Steps After Conversion", fontweight='bold')
        ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, "snn_recovery_dynamics.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plotting] ✅ SNN Phase visual saved to {save_path}")
    
    
# ------------------------------------------------------------------------- intra episode values (plot of actual critic values on average in an episode) -------------------------------------------------------------------------
def plot_intra_episode_values(
    values: Union[List[float], np.ndarray, Sequence[Union[List[float], np.ndarray]]],
    save_path: str,
    exp_name: str = "ann_baseline",
    config: Optional[Dict] = None,
    env_name: Optional[str] = None,
    title: str = "Intra-Episode Value Dynamics",
    **kwargs
):
    """
    Plots critic-value dynamics versus step index within an evaluation episode.
    Input can be either a single episode trace or multiple seed traces.
    """
    multi_seed = False
    seed_arrays: List[np.ndarray] = []
    if isinstance(values, (list, tuple)) and len(values) > 0:
        first = values[0]
        if isinstance(first, (list, tuple, np.ndarray, pd.Series, torch.Tensor)):
            multi_seed = True
            for v in values:  # type: ignore[assignment]
                arr = _ensure_numpy(v).reshape(-1)
                if arr.size > 0:
                    seed_arrays.append(arr.astype(float))
    if multi_seed:
        if not seed_arrays:
            _placeholder_plot("No value data available for this episode", save_path)
            return
        min_len = min(len(a) for a in seed_arrays)
        if min_len == 0:
            _placeholder_plot("No value data available for this episode", save_path)
            return
        aligned = np.stack([a[:min_len] for a in seed_arrays], axis=0)
        values_mean = np.nanmean(aligned, axis=0)
        values_lo = np.nanpercentile(aligned, 2.5, axis=0)
        values_hi = np.nanpercentile(aligned, 97.5, axis=0)
        timesteps = np.arange(min_len)
    else:
        values = _ensure_numpy(values).reshape(-1).astype(float)
        if len(values) == 0:
            _placeholder_plot("No value data available for this episode", save_path)
            return
        timesteps = np.arange(len(values))

    # --- Dynamic Configuration Extraction ---
    if config is not None:
        if env_name is None:
            env_name = config.get("env", {}).get("id", config.get("env_name", "Unknown Env"))
            
    env_name = env_name or "Unknown Env"

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Standard styles to keep your visual language consistent across the paper
    styles = {
        'ann_baseline': {'color': 'tab:blue', 'ls': '-', 'lw': 3},
        'snn_actor_ann_critic': {'color': 'tab:green', 'ls': '--', 'lw': 2.5},
        'ann2snn_actor': {'color': 'tab:orange', 'ls': '-.', 'lw': 2.5},
        'ann2snn_both': {'color': 'tab:red', 'ls': ':', 'lw': 2.5},
        'snn_actor_snn_timing_critic': {'color': 'tab:purple', 'ls': '-', 'lw': 3, 'marker': 'o', 'markersize': 5}
    }
    
    # Get standard style, or fallback to a default if exp_name is custom
    style = styles.get(exp_name, {'color': 'tab:blue', 'ls': '-', 'lw': 2.5})
    
    # Plot line(s)
    if multi_seed:
        line_label = "Eval Critic Value Mean"
        ax.plot(timesteps, values_mean, label=line_label, **style)
        ax.fill_between(timesteps, values_lo, values_hi, color=style.get("color", "tab:blue"), alpha=0.15, label="95% CI")
    else:
        line_label = "ANN Critic Value" if exp_name == "ann" else f"{exp_name} (Single Seed)"
        ax.plot(timesteps, values, label=line_label, **style)

    # Formatting and Labels
    ax.set_xlabel('Step within Evaluation Episode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Critic Value $V(s_t)$', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # Dynamic Axes Constraints
    ax.set_xlim(0, max(1, len(timesteps) - 1))
    
    # Dynamically scale Y-axis around the observed critic-value range.
    # Avoid forcing y-min to 0; that flattens informative variation for near-constant ANN critics.
    y_data = values_mean if multi_seed else values
    y_min, y_max = float(np.min(y_data)), float(np.max(y_data))
    y_span = y_max - y_min
    if y_span > 0:
        margin = max(0.02 * y_span, 1e-4)
    else:
        # Flat trace fallback: keep a narrow window around the constant value.
        margin = max(0.01 * max(1.0, abs(y_max)), 1e-3)
    ax.set_ylim(y_min - margin, y_max + margin)
    
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    
    # Use your custom title formatter
    default_desc = "Multi-seed aligned profile (mean ±95% CI)" if multi_seed else "Single episode evaluation"
    profile_desc = str(kwargs.get("profile_desc", default_desc))
    _set_titles(ax, f"{title} - {env_name}", subtitle=profile_desc)
    
    # Save using your custom wrapper
    _savefig(fig, save_path)


def plot_eval_checkpoint_value_trend(
    df: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    save_path: str,
    exp_name: str = "snn_actor_ann_critic",
    env_name: str = "tmaze-v0",
    title: str = "Critic Value Trend Across Evaluation Checkpoints",
    **kwargs,
):
    """
    Secondary figure for thesis reporting:
    checkpoint-level eval critic value trend (not intra-episode dynamics).
    """
    seeds = _as_seed_dfs(df)
    if not seeds:
        _placeholder_plot("No eval critic-value checkpoint data", save_path)
        return

    mean_col = None
    for col in ("eval/critic_value_mean", "eval_critic_value_mean"):
        if any(col in s.columns for s in seeds):
            mean_col = col
            break
    std_col = None
    for col in ("eval/critic_value_std", "eval_critic_value_std"):
        if any(col in s.columns for s in seeds):
            std_col = col
            break
    if mean_col is None:
        _placeholder_plot("No eval critic-value checkpoint data", save_path)
        return

    multi_seed = len(seeds) > 1
    fig, ax = plt.subplots(figsize=(10, 6))

    if multi_seed:
        agg = _aggregate_seed_stat(seeds, mean_col, prefer_steps=("total_timesteps", "update"))
        if agg.empty:
            _placeholder_plot("No valid eval critic-value checkpoints", save_path)
            return
        x = np.arange(len(agg), dtype=int)
        y_np = agg["mean"].to_numpy(dtype=float)
        ax.plot(x, y_np, color="tab:green", lw=2.2, marker="o", ms=4, label="Eval Critic Value Mean")
        ax.fill_between(x, agg["lo"].to_numpy(dtype=float), agg["hi"].to_numpy(dtype=float), color="tab:green", alpha=0.15, label="95% CI")
    else:
        df_single = seeds[0]
        y = pd.to_numeric(df_single[mean_col], errors="coerce")
        if y.isna().all():
            _placeholder_plot("No valid eval critic-value checkpoints", save_path)
            return
        y = y.dropna().reset_index(drop=True)
        x = np.arange(len(y), dtype=int)
        y_np = y.to_numpy(dtype=float)
        ax.plot(x, y_np, color="tab:green", lw=2.2, marker="o", ms=4, label="Eval Critic Value Mean")

        if std_col and std_col in df_single.columns:
            s = pd.to_numeric(df_single[std_col], errors="coerce").dropna().reset_index(drop=True)
            if len(s) == len(y):
                s_np = s.to_numpy(dtype=float)
                ax.fill_between(x, y_np - s_np, y_np + s_np, color="tab:green", alpha=0.15, label="Mean ±1 Std")

    y_min, y_max = float(np.min(y_np)), float(np.max(y_np))
    span = y_max - y_min
    margin = max(0.02 * span, 1e-3) if span > 0 else max(0.01 * max(1.0, abs(y_max)), 1e-3)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlim(0, max(1, len(x) - 1))

    ax.set_xlabel("Evaluation Checkpoint Index", fontsize=14, fontweight="bold")
    ax.set_ylabel("Critic Value $V(s_t)$", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.7)
    subtitle = "Checkpoint-level eval mean (multi-seed ±95% CI)" if multi_seed else "Checkpoint-level eval mean (not intra-episode)"
    _set_titles(ax, f"{title} - {env_name}", subtitle=subtitle)
    _safe_legend(ax, loc="best", framealpha=0.9)
    _savefig(fig, save_path)

# ------------------------------------------------------------------------- snn_timing_critic_dynamics --------------------------------------------------------------------------------------------------------------------------
def plot_timing_critic_dynamics(
    taus: Union[List[int], np.ndarray, Sequence[Union[List[int], np.ndarray]]],
    values: Union[List[float], np.ndarray, Sequence[Union[List[float], np.ndarray]]],
    save_path: str,
    config: Optional[Dict] = None,
    actor_steps: Optional[int] = None,
    critic_internal_time: Optional[int] = None,
    env_name: Optional[str] = None,
    title: str = "Timing Critic Internal Dynamics",
    max_actor_steps: Optional[int] = 16,
    max_annotations: int = 16,
    **kwargs
):
    """
    Visualizes the internal timing critic's spike times (tau) and values 
    within each actor step for a single rollout/episode.
    Dynamically scales based on the provided config.
    """
    # --- Dynamic Configuration Extraction ---
    if config is not None:
        if env_name is None:
            env_name = config.get("env", {}).get("id", config.get("env_name", "Unknown Env"))
        if critic_internal_time is None:
            # Prefer canonical config key (critic_T), then legacy aliases.
            snn_cfg = config.get("snn", {})
            critic_internal_time = snn_cfg.get(
                "critic_T",
                snn_cfg.get("sim_time", snn_cfg.get("critic_sim_time", snn_cfg.get("T", 32))),
            )
            
    # Fallbacks in case config is missing or keys aren't found
    env_name = env_name or "Unknown Env"
    critic_internal_time = critic_internal_time or 32

    # Multi-seed mode: aggregate across aligned actor steps.
    multi_seed = False
    seed_taus: List[np.ndarray] = []
    seed_values: List[np.ndarray] = []
    if isinstance(taus, (list, tuple)) and len(taus) > 0 and isinstance(taus[0], (list, tuple, np.ndarray, pd.Series, torch.Tensor)):
        if isinstance(values, (list, tuple)) and len(values) > 0 and isinstance(values[0], (list, tuple, np.ndarray, pd.Series, torch.Tensor)):
            multi_seed = True
            n_seed = min(len(taus), len(values))  # type: ignore[arg-type]
            for i in range(n_seed):
                tau_arr = _ensure_numpy(taus[i]).reshape(-1).astype(float)  # type: ignore[index]
                val_arr = _ensure_numpy(values[i]).reshape(-1).astype(float)  # type: ignore[index]
                n = min(len(tau_arr), len(val_arr))
                if n == 0:
                    continue
                if actor_steps is not None:
                    n = min(n, int(actor_steps))
                if max_actor_steps is not None:
                    n = min(n, int(max_actor_steps))
                if n == 0:
                    continue
                seed_taus.append(tau_arr[:n])
                seed_values.append(val_arr[:n])

    if multi_seed:
        if not seed_taus or not seed_values:
            _placeholder_plot("No timing critic data available", save_path)
            return
        min_len = min(min(len(a) for a in seed_taus), min(len(a) for a in seed_values))
        if min_len == 0:
            _placeholder_plot("No timing critic data available", save_path)
            return
        tau_aligned = np.stack([a[:min_len] for a in seed_taus], axis=0)
        val_aligned = np.stack([a[:min_len] for a in seed_values], axis=0)
        x = np.arange(min_len, dtype=int)
        tau_mean = np.nanmean(tau_aligned, axis=0)
        tau_lo = np.nanpercentile(tau_aligned, 2.5, axis=0)
        tau_hi = np.nanpercentile(tau_aligned, 97.5, axis=0)
        val_mean = np.nanmean(val_aligned, axis=0)
        val_lo = np.nanpercentile(val_aligned, 2.5, axis=0)
        val_hi = np.nanpercentile(val_aligned, 97.5, axis=0)

        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axs[0].plot(x, tau_mean, color="tab:blue", lw=2.5, label="Mean $\\tau$")
        axs[0].fill_between(x, tau_lo, tau_hi, color="tab:blue", alpha=0.2, label="95% CI")
        axs[0].axhline(critic_internal_time, color="gray", linestyle="--", alpha=0.6, label=f"Max Internal Time ({critic_internal_time})")
        axs[0].set_ylabel("Spike Time ($\\tau$)", fontweight="bold")
        axs[0].set_ylim(0, critic_internal_time * 1.15)
        axs[0].grid(True, axis="y", alpha=0.3)
        axs[0].legend(loc="upper right")

        axs[1].plot(x, val_mean, color="tab:green", lw=2.5, label="Mean Value")
        axs[1].fill_between(x, val_lo, val_hi, color="tab:green", alpha=0.2, label="95% CI")
        axs[1].set_ylabel("Predicted Value", fontweight="bold")
        axs[1].set_xlabel("Actor Step (Aligned across seeds)", fontweight="bold")
        axs[1].grid(True, axis="y", alpha=0.3)
        axs[1].legend(loc="best")

        _set_titles(
            axs[0],
            f"{title} - {env_name}",
            subtitle="Multi-seed micro dynamics (aligned actor steps, mean ±95% CI)",
        )
        plt.tight_layout()
        _savefig(fig, save_path)
        return

    taus = _ensure_numpy(taus).reshape(-1)
    values = _ensure_numpy(values).reshape(-1)
    # If actor_steps isn't explicitly provided, dynamically use the length of the episode
    n = min(len(taus), len(values))
    if n == 0:
        _placeholder_plot("No timing critic data available", save_path)
        return
    taus = taus[:n]
    values = values[:n]

    actor_steps = int(actor_steps or n)
    actor_steps = max(1, actor_steps)
    if max_actor_steps is not None:
        actor_steps = min(actor_steps, int(max_actor_steps))
    n_plot = min(actor_steps, n)
    taus = taus[:n_plot]
    values = values[:n_plot]
    total_timesteps = actor_steps * critic_internal_time

    fig, ax = plt.subplots(figsize=(15, 6))

    # Draw alternating decision windows to mirror actor-step boundaries.
    for i in range(actor_steps):
        # 1. Background Shading (Banding)
        if i % 2 == 0:
            ax.axvspan(i * critic_internal_time, (i + 1) * critic_internal_time, facecolor='lightgray', alpha=0.25)

    taus_i = np.rint(taus).astype(int)
    taus_i = np.clip(taus_i, 0, max(0, int(critic_internal_time) - 1))
    values_f = values.astype(float)

    annotate_count = min(max_annotations, n_plot)
    annotate_idx = set(np.linspace(0, n_plot - 1, annotate_count, dtype=int).tolist()) if annotate_count > 0 else set()

    for i in range(n_plot):
        tau = int(taus_i[i])
        spike_value = float(values_f[i])
        global_tau = i * critic_internal_time + tau

        # 2. Stem Plot (Lollipop Chart)
        markerline, stemlines, baseline = ax.stem([global_tau], [spike_value], basefmt=" ", linefmt='b-', markerfmt='bo')
        plt.setp(stemlines, 'linewidth', 1.5)
        plt.setp(markerline, 'markersize', 6)

        # 3. Controlled annotation density for readability.
        if i in annotate_idx:
            y_offset = 12 if i % 2 == 0 else -25
            va_align = 'bottom' if i % 2 == 0 else 'top'
            annotation_text = f'$\\tau={tau}$\nv={spike_value:.2f}'
            ax.annotate(
                annotation_text,
                xy=(global_tau, spike_value),
                xytext=(0, y_offset),
                textcoords='offset points',
                ha='center',
                va=va_align,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
            )

    # Formatting and Labels
    ax.set_ylabel('Tau Value', fontweight='bold')
    
    # Dynamic Y-axis bounds based on actual data
    y_max = np.max(values_f) if len(values_f) > 0 else 1.0
    ax.set_ylim(0, y_max * 1.3) 
    
    # Grid and Bottom X-axis
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_xlim(0, total_timesteps)
    tick_stride = max(1, int(np.ceil(actor_steps / 16.0)))
    tick_positions = [j * critic_internal_time for j in range(0, actor_steps + 1, tick_stride)]
    if tick_positions[-1] != total_timesteps:
        tick_positions.append(total_timesteps)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f'{pos}' for pos in tick_positions])
    
    # Top X-axis (actor_T centered)
    ax_top = ax.twiny()
    ax_top.set_xlim(0, total_timesteps)
    
    top_stride = max(1, int(np.ceil(actor_steps / 16.0)))
    midpoints = [j * critic_internal_time + (critic_internal_time / 2) for j in range(0, actor_steps, top_stride)]
    top_labels = [f'T={j}' for j in range(0, actor_steps, top_stride)]
    ax_top.set_xticks(midpoints)
    ax_top.set_xticklabels(top_labels)
    ax_top.tick_params(axis='x', length=0)
    ax_top.set_xlabel('Actor Timestep (actor_T)', fontsize=12, fontweight='bold', labelpad=10)

    ax.set_xlabel(f'Total Timesteps (1 Actor Step = {critic_internal_time} Critic Internal Steps)', fontsize=12, fontweight='bold')
    
    _set_titles(ax, f"{title} - {env_name}")
    _savefig(fig, save_path)


def plot_timing_critic_macro_dynamics(
    combined_df: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    save_path: str,
    config: Optional[Dict] = None,
    tau_col: str = "eval/mean_tau",
    val_col: str = "eval/mean_value",
    window: int = 10,
    title_prefix: str = "Timing Critic",
    critic_internal_time: Optional[int] = None,
    env_name: Optional[str] = None,
    **kwargs
):
    """
    Plots the macro-level learning dynamics of the timing critic over millions of steps.
    Dynamically scales the Y-axis constraints based on the config parameters.
    """
    seeds = _as_seed_dfs(combined_df)
    if not seeds:
        _placeholder_plot("Missing tau/value columns for macro dynamics plot.", save_path)
        return
    if not any(tau_col in s.columns and val_col in s.columns for s in seeds):
        _placeholder_plot("Missing tau/value columns for macro dynamics plot.", save_path)
        return

    # --- Dynamic Configuration Extraction ---
    if config is not None:
        if env_name is None:
            env_name = config.get("env", {}).get("id", config.get("env_name", "Unknown Env"))
        if critic_internal_time is None:
            snn_cfg = config.get("snn", {})
            critic_internal_time = snn_cfg.get(
                "critic_T",
                snn_cfg.get("sim_time", snn_cfg.get("critic_sim_time", snn_cfg.get("T", 32))),
            )
            
    env_name = env_name or "Unknown Env"
    critic_internal_time = critic_internal_time or 32

    rows_tau = []
    rows_val = []
    for sid, seed in enumerate(seeds):
        if tau_col not in seed.columns or val_col not in seed.columns:
            continue
        steps = _as_numeric_1d(_get_steps(seed, ["total_timesteps", "update"]))
        tau_v = _as_numeric_1d(seed[tau_col])
        val_v = _as_numeric_1d(seed[val_col])
        n = min(len(steps), len(tau_v), len(val_v))
        if n == 0:
            continue
        local = pd.DataFrame({"seed": sid, "step": steps[:n], "tau": tau_v[:n], "val": val_v[:n]}).dropna()
        local = local[np.isfinite(local["step"]) & np.isfinite(local["tau"]) & np.isfinite(local["val"])]
        local = local.sort_values("step").drop_duplicates(subset=["step"], keep="last")
        if local.empty:
            continue
        tau_sm = pd.Series(local["tau"].to_numpy()).ewm(span=window).mean().to_numpy()
        val_sm = pd.Series(local["val"].to_numpy()).ewm(span=window).mean().to_numpy()
        rows_tau.append(pd.DataFrame({"seed": sid, "step": local["step"], "raw": local["tau"], "smooth": tau_sm}))
        rows_val.append(pd.DataFrame({"seed": sid, "step": local["step"], "raw": local["val"], "smooth": val_sm}))
    if not rows_tau or not rows_val:
        _placeholder_plot("No valid timing-critic macro data.", save_path)
        return

    tau_df = pd.concat(rows_tau, ignore_index=True)
    val_df = pd.concat(rows_val, ignore_index=True)
    tau_agg = tau_df.groupby("step", as_index=False).agg(
        mean=("smooth", "mean"),
        lo=("smooth", lambda a: np.percentile(a, 2.5)),
        hi=("smooth", lambda a: np.percentile(a, 97.5)),
    )
    val_agg = val_df.groupby("step", as_index=False).agg(
        mean=("smooth", "mean"),
        lo=("smooth", lambda a: np.percentile(a, 2.5)),
        hi=("smooth", lambda a: np.percentile(a, 97.5)),
    )
    x_values = tau_agg["step"].to_numpy()

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top Plot: Tau Dynamics
    axs[0].plot(x_values, tau_agg["mean"], color="tab:blue", lw=2.5, label="Mean $\\tau$")
    axs[0].fill_between(x_values, tau_agg["lo"], tau_agg["hi"], color="tab:blue", alpha=0.2, label="95% CI")
    
    # Dynamically draw the max threshold and set y-limits based on the configured simulation time
    axs[0].axhline(critic_internal_time, color="gray", linestyle="--", alpha=0.5, label=f"Max Internal Time ({critic_internal_time})")
    axs[0].set_ylim(0, critic_internal_time * 1.15)
    
    axs[0].set_ylabel("Spike Time ($\\tau$)", fontweight="bold")
    _set_titles(axs[0], f"{title_prefix}: Evolution of Spike Time ($\\tau$) - {env_name}")
    axs[0].legend(loc="upper right")

    # Bottom Plot: Value Dynamics
    axs[1].plot(val_agg["step"], val_agg["mean"], color="tab:green", lw=2.5, label="Mean Value")
    axs[1].fill_between(val_agg["step"], val_agg["lo"], val_agg["hi"], color="tab:green", alpha=0.2, label="95% CI")
    axs[1].set_ylabel("Predicted Value", fontweight="bold")
    axs[1].set_xlabel("Environment Steps", fontweight="bold")
    _set_titles(axs[1], f"{title_prefix}: Evolution of Predicted Value - {env_name}")
    axs[1].legend(loc="lower right")

    # Format X-axis
    _format_steps_axis(axs[1])

    plt.tight_layout()
    _savefig(fig, save_path)
