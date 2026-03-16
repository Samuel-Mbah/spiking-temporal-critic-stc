

"""
plotting.py

Research-grade visualization utilities for RL + NeuroAI experiments.
Generates publication-ready plots for training dynamics, energy efficiency, and SNN conversion.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from typing import Optional, Tuple, Dict, Union, Iterable, List

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

def _rolling_mean(x: np.ndarray, window: int) -> Tuple[np.ndarray, int]:
    """Computes a centered rolling mean; returns the smoothed array and window used."""
    if len(x) == 0:
        return np.array([]), 0

    # Dynamic Window: If data is shorter than window, shrink window to 20% of data
    real_window = max(1, int(len(x) * 0.2)) if len(x) < window else window
    real_window = max(1, real_window)
    
    # Backward rolling (center=False)
    roll = pd.Series(x).rolling(window=real_window, min_periods=1, center=False)
    mean = roll.mean().to_numpy()
    std = roll.std().fillna(0).to_numpy()
    
    return mean, std, real_window


def _as_run_list(per_episode_data: Union[pd.DataFrame, Iterable[pd.DataFrame], Dict[str, pd.DataFrame]]) -> List[pd.DataFrame]:
    if isinstance(per_episode_data, pd.DataFrame):
        return [per_episode_data]
    if isinstance(per_episode_data, dict):
        return list(per_episode_data.values())
    return list(per_episode_data)


def _get_steps(df: pd.DataFrame, prefer_cols: Iterable[str]) -> np.ndarray:
    for col in prefer_cols:
        if col in df.columns:
            return _ensure_numpy(df[col])
    try:
        return _ensure_numpy(calculate_cumulative_steps(df))
    except Exception:
        return np.arange(len(df))


def _has_spike_activity(dfs: Iterable[pd.DataFrame]) -> bool:
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
    for df in dfs:
        for col in spike_cols:
            if col in df.columns:
                series = df[col].fillna(0)
                if series.abs().sum() > 0:
                    return True
    return False


def _z_value(ci: float) -> float:
    if ci >= 0.995:
        return 2.807
    if ci >= 0.99:
        return 2.576
    if ci >= 0.975:
        return 2.241
    if ci >= 0.95:
        return 1.96
    if ci >= 0.90:
        return 1.645
    return 1.96


def _aggregate_runs(
    dfs: List[pd.DataFrame],
    metric_col: str,
    step_cols: Iterable[str],
    band: str = "std",
    ci: float = 0.95,
    interpolate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """Aggregate multiple runs into mean and variability bands."""
    series_list = []
    for df in dfs:
        if metric_col not in df.columns:
            continue
        df_valid = df.dropna(subset=[metric_col])
        if df_valid.empty:
            continue
        steps = _get_steps(df_valid, step_cols)
        values = _ensure_numpy(df_valid[metric_col])
        s = pd.Series(values, index=steps).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        series_list.append(s)

    if not series_list:
        return np.array([]), np.array([]), None, None, np.array([])

    if len(series_list) == 1:
        s = series_list[0]
        return s.index.to_numpy(), s.to_numpy(), None, None, np.array([len(s)])

    common_steps = np.unique(np.concatenate([s.index.to_numpy() for s in series_list]))
    aligned = []
    for s in series_list:
        s_re = s.reindex(common_steps)
        if interpolate:
            s_re = s_re.interpolate(method="linear", limit_area="inside")
        aligned.append(s_re)

    df_aligned = pd.concat(aligned, axis=1)
    mean = df_aligned.mean(axis=1, skipna=True).to_numpy()
    std = df_aligned.std(axis=1, ddof=1, skipna=True).to_numpy()
    n = df_aligned.count(axis=1).to_numpy()

    lower = upper = None
    if band == "std":
        lower = mean - std
        upper = mean + std
    elif band == "sem":
        sem = std / np.sqrt(np.maximum(n, 1))
        lower = mean - sem
        upper = mean + sem
    elif band == "ci":
        z = _z_value(ci)
        sem = std / np.sqrt(np.maximum(n, 1))
        half = z * sem
        lower = mean - half
        upper = mean + half

    return common_steps, mean, lower, upper, n


def _plot_band(ax: plt.Axes, steps: np.ndarray, lower: np.ndarray, upper: np.ndarray, label: str, color: str):
    ax.fill_between(steps, lower, upper, color=color, alpha=0.2, label=label)

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
    ax.set_title(title, fontweight="bold", pad=12)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, 
                ha="center", va="bottom", fontsize=10, color="gray")

def _savefig(fig: Figure, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout(pad=2.0)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _placeholder_plot(message: str, save_path: str):
    """Generates an empty plot with a message if data is missing."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="gray")
    _savefig(fig, save_path)


# --------------------------------------------------------------------- training rollout reward --------------------------------------------------------------------------------------------------------------------------------

def plot_train_rollout_vs_steps(
    df: pd.DataFrame,
    save_path: str,
    exp_name: str = "ann_baseline",
    config: Optional[Dict] = None,
    env_name: Optional[str] = None,
    window: int = 50,
    title: str = "Training Dynamics",
    **kwargs
):
    """
    Enhanced single-seed plot: Shows reward, policy stability (rolling std), 
    peak performance annotation, and optionally episode length.
    """
    if "train_reward_unscaled" in df.columns:
        metric_col = "train_reward_unscaled"
    elif "train_reward" in df.columns:
        metric_col = "train_reward"
    else:
        _placeholder_plot("No training rollout data", save_path)
        return

    df_valid = df.dropna(subset=[metric_col])
    if df_valid.empty: return

    # Extract steps, reward, and calculate backward stats
    steps = _get_steps(df_valid, ["total_timesteps", "update"])
    raw_reward = _ensure_numpy(df_valid[metric_col])
    smooth_reward, std_reward, used_window = _rolling_mean(raw_reward, window)

    # Dynamic configuration
    env_name = env_name or (config.get("env", {}).get("id", "Unknown Env") if config else "Unknown Env")
    
    env_lower = env_name.lower()
    if "cartpole" in env_lower:
        threshold, expected_max = 475.0, 500.0
    elif "tmaze" in env_lower or "t_maze" in env_lower or "t-maze" in env_lower:
        threshold, expected_max = 1.0, 1.0
    else:
        threshold, expected_max = None, np.max(raw_reward)

    fig, ax1 = plt.subplots(figsize=(12, 7))
    color_rw = "tab:blue" # Fixed color for Reward

    # 1. Plot Reward & Rolling Stability Band
    ax1.plot(steps, raw_reward, alpha=0.15, color=color_rw, lw=1, label="Raw Reward")
    ax1.plot(steps, smooth_reward, lw=2.5, color=color_rw, label=f"Smoothed Reward (w={used_window})")
    ax1.fill_between(steps, smooth_reward - std_reward, smooth_reward + std_reward, 
                     color=color_rw, alpha=0.15, label="Policy Stability (±1 Std)")

    # 2. Annotate Peak Performance
    max_idx = np.argmax(smooth_reward)
    ax1.plot(steps[max_idx], smooth_reward[max_idx], marker='*', color='gold', markersize=12, markeredgecolor='black', zorder=10)
    ax1.annotate(f'Peak: {smooth_reward[max_idx]:.2f}', 
                 xy=(steps[max_idx], smooth_reward[max_idx]),
                 xytext=(-30, 15), textcoords='offset points', fontweight='bold', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gold", lw=1.5))

    if threshold is not None:
        ax1.axhline(threshold, ls="--", lw=2, color="red", alpha=0.8, label="Solved Threshold")

    ax1.set_xlabel("Environment Steps", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Rollout Reward", fontsize=12, fontweight="bold", color=color_rw)
    ax1.tick_params(axis='y', labelcolor=color_rw)
    ax1.grid(True, linestyle=":", alpha=0.7)
    
    actual_max = max(expected_max, np.max(raw_reward))
    ax1.set_ylim(min(0, np.min(raw_reward)), actual_max + (actual_max * 0.15))

    # 3. Secondary Y-Axis for Episode Length (if data exists)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = [], []
    
    if "episode_length" in df_valid.columns or "episode_length_steps" in df_valid.columns:
        ep_col = "episode_length" if "episode_length" in df_valid.columns else "episode_length_steps"
        raw_ep_len = _ensure_numpy(df_valid[ep_col])
        smooth_ep_len, _, _ = _rolling_mean(raw_ep_len, window)
        
        ax2 = ax1.twinx()
        color_el = "tab:orange"
        ax2.plot(steps, smooth_ep_len, lw=2.5, ls="-.", color=color_el, label="Smoothed Episode Length")
        ax2.set_ylabel("Episode Length (Steps)", fontsize=12, fontweight="bold", color=color_el)
        ax2.tick_params(axis='y', labelcolor=color_el)
        ax2.set_ylim(0, np.max(smooth_ep_len) * 1.1)
        
        lines_2, labels_2 = ax2.get_legend_handles_labels()

    # Combine Legends
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right", framealpha=0.9, fontsize=10)
    
    _format_steps_axis(ax1)
    _set_titles(ax1, f"{title} - {env_name}", subtitle=f"{exp_name} (Single Seed)")
    
    _savefig(fig, save_path)



# --------------------------------------------------------------------- evaluation performance --------------------------------------------------------------------------------------------------------------------------------
def plot_eval_return_vs_steps(
    df: pd.DataFrame,
    save_path: str,
    exp_name: str = "ann_baseline",
    config: Optional[Dict] = None,
    env_name: Optional[str] = None,
    title: str = "Evaluation Performance",
    **kwargs
):
    """
    Enhanced single-seed plot for Evaluation Performance.
    Shows sparse evaluation returns, peak performance annotation, and optionally eval episode length.
    """
    # 1. Determine Evaluation Metric Column
    if "test_reward" in df.columns:
        metric_col = "test_reward"
    elif "eval_reward" in df.columns:
        metric_col = "eval_reward"
    else:
        _placeholder_plot("No evaluation data", save_path)
        return

    df_valid = df.dropna(subset=[metric_col])
    if df_valid.empty:
        _placeholder_plot("No valid evaluation points", save_path)
        return

    steps = _get_steps(df_valid, ["total_timesteps", "update"])
    eval_reward = _ensure_numpy(df_valid[metric_col])

    # 2. Dynamic Environment Configuration
    env_name = env_name or (config.get("env", {}).get("id", "Unknown Env") if config else "Unknown Env")
    
    env_lower = env_name.lower()
    if "cartpole" in env_lower:
        threshold, expected_max = 475.0, 500.0
    elif "tmaze" in env_lower or "t_maze" in env_lower or "t-maze" in env_lower:
        threshold, expected_max = 1.0, 1.0
    else:
        threshold, expected_max = None, np.max(eval_reward)

    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Standardized styling
    styles = {
        'ann_baseline': {'color': 'tab:blue'},
        'snn_actor_ann_critic': {'color': 'tab:green'},
        'ann2snn_actor': {'color': 'tab:orange'},
        'ann2snn_both': {'color': 'tab:red'},
        'snn_actor_snntiming_critic': {'color': 'tab:purple'}
    }
    color_rw = styles.get(exp_name, {'color': 'tab:orange'})['color']

    # 3. Plot Evaluation Reward with Explicit Markers
    ax1.plot(steps, eval_reward, "o-", lw=2.5, markersize=7, color=color_rw, label="Evaluation Return")

    # 4. Annotate Peak Performance
    max_idx = np.argmax(eval_reward)
    ax1.plot(steps[max_idx], eval_reward[max_idx], marker='*', color='gold', markersize=15, markeredgecolor='black', zorder=10)
    ax1.annotate(f'Peak Eval: {eval_reward[max_idx]:.2f}', 
                 xy=(steps[max_idx], eval_reward[max_idx]),
                 xytext=(-40, 15), textcoords='offset points', fontweight='bold', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gold", lw=1.5))

    if threshold is not None:
        ax1.axhline(threshold, ls="--", lw=2, color="red", alpha=0.8, label="Solved Threshold")

    # Formatting axes
    ax1.set_xlabel("Environment Steps", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Episode Return", fontsize=12, fontweight="bold", color=color_rw)
    ax1.tick_params(axis='y', labelcolor=color_rw)
    ax1.grid(True, linestyle=":", alpha=0.7)
    
    actual_max = max(expected_max, np.max(eval_reward))
    ax1.set_ylim(min(0, np.min(eval_reward)), actual_max + (actual_max * 0.15))

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
        eval_ep_len = _ensure_numpy(df_valid[ep_len_col])
        ax2 = ax1.twinx()
        color_el = "gray"
        ax2.plot(steps, eval_ep_len, "s--", lw=2.0, markersize=5, color=color_el, alpha=0.6, label="Eval Episode Length")
        ax2.set_ylabel("Episode Length (Steps)", fontsize=12, fontweight="bold", color=color_el)
        ax2.tick_params(axis='y', labelcolor=color_el)
        ax2.set_ylim(0, np.max(eval_ep_len) * 1.15)
        
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
    df: pd.DataFrame,
    save_path: str,
    exp_name: str = "ann_baseline",
    config: Optional[Dict] = None,
    env_name: Optional[str] = None,
    window: int = 50,
    title: str = "Success Rate",
    **kwargs
):
    """
    Industry-standard single-seed Success Rate plot.
    Uses a backward rolling window and dynamic environment thresholds.
    """
    if "test_reward" not in df.columns:
        _placeholder_plot("No evaluation data for success rate", save_path)
        return

    df_valid = df.dropna(subset=["test_reward"])
    if df_valid.empty: return

    # --- Dynamic Environment Configuration ---
    env_name = env_name or (config.get("env", {}).get("id", "Unknown Env") if config else "Unknown Env")
    env_lower = env_name.lower()
    
    # Standard RL Solved Thresholds
    if "cartpole" in env_lower:
        threshold = 475.0
    elif "tmaze" in env_lower or "t_maze" in env_lower or "t-maze" in env_lower:
        threshold = 1.0
    else:
        threshold = kwargs.get("threshold", 0.0)

    # 1. Calculate binary 'hits' (1.0 if reward >= threshold, else 0.0)
    rewards = _ensure_numpy(df_valid["test_reward"])
    steps = _get_steps(df_valid, ["total_timesteps", "update"])
    hits = (rewards >= threshold).astype(float) * 100.0

    # 2. Calculate Backward Rolling Success Rate (center=False)
    # Using the updated _rolling_mean you modified earlier
    success_smooth, used_window = _rolling_mean(hits, window)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Standard experiment coloring
    styles = {
        'ann_baseline': 'tab:blue',
        'snn_actor_snntiming_critic': 'tab:purple',
        'ann2snn_both': 'tab:red',
        'ann2snn_actor': 'tab:orange',
        'snn_actor_ann_critic': 'tab:green'
        
    }
    color = styles.get(exp_name, 'tab:green')

    # 3. Plot Success Rate with Area Fill
    ax.plot(steps, success_smooth, lw=2.5, color=color, label=f"Success Rate (w={used_window})")
    ax.fill_between(steps, 0, success_smooth, color=color, alpha=0.1)

    # 4. Annotate Peak Success
    max_idx = np.argmax(success_smooth)
    ax.plot(steps[max_idx], success_smooth[max_idx], marker='*', color='gold', 
            markersize=12, markeredgecolor='black', zorder=10)
    
    ax.annotate(f'Peak: {success_smooth[max_idx]:.1f}%', 
                 xy=(steps[max_idx], success_smooth[max_idx]),
                 xytext=(-30, 15), textcoords='offset points', fontweight='bold', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gold", lw=1.5))

    # Formatting
    ax.set_ylim(-5, 105)
    xlabel = _xlabel_from_cols([df_valid], ["total_timesteps", "update"])
    if xlabel == "Environment Steps":
        _format_steps_axis(ax)
    else:
        ax.set_xlabel(xlabel, fontweight="bold")
        
    ax.set_ylabel("Success Rate (%)", fontweight="bold")
    _set_titles(ax, f"{title} - {env_name}", subtitle=f"{exp_name} (Threshold >= {threshold})")
    ax.legend(frameon=True, loc="lower right")
    
    _savefig(fig, save_path)


# ----------------------------------------------------------------------------- energy efficiency --------------------------------------------------------------------------------------------------------------------------------
def plot_single_seed_energy_vs_steps(
    df: pd.DataFrame,
    save_path: str,
    exp_name: str = "ann_baseline",
    config: Optional[Dict] = None,
    env_name: Optional[str] = None,
    window: int = 50,
    title: str = "Energy Consumption & Efficiency",
    **kwargs
):
    """
    Industry-standard single-seed energy plot.
    Shows efficiency (J/step), cumulative energy (kJ), and policy stability.
    """
    df_local = df.copy()
    
    # 1. Pre-process Energy Metrics
    # Calculate Per-Step Energy and Cumulative Total
    if "total_timesteps" in df_local.columns:
        step_delta = df_local["total_timesteps"].diff().fillna(df_local["total_timesteps"])
        step_delta = step_delta.replace(0, np.nan)
    else:
        _placeholder_plot("Missing timestep data for energy calculation", save_path)
        return

    # Determine metric to plot (Train or Inference)
    if "train_rollout_energy" in df_local.columns:
        metric_col = "train_energy_per_step"
        raw_energy = df_local["train_rollout_energy"]
        df_local[metric_col] = raw_energy / step_delta
        y_label = "Training Energy (J/step)"
    elif "inference_energy" in df_local.columns:
        metric_col = "inference_energy_per_step"
        # Often inference energy is logged per episode
        ep_len_col = "episode_length_steps" if "episode_length_steps" in df_local.columns else "episode_length"
        if ep_len_col in df_local.columns:
             df_local[metric_col] = df_local["inference_energy"] / df_local[ep_len_col]
        else:
             df_local[metric_col] = df_local["inference_energy"] / step_delta
        y_label = "Inference Energy (J/step)"
    else:
        _placeholder_plot("No energy data found in DataFrame", save_path)
        return

    df_valid = df_local.dropna(subset=[metric_col])
    steps = _get_steps(df_valid, ["total_timesteps", "update"])
    raw_efficiency = _ensure_numpy(df_valid[metric_col])
    
    # Calculate Cumulative Energy in kiloJoules (kJ)
    # Using 'train_rollout_energy' or 'inference_energy' if available
    energy_source = "train_rollout_energy" if "train_rollout_energy" in df_valid.columns else "inference_energy"
    cumulative_energy_kj = np.cumsum(df_valid[energy_source].fillna(0)) / 1000.0

    # Calculate backward stats for the efficiency line
    smooth_eff, std_eff, used_window = _rolling_mean(raw_efficiency, window)

    # Dynamic configuration
    env_name = env_name or (config.get("env", {}).get("id", "Unknown Env") if config else "Unknown Env")
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    color_eff = "purple" if "inference" in metric_col else "tab:gray"

    # 2. Plot Efficiency (Left Axis)
    ax1.plot(steps, raw_efficiency, alpha=0.15, color=color_eff, lw=1, label=f"Raw {y_label}")
    ax1.plot(steps, smooth_eff, lw=2.5, color=color_eff, label=f"Smoothed Efficiency (w={used_window})")
    ax1.fill_between(steps, smooth_eff - std_eff, smooth_eff + std_eff, 
                     color=color_eff, alpha=0.15, label="Firing Stability (±1 Std)")

    # 3. Annotate Final Efficiency
    ax1.plot(steps[-1], smooth_eff[-1], marker='o', color=color_eff, markersize=8)
    ax1.annotate(f'Final: {smooth_eff[-1]:.4f} J/step', 
                 xy=(steps[-1], smooth_eff[-1]),
                 xytext=(-120, 10), textcoords='offset points', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_eff, lw=1.5))

    ax1.set_xlabel("Environment Steps", fontsize=12, fontweight="bold")
    ax1.set_ylabel(y_label, fontsize=12, fontweight="bold", color=color_eff)
    ax1.tick_params(axis='y', labelcolor=color_eff)
    ax1.grid(True, linestyle=":", alpha=0.7)

    # 4. Plot Cumulative Energy (Right Axis)
    ax2 = ax1.twinx()
    color_total = "tab:red"
    ax2.plot(steps, cumulative_energy_kj, lw=2, ls="--", color=color_total, label="Total Energy Budget")
    ax2.set_ylabel("Total Energy Consumed (kJ)", fontsize=12, fontweight="bold", color=color_total)
    ax2.tick_params(axis='y', labelcolor=color_total)
    ax2.set_ylim(0, max(cumulative_energy_kj) * 1.2)

    # Combine Legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper center", ncol=2, framealpha=0.9)

    _format_steps_axis(ax1)
    _set_titles(ax1, f"{title} - {env_name}", subtitle=f"{exp_name} (Single Seed)")
    _savefig(fig, save_path)
    

# --------------------------------------------------------------------- spike activity --------------------------------------------------------------------------------------------------------------------------------    
def plot_spikes_vs_steps(
    df: pd.DataFrame,
    save_path: str,
    exp_name: str = "snn_actor_snntiming_critic",
    config: Optional[Dict] = None,
    **kwargs
):
    """
    Plots decomposed spike activity for Actor and Critic components.
    Perfect for 'both' experiments where multiple networks spike.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define components to look for
    components = {
        'spikes/actor': {'color': 'tab:red', 'label': 'Actor Spikes', 'alpha': 0.8},
        'spikes/critic': {'color': 'tab:orange', 'label': 'Critic Spikes', 'alpha': 0.8},
    }
    
    steps = _get_steps(df, ["total_timesteps", "update"])
    
    for col, style in components.items():
        if col in df.columns:
            raw = _ensure_numpy(df[col].fillna(0))
            smooth, std, _ = _rolling_mean(raw, window=50)
            
            # Plot smoothed line
            ax.plot(steps, smooth, lw=2.5, color=style['color'], label=style['label'])
            # Plot stability band
            ax.fill_between(steps, smooth - std, smooth + std, 
                             color=style['color'], alpha=0.1)

    # Add a "Total" line if both exist
    if 'spikes/actor' in df.columns and 'spikes/critic' in df.columns:
        total = df['spikes/actor'].fillna(0) + df['spikes/critic'].fillna(0)
        smooth_total, _, _ = _rolling_mean(total, window=50)
        ax.plot(steps, smooth_total, lw=1.5, ls='--', color='black', label='Total Activity', alpha=0.6)

    ax.set_ylabel("Spikes / Step", fontweight="bold")
    ax.set_xlabel("Environment Steps", fontweight="bold")
    _format_steps_axis(ax)
    ax.legend(loc="upper right", frameon=True)
    _set_titles(ax, "Decomposed Spike Activity", subtitle=f"{exp_name}")
    
    _savefig(fig, save_path)


# --------------------------------------------------------------------- latency analysis --------------------------------------------------------------------------------------------------------------------------------
def plot_single_seed_latency(
    df: pd.DataFrame,
    save_path: str,
    exp_name: str = "snn_actor_snntiming_critic",
    config: Optional[Dict] = None,
    env_name: Optional[str] = None,
    window: int = 50,
    title: str = "Decision Latency & Timing",
    **kwargs
):
    """
    Industry-standard single-seed latency plot.
    Handles decision-time (steps) for SNNs and wall-clock (ms) for ANNs.
    """
    df_valid = df.copy()
    steps = _get_steps(df_valid, ["total_timesteps", "update"])
    
    # 1. Identify Latency Type (Decision Time vs Wall-Clock)
    # Priority: Spike Timing (Internal steps) > Wall-Clock (ms)
    snn_cols = ["latency/actor_spike_timing_steps", "latency/critic_spike_timing_steps", "latency/spike_timing_steps"]
    ann_cols = ["latency_mean_ms", "latency/eval_wall_clock_ms"]
    
    active_snn_cols = [c for c in snn_cols if c in df_valid.columns]
    active_ann_cols = [c for c in ann_cols if c in df_valid.columns]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Standard Latency Styling
    styles = {
        'actor': {'color': 'tab:red', 'label': 'Actor Decision Time'},
        'critic': {'color': 'tab:orange', 'label': 'Critic Decision Time'},
        'general': {'color': 'tab:blue', 'label': 'Decision Latency'},
        'ann': {'color': 'tab:brown', 'label': 'ANN Wall-Clock'}
    }

    # 2. Plotting Logic
    if active_snn_cols:
        ylabel = "Decision Latency (Internal Steps $\\tau$)"
        for col in active_snn_cols:
            raw = _ensure_numpy(df_valid[col].fillna(0))
            smooth, std, w = _rolling_mean(raw, window)
            
            # Determine sub-label
            label = styles['actor']['label'] if 'actor' in col else (styles['critic']['label'] if 'critic' in col else styles['general']['label'])
            color = styles['actor']['color'] if 'actor' in col else (styles['critic']['color'] if 'critic' in col else styles['general']['color'])
            
            ax.plot(steps, smooth, lw=2.5, color=color, label=f"{label} (w={w})")
            ax.fill_between(steps, smooth - std, smooth + std, color=color, alpha=0.15)
            
            # Annotate Minimum Decision Time (The "Fast" Point)
            min_idx = np.argmin(smooth)
            ax.plot(steps[min_idx], smooth[min_idx], marker='*', color='gold', markersize=10, markeredgecolor='black')

    elif active_ann_cols:
        ylabel = "Rollout Wall-Clock Latency (ms)"
        for col in active_ann_cols:
            raw = _ensure_numpy(df_valid[col].fillna(0))
            smooth, _, w = _rolling_mean(raw, window)
            ax.plot(steps, smooth, lw=2.5, color=styles['ann']['color'], label=f"ANN Latency (w={w})")

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
    _set_titles(ax, f"{title} - {env_name}", subtitle=f"{exp_name} (Single Seed)")
    ax.legend(loc="upper right", frameon=True)
    
    _savefig(fig, save_path)


# def plot_activation_counts(
#     activations: dict, 
#     temporal: bool, 
#     title: str, 
#     save_path: str = None, 
#     **kwargs
# ):
#     """Visualizes layer-wise activation statistics."""
#     fig, ax = plt.subplots()
    
#     for name, arr in activations.items():
#         if hasattr(arr, "detach"): 
#             arr = arr.detach().cpu().numpy()
#         arr = np.asarray(arr)
        
#         # Handle 1D Time Series vs 2D [Time, Batch] ---
#         if temporal:
#             if arr.ndim == 1:
#                 # Direct time series (e.g. pre-averaged)
#                 ax.plot(arr, label=name, lw=2)
#             elif arr.ndim >= 2:
#                 # Average over batch/features if multi-dim
#                 mean_act = arr.mean(axis=tuple(range(1, arr.ndim)))
#                 ax.plot(mean_act, label=name, lw=2)
#         else:
#             # Bar chart for scalar means
#             ax.bar(name, float(arr.mean()))

#     ax.set_ylabel("Mean Activations / Spikes", fontweight="bold")
#     _set_titles(ax, title)
    
#     if temporal:
#         ax.set_xlabel("Time Step (Validation)")
#         ax.legend(frameon=True)
        
#     if save_path:
#         _savefig(fig, save_path)
#     else:
#         plt.show()


# --------------------------------------------------------------------- output readout validation --------------------------------------------------------------------------------------------------------------------------------
def plot_output_readout_validation(
    output_potential: np.ndarray,
    output_spikes: np.ndarray,
    save_path: str,
    threshold: float = 1.0,
    reset_val: float = 0.0,
    is_internal_window: bool = True, # New flag
    exp_name: str = "snn_actor_snntiming_critic",
    title: str = "Critic Output Neuron Dynamics",
    **kwargs,
):
    """
    Plots critic membrane potential and spikes across internal timesteps (tau).
    """
    pot = _ensure_numpy(output_potential).ravel()
    spikes = _ensure_numpy(output_spikes).ravel()
    
    min_len = min(pot.size, spikes.size)
    pot, spikes = pot[:min_len], spikes[:min_len]
    steps = np.arange(min_len)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Potential and Threshold
    ax.plot(steps, pot, color="tab:blue", lw=2, label="Critic Potential $U(\\tau)$")
    ax.axhline(y=threshold, color="tab:red", ls="--", label="Threshold $\\theta$")

    # Spikes
    spike_indices = np.where(spikes > 0)[0]
    if len(spike_indices) > 0:
        ax.scatter(spike_indices, [threshold] * len(spike_indices), 
                   color="tab:green", marker="^", s=100, label="Value-Encoding Spike", zorder=3)
        for idx in spike_indices:
            ax.axvline(x=idx, color="tab:green", alpha=0.2, ls=':')

    # Dynamic Labeling
    x_label = "Internal Simulation Step ($\\tau$)" if is_internal_window else "Environment Step ($T$)"
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel("Membrane Potential", fontsize=12, fontweight="bold")
    
    _set_titles(ax, title, subtitle=f"Experiment: {exp_name}")
    ax.legend(loc="upper right")
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
    Plots the competition between different action neurons in the Actor
    across the internal simulation window (tau).
    """
    pots = _ensure_numpy(output_potentials) # [N, tau]
    spks = _ensure_numpy(output_spikes)     # [N, tau]
    
    num_actions, tau = pots.shape
    steps = np.arange(tau)
    
    if action_names is None:
        action_names = [f"Action {i}" for i in range(num_actions)]

    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Use a colormap to distinguish actions
    colors = plt.cm.get_cmap('tab10', num_actions)

    for i in range(num_actions):
        # 1. Plot Membrane Potential for this action
        ax.plot(steps, pots[i], color=colors(i), lw=2, 
                label=f"$U(\\tau)$ - {action_names[i]}", alpha=0.8)
        
        # 2. Plot Spikes for this action
        spike_idx = np.where(spks[i] > 0)[0]
        if len(spike_idx) > 0:
            ax.scatter(spike_idx, [threshold + (i*0.05)] * len(spike_idx), 
                       color=colors(i), marker="d", s=70, 
                       label=f"Spike - {action_names[i]}", zorder=5)

    # 3. Add Threshold Line
    ax.axhline(y=threshold, color="black", ls="--", alpha=0.6, label="Firing Threshold")

    # Formatting
    ax.set_xlabel("Internal Simulation Step ($\\tau$)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Membrane Potential", fontsize=12, fontweight="bold")
    
    _set_titles(ax, title, subtitle=f"Experiment: {exp_name} | Decision Window")
    
    # Move legend outside if there are many actions
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    _savefig(fig, save_path)




# def plot_conversion_validation(ann_values, snn_values, save_path, **kwargs):
#     """Scatter plot comparing ANN logits vs SNN logits."""
#     if ann_values is None or snn_values is None: return
    
#     ann, snn = np.asarray(ann_values).ravel(), np.asarray(snn_values).ravel()
    
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.scatter(ann, snn, s=15, alpha=0.5, color="tab:blue", edgecolors='none')
    
#     # Identity line
#     lims = (min(ann.min(), snn.min()), max(ann.max(), snn.max()))
#     ax.plot(lims, lims, "r--", lw=1.5, label="Ideal (y=x)")
    
#     ax.set_xlabel("ANN Output", fontweight="bold")
#     ax.set_ylabel("SNN Output", fontweight="bold")
#     _set_titles(ax, "ANN-SNN Conversion Fidelity")
#     ax.legend()
#     ax.grid(True, linestyle=":")
#     _savefig(fig, save_path)


# def plot_timing_critic_correlation(timing, values, save_path, **kwargs):
#     """Scatter plot + binned mean/std + correlation stats for SNN Timing Critic."""
#     if timing is None or values is None:
#         return

#     x = np.asarray(timing).ravel()
#     y = np.asarray(values).ravel()
#     mask = np.isfinite(x) & np.isfinite(y)
#     x = x[mask]
#     y = y[mask]
#     if x.size == 0 or y.size == 0:
#         _placeholder_plot("No timing/value data", save_path)
#         return

#     bins = int(kwargs.get("bins", 20))

#     # Correlations (prefer scipy if available for p-values)
#     pearson_r = np.nan
#     pearson_p = np.nan
#     spearman_r = np.nan
#     spearman_p = np.nan
#     if x.size >= 2 and y.size >= 2 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
#         try:
#             from scipy import stats as _stats
#             pearson_r, pearson_p = _stats.pearsonr(x, y)
#             spearman_r, spearman_p = _stats.spearmanr(x, y)
#         except Exception:
#             # Fallback to correlation only (no p-values)
#             try:
#                 pearson_r = float(np.corrcoef(x, y)[0, 1])
#                 # Approx spearman via ranks
#                 rx = np.argsort(np.argsort(x))
#                 ry = np.argsort(np.argsort(y))
#                 spearman_r = float(np.corrcoef(rx, ry)[0, 1])
#             except Exception:
#                 pass

#     fig, (ax_hist, ax) = plt.subplots(
#         2, 1, figsize=(7.5, 7.5), gridspec_kw={"height_ratios": [1, 3]}
#     )

#     # Histogram of timing distribution (tau)
#     ax_hist.hist(x, bins=bins, color="tab:gray", alpha=0.7, edgecolor="white")
#     tau_mean = float(np.mean(x)) if x.size else float("nan")
#     tau_std = float(np.std(x)) if x.size else float("nan")
#     tau_unique = float(len(np.unique(x))) if x.size else 0.0
#     ax_hist.text(
#         0.02,
#         0.95,
#         f"tau mean={tau_mean:.3f}\\\\\n"
#         f"tau std={tau_std:.3f}\\\\\n"
#         f"unique tau={tau_unique:.0f}",
#         transform=ax_hist.transAxes,
#         va="top",
#         ha="left",
#         fontsize=8,
#         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
#     )
#     ax_hist.set_ylabel("Count", fontweight="bold")
#     ax_hist.set_xlabel("First Spike Time (steps)", fontweight="bold")
#     ax_hist.grid(True, linestyle=":", alpha=0.6)

#     # Scatter + binned mean/std
#     ax.scatter(x, y, alpha=0.45, c=y, cmap="viridis", s=18, edgecolors="none")
#     centers, mean, std = _bin_stats(x, y, bins=bins)
#     if len(centers) > 0 and np.isfinite(mean).any():
#         ax.plot(centers, mean, color="black", lw=2.0, label="Binned Mean")
#         if np.isfinite(std).any():
#             ax.fill_between(centers, mean - std, mean + std, color="black", alpha=0.2, label="±1 Std. Dev.")

#     # Annotate correlation stats
#     stat_lines = [
#         f"Pearson r={pearson_r:.3f}, p={pearson_p:.3g}" if np.isfinite(pearson_r) else "Pearson r=NA, p=NA",
#         f"Spearman r={spearman_r:.3f}, p={spearman_p:.3g}" if np.isfinite(spearman_r) else "Spearman r=NA, p=NA",
#     ]
#     ax.text(
#         0.02,
#         0.98,
#         "\n".join(stat_lines),
#         transform=ax.transAxes,
#         va="top",
#         ha="left",
#         fontsize=9,
#         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
#     )

#     ax.set_xlabel("First Spike Time (steps)", fontweight="bold")
#     ax.set_ylabel("Predicted Value", fontweight="bold")
#     _set_titles(ax, "Timing vs. Value Correlation")
#     _safe_legend(ax)
#     ax.grid(True, linestyle=":", alpha=0.6)
#     fig.tight_layout()
#     _savefig(fig, save_path)


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
    r2 = r2_score(ann, snn)
    mse = mean_squared_error(ann, snn)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Color mapping based on error magnitude
    errors = np.abs(ann - snn)
    scatter = ax.scatter(ann, snn, c=errors, cmap='viridis_r', s=15, alpha=0.6, edgecolors='none')
    
    # Identity Line
    mn, mx = min(ann.min(), snn.min()), max(ann.max(), snn.max())
    lims = [mn, mx]
    ax.plot(lims, lims, "r--", lw=2, label="Ideal Fidelity ($y=x$)", zorder=5)
    
    # Left Box: Performance Metrics
    stats_text = f"$R^2$: {r2:.4f}\n$MSE$: {mse:.4f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

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
    
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Absolute Conversion Error', fontweight='bold')
    
    _savefig(fig, save_path)


def _select_first_available(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def _bin_stats(x: np.ndarray, y: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])
    edges = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(bins, np.nan)
    std = np.full(bins, np.nan)
    for i in range(bins):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if mask.any():
            mean[i] = np.nanmean(y[mask])
            std[i] = np.nanstd(y[mask])
    return centers, mean, std


def _scatter_with_bands(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 20,
    color: str = "tab:blue",
    label: Optional[str] = None,
    min_points: int = 20,
):
    ax.scatter(x, y, s=18, alpha=0.25, color=color, edgecolors="none", label=label)
    if len(x) < min_points:
        return
    centers, mean, std = _bin_stats(x, y, bins)
    if len(centers) > 0:
        ax.plot(centers, mean, color=color, lw=2.5, label="Binned Mean")
        ax.fill_between(centers, mean - std, mean + std, color=color, alpha=0.2, label="±1 Std. Dev.")


def _safe_legend(ax: plt.Axes):
    handles, labels = ax.get_legend_handles_labels()
    if any(l for l in labels):
        ax.legend(frameon=True)



# --------------------------------------------------------------------- energy vs spikes correlation --------------------------------------------------------------------------------------------------------------------------------
def plot_energy_vs_spikes(
    df: pd.DataFrame,
    save_path: str,
    exp_name: str = "snn_actor_snntiming_critic",
    env_name: str = "T-Maze",
    **kwargs
):
    """
    Industry-standard Efficiency Correlation plot.
    Calculates cost-per-spike and static energy overhead.
    """
    # 1. Select the most relevant columns
    x_col = _select_first_available(df, ["spikes/per_step", "spikes/firing_rate", "spikes/total"])
    y_col = _select_first_available(df, ["inference_energy", "train_rollout_energy", "total_energy"])
    
    if not x_col or not y_col:
        _placeholder_plot("Missing Spike or Energy Data", save_path)
        return

    data = df[[x_col, y_col]].dropna()
    x = data[x_col].to_numpy()
    y = data[y_col].to_numpy()

    # 2. Linear Regression (The Efficiency Law)
    slope, intercept, r_val, p_val, std_err = linregress(x, y)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 3. Density Scatter (Color by energy magnitude)
    scatter = ax.scatter(x, y, c=y, cmap='magma', s=35, alpha=0.6, edgecolors='none', label="Rollout Samples")
    
    # 4. Regression Line
    x_range = np.array([x.min(), x.max()])
    ax.plot(x_range, intercept + slope * x_range, 'r--', lw=2, label=f"Linear Fit ($R^2={r_val**2:.3f}$)")

    # 5. Stats Annotation
    # Slope is J/spike, intercept is static overhead J
    stats_text = (f"Cost/Spike: {slope*1000:.2f} mJ\n"
                  f"Static Overhead: {intercept*1000:.2f} mJ\n"
                  f"Pearson R: {r_val:.3f}")
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Formatting
    ax.set_xlabel("Spike Activity (Spikes/Step)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Energy Consumption (Joules)", fontsize=12, fontweight="bold")
    
    _set_titles(ax, "Energy-Spike Efficiency Correlation", subtitle=f"{exp_name} | {env_name}")
    
    ax.grid(True, linestyle=":", alpha=0.6)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Energy Magnitude (J)', fontweight='bold')
    
    ax.legend(loc='lower right', frameon=True)
    _savefig(fig, save_path)


# --------------------------------------------------------------------- reward vs spikes correlation --------------------------------------------------------------------------------------------------------------------------------
def plot_reward_vs_spikes(
    df: pd.DataFrame,
    save_path: str,
    exp_name: str = "snn_actor_snntiming_critic",
    env_name: str = "CartPole",
    **kwargs
):
    """
    Industry-standard Reward-Sparsity Pareto Analysis.
    Visualizes the trade-off between neural activity and behavioral performance.
    """
    # 1. Select Columns
    x_col = _select_first_available(df, ["spikes/per_step", "spikes/firing_rate", "spikes/total"])
    y_col = _select_first_available(df, ["test_reward", "train_reward"])
    
    if not x_col or not y_col:
        _placeholder_plot("Missing Spike or Reward Data", save_path)
        return

    data = df[[x_col, y_col]].dropna().sort_values(by=x_col)
    spikes = data[x_col].to_numpy()
    reward = data[y_col].to_numpy()

    # Determine Solved Threshold
    env_lower = env_name.lower()
    threshold = 475.0 if "cartpole" in env_lower else (1.0 if "tmaze" in env_lower else None)

    fig, ax = plt.subplots(figsize=(10, 7))

    # 2. Density Scatter (Color by Reward)
    scatter = ax.scatter(spikes, reward, c=reward, cmap='viridis', s=40, alpha=0.7, edgecolors='none', label="Rollout Samples")

    # 3. Efficiency Frontier (Moving average of reward across spike bins)
    smooth_reward = data[y_col].rolling(window=len(data)//10, min_periods=1, center=True).mean()
    ax.plot(spikes, smooth_reward, color='tab:red', lw=3, label="Efficiency Frontier", zorder=10)

    # 4. Solved Threshold & Shading
    if threshold is not None:
        ax.axhline(threshold, color='red', linestyle='--', alpha=0.6, label=f"Solved ({threshold})")
        ax.axhspan(threshold, max(reward)*1.1 if max(reward) > threshold else threshold*1.1, 
                   color='green', alpha=0.05, label="Target Performance")

        # 5. Sparsity Annotation
        solved_data = data[data[y_col] >= threshold]
        if not solved_data.empty:
            avg_spikes_solved = solved_data[x_col].mean()
            ax.annotate(f'Sparsity at Solved:\n{avg_spikes_solved:.2f} Spikes/Step',
                        xy=(avg_spikes_solved, threshold), xytext=(avg_spikes_solved + 0.3, threshold * 0.8),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1),
                        fontweight='bold')

    # Formatting
    ax.set_xlabel("Neural Activity (Spikes/Step)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rollout Reward", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    _set_titles(ax, "Reward-Sparsity Pareto Analysis", subtitle=f"{exp_name} | {env_name}")
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Reward Level', fontweight='bold')
    
    ax.legend(loc='lower left', framealpha=0.9)
    _savefig(fig, save_path)
    
    
# --------------------------------------------------------------------- latency vs spikes correlation --------------------------------------------------------------------------------------------------------------------------------
def plot_latency_vs_spikes(
    df: pd.DataFrame,
    save_path: str,
    exp_name: str = "snn_actor_snntiming_critic",
    env_name: str = "CartPole",
    component: str = "Actor", # 'Actor' or 'Critic'
    **kwargs
):
    """
    Industry-standard plot for the Neural Speed-Cost Trade-off.
    Specially tuned to differentiate between Actor reaction time and Critic estimation time.
    """
    # 1. Select Columns based on component
    if component.lower() == "actor":
        x_col = _select_first_available(df, ["spikes/actor", "spikes/per_step", "spikes/firing_rate"])
        y_col = _select_first_available(df, ["latency/actor_spike_timing_steps", "latency/spike_timing_steps"])
        color_map = 'Reds'
    else:
        x_col = _select_first_available(df, ["spikes/critic", "spikes/per_step", "spikes/firing_rate"])
        y_col = _select_first_available(df, ["latency/critic_spike_timing_steps", "latency/spike_timing_steps"])
        color_map = 'Oranges'

    if not x_col or not y_col:
        _placeholder_plot(f"Missing {component} Data", save_path)
        return

    data = df[[x_col, y_col]].dropna()
    x, y = data[x_col].to_numpy(), data[y_col].to_numpy()

    # 2. Regression Analysis
    slope, intercept, r_val, _, _ = linregress(x, y)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 3. Density Scatter (Red for Actor, Orange for Critic)
    scatter = ax.scatter(x, y, c=y, cmap=color_map, s=40, alpha=0.7, edgecolors='none')
    
    # 4. Correlation Line
    x_range = np.array([x.min(), x.max()])
    ax.plot(x_range, intercept + slope * x_range, 'k--', lw=2, label=f"{component} Trend ($R^2={r_val**2:.3f}$)")

    # 5. Stats Annotation
    y_unit = "$\\tau$"
    stats_text = (f"{component} Reactivity: {slope:.2f} {y_unit}/spike\n"
                  f"Min Latency: {intercept:.2f} {y_unit}\n"
                  f"Pearson R: {r_val:.3f}")
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Formatting
    ax.set_xlabel(f"{component} Activity (Spikes/Step)", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Decision Latency ({y_unit})", fontsize=12, fontweight="bold")
    _set_titles(ax, f"{component} Speed-Cost Trade-off", subtitle=f"{exp_name} | {env_name}")
    
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.colorbar(scatter, ax=ax).set_label(f"Latency ({y_unit})", fontweight='bold')
    ax.legend(loc='lower right')
    
    _savefig(fig, save_path)

# def plot_episode_length_vs_energy(
#     per_episode_data: Union[pd.DataFrame, Iterable[pd.DataFrame], Dict[str, pd.DataFrame]],
#     save_path: str,
#     bins: int = 20,
#     title: str = "Episode Length vs Energy",
#     **kwargs,
# ):
#     dfs = _as_run_list(per_episode_data)
#     if not dfs:
#         _placeholder_plot("No data for episode length vs energy", save_path)
#         return
#     x_col = _select_first_available(dfs[0], ["episode_length_steps"])
#     y_col = _select_first_available(dfs[0], ["inference_energy", "total_energy"])
#     if not x_col or not y_col:
#         _placeholder_plot("Missing episode length or energy data", save_path)
#         return
#     data = pd.concat([df[[x_col, y_col]].dropna() for df in dfs], ignore_index=True)
#     if data[x_col].nunique() <= 1 or data[x_col].std() < 1.0:
#         _placeholder_plot("Episode length near-constant", save_path)
#         return
#     fig, ax = plt.subplots()
#     _scatter_with_bands(ax, data[x_col].to_numpy(), data[y_col].to_numpy(), bins=bins, color="tab:orange")
#     ax.set_xlabel("Episode Length (steps)", fontweight="bold")
#     ax.set_ylabel("Energy (Joules)", fontweight="bold")
#     _set_titles(ax, title)
#     _safe_legend(ax)
#     _savefig(fig, save_path)



def plot_snn_phase(log_dir, save_dir=None):
    """
    Plots metrics for SNN Zero-Shot (Phase 3) and Fine-Tuning (Phase 4).
    Aligns time series to start from the Zero-Shot point (t=0).
    UPDATED: Now includes ANN Baseline comparison line.
    """
    if isinstance(log_dir, dict):
        logger_obj = log_dir.get("logger")
        if logger_obj is not None and hasattr(logger_obj, "log_dir"):
            log_dir = str(logger_obj.log_dir)
        else:
            log_dir = log_dir.get("log_dir")
        if log_dir is None:
            print("[Plotting] Error: log_dir is missing in result dict.")
            return
    log_dir = os.fspath(log_dir)
    if save_dir is None:
        save_dir = os.path.join(log_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(log_dir, "per_episode_metrics.csv")
    # Support standard monitor.csv as fallback if using Stable Baselines logger style
    if not os.path.exists(csv_path):
        csv_path = os.path.join(log_dir, "progress.csv")
        
    if not os.path.exists(csv_path):
        print(f"[Plotting] Error: Could not find metrics CSV at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # --- Configuration ---
    keys = {
        'ann_reward': 'train/rollout_reward',  # Phase 1 Key
        'zs_reward': 'post_conversion/zero_shot_reward',
        'zs_energy': 'post_conversion/inference_energy',
        'zs_latency': 'post_conversion/mean_latency',
        'ft_train_reward': 'post_conversion_ft/train_reward',
        'ft_eval_reward': 'post_conversion_ft/eval_reward',
        'ft_energy': 'post_conversion_ft/energy/inference',
        'ft_latency': 'post_conversion_ft/train_latency',
        'time': 'time/total_timesteps_snn'
    }

    # --- 1. Extract ANN Baseline (Phase 1) ---
    ann_baseline = None
    if keys['ann_reward'] in df.columns:
        # Get the average of the last 10 episodes of Phase 1
        # We assume Phase 1 data exists where 'zs_reward' is NaN and 'ft_train_reward' is NaN
        ann_data = df[df[keys['ann_reward']].notna()]
        if not ann_data.empty:
            # Take mean of last 5% of ANN training to get stable performance
            tail_len = max(1, int(len(ann_data) * 0.05))
            ann_baseline = ann_data[keys['ann_reward']].iloc[-tail_len:].mean()
            print(f"[Plotting] ANN Baseline Reward: {ann_baseline:.2f}")

    # --- 2. Extract Zero-Shot Baseline ---
    zs_data = None
    start_step = 0
    
    if keys['zs_reward'] in df.columns:
        zs_rows = df[df[keys['zs_reward']].notna()]
        if not zs_rows.empty:
            row = zs_rows.iloc[-1]
            start_step = row.get(keys['time'], 0)
            
            zs_data = {
                'reward': row.get(keys['zs_reward'], np.nan),
                'energy': row.get(keys['zs_energy'], np.nan),
                'latency': row.get(keys['zs_latency'], np.nan)
            }

    # --- 3. Extract Fine-Tuning Data ---
    ft_df = pd.DataFrame()
    if keys['ft_train_reward'] in df.columns:
        ft_df = df[df[keys['ft_train_reward']].notna()].copy()

    has_ft_data = not ft_df.empty
    
    if not has_ft_data and zs_data is None:
        print("[Plotting] No SNN post-conversion data found.")
        return

    # Normalize X-axis
    if has_ft_data:
        time_col = keys['time']
        if time_col in ft_df.columns:
            ft_df['rel_step'] = ft_df[time_col] - start_step
        else:
            ft_df['rel_step'] = np.arange(1, len(ft_df) + 1) * 2048

    # --- 4. Generate 2x2 Grid ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SNN Post-Conversion Dynamics', fontsize=18, fontweight='bold')

    # Plot A: Fine-Tuning Stability (Train Reward)
    ax = axes[0, 0]
    
    # Draw ANN Baseline
    if ann_baseline is not None:
        ax.axhline(ann_baseline, color='gray', linestyle='--', linewidth=1.5, label=f'ANN Baseline ({ann_baseline:.0f})')
        
    if has_ft_data and keys['ft_train_reward'] in ft_df.columns:
        ax.plot(ft_df['rel_step'], ft_df[keys['ft_train_reward']], 
                color='purple', alpha=0.2, label='Raw')
        ma = ft_df[keys['ft_train_reward']].rolling(window=10, min_periods=1).mean()
        ax.plot(ft_df['rel_step'], ma, color='purple', lw=2, label='Smoothed')
    
    # Always include Zero-Shot point on the training graph for continuity
    if zs_data:
        ax.plot(0, zs_data['reward'], marker='*', markersize=12, color='gold', markeredgecolor='black', label='Zero-Shot')

    ax.set_title("Fine-Tuning Stability")
    ax.set_ylabel("Rollout Reward")
    ax.set_xlabel("Steps after Conversion")
    ax.legend(loc='lower right')

    # Plot B: Evaluation Performance (Zero-Shot + FT)
    ax = axes[0, 1]
    
    if ann_baseline is not None:
        ax.axhline(ann_baseline, color='gray', linestyle='--', linewidth=1.5, label='ANN Baseline')

    # Zero-Shot Point
    if zs_data:
        ax.plot(0, zs_data['reward'], marker='*', markersize=18, color='gold', 
                markeredgecolor='black', label='Zero-Shot', zorder=10)
    
    # FT Points
    if has_ft_data and keys['ft_eval_reward'] in ft_df.columns:
        ft_eval = ft_df.dropna(subset=[keys['ft_eval_reward']])
        ax.plot(ft_eval['rel_step'], ft_eval[keys['ft_eval_reward']], 
                'o-', color='tab:green', lw=2, label='FT Eval')

    ax.set_title("Evaluation Performance")
    ax.set_ylabel("Episode Reward")
    ax.legend()

    # Plot C: Energy
    ax = axes[1, 0]
    if zs_data and not np.isnan(zs_data['energy']):
        ax.plot(0, zs_data['energy'], marker='*', markersize=15, color='gold', 
                markeredgecolor='black', zorder=10, label="Zero-Shot")
        
    if has_ft_data and keys['ft_energy'] in ft_df.columns:
        ft_energy = ft_df.dropna(subset=[keys['ft_energy']])
        ax.plot(ft_energy['rel_step'], ft_energy[keys['ft_energy']], 
                '^-', color='tab:orange', label='Inference Energy')

    ax.set_title("Inference Energy")
    ax.set_ylabel("Joules / Episode")
    ax.set_xlabel("Steps after Conversion")
    ax.legend()

    # Plot D: Latency
    ax = axes[1, 1]
    if zs_data and not np.isnan(zs_data['latency']):
        ax.axhline(zs_data['latency'], color='blue', linestyle='--', label='Zero-Shot Latency')
    
    if has_ft_data and keys['ft_latency'] in ft_df.columns:
        ft_lat = ft_df.dropna(subset=[keys['ft_latency']])
        ax.plot(ft_lat['rel_step'], ft_lat[keys['ft_latency']], 
                color='tab:red', label='FT Latency')

    ax.set_title("Latency Metric")
    ax.set_ylabel("Time (ms) or Spikes")
    ax.legend()
    
    # Plot E: Energy vs Spikes Scatter
    # Plot F: reward vs Spikes Scatter
    # Plot G: Latency vs Spikes Scatter
    # Plot H: Episode Length vs Energy Scatter

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, "snn_post_conversion_metrics.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plotting] ✅ Saved SNN Post-Conversion plots to {save_path}")
    
    # Cross-metric relationships (post-conversion only for spike-related plots)
    try:
        post_cols = [keys['zs_reward'], keys['ft_train_reward'], keys['ft_eval_reward']]
        available_post_cols = [c for c in post_cols if c in df.columns]
        if available_post_cols:
            df_post = df[df[available_post_cols].notna().any(axis=1)]
        else:
            df_post = pd.DataFrame()

        has_spikes = _select_first_available(
            df_post,
            ["spikes/per_step", "spikes/firing_rate", "spikes/total", "spike_count_total", "post_conversion/total_spikes"],
        ) is not None
        has_energy = _select_first_available(df, ["inference_energy", "total_energy"]) is not None
        has_ep_len = "episode_length_steps" in df.columns

        if has_spikes and not df_post.empty:
            plot_energy_vs_spikes(df_post, os.path.join(save_dir, "snn_energy_vs_spikes.png"))
            plot_reward_vs_spikes(df_post, os.path.join(save_dir, "snn_reward_vs_spikes.png"))
            plot_latency_vs_spikes(df_post, os.path.join(save_dir, "snn_latency_vs_spikes.png"))
        if has_energy and has_ep_len:
            plot_episode_length_vs_energy(df, os.path.join(save_dir, "snn_episode_length_vs_energy.png"))
    except Exception as e:
        print(f"[Plotting] Warning: Failed to generate cross-metric plots: {e}")
    

# def plot_research_results(
#     combined_df: pd.DataFrame,
#     save_path: str,
#     metric: str = "eval_episode_reward",
#     title: str = "Agent Performance",
#     xlabel: str = "Environment Steps",
#     ylabel: str = "Average Return",
#     window: int = 5,
#     max_steps: int = None,
#     ylim: tuple = None
# ):
#     """
#     Generates a publication-grade plot with Mean +/- Standard Deviation error bands.
#     Aggregates data from multiple seeds.
#     """
#     if metric not in combined_df.columns:
#         print(f"[Warning] Metric '{metric}' not found in dataframe. Skipping plot.")
#         return

#     # Clean data
#     df = combined_df.dropna(subset=[metric]).copy()
    
#     # X-Axis Alignment
#     if "total_timesteps" in df.columns:
#         # Group by 'update' to calculate mean timestep for that update index
#         grouped = df.groupby("update")
#         x_values = grouped["total_timesteps"].mean()
#     else:
#         grouped = df.groupby("update")
#         x_values = grouped.index

#     # Statistics
#     mean_values = grouped[metric].mean()
#     std_values = grouped[metric].std().fillna(0.0)

#     # Smoothing (Exponential Moving Average)
#     mean_smooth = mean_values.ewm(span=window, adjust=False).mean()
#     std_smooth = std_values.ewm(span=window, adjust=False).mean()

#     # Plot
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     color_line = "#1f77b4"
#     color_fill = "#a6cee3"

#     ax.plot(x_values, mean_smooth, color=color_line, lw=2.5, label="Mean")
#     ax.fill_between(
#         x_values, 
#         mean_smooth - std_smooth, 
#         mean_smooth + std_smooth, 
#         color=color_fill, alpha=0.4, label="±1 Std. Dev."
#     )

#     # Threshold Line (if relevant)
#     if "reward" in metric.lower() or "return" in ylabel.lower():
#         ax.axhline(475.0, ls="--", color="#d62728", lw=2, alpha=0.8, label="Solved (475)")
#         if ylim is None:
#             ax.set_ylim(bottom=0)

#     if max_steps:
#         ax.set_xlim(0, max_steps)
#     if ylim:
#         ax.set_ylim(ylim)

#     # Formatting
#     def human_format(num, pos):
#         return f'{num/1e6:.1f}M' if num >= 1e6 else f'{int(num)}'

#     ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    
#     ax.set_title(title, fontsize=18, fontweight="bold", pad=15)
#     ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
#     ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
#     ax.tick_params(labelsize=12)
#     ax.grid(True, linestyle=":", alpha=0.6)
#     ax.legend(fontsize=12, frameon=True, loc="best")

#     print(f"[Plotting] Saving research plot to {save_path}")
#     _savefig(fig, save_path)
    
    
    
# ------------------------------------------------------------------------- intra episode values (plot of actual critic values on average in an episode) -------------------------------------------------------------------------
def plot_intra_episode_values(
    values: Union[List[float], np.ndarray],
    save_path: str,
    exp_name: str = "ann_baseline",
    config: Optional[Dict] = None,
    env_name: Optional[str] = None,
    title: str = "Intra-Episode Value Dynamics",
    **kwargs
):
    """
    Plots the critic value at each timestep within a SINGLE episode
    for a SINGLE experiment/seed.
    
    Args:
        values: 1D array or list of predicted values across the episode.
        save_path: Path to save the plot.
        exp_name: The name of the experiment (used for standardized styling).
        config: Optional config dict to dynamically extract environment settings.
        env_name: Explicit environment name (overrides config if provided).
    """
    values = _ensure_numpy(values)
    if len(values) == 0:
        _placeholder_plot("No value data available for this episode", save_path)
        return

    # --- Dynamic Configuration Extraction ---
    if config is not None:
        if env_name is None:
            env_name = config.get("env", {}).get("id", config.get("env_name", "Unknown Env"))
            
    env_name = env_name or "Unknown Env"
    timesteps = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Standard styles to keep your visual language consistent across the paper
    styles = {
        'ann_baseline': {'color': 'tab:blue', 'ls': '-', 'lw': 3},
        'snn_actor_ann_critic': {'color': 'tab:green', 'ls': '--', 'lw': 2.5},
        'ann2snn_actor': {'color': 'tab:orange', 'ls': '-.', 'lw': 2.5},
        'ann2snn_both': {'color': 'tab:red', 'ls': ':', 'lw': 2.5},
        'snn_actor_snntiming_critic': {'color': 'tab:purple', 'ls': '-', 'lw': 3, 'marker': 'o', 'markersize': 5}
    }
    
    # Get standard style, or fallback to a default if exp_name is custom
    style = styles.get(exp_name, {'color': 'tab:blue', 'ls': '-', 'lw': 2.5})
    
    # Plot the single line
    ax.plot(timesteps, values, label=f"{exp_name} (Single Seed)", **style)

    # Formatting and Labels
    ax.set_xlabel('Timestep within Episode (actor_T)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted Value $V(s_t)$', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # Dynamic Axes Constraints
    ax.set_xlim(0, max(1, len(values) - 1))
    
    # Dynamically scale Y-axis with a 10% margin so curves aren't pressed against the ceiling.
    # This handles CartPole (values near 1.0) and T-Maze (values ramping from 0 to scale) seamlessly.
    y_min, y_max = np.min(values), np.max(values)
    margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
    ax.set_ylim(min(0, y_min - margin), y_max + margin)
    
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    
    # Use your custom title formatter
    _set_titles(ax, f"{title} - {env_name}", subtitle="Single Episode Evaluation")
    
    # Save using your custom wrapper
    _savefig(fig, save_path)

# ------------------------------------------------------------------------- snn_timing_critic_dynamics --------------------------------------------------------------------------------------------------------------------------
def plot_timing_critic_dynamics(
    taus: Union[List[int], np.ndarray],
    values: Union[List[float], np.ndarray],
    save_path: str,
    config: Optional[Dict] = None,
    actor_steps: Optional[int] = None,
    critic_internal_time: Optional[int] = None,
    env_name: Optional[str] = None,
    title: str = "Timing Critic Internal Dynamics",
    **kwargs
):
    """
    Visualizes the internal timing critic's spike times (tau) and values 
    within each actor step for a single rollout/episode.
    Dynamically scales based on the provided config.
    """
    taus = _ensure_numpy(taus)
    values = _ensure_numpy(values)
    
    # --- Dynamic Configuration Extraction ---
    if config is not None:
        if env_name is None:
            env_name = config.get("env", {}).get("id", config.get("env_name", "Unknown Env"))
        if critic_internal_time is None:
            # Try common SNN config keys for simulation time
            critic_internal_time = config.get("snn", {}).get("sim_time", config.get("snn", {}).get("critic_sim_time", 32))
            
    # Fallbacks in case config is missing or keys aren't found
    env_name = env_name or "Unknown Env"
    critic_internal_time = critic_internal_time or 32
    # If actor_steps isn't explicitly provided, dynamically use the length of the episode
    actor_steps = actor_steps or len(taus) 
    
    total_timesteps = actor_steps * critic_internal_time
    
    if len(taus) == 0 or len(values) == 0:
        _placeholder_plot("No timing critic data available", save_path)
        return

    fig, ax = plt.subplots(figsize=(15, 6))
    
    for i in range(min(actor_steps, len(taus))):
        # 1. Background Shading (Banding)
        if i % 2 == 0:
            ax.axvspan(i * critic_internal_time, (i + 1) * critic_internal_time, facecolor='lightgray', alpha=0.25)
            
        tau = int(taus[i])
        spike_value = float(values[i])
        global_tau = i * critic_internal_time + tau
        
        # 2. Stem Plot (Lollipop Chart)
        markerline, stemlines, baseline = ax.stem([global_tau], [spike_value], basefmt=" ", linefmt='b-', markerfmt='bo')
        plt.setp(stemlines, 'linewidth', 1.5)
        plt.setp(markerline, 'markersize', 6)
        
        # 3. Dynamic Annotation Placement
        y_offset = 12 if i % 2 == 0 else -25
        va_align = 'bottom' if i % 2 == 0 else 'top'
        
        annotation_text = f'$\\tau={tau}$\nv={spike_value:.2f}'
        ax.annotate(annotation_text, xy=(global_tau, spike_value), 
                    xytext=(0, y_offset), textcoords='offset points', ha='center', va=va_align, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    # Formatting and Labels
    ax.set_ylabel('Tau Value', fontweight='bold')
    
    # Dynamic Y-axis bounds based on actual data
    y_max = np.max(values) if len(values) > 0 else 1.0
    ax.set_ylim(0, y_max * 1.3) 
    
    # Grid and Bottom X-axis
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_xlim(0, total_timesteps)
    ax.set_xticks([j * critic_internal_time for j in range(actor_steps + 1)])
    ax.set_xticklabels([f'{j * critic_internal_time}' for j in range(actor_steps + 1)])
    
    # Top X-axis (actor_T centered)
    ax_top = ax.twiny()
    ax_top.set_xlim(0, total_timesteps)
    
    midpoints = [j * critic_internal_time + (critic_internal_time / 2) for j in range(actor_steps)]
    ax_top.set_xticks(midpoints)
    ax_top.set_xticklabels([f'T={j}' for j in range(actor_steps)])
    ax_top.tick_params(axis='x', length=0)
    ax_top.set_xlabel('Actor Timestep (actor_T)', fontsize=12, fontweight='bold', labelpad=10)

    ax.set_xlabel(f'Total Timesteps (1 Actor Step = {critic_internal_time} Critic Internal Steps)', fontsize=12, fontweight='bold')
    
    _set_titles(ax, f"{title} - {env_name}")
    _savefig(fig, save_path)


def plot_timing_critic_macro_dynamics(
    combined_df: pd.DataFrame,
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
    if tau_col not in combined_df.columns or val_col not in combined_df.columns:
        print(f"[Warning] Missing tau/value columns for macro dynamics plot.")
        return

    # --- Dynamic Configuration Extraction ---
    if config is not None:
        if env_name is None:
            env_name = config.get("env", {}).get("id", config.get("env_name", "Unknown Env"))
        if critic_internal_time is None:
            critic_internal_time = config.get("snn", {}).get("sim_time", config.get("snn", {}).get("critic_sim_time", 32))
            
    env_name = env_name or "Unknown Env"
    critic_internal_time = critic_internal_time or 32

    df = combined_df.dropna(subset=[tau_col, val_col]).copy()
    
    # Group by update to get x-axis (timesteps)
    grouped = df.groupby("update")
    x_values = grouped["total_timesteps"].mean() if "total_timesteps" in df.columns else grouped.index

    # Calculate smoothing for Tau
    tau_mean = grouped[tau_col].mean().ewm(span=window).mean()
    tau_std = grouped[tau_col].std().fillna(0.0).ewm(span=window).mean()

    # Calculate smoothing for Value
    val_mean = grouped[val_col].mean().ewm(span=window).mean()
    val_std = grouped[val_col].std().fillna(0.0).ewm(span=window).mean()

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top Plot: Tau Dynamics
    axs[0].plot(x_values, tau_mean, color="tab:blue", lw=2.5, label="Mean $\\tau$")
    axs[0].fill_between(x_values, tau_mean - tau_std, tau_mean + tau_std, color="tab:blue", alpha=0.2)
    
    # Dynamically draw the max threshold and set y-limits based on the configured simulation time
    axs[0].axhline(critic_internal_time, color="gray", linestyle="--", alpha=0.5, label=f"Max Internal Time ({critic_internal_time})")
    axs[0].set_ylim(0, critic_internal_time * 1.15)
    
    axs[0].set_ylabel("Spike Time ($\\tau$)", fontweight="bold")
    _set_titles(axs[0], f"{title_prefix}: Evolution of Spike Time ($\\tau$) - {env_name}")
    axs[0].legend(loc="upper right")

    # Bottom Plot: Value Dynamics
    axs[1].plot(x_values, val_mean, color="tab:green", lw=2.5, label="Mean Value")
    axs[1].fill_between(x_values, val_mean - val_std, val_mean + val_std, color="tab:green", alpha=0.2)
    axs[1].set_ylabel("Predicted Value", fontweight="bold")
    axs[1].set_xlabel("Environment Steps", fontweight="bold")
    _set_titles(axs[1], f"{title_prefix}: Evolution of Predicted Value - {env_name}")
    axs[1].legend(loc="lower right")

    # Format X-axis
    _format_steps_axis(axs[1])

    plt.tight_layout()
    _savefig(fig, save_path)