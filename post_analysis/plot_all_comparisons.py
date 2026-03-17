import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import yaml
from typing import List, Dict, Any

from src.utils.plotting import (
    plot_train_rollout_vs_steps,
    plot_eval_return_vs_steps,
    plot_success_rate_vs_steps,
    plot_energy_vs_steps,
    plot_latency_vs_steps,
    plot_spikes_vs_steps,
    plot_energy_vs_spikes,
    plot_reward_vs_spikes,
    plot_latency_vs_spikes,
    plot_eval_checkpoint_value_trend,
    plot_timing_critic_macro_dynamics,
)

# --- 1. Research-Grade Styling ---
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 2.5

# --- 2. Configuration ---
EXPERIMENTS = {
    "ANN Baseline": {"path": "results/logs/ann_baseline_no_fs", "color": "#1f77b4"},
    "SNN Actor (ANN Critic)": {"path": "results/logs/snn_actor_ann_critic_no_fs", "color": "#ff7f0e"},
    "SNN Actor (Timing Critic)": {"path": "results/logs/snn_actor_snn_timing_critic_no_fs", "color": "#9467bd"},
    "ANN2SNN Full": {"path": "results/logs/ann2snn_full_no_fs", "color": "#d62728"},
    "ANN2SNN Actor": {"path": "results/logs/ann2snn_actor_no_fs", "color": "#9467bd"}
}

EXPERIMENT_CONFIGS = {
    "ANN Baseline": "configs/cartpole/ann_baseline.yaml",
    "SNN Actor (ANN Critic)": "configs/cartpole/snn_actor_ann_critic.yaml",
    "SNN Actor (Timing Critic)": "configs/cartpole/snn_actor_snn_timing_critic.yaml",
    "ANN2SNN Full": "configs/cartpole/ann2snn_full.yaml",
    "ANN2SNN Actor": "configs/cartpole/ann2snn_actor.yaml",
}

# Metric Definitions: (Display Name, [List of possible CSV column names])
METRICS = [
    ("Evaluation Reward", ["test_reward", "eval_reward", "post_conversion_ft/eval_reward", "eval_episode_reward"]),
    ("Training Reward", []),
    ("Success Rate", []),
    ("Training Energy (J/step)", []),
    ("Inference Energy (J/step)", []),
    ("Energy Proxy (J/step, activity-normalized)", []),
    ("Energy Proxy (J/step, sparsity-adjusted)", []),
    ("Latency (Wall-Clock ms)", []),
    ("Latency (SNN Internal tau)", []),
    ("Spike Count", ["spike_count_total", "post_conversion/total_spikes", "spikes/total"]),
]

def load_config(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _canonical_exp_name(display_name: str) -> str:
    name = display_name.lower()
    if "ann baseline" in name:
        return "ann_baseline"
    if "timing critic" in name:
        return "snn_actor_snn_timing_critic"
    if "snn actor (ann critic)" in name:
        return "snn_actor_ann_critic"
    if "ann2snn full" in name:
        return "ann2snn_both"
    if "ann2snn actor" in name:
        return "ann2snn_actor"
    return "ann_baseline"


def _load_seed_dataframes(base_path: str) -> List[pd.DataFrame]:
    seed_files = [
        os.path.join(base_path, d, "per_episode_metrics.csv")
        for d in sorted(os.listdir(base_path))
        if d.startswith("seed_")
    ] if os.path.exists(base_path) else []
    dfs: List[pd.DataFrame] = []
    for f in seed_files:
        if not os.path.exists(f):
            continue
        try:
            df = pd.read_csv(f)
            if not df.empty:
                dfs.append(df)
        except Exception:
            continue

    # Backward compatibility: single-seed logs at experiment root.
    if not dfs:
        root_csv = os.path.join(base_path, "per_episode_metrics.csv")
        if os.path.exists(root_csv):
            try:
                df = pd.read_csv(root_csv)
                if not df.empty:
                    dfs.append(df)
            except Exception:
                pass
    return dfs

# --- 3. Data Loading Logic ---
def _get_steps(df: pd.DataFrame) -> np.ndarray:
    if "total_timesteps_snn" in df.columns:
        return df["total_timesteps_snn"].values
    if "total_timesteps" in df.columns:
        return df["total_timesteps"].values
    if "episode_length_steps" in df.columns:
        return df["episode_length_steps"].cumsum().values
    return np.arange(len(df))


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) == 0:
        return x
    window = max(1, int(window))
    return pd.Series(x).rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def _get_eval_rewards(df: pd.DataFrame) -> np.ndarray:
    for col in ("test_reward", "eval_reward", "post_conversion_ft/eval_reward", "eval_episode_reward"):
        if col in df.columns:
            return df[col].to_numpy()
    return np.array([])

def _get_eval_success_rate(df: pd.DataFrame) -> np.ndarray:
    if "eval/success_rate" in df.columns:
        return df["eval/success_rate"].to_numpy()
    return np.array([])


def _compute_success_rate(df: pd.DataFrame, reward_threshold: float, eval_window: int) -> np.ndarray:
    sr = _get_eval_success_rate(df)
    if sr.size > 0:
        mask = ~np.isnan(sr)
        return sr[mask]
    rewards = _get_eval_rewards(df)
    if rewards.size == 0:
        return np.array([])
    mask = ~np.isnan(rewards)
    if not np.any(mask):
        return np.array([])
    rewards = rewards[mask]
    hits = (rewards >= reward_threshold).astype(float) * 100.0
    return _rolling_mean(hits, eval_window)

def _compute_training_reward(df: pd.DataFrame, exp_name: str) -> np.ndarray:
    # For ANN2SNN experiments, prefer finetuning reward if present.
    if "ANN2SNN" in exp_name:
        if "post_conversion_ft/train_reward" in df.columns:
            return df["post_conversion_ft/train_reward"].to_numpy()
    if "train_reward" in df.columns:
        return df["train_reward"].to_numpy()
    return np.array([])


def _compute_train_energy_per_step(df: pd.DataFrame) -> np.ndarray:
    energy_col = None
    if "train_rollout_dynamic_energy" in df.columns:
        energy_col = "train_rollout_dynamic_energy"
    elif "train_rollout_energy" in df.columns:
        energy_col = "train_rollout_energy"
    if energy_col is None:
        return np.array([])

    energy = pd.to_numeric(df[energy_col], errors="coerce")
    if "train_rollout_steps" in df.columns:
        denom = pd.to_numeric(df["train_rollout_steps"], errors="coerce").replace(0, np.nan)
        return (energy / denom).to_numpy()

    steps = _get_steps(df)
    delta_steps = np.diff(steps, prepend=steps[0])
    delta_steps = np.where(delta_steps == 0, np.nan, delta_steps)
    return energy.to_numpy() / delta_steps


def _compute_inference_energy_per_step(df: pd.DataFrame) -> np.ndarray:
    energy_col = None
    if "inference_dynamic_energy" in df.columns:
        energy_col = "inference_dynamic_energy"
    elif "inference_energy" in df.columns:
        energy_col = "inference_energy"
    if energy_col is None:
        return np.array([])

    energy = pd.to_numeric(df[energy_col], errors="coerce")

    # Keep denominator consistent with src/utils/plotting.py::plot_energy_vs_steps
    # so comparison and multiseed figures report the same J/step values.
    if "eval/n_eval_episodes" in df.columns and "eval_episode_length" in df.columns:
        n_eval = pd.to_numeric(df["eval/n_eval_episodes"], errors="coerce")
        eval_len = pd.to_numeric(df["eval_episode_length"], errors="coerce")
        denom = (n_eval * eval_len).replace(0, np.nan)
        return (energy / denom).to_numpy()

    if "episode_length_steps" in df.columns:
        denom = pd.to_numeric(df["episode_length_steps"], errors="coerce").replace(0, np.nan)
        return (energy / denom).to_numpy()

    steps = _get_steps(df)
    delta_steps = np.diff(steps, prepend=steps[0])
    delta_steps = np.where(delta_steps == 0, np.nan, delta_steps)
    return energy.to_numpy() / delta_steps

def _compute_spikes_per_step(df: pd.DataFrame) -> np.ndarray:
    if "spikes/per_step" in df.columns:
        return df["spikes/per_step"].to_numpy()
    if "post_conversion/total_spikes" in df.columns and "episode_length_steps" in df.columns:
        return df["post_conversion/total_spikes"].to_numpy() / df["episode_length_steps"].to_numpy()
    if "spike_count_total" in df.columns and "episode_length_steps" in df.columns:
        return df["spike_count_total"].to_numpy() / df["episode_length_steps"].to_numpy()
    return np.array([])


def _compute_energy_proxy(df: pd.DataFrame) -> np.ndarray:
    energy = _compute_inference_energy_per_step(df)
    if energy.size == 0:
        return np.array([])
    spikes = _compute_spikes_per_step(df)
    if spikes.size == 0 or np.nanmean(spikes) == 0:
        return energy
    scale = spikes / np.nanmean(spikes)
    return energy * scale


def _compute_energy_proxy_sparsity(df: pd.DataFrame) -> np.ndarray:
    energy = _compute_inference_energy_per_step(df)
    if energy.size == 0:
        return np.array([])
    sparsity = None
    for col in ("spikes/sparsity", "post_conversion/sparsity", "spikes/eval_sparsity"):
        if col in df.columns:
            sparsity = df[col].to_numpy()
            break
    if sparsity is None or np.nanmean(sparsity) == 0:
        return energy
    active = 1.0 - sparsity
    if np.nanmean(active) == 0:
        return energy
    scale = active / np.nanmean(active)
    return energy * scale


def _compute_latency(
    df: pd.DataFrame,
    is_ann: bool,
    exp_name: str,
    latency_component: str = "auto",
) -> np.ndarray:
    # Prefer post-conversion latency for ANN2SNN experiments when available.
    if "ANN2SNN" in exp_name and "post_conversion/mean_latency" in df.columns:
        return df["post_conversion/mean_latency"].to_numpy()
    if is_ann:
        if "latency_mean_ms" in df.columns:
            return df["latency_mean_ms"].to_numpy()
        if "latency/eval_wall_clock_ms" in df.columns:
            return df["latency/eval_wall_clock_ms"].to_numpy()
        return np.array([])
    # SNN latency uses spike timing (component-selectable for compare plots).
    if latency_component == "critic":
        if "latency/critic_spike_timing_steps" in df.columns:
            return df["latency/critic_spike_timing_steps"].to_numpy()
        if "latency/critic_eval_spike_timing_steps" in df.columns:
            return df["latency/critic_eval_spike_timing_steps"].to_numpy()
        if "latency/spike_timing_steps" in df.columns:
            return df["latency/spike_timing_steps"].to_numpy()
    elif latency_component == "actor":
        if "latency/actor_spike_timing_steps" in df.columns:
            return df["latency/actor_spike_timing_steps"].to_numpy()
        if "latency/spike_timing_steps" in df.columns:
            return df["latency/spike_timing_steps"].to_numpy()
    else:
        if "latency/actor_spike_timing_steps" in df.columns:
            return df["latency/actor_spike_timing_steps"].to_numpy()
        if "latency/critic_spike_timing_steps" in df.columns:
            return df["latency/critic_spike_timing_steps"].to_numpy()
        if "latency/critic_eval_spike_timing_steps" in df.columns:
            return df["latency/critic_eval_spike_timing_steps"].to_numpy()
        if "latency/spike_timing_steps" in df.columns:
            return df["latency/spike_timing_steps"].to_numpy()
    if "post_conversion/mean_latency" in df.columns:
        return df["post_conversion/mean_latency"].to_numpy()
    return np.array([])


def _compute_wall_clock_latency(df: pd.DataFrame) -> np.ndarray:
    # Unified latency for fair ANN-vs-SNN speed comparisons.
    if "latency_mean_ms" in df.columns:
        return df["latency_mean_ms"].to_numpy()
    if "latency/mean_ms" in df.columns:
        return df["latency/mean_ms"].to_numpy()
    if "latency/eval_wall_clock_ms" in df.columns:
        return df["latency/eval_wall_clock_ms"].to_numpy()
    return np.array([])


def load_and_align_data(
    base_path,
    target_cols,
    metric_name,
    config,
    exp_name: str,
    latency_component: str = "auto",
):
    """
    Loads all seeds for a specific experiment and metric (target_cols).
    Returns interpolated common_x, mean_y, std_y.
    """
    if not os.path.exists(base_path): return None, None, None

    all_x, all_y = [], []
    
    # Identify Seed Files
    files = [os.path.join(base_path, d, "per_episode_metrics.csv") 
             for d in os.listdir(base_path) if d.startswith("seed_")]
    
    # Fallback to root if no seeds
    if not files and os.path.exists(os.path.join(base_path, "per_episode_metrics.csv")):
        files = [os.path.join(base_path, "per_episode_metrics.csv")]

    for f in files:
        if not os.path.exists(f): continue
        try:
            df = pd.read_csv(f)
            
            x = _get_steps(df)

            if metric_name == "Training Reward":
                y = _compute_training_reward(df, exp_name)
            elif metric_name == "Success Rate":
                eval_window = int(config.get("ppo", {}).get("eval_episodes", 20))
                reward_threshold = float(config.get("ppo", {}).get("reward_threshold", 475.0))

                sr = _get_eval_success_rate(df)
                if sr.size > 0:
                    mask = ~np.isnan(sr)
                    y = sr[mask]
                    x = x[mask]
                else:
                    rewards = _get_eval_rewards(df)
                    if rewards.size == 0:
                        y = np.array([])
                    else:
                        mask = ~np.isnan(rewards)
                        rewards = rewards[mask]
                        hits = (rewards >= reward_threshold).astype(float) * 100.0
                        y = _rolling_mean(hits, eval_window)
                        x = x[mask]

                if os.environ.get("DEBUG_PLOTS"):
                    print(f"[Debug] Success Rate {exp_name}: {y.size} points from {os.path.basename(f)}")
            elif metric_name == "Training Energy (J/step)":
                y = _compute_train_energy_per_step(df)
            elif metric_name == "Inference Energy (J/step)":
                y = _compute_inference_energy_per_step(df)
            elif metric_name == "Energy Proxy (J/step, activity-normalized)":
                y = _compute_energy_proxy(df)
            elif metric_name == "Energy Proxy (J/step, sparsity-adjusted)":
                y = _compute_energy_proxy_sparsity(df)
            elif metric_name == "Latency (Wall-Clock ms)":
                y = _compute_wall_clock_latency(df)
            elif metric_name == "Latency (SNN Internal tau)":
                is_ann = config.get("model", {}).get("mode") == "ann" and not config.get("snn")
                if is_ann:
                    y = np.array([])
                else:
                    y = _compute_latency(
                        df,
                        is_ann=is_ann,
                        exp_name=exp_name,
                        latency_component=latency_component,
                    )
            elif metric_name == "Latency (ms / steps)":
                # Backward compatibility for any external caller.
                is_ann = config.get("model", {}).get("mode") == "ann" and not config.get("snn")
                y = _compute_latency(
                    df,
                    is_ann=is_ann,
                    exp_name=exp_name,
                    latency_component=latency_component,
                )
            else:
                y_col = next((c for c in target_cols if c in df.columns), None)
                if not y_col:
                    continue
                y = df[y_col].values
            
            # Clean NaNs
            mask = np.isfinite(y) & np.isfinite(x)
            if np.any(mask):
                all_x.append(x[mask])
                all_y.append(y[mask])
        except: continue

    if not all_x: return None, None, None

    if metric_name == "Success Rate":
        return all_x, all_y, None

    # Interpolate to common grid for dense metrics
    min_t = min(x.min() for x in all_x)
    max_t = max(x.max() for x in all_x)
    common_x = np.linspace(min_t, max_t, 1000)
    
    interp_ys = []
    for x, y in zip(all_x, all_y):
        # Sort X for interpolation
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        interp = np.interp(common_x, x_sorted, y_sorted)
        interp[(common_x < x_sorted.min()) | (common_x > x_sorted.max())] = np.nan
        interp_ys.append(interp)
    
    mean_y = np.nanmean(interp_ys, axis=0)
    std_y = np.nanstd(interp_ys, axis=0)
    
    return common_x, mean_y, std_y

# --- 4. Plotting Loop ---
def _slugify(text: str) -> str:
    s = str(text).strip().lower().replace(" ", "_")
    out = []
    for ch in s:
        if ch.isalnum() or ch == "_":
            out.append(ch)
    return "".join(out).strip("_")


def _last_finite_xy(x: np.ndarray, y) -> tuple[float, float] | tuple[None, None]:
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    n = min(len(x_arr), len(y_arr))
    if n == 0:
        return None, None
    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.any(mask):
        return None, None
    idx = np.where(mask)[0][-1]
    return float(x_arr[idx]), float(y_arr[idx])


def plot_all_metrics(output_dir="results/plots", context_label: str = ""):
    os.makedirs(output_dir, exist_ok=True)
    context_label = str(context_label or "").strip()
    context_slug = _slugify(context_label) if context_label else ""
    audit_rows: List[Dict[str, Any]] = []

    for metric_name, cols in METRICS:
        print(f"Generating plot for: {metric_name}...")
        fig, ax = plt.subplots(figsize=(12, 7))
        has_data = False
        uncertainty_note = ""
        reward_thresholds = []
        final_label_idx = 0

        for name, config in EXPERIMENTS.items():
            cfg_path = EXPERIMENT_CONFIGS.get(name, "")
            cfg = load_config(cfg_path)
            # Load data for this specific metric
            x, mean, std = load_and_align_data(config["path"], cols, metric_name, cfg, exp_name=name)
            
            if x is not None:
                has_data = True
                if metric_name in ("Evaluation Reward", "Training Reward"):
                    thr = cfg.get("ppo", {}).get("reward_threshold")
                    if thr is not None:
                        try:
                            reward_thresholds.append(float(thr))
                        except Exception:
                            pass
                label_name = name
                is_timing_critic = ("timing critic" in name.lower())
                if metric_name == "Training Reward" and "ANN2SNN" in name:
                    label_name = f"{name} (Finetune reward)"
                if metric_name == "Latency (SNN Internal tau)" and is_timing_critic:
                    label_name = f"{name} (Actor latency, tau)"
                if metric_name == "Success Rate":
                    if not mean:
                        continue
                    xs = np.concatenate(x)
                    ys = np.concatenate(mean)
                    df_sr = pd.DataFrame({"x": xs, "y": ys}).dropna()
                    if df_sr.empty:
                        continue
                    grouped = df_sr.groupby("x")["y"]
                    steps = grouped.mean().index.to_numpy()
                    y_mean = grouped.mean().to_numpy()
                    y_std = grouped.std(ddof=1).fillna(0.0).to_numpy()
                    ax.errorbar(
                        steps,
                        y_mean,
                        yerr=y_std,
                        fmt="o-",
                        color=config["color"],
                        label=label_name,
                        capsize=2,
                        alpha=0.9,
                    )
                    uncertainty_note = "Uncertainty: error bars are mean ±1 std across seeds."
                    continue

                # Apply Smoothing (EMA) for cleaner visuals
                mean_smooth = pd.Series(mean).ewm(span=10).mean()
                std_smooth = pd.Series(std).ewm(span=10).mean()
                if metric_name == "Latency (Wall-Clock ms)":
                    # Log-scale plotting requires positive values.
                    mean_smooth = mean_smooth.where(mean_smooth > 0, np.nan)
                
                ax.plot(x, mean_smooth, label=label_name, color=config["color"])
                ax.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth, 
                                color=config["color"], alpha=0.15)
                uncertainty_note = "Uncertainty: shaded band is mean ±1 std across seeds (EMA-smoothed)."
                if metric_name == "Latency (Wall-Clock ms)":
                    last_x, last_y = _last_finite_xy(x, mean_smooth)
                    if last_x is not None and last_y is not None:
                        ax.scatter([last_x], [last_y], color=config["color"], s=34, zorder=8)
                        ax.annotate(
                            f"{label_name} final: {last_y:.3f} ms",
                            xy=(last_x, last_y),
                            xytext=(-190, -16 - (14 * final_label_idx)),
                            textcoords="offset points",
                            fontsize=9,
                            fontweight="bold",
                            color=config["color"],
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=config["color"], lw=1.0, alpha=0.9),
                            arrowprops=dict(arrowstyle="-|>", color=config["color"], lw=0.9, alpha=0.85),
                        )
                        final_label_idx += 1
                if metric_name == "Latency (SNN Internal tau)" and is_timing_critic:
                    x_c, mean_c, std_c = load_and_align_data(
                        config["path"],
                        cols,
                        metric_name,
                        cfg,
                        exp_name=name,
                        latency_component="critic",
                    )
                    if x_c is not None:
                        mean_c_smooth = pd.Series(mean_c).ewm(span=10).mean()
                        std_c_smooth = pd.Series(std_c).ewm(span=10).mean()
                        ax.plot(
                            x_c,
                            mean_c_smooth,
                            label=f"{name} (Critic latency, tau)",
                            color=config["color"],
                            linestyle="--",
                            linewidth=2.0,
                        )
                        ax.fill_between(
                            x_c,
                            mean_c_smooth - std_c_smooth,
                            mean_c_smooth + std_c_smooth,
                            color=config["color"],
                            alpha=0.08,
                        )

                # Add final-value marker/label for J/step metrics and record audit row.
                if "J/step" in metric_name:
                    last_x, last_y = _last_finite_xy(x, mean_smooth)
                    if last_x is not None and last_y is not None:
                        ax.scatter([last_x], [last_y], color=config["color"], s=42, zorder=8)
                        ax.annotate(
                            f"Final: {last_y:.4f} J/step",
                            xy=(last_x, last_y),
                            xytext=(-170, -18 - (14 * final_label_idx)),
                            textcoords="offset points",
                            fontsize=9,
                            fontweight="bold",
                            color=config["color"],
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=config["color"], lw=1.0, alpha=0.9),
                            arrowprops=dict(arrowstyle="-|>", color=config["color"], lw=0.9, alpha=0.85),
                        )
                        final_label_idx += 1
                        print(f"  Final J/step | {metric_name} | {label_name}: {last_y:.6f}")
                        audit_rows.append(
                            {
                                "context_label": context_label,
                                "metric": metric_name,
                                "experiment": label_name,
                                "logs_path": config["path"],
                                "final_step": last_x,
                                "final_j_per_step": last_y,
                            }
                        )
                if metric_name in ("Evaluation Reward", "Training Reward") and len(mean_smooth) > 0:
                    peak_idx = int(np.nanargmax(mean_smooth))
                    peak_x = float(x[peak_idx])
                    peak_y = float(mean_smooth.iloc[peak_idx] if hasattr(mean_smooth, "iloc") else mean_smooth[peak_idx])
                    ax.scatter(
                        [peak_x],
                        [peak_y],
                        marker="*",
                        s=170,
                        color="gold",
                        edgecolor="black",
                        linewidths=0.9,
                        zorder=10,
                    )
                    ax.annotate(
                        f"Peak: {peak_y:.2f}",
                        xy=(peak_x, peak_y),
                        xytext=(-36, 12),
                        textcoords="offset points",
                        fontsize=9,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gold", lw=1.3, alpha=0.95),
                    )

                # --- Special Case: Zero-Shot Marker (Only for Reward) ---
                if metric_name == "Evaluation Reward" and name == "ANN2SNN Actor":
                    zs_path = os.path.join(config["path"], "per_episode_metrics.csv")
                    if os.path.exists(zs_path):
                        try:
                            df = pd.read_csv(zs_path)
                            if "post_conversion/zero_shot_reward" in df.columns:
                                zs = df["post_conversion/zero_shot_reward"].mean()
                                ax.scatter([0], [zs], color=config["color"], marker='*', s=300, 
                                           edgecolor='white', zorder=10, label="ANN2SNN (Zero-Shot)")
                        except: pass

        if has_data:
            # --- Formatting ---
            if metric_name in ("Evaluation Reward", "Training Reward"):
                solved = None
                if "cartpole" in context_slug:
                    solved = 475.0
                elif reward_thresholds:
                    solved = float(np.median(np.asarray(reward_thresholds, dtype=float)))
                if solved is not None:
                    ax.axhline(
                        solved,
                        color="#ff3b30",
                        linestyle="--",
                        lw=2.0,
                        alpha=0.85,
                        label=f"Solved ({solved:g})",
                    )
            
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1e6:.1f}M' if x >= 1e6 else f'{int(x)}'))
            
            title = f"{metric_name} Comparison"
            if context_label:
                title = f"{title} | {context_label}"
            ax.set_title(title, fontsize=18, fontweight="bold", pad=15)
            ax.set_xlabel("Environment Steps", fontsize=14, fontweight="bold")
            if metric_name == "Latency (Wall-Clock ms)":
                ax.set_ylabel("Latency (ms)", fontsize=14, fontweight="bold")
                ax.set_yscale("log")
            elif metric_name == "Latency (SNN Internal tau)":
                ax.set_ylabel("Latency (tau steps)", fontsize=14, fontweight="bold")
            else:
                ax.set_ylabel(metric_name, fontsize=14, fontweight="bold")
            ax.grid(True, linestyle=":", alpha=0.6)
            if uncertainty_note:
                ax.text(
                    0.01,
                    0.99,
                    uncertainty_note,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    color="dimgray",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="lightgray"),
                )
            ax.legend(fontsize=12, loc="best", framealpha=0.95)
            
            # Save
            metric_slug = _slugify(metric_name.replace("/", "_"))
            if context_slug:
                filename = f"compare_{metric_slug}_{context_slug}.png"
            else:
                filename = f"compare_{metric_slug}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()
        else:
            plt.close()
            print(f"  Skipping {metric_name} (No data found)")

    if audit_rows:
        audit_df = pd.DataFrame(audit_rows)
        if not audit_df.empty:
            audit_csv = os.path.join(output_dir, "table_final_j_per_step_audit.csv")
            audit_df.to_csv(audit_csv, index=False)
            print(f"Saved final J/step audit CSV: {audit_csv}")


def plot_multiseed_dynamics(output_dir: str = "results/plots/multiseed_dynamics"):
    os.makedirs(output_dir, exist_ok=True)
    for name, exp_cfg in EXPERIMENTS.items():
        base_path = exp_cfg["path"]
        cfg_path = EXPERIMENT_CONFIGS.get(name, "")
        cfg = load_config(cfg_path)
        env_name = cfg.get("env", {}).get("id", "Unknown Env")
        exp_name = _canonical_exp_name(name)
        seeds = _load_seed_dataframes(base_path)
        if not seeds:
            print(f"Skipping multi-seed dynamics for {name} (no seed CSVs found)")
            continue

        out_dir = os.path.join(output_dir, exp_name)
        os.makedirs(out_dir, exist_ok=True)
        data_arg = seeds if len(seeds) > 1 else seeds[0]
        primary_df = seeds[0]

        try:
            plot_train_rollout_vs_steps(
                data_arg,
                save_path=os.path.join(out_dir, "train_rollout_vs_steps.png"),
                exp_name=exp_name,
                config=cfg,
                env_name=env_name,
            )
            plot_eval_return_vs_steps(
                data_arg,
                save_path=os.path.join(out_dir, "eval_return_vs_steps.png"),
                exp_name=exp_name,
                config=cfg,
                env_name=env_name,
            )
            plot_success_rate_vs_steps(
                data_arg,
                save_path=os.path.join(out_dir, "success_rate_vs_steps.png"),
                exp_name=exp_name,
                config=cfg,
                env_name=env_name,
            )
            plot_energy_vs_steps(
                data_arg,
                save_path=os.path.join(out_dir, "energy_vs_steps.png"),
                exp_name=exp_name,
                config=cfg,
                env_name=env_name,
            )
            plot_latency_vs_steps(
                data_arg,
                save_path=os.path.join(out_dir, "latency_vs_steps.png"),
                exp_name=exp_name,
                config=cfg,
                env_name=env_name,
            )
            plot_spikes_vs_steps(
                data_arg,
                save_path=os.path.join(out_dir, "spikes_vs_steps.png"),
                exp_name=exp_name,
                config=cfg,
            )
            plot_energy_vs_spikes(
                data_arg,
                save_path=os.path.join(out_dir, "energy_vs_spikes.png"),
                exp_name=exp_name,
                env_name=env_name,
            )
            plot_reward_vs_spikes(
                data_arg,
                save_path=os.path.join(out_dir, "reward_vs_spikes.png"),
                exp_name=exp_name,
                env_name=env_name,
                config=cfg,
            )
            plot_latency_vs_spikes(
                data_arg,
                save_path=os.path.join(out_dir, "latency_vs_spikes_actor.png"),
                exp_name=exp_name,
                env_name=env_name,
                component="Actor",
            )
            plot_eval_checkpoint_value_trend(
                data_arg,
                save_path=os.path.join(out_dir, "eval_checkpoint_value_trend.png"),
                exp_name=exp_name,
                env_name=env_name,
            )
            plot_timing_critic_macro_dynamics(
                data_arg,
                save_path=os.path.join(out_dir, "timing_critic_macro_dynamics.png"),
                config=cfg,
                env_name=env_name,
            )
            print(f"Generated multi-seed dynamics for {name} -> {out_dir}")
        except Exception as exc:
            print(f"Failed multi-seed dynamics for {name}: {exc}")

if __name__ == "__main__":
    plot_all_metrics()
    plot_multiseed_dynamics()
