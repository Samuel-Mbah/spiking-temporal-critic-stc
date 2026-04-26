"""
Surrogate gradient ablation comparison for tmaze_active snn_timing_critic.

Compares three conditions:
  1. Different surrogate gradients (actor=fast_sigmoid, critic=cosh) — default, 10 seeds
  2. Same surrogate — both fast_sigmoid — 5 seeds
  3. Same surrogate — both cosh — 5 seeds

Output: results/neurips/tmaze_active/surrogate_ablation_comparison/
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

BASE = "results/neurips/tmaze_active/snn_timing_critic"

EXPERIMENTS = {
    "Different (actor=FastSigmoid, critic=Cosh)": {
        "path": BASE,
        "color": "#2196F3",
        "linestyle": "-",
    },
    "Same — Both FastSigmoid": {
        "path": os.path.join(BASE, "ablation_both_fastsigmoid"),
        "color": "#FF5722",
        "linestyle": "--",
    },
    "Same — Both Cosh": {
        "path": os.path.join(BASE, "ablation_both_cosh"),
        "color": "#4CAF50",
        "linestyle": ":",
    },
}

OUTPUT_DIR = "results/neurips/tmaze_active/surrogate_ablation_comparison"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["lines.linewidth"] = 2.5

# ---------------------------------------------------------------------------
# Metrics to plot: (title, csv_column_or_derived, y_label)
# ---------------------------------------------------------------------------

def _steps(df: pd.DataFrame) -> np.ndarray:
    if "total_timesteps" in df.columns:
        return df["total_timesteps"].values
    if "update" in df.columns:
        return df["update"].values
    return np.arange(len(df))


def _extract(df: pd.DataFrame, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) arrays for a given metric key, NaN-cleaned."""
    x = _steps(df)

    if key == "eval/success_rate":
        y = df["eval/success_rate"].values if "eval/success_rate" in df.columns else np.full(len(x), np.nan)
    elif key == "test_reward":
        y = df["test_reward"].values if "test_reward" in df.columns else np.full(len(x), np.nan)
    elif key == "train_reward":
        y = df["train_reward"].values if "train_reward" in df.columns else np.full(len(x), np.nan)
    elif key == "inference_energy_per_step":
        energy = pd.to_numeric(df.get("inference_energy", pd.Series(dtype=float)), errors="coerce")
        ep_len = pd.to_numeric(df.get("eval_episode_length", pd.Series(dtype=float)), errors="coerce").replace(0, np.nan)
        n_ep = pd.to_numeric(df.get("eval/n_eval_episodes", pd.Series(dtype=float)), errors="coerce").replace(0, np.nan)
        denom = (ep_len * n_ep).replace(0, np.nan)
        y = (energy / denom).values if len(energy) == len(x) else np.full(len(x), np.nan)
    elif key == "spikes/per_step":
        y = df["spikes/per_step"].values if "spikes/per_step" in df.columns else np.full(len(x), np.nan)
    elif key == "eval/spikes_per_step":
        y = df["eval/spikes_per_step"].values if "eval/spikes_per_step" in df.columns else np.full(len(x), np.nan)
    elif key == "latency/actor_spike_timing_steps":
        y = df["latency/actor_spike_timing_steps"].values if "latency/actor_spike_timing_steps" in df.columns else np.full(len(x), np.nan)
    elif key == "latency/critic_spike_timing_steps":
        col = "latency/critic_eval_spike_timing_steps" if "latency/critic_eval_spike_timing_steps" in df.columns else "latency/critic_spike_timing_steps"
        y = df[col].values if col in df.columns else np.full(len(x), np.nan)
    else:
        y = df[key].values if key in df.columns else np.full(len(x), np.nan)

    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


METRICS = [
    ("Eval Success Rate (%)", "eval/success_rate", "Success Rate (%)"),
    ("Eval Reward", "test_reward", "Eval Reward"),
    ("Train Reward", "train_reward", "Train Reward"),
    ("Actor Spike Timing Latency (tau steps)", "latency/actor_spike_timing_steps", "Latency (tau steps)"),
    ("Critic Spike Timing Latency (tau steps)", "latency/critic_spike_timing_steps", "Latency (tau steps)"),
    ("Train Spikes per Step", "spikes/per_step", "Spikes / Step"),
    ("Eval Spikes per Step", "eval/spikes_per_step", "Spikes / Step"),
    ("Inference Energy per Step (J)", "inference_energy_per_step", "Energy (J/step)"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_seeds(base_path: str) -> list[pd.DataFrame]:
    """Load all seed CSVs from seed_* subdirectories."""
    dfs = []
    if not os.path.exists(base_path):
        return dfs
    for d in sorted(os.listdir(base_path)):
        if not d.startswith("seed_"):
            continue
        csv = os.path.join(base_path, d, "per_episode_metrics.csv")
        if not os.path.exists(csv):
            continue
        try:
            df = pd.read_csv(csv)
            if not df.empty:
                dfs.append(df)
        except Exception:
            pass
    return dfs


def _interpolate_seeds(
    dfs: list[pd.DataFrame], key: str, n_points: int = 500
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (common_x, mean_y, std_y, n_seeds) interpolated across seeds."""
    all_x, all_y = [], []
    for df in dfs:
        x, y = _extract(df, key)
        if x.size == 0:
            continue
        sort_idx = np.argsort(x)
        all_x.append(x[sort_idx])
        all_y.append(y[sort_idx])

    if not all_x:
        return None, None, None, 0

    min_t = min(x.min() for x in all_x)
    max_t = max(x.max() for x in all_x)
    common_x = np.linspace(min_t, max_t, n_points)

    interp_ys = []
    for x, y in zip(all_x, all_y):
        yi = np.interp(common_x, x, y)
        yi[(common_x < x.min()) | (common_x > x.max())] = np.nan
        interp_ys.append(yi)

    mean_y = np.nanmean(interp_ys, axis=0)
    std_y = np.nanstd(interp_ys, axis=0)
    return common_x, mean_y, std_y, len(interp_ys)


def _ema(arr: np.ndarray, span: int = 8) -> np.ndarray:
    return pd.Series(arr).ewm(span=span).mean().to_numpy()


def _fmt_steps(x, _):
    if x >= 1e6:
        return f"{x/1e6:.1f}M"
    if x >= 1e3:
        return f"{x/1e3:.0f}k"
    return f"{int(x)}"


# ---------------------------------------------------------------------------
# Main comparison plot
# ---------------------------------------------------------------------------

def plot_surrogate_comparison():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pre-load seeds for all experiments
    loaded = {name: _load_seeds(cfg["path"]) for name, cfg in EXPERIMENTS.items()}
    for name, dfs in loaded.items():
        print(f"  {name}: {len(dfs)} seed(s) loaded")

    for title, key, ylabel in METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False

        for name, cfg in EXPERIMENTS.items():
            dfs = loaded[name]
            if not dfs:
                continue

            common_x, mean_y, std_y, n = _interpolate_seeds(dfs, key)
            if common_x is None:
                continue

            has_data = True
            mean_s = _ema(mean_y)
            std_s = _ema(std_y)

            label = f"{name} (n={n})"
            ax.plot(
                common_x,
                mean_s,
                label=label,
                color=cfg["color"],
                linestyle=cfg["linestyle"],
                linewidth=2.5,
            )
            ax.fill_between(
                common_x,
                mean_s - std_s,
                mean_s + std_s,
                color=cfg["color"],
                alpha=0.15,
            )

        if not has_data:
            plt.close()
            print(f"  Skipping '{title}' — no data found")
            continue

        ax.set_title(
            f"{title}\nTmaze Active — Surrogate Gradient Ablation",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Environment Steps", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_steps))
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=10, loc="best", framealpha=0.95)
        ax.text(
            0.01,
            0.99,
            "Shaded band: mean ± 1 std (EMA-smoothed)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="dimgray",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="lightgray"),
        )

        slug = key.replace("/", "_").replace(" ", "_")
        out_path = os.path.join(OUTPUT_DIR, f"surrogate_ablation_{slug}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")

    # Summary table of final-value statistics
    _print_summary_table(loaded)


def _print_summary_table(loaded: dict):
    """Print and save a CSV summary of final-epoch mean ± std across seeds."""
    rows = []
    summary_metrics = [
        ("eval/success_rate", "Final Success Rate (%)"),
        ("test_reward", "Final Eval Reward"),
        ("spikes/per_step", "Final Train Spikes/Step"),
        ("eval/spikes_per_step", "Final Eval Spikes/Step"),
        ("latency/actor_spike_timing_steps", "Final Actor Latency (tau)"),
        ("latency/critic_spike_timing_steps", "Final Critic Latency (tau)"),
    ]

    for name, dfs in loaded.items():
        if not dfs:
            continue
        row = {"Experiment": name, "n_seeds": len(dfs)}
        for key, label in summary_metrics:
            finals = []
            for df in dfs:
                x, y = _extract(df, key)
                if y.size > 0:
                    # Last 5% of training steps — stable final value
                    cutoff = x.max() * 0.95
                    tail = y[x >= cutoff]
                    if tail.size > 0:
                        finals.append(np.nanmean(tail))
            if finals:
                row[f"{label} mean"] = round(float(np.mean(finals)), 4)
                row[f"{label} std"] = round(float(np.std(finals)), 4)
            else:
                row[f"{label} mean"] = None
                row[f"{label} std"] = None
        rows.append(row)

    if not rows:
        return

    df_summary = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "surrogate_ablation_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSummary table saved: {csv_path}")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    print("=== Surrogate Gradient Ablation Comparison: Tmaze Active ===")
    plot_surrogate_comparison()
    print("\nDone.")
