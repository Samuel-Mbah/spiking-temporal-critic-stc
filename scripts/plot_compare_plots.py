"""
plot_results.py
───────────────
Generates training return curves and evaluation return bar charts
for all four environments:
  - CartPole (fully observable)
  - PO-CartPole (partially observable)
  - T-Maze Passive
  - T-Maze Active

HOW TO USE WITH REAL DATA
─────────────────────────
Replace the contents of REAL_DATA at the bottom of this file.
Each entry is structured as:

REAL_DATA[env_name][variant_name] = {
    "train_steps"  : 1-D np.array of shape (C,)       ← checkpoint steps
    "train_seeds"  : 2-D np.array of shape (N_seeds, C) ← per-seed train returns
    "eval_seeds"   : 1-D np.array of shape (N_seeds,)  ← final eval return per seed
}

If REAL_DATA is None (default), synthetic data is generated so
you can verify the plot style immediately.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import argparse
import glob
import json
import os
import warnings

# ═══════════════════════════════════════════════════════════════════════════════
#  STYLE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
PALETTE = {
    "ANN Baseline":                "#2C6FBF",
    "SNN Actor + ANN Critic":      "#E07B39",
    "ANN2SNN-Actor":               "#8B55A6",
    "ANN2SNN-Both":                "#C0392B",
    "STC (our model)":             "#28A745",
}
LINESTYLES = {
    "ANN Baseline":                (0, ()),
    "SNN Actor + ANN Critic":      (0, (5, 1.5)),
    "ANN2SNN-Actor":               (0, (3, 1, 1, 1)),
    "ANN2SNN-Both":                (0, (1, 1)),
    "STC (our model)":             (0, (5, 1, 1, 1, 1, 1)),
}
VARIANTS = [
    "ANN Baseline",
    "STC (our model)",
    "SNN Actor + ANN Critic",
    "ANN2SNN-Actor",
    "ANN2SNN-Both",
]
BAR_WIDTH   = 0.14
N_SEEDS     = 5
TOTAL_STEPS = 2_000_000
N_CKPTS     = 100
LW          = 1.9
ALPHA_SHADE = 0.13
ALPHA_SEED  = 0.45
SEED_DOT_SZ = 18

ENVS = {
    "CartPole":        {"threshold": 475.0, "chance": None,  "ymax": 540,  "ylabel": "Mean Episodic Return"},
    "PO-CartPole":     {"threshold": 475.0, "chance": None,  "ymax": 540,  "ylabel": "Mean Episodic Return"},
    "T-Maze Passive":  {"threshold": 0.95,  "chance": 0.50,  "ymax": 1.08, "ylabel": "Mean Episodic Return"},
    "T-Maze Active":   {"threshold": 0.95,  "chance": 0.50,  "ymax": 1.08, "ylabel": "Mean Episodic Return"},
}

ENV_LABELS = {
    "CartPole":       "CartPole (MDP)",
    "PO-CartPole":    "PO-CartPole (POMDP)",
    "T-Maze Passive": "T-Maze Passive (POMDP)",
    "T-Maze Active":  "T-Maze Active (POMDP)",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA GENERATOR  (replace with real data — see header)
# ═══════════════════════════════════════════════════════════════════════════════
def make_synthetic(rng):
    steps = np.linspace(0, TOTAL_STEPS, N_CKPTS)

    def sig(steps, onset, slope, plat, noise, smooth=5):
        base  = plat / (1 + np.exp(-slope * (steps - onset)))
        noisy = base + rng.normal(0, noise, len(steps))
        return gaussian_filter1d(noisy, sigma=smooth)

    # (onset_mean, onset_std, slope, plateau_mean, plateau_std, noise_scale)
    SYNTH_PARAMS = {
        "CartPole": {
            "ANN Baseline":                (400e3, 50e3,  8e-6, 495, 3,  10),
            "STC (our model)":             (560e3, 80e3,  7e-6, 488, 6,  18),
            "SNN Actor + ANN Critic":      (520e3, 70e3,  7e-6, 483, 7,  16),
            "ANN2SNN-Actor":               (750e3, 120e3, 5e-6, 420, 28, 28),
            "ANN2SNN-Both":                (900e3, 150e3, 4e-6, 368, 42, 38),
        },
        "PO-CartPole": {
            "ANN Baseline":                (550e3, 70e3,  7e-6, 445, 12, 18),
            "STC (our model)":             (620e3, 90e3,  6e-6, 438, 14, 22),
            "SNN Actor + ANN Critic":      (680e3, 100e3, 6e-6, 420, 18, 22),
            "ANN2SNN-Actor":               (950e3, 160e3, 4e-6, 320, 55, 40),
            "ANN2SNN-Both":                (1.1e6, 180e3, 3e-6, 255, 65, 48),
        },
        "T-Maze Passive": {
            "ANN Baseline":                (450e3, 60e3,  9e-6, 0.96, 0.02, 0.04),
            "STC (our model)":             (500e3, 80e3,  8e-6, 0.95, 0.03, 0.05),
            "SNN Actor + ANN Critic":      (600e3, 90e3,  7e-6, 0.93, 0.04, 0.06),
            "ANN2SNN-Actor":               (800e3, 130e3, 5e-6, 0.78, 0.10, 0.08),
            "ANN2SNN-Both":                (1.1e6, 160e3, 4e-6, 0.65, 0.14, 0.10),
        },
        "T-Maze Active": {
            "ANN Baseline":                (600e3, 90e3,  7e-6, 0.93, 0.05, 0.06),
            "STC (our model)":             (650e3, 110e3, 6e-6, 0.91, 0.07, 0.08),
            "SNN Actor + ANN Critic":      (800e3, 130e3, 5e-6, 0.85, 0.10, 0.09),
            "ANN2SNN-Actor":               (1.2e6, 200e3, 3e-6, 0.63, 0.18, 0.11),
            "ANN2SNN-Both":                (1.5e6, 230e3, 2e-6, 0.54, 0.22, 0.12),
        },
    }

    data = {}
    for env, vparams in SYNTH_PARAMS.items():
        lo  = 0.48 if "Maze" in env else 9.0
        hi  = 1.0  if "Maze" in env else 500.0
        data[env] = {}
        for v, (om, os_, sl, pm, ps, ns) in vparams.items():
            seeds = []
            for _ in range(N_SEEDS):
                onset = rng.normal(om, os_)
                plat  = float(np.clip(rng.normal(pm, ps), lo, hi))
                seeds.append(sig(steps, onset, sl, plat, ns))
            train_arr = np.clip(np.array(seeds), lo, hi)
            eval_arr  = np.clip(
                rng.normal(pm, ps * 0.6, N_SEEDS), lo, hi
            ).astype(float)
            data[env][v] = {
                "train_steps": steps,
                "train_seeds": train_arr,
                "eval_seeds":  eval_arr,
            }
    return data


# ═══════════════════════════════════════════════════════════════════════════════
#  AXIS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def style_ax(ax):
    ax.yaxis.grid(True, color="#E4E4E4", lw=0.7, zorder=0)
    ax.xaxis.grid(True, color="#E4E4E4", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#CCCCCC")
    ax.tick_params(axis="both", labelsize=8.5, length=3, color="#AAAAAA")


def threshold_line(ax, val, label, color="#444444"):
    ax.axhline(val, color=color, lw=1.1,
               linestyle=(0, (6, 4)), zorder=1)
    ax.text(1.01, val, label, transform=ax.get_yaxis_transform(),
            fontsize=7, color=color, va="center", clip_on=False)


def chance_line(ax, val):
    ax.axhline(val, color="#AAAAAA", lw=0.9,
               linestyle=(0, (3, 4)), zorder=1)
    ax.text(1.01, val, "chance", transform=ax.get_yaxis_transform(),
            fontsize=7, color="#AAAAAA", va="center", clip_on=False)


def env_step_limit(env_data):
    max_step = 0.0
    for v in VARIANTS:
        steps = np.asarray(env_data[v]["train_steps"], dtype=float)
        if steps.size:
            max_step = max(max_step, float(np.nanmax(steps)))
    return max_step if max_step > 0 else float(TOTAL_STEPS)


# ═══════════════════════════════════════════════════════════════════════════════
#  STATISTICAL HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def sig_vs_baseline(eval_data, variant, baseline="ANN Baseline"):
    """
    Returns significance marker string.
    Paired: Shapiro-Wilk on differences -> t-test or Wilcoxon.
    """
    a = np.asarray(eval_data[baseline]["eval_seeds"], dtype=float)
    b = np.asarray(eval_data[variant]["eval_seeds"], dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]
    if len(a) < 2:
        return "ns"

    d = b - a
    if np.allclose(d, 0.0):
        return "ns"
    # Shapiro is undefined/unstable for zero-range samples.
    if np.ptp(d) == 0.0:
        return "***"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        if len(d) >= 3:
            p_sw = float(stats.shapiro(d).pvalue)
        else:
            p_sw = 1.0

        if p_sw > 0.05:
            p = float(stats.ttest_rel(a, b).pvalue)
        else:
            # zsplit/approx is more stable with ties/zeros at small N.
            p = float(stats.wilcoxon(a, b, zero_method="zsplit", method="approx").pvalue)

    if not np.isfinite(p):
        return "ns"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — TRAINING CURVES  (one figure per environment, 180 dpi)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_training(env, env_data, cfg, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")
    style_ax(ax)

    for v in VARIANTS:
        d     = env_data[v]["train_seeds"]
        steps = env_data[v]["train_steps"]
        mu    = d.mean(axis=0)
        sigma = d.std(axis=0)
        col   = PALETTE[v]
        ls    = LINESTYLES[v]
        ax.fill_between(steps, mu - sigma, mu + sigma,
                        color=col, alpha=ALPHA_SHADE, zorder=2)
        ax.plot(steps, mu, color=col, lw=LW, linestyle=ls,
                label=v, zorder=3)
        if v == "ANN2SNN-Actor":
            ft_steps = env_data[v].get("ft_train_steps")
            ft_seeds = env_data[v].get("ft_train_seeds")
            if ft_steps is not None and ft_seeds is not None and len(ft_steps) > 0:
                ft_mu = ft_seeds.mean(axis=0)
                ft_sigma = ft_seeds.std(axis=0)
                ax.fill_between(
                    ft_steps, ft_mu - ft_sigma, ft_mu + ft_sigma,
                    color=col, alpha=ALPHA_SHADE + 0.04, zorder=4
                )
                ax.plot(
                    ft_steps, ft_mu, color=col, lw=LW + 0.4,
                    linestyle=(0, ()), zorder=5
                )

    # ANN2SNN-Actor zero-shot marker (mean across seeds when available)
    a2s = env_data.get("ANN2SNN-Actor", {})
    zs_steps = np.asarray(a2s.get("zero_shot_steps", []), dtype=float)
    zs_vals = np.asarray(a2s.get("zero_shot_vals", []), dtype=float)
    zs_mask = np.isfinite(zs_steps) & np.isfinite(zs_vals)
    if zs_mask.any():
        x_star = float(np.nanmean(zs_steps[zs_mask]))
        y_star = float(np.nanmean(zs_vals[zs_mask]))
        ax.scatter(
            [x_star], [y_star], marker="*", s=120,
            color=PALETTE["ANN2SNN-Actor"], edgecolors="#1a1a1a",
            linewidths=0.9, zorder=7
        )
        y_off = 10.0 if cfg["chance"] is None else 0.03
        ax.annotate(
            "zero-shot *",
            xy=(x_star, y_star),
            xytext=(x_star, y_star + y_off),
            ha="center", va="bottom", fontsize=8,
            color=PALETTE["ANN2SNN-Actor"],
            arrowprops=dict(arrowstyle="-", lw=0.8, color=PALETTE["ANN2SNN-Actor"]),
            zorder=8,
        )

    # Reference lines
    threshold_line(ax, cfg["threshold"],
                   f"τ = {cfg['threshold']}")
    if cfg["chance"] is not None:
        chance_line(ax, cfg["chance"])

    x_max = env_step_limit(env_data)
    ax.set_xlim(0, x_max * 1.01)
    ax.set_ylim(0.0, cfg["ymax"])
    ax.set_xlabel("Environment Steps", fontsize=10, labelpad=5)
    ax.set_ylabel(cfg["ylabel"], fontsize=10, labelpad=5)
    ax.set_title(f"{ENV_LABELS[env]} — Training Return Curves",
                 fontsize=11, fontweight="bold", pad=8)

    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x, _: f"{x/1e6:.1f}M" if x > 0 else "0"))

    leg = ax.legend(loc="lower right", fontsize=8, framealpha=0.92,
                    edgecolor="#CCCCCC", fancybox=False,
                    handlelength=2.2, labelspacing=0.4, borderpad=0.6)
    leg.get_frame().set_linewidth(0.8)
    for t in leg.get_texts():
        if "our model" in t.get_text():
            t.set_fontweight("bold")

    ax.text(0.02, 0.97,
            f"Mean ± 1 std  |  {N_SEEDS} seeds",
            transform=ax.transAxes, fontsize=7.5,
            color="#666666", va="top")
    if "ft_train_steps" in a2s:
        ax.text(0.02, 0.91,
                "ANN2SNN-Actor: * zero-shot, solid continuation = fine-tuning",
                transform=ax.transAxes, fontsize=7.2,
                color=PALETTE["ANN2SNN-Actor"], va="top")

    plt.tight_layout(pad=1.2)
    fname = env.lower().replace(" ", "_").replace("-", "_")
    path  = f"{out_dir}/{fname}_training.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — EVALUATION BAR CHART  (one figure per environment, 180 dpi)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_eval(env, env_data, cfg, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")
    style_ax(ax)

    x      = np.arange(len(VARIANTS))
    is_tmaze = "Maze" in env

    for i, v in enumerate(VARIANTS):
        seeds = env_data[v]["eval_seeds"]
        mu    = seeds.mean()
        se    = seeds.std(ddof=1) / np.sqrt(N_SEEDS)
        col   = PALETTE[v]
        edge  = "#1a1a1a" if "our model" in v else col

        ax.bar(i, mu, width=BAR_WIDTH * 4.5,
               color=col, edgecolor=edge,
               linewidth=1.5 if "our model" in v else 0.6,
               zorder=3, alpha=0.88)

        # Error bar
        ax.errorbar(i, mu, yerr=se, fmt="none",
                    ecolor="#333333", elinewidth=1.3,
                    capsize=4, capthick=1.3, zorder=5)

        # Individual seed dots
        jitter = np.linspace(-0.06, 0.06, N_SEEDS)
        ax.scatter(np.full(N_SEEDS, i) + jitter, seeds,
                   color="white", edgecolors=col,
                   s=SEED_DOT_SZ, lw=1.2, zorder=6,
                   alpha=ALPHA_SEED + 0.2)

        # Significance vs ANN baseline
        if v != "ANN Baseline":
            sig = sig_vs_baseline(env_data, v)
            col_sig = "#333333" if sig != "ns" else "#AAAAAA"
            ax.text(i, mu + se + (0.012 if is_tmaze else 6),
                    sig, ha="center", va="bottom",
                    fontsize=8.5, color=col_sig, fontweight="bold")

    # Reference lines
    threshold_line(ax, cfg["threshold"],
                   f"τ = {cfg['threshold']}")
    if cfg["chance"] is not None:
        chance_line(ax, cfg["chance"])

    # x-axis labels (abbreviated for readability)
    xlabels = [
        v.replace("STC (our model)", "STC\n(our model)")
         .replace("SNN Actor + ANN Critic", "SNN Actor\n+ ANN Critic")
         .replace("ANN Baseline", "ANN\nBaseline")
         .replace("ANN2SNN-Actor", "ANN2SNN\nActor")
         .replace("ANN2SNN-Both", "ANN2SNN\nBoth")
        for v in VARIANTS
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)

    y_lo = 0.0 if is_tmaze else 0
    ax.set_ylim(y_lo, cfg["ymax"])
    ax.set_ylabel(cfg["ylabel"], fontsize=10, labelpad=5)
    ax.set_title(f"{ENV_LABELS[env]} — Final Evaluation Returns",
                 fontsize=11, fontweight="bold", pad=8)

    ax.text(0.02, 0.97,
            "Bars: mean ± SE  |  Dots: individual seeds  |"
            "  * p<0.05  ** p<0.01  *** p<0.001  vs ANN Baseline",
            transform=ax.transAxes, fontsize=7, color="#666666", va="top")

    plt.tight_layout(pad=1.2)
    fname = env.lower().replace(" ", "_").replace("-", "_")
    path  = f"{out_dir}/{fname}_eval.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 3 — 4×2 OVERVIEW PANEL  (training top row, eval bottom row)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_overview_panel(all_data, out_dir):
    env_list = list(ENVS.keys())
    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.38, wspace=0.32,
                           left=0.055, right=0.96,
                           top=0.91, bottom=0.09)

    row_labels = ["Training Return Curves", "Final Evaluation Returns"]

    for col_idx, env in enumerate(env_list):
        cfg      = ENVS[env]
        env_data = all_data[env]
        is_tmaze = "Maze" in env
        x_max = env_step_limit(env_data)

        # ── Training row ─────────────────────────────────────────────────────
        ax_tr = fig.add_subplot(gs[0, col_idx])
        ax_tr.set_facecolor("#FAFAFA")
        style_ax(ax_tr)

        for v in VARIANTS:
            d     = env_data[v]["train_seeds"]
            steps = env_data[v]["train_steps"]
            mu    = d.mean(axis=0)
            sigma = d.std(axis=0)
            col   = PALETTE[v]
            ax_tr.fill_between(steps, mu - sigma, mu + sigma,
                               color=col, alpha=ALPHA_SHADE, zorder=2)
            ax_tr.plot(steps, mu, color=col, lw=LW - 0.3,
                       linestyle=LINESTYLES[v], zorder=3,
                       label=v)
            if v == "ANN2SNN-Actor":
                ft_steps = env_data[v].get("ft_train_steps")
                ft_seeds = env_data[v].get("ft_train_seeds")
                if ft_steps is not None and ft_seeds is not None and len(ft_steps) > 0:
                    ft_mu = ft_seeds.mean(axis=0)
                    ft_sigma = ft_seeds.std(axis=0)
                    ax_tr.fill_between(
                        ft_steps, ft_mu - ft_sigma, ft_mu + ft_sigma,
                        color=col, alpha=ALPHA_SHADE + 0.04, zorder=4
                    )
                    ax_tr.plot(
                        ft_steps, ft_mu, color=col, lw=LW,
                        linestyle=(0, ()), zorder=5
                    )

        threshold_line(ax_tr, cfg["threshold"],
                       f"τ={cfg['threshold']}")
        if cfg["chance"] is not None:
            chance_line(ax_tr, cfg["chance"])

        ax_tr.set_xlim(0, x_max * 1.01)
        ax_tr.set_ylim(0.0, cfg["ymax"])
        ax_tr.xaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda x, _: f"{x/1e6:.0f}M" if x > 0 else "0"))
        ax_tr.set_xlabel("Steps", fontsize=8, labelpad=3)
        if col_idx == 0:
            ax_tr.set_ylabel("Mean Episodic Return", fontsize=8.5)
        ax_tr.set_title(ENV_LABELS[env], fontsize=9,
                        fontweight="bold", pad=5)

        # ── Eval row ─────────────────────────────────────────────────────────
        ax_ev = fig.add_subplot(gs[1, col_idx])
        ax_ev.set_facecolor("#FAFAFA")
        style_ax(ax_ev)

        x = np.arange(len(VARIANTS))
        for i, v in enumerate(VARIANTS):
            seeds = env_data[v]["eval_seeds"]
            mu    = seeds.mean()
            se    = seeds.std(ddof=1) / np.sqrt(N_SEEDS)
            col   = PALETTE[v]
            edge  = "#1a1a1a" if "our model" in v else col
            ax_ev.bar(i, mu, width=0.55,
                      color=col, edgecolor=edge,
                      linewidth=1.4 if "our model" in v else 0.5,
                      zorder=3, alpha=0.88)
            ax_ev.errorbar(i, mu, yerr=se, fmt="none",
                           ecolor="#333333", elinewidth=1.2,
                           capsize=3, capthick=1.2, zorder=5)
            jitter = np.linspace(-0.07, 0.07, N_SEEDS)
            ax_ev.scatter(np.full(N_SEEDS, i) + jitter, seeds,
                          color="white", edgecolors=col,
                          s=14, lw=1.0, zorder=6,
                          alpha=0.65)
            if v != "ANN Baseline":
                sig = sig_vs_baseline(env_data, v)
                col_sig = "#333333" if sig != "ns" else "#BBBBBB"
                ax_ev.text(
                    i,
                    mu + se + (0.015 if is_tmaze else 7),
                    sig, ha="center", va="bottom",
                    fontsize=7.5, color=col_sig, fontweight="bold")

        threshold_line(ax_ev, cfg["threshold"],
                       f"τ={cfg['threshold']}")
        if cfg["chance"] is not None:
            chance_line(ax_ev, cfg["chance"])

        xlabels = ["ANN", "STC*", "SNN+ANN", "A2S-A", "A2S-B"]
        ax_ev.set_xticks(x)
        ax_ev.set_xticklabels(xlabels, fontsize=8)
        ax_ev.set_ylim(0.0 if is_tmaze else 0, cfg["ymax"])
        if col_idx == 0:
            ax_ev.set_ylabel("Final Eval Return", fontsize=8.5)

    # Row labels on left margin
    for row_i, label in enumerate(row_labels):
        fig.text(0.008, 0.73 - row_i * 0.46, label,
                 fontsize=10, fontweight="bold", color="#333333",
                 rotation=90, va="center")

    # Shared legend at top
    handles = [
        plt.Line2D([0], [0], color=PALETTE[v], lw=2.0,
                   linestyle=LINESTYLES[v],
                   label=v)
        for v in VARIANTS
    ]
    leg = fig.legend(handles=handles,
                     loc="upper center", ncol=5,
                     fontsize=8.5, framealpha=0.92,
                     edgecolor="#CCCCCC", fancybox=False,
                     handlelength=2.0, columnspacing=1.0,
                     bbox_to_anchor=(0.5, 1.0))
    leg.get_frame().set_linewidth(0.8)
    for t in leg.get_texts():
        if "our model" in t.get_text():
            t.set_fontweight("bold")

    # Caption note
    fig.text(0.5, 0.005,
             "* STC = STC (our model)  |  "
             "A2S-A = ANN2SNN-Actor  |  A2S-B = ANN2SNN-Both  |  "
             "A2S-A * = zero-shot, solid continuation = fine-tuning  |  "
             "Error bars: ±SE  |  Dots: individual seeds  |  "
             "* p<0.05  ** p<0.01  *** p<0.001  vs ANN Baseline",
             ha="center", fontsize=7, color="#666666")

    fig.suptitle("Training and Evaluation Returns — All Environments",
                 fontsize=13, fontweight="bold", y=1.025)

    path = f"{out_dir}/overview_all_environments.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  REAL-DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════
ENV_DIR_TO_NAME = {
    "cartpole": "CartPole",
    "partial_cartpole": "PO-CartPole",
    "tmaze_passive": "T-Maze Passive",
    "tmaze_active": "T-Maze Active",
}

VARIANT_DIR_TO_NAME = {
    "ann_baseline": "ANN Baseline",
    "poc_ann_baseline": "ANN Baseline",
    "tmaze_ann_baseline_active": "ANN Baseline",
    "tmaze_ann_baseline_passive": "ANN Baseline",
    "snn_actor_snntiming_critic": "STC (our model)",
    "poc_snn_actor_snntiming_critic": "STC (our model)",
    "tmaze_snn_actor_snntiming_critic_active": "STC (our model)",
    "tmaze_snn_actor_snntiming_critic_passive": "STC (our model)",
    "snn_actor_ann_critic": "SNN Actor + ANN Critic",
    "poc_snn_actor_ann_critic": "SNN Actor + ANN Critic",
    "tmaze_snn_actor_ann_critic_active": "SNN Actor + ANN Critic",
    "tmaze_snn_actor_ann_critic_passive": "SNN Actor + ANN Critic",
    "ann2snn_actor": "ANN2SNN-Actor",
    "poc_ann2snn_actor": "ANN2SNN-Actor",
    "tmaze_ann2snn_actor_active": "ANN2SNN-Actor",
    "tmaze_ann2snn_actor_passive": "ANN2SNN-Actor",
    "ann2snn_both": "ANN2SNN-Both",
    "poc_ann2snn_full": "ANN2SNN-Both",
    "tmaze_ann2snn_full_active": "ANN2SNN-Both",
    "tmaze_ann2snn_full_passive": "ANN2SNN-Both",
}


def _extract_series(metrics, key):
    events = metrics.get(key, [])
    if not events:
        return None, None
    steps = np.array([e["step"] for e in events], dtype=float)
    vals = np.array([e["value"] for e in events], dtype=float)
    mask = np.isfinite(steps) & np.isfinite(vals)
    return steps[mask], vals[mask]


def _align_seed_series(seed_steps, seed_vals):
    if not seed_steps:
        raise ValueError("No seed series to align.")
    lengths = [len(s) for s in seed_steps]
    ref_idx = int(np.argmax(lengths))
    steps_ref = np.asarray(seed_steps[ref_idx], dtype=float)

    aligned = []
    for s, v in zip(seed_steps, seed_vals):
        s_arr = np.asarray(s, dtype=float)
        v_arr = np.asarray(v, dtype=float)
        order = np.argsort(s_arr)
        s_arr = s_arr[order]
        v_arr = v_arr[order]
        interp = np.interp(steps_ref, s_arr, v_arr, left=np.nan, right=np.nan)
        aligned.append(interp)
    aligned = np.asarray(aligned, dtype=float)

    keep = np.all(np.isfinite(aligned), axis=0)
    if np.sum(keep) >= 3:
        return steps_ref[keep], aligned[:, keep]

    min_len = min(lengths)
    steps_trim = np.asarray(seed_steps[0][:min_len], dtype=float)
    train_trim = np.asarray([np.asarray(v[:min_len], dtype=float) for v in seed_vals], dtype=float)
    return steps_trim, train_trim


def _load_seed_metrics(path, variant_name=None):
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    train_steps, train_vals = _extract_series(metrics, "eval/rolling_reward")
    if train_steps is None or train_vals is None or len(train_steps) == 0:
        raise ValueError(f"Missing eval/rolling_reward in {path}")
    ft_steps, ft_vals = _extract_series(metrics, "post_conversion_ft/train_reward")
    if ft_steps is not None and ft_vals is not None and len(ft_steps) > 0:
        keep = ft_steps > float(train_steps[-1])
        if np.any(keep):
            ft_steps = ft_steps[keep]
            ft_vals = ft_vals[keep]
        else:
            ft_steps, ft_vals = None, None

    zs_steps, zs_vals = _extract_series(metrics, "post_conversion/zero_shot_reward")
    if zs_steps is None or zs_vals is None or len(zs_steps) == 0:
        zs_steps, zs_vals = _extract_series(metrics, "snn_zero_shot/reward")

    zs_step = float(zs_steps[-1]) if zs_steps is not None and len(zs_steps) > 0 else np.nan
    zs_val = float(zs_vals[-1]) if zs_vals is not None and len(zs_vals) > 0 else np.nan

    if variant_name in {"ANN2SNN-Actor", "ANN2SNN-Both"}:
        if np.isfinite(zs_val):
            final_eval = float(zs_val)
        else:
            raise ValueError(f"Missing zero-shot reward for ANN2SNN variant in {path}")
    else:
        _, eval_vals_ft = _extract_series(metrics, "post_conversion_ft/eval_reward")
        _, eval_vals = _extract_series(metrics, "eval/reward")
        if eval_vals_ft is not None and len(eval_vals_ft) > 0:
            final_eval = float(eval_vals_ft[-1])
        elif eval_vals is not None and len(eval_vals) > 0:
            final_eval = float(eval_vals[-1])
        else:
            raise ValueError(f"Missing eval/reward in {path}")

    return train_steps, train_vals, final_eval, zs_step, zs_val, ft_steps, ft_vals


def load_real_data(logs_root):
    logs_root = os.path.abspath(logs_root)
    if not os.path.isdir(logs_root):
        raise FileNotFoundError(f"Logs root does not exist: {logs_root}")

    raw = {env: {} for env in ENVS.keys()}
    for env_dir in sorted(os.listdir(logs_root)):
        env_name = ENV_DIR_TO_NAME.get(env_dir)
        if env_name is None:
            continue
        env_path = os.path.join(logs_root, env_dir)
        if not os.path.isdir(env_path):
            continue
        for variant_dir in sorted(os.listdir(env_path)):
            variant_name = VARIANT_DIR_TO_NAME.get(variant_dir)
            if variant_name is None:
                continue
            seed_glob = os.path.join(env_path, variant_dir, "seed_*", "metrics_raw.json")
            seed_paths = sorted(glob.glob(seed_glob))
            if not seed_paths:
                continue
            seed_train = []
            seed_steps = []
            seed_eval = []
            seed_zs_steps = []
            seed_zs_vals = []
            seed_ft_steps = []
            seed_ft_vals = []
            for p in seed_paths:
                try:
                    steps, train_vals, final_eval, zs_step, zs_val, ft_steps, ft_vals = _load_seed_metrics(
                        p,
                        variant_name=variant_name,
                    )
                except ValueError:
                    continue
                seed_steps.append(steps)
                seed_train.append(train_vals)
                seed_eval.append(final_eval)
                seed_zs_steps.append(zs_step)
                seed_zs_vals.append(zs_val)
                if ft_steps is not None and ft_vals is not None and len(ft_steps) > 0:
                    seed_ft_steps.append(ft_steps)
                    seed_ft_vals.append(ft_vals)
            if not seed_train:
                continue
            steps_ref, train_arr = _align_seed_series(seed_steps, seed_train)
            eval_arr = np.array(seed_eval, dtype=float)
            entry = {
                "train_steps": steps_ref,
                "train_seeds": train_arr,
                "eval_seeds": eval_arr,
            }
            if np.isfinite(seed_zs_steps).any() and np.isfinite(seed_zs_vals).any():
                entry["zero_shot_steps"] = np.asarray(seed_zs_steps, dtype=float)
                entry["zero_shot_vals"] = np.asarray(seed_zs_vals, dtype=float)
            if seed_ft_steps:
                ft_steps_ref, ft_arr = _align_seed_series(seed_ft_steps, seed_ft_vals)
                entry["ft_train_steps"] = ft_steps_ref
                entry["ft_train_seeds"] = ft_arr
            raw[env_name][variant_name] = entry

    for env_name in ENVS.keys():
        missing = [v for v in VARIANTS if v not in raw[env_name]]
        if missing:
            raise RuntimeError(
                f"Missing variants for {env_name}: {missing}. "
                f"Check logs under {logs_root}."
            )
    return raw


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare training/eval plots across 4 environments."
    )
    parser.add_argument(
        "--logs-root",
        default="results/logs/masters",
        help="Root directory containing env/variant/seed_*/metrics_raw.json logs.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/plots/compare_plots",
        help="Directory where plot images are saved.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of loading real logs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.synthetic:
        print("Using synthetic data.")
        rng = np.random.default_rng(42)
        data = make_synthetic(rng)
    else:
        print(f"Loading real data from: {os.path.abspath(args.logs_root)}")
        data = load_real_data(args.logs_root)

    print("\nGenerating individual environment plots...")
    for env, cfg in ENVS.items():
        print(f"\n  [{env}]")
        plot_training(env, data[env], cfg, out_dir)
        plot_eval(env, data[env], cfg, out_dir)

    print("\nGenerating overview panel...")
    plot_overview_panel(data, out_dir)
    print("\nAll plots saved to", out_dir)


if __name__ == "__main__":
    main()
