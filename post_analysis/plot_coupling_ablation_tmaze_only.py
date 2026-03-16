#!/usr/bin/env python3
"""
Plot critic-to-actor coupling ablation for T-Maze passive/active from fixed table values.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main() -> None:
    out_dir = Path("results/thesis_plots/coupling_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Values provided by user table (mean ± std).
    conditions = ["Off (baseline)", "On (detached)", "On (no detach)"]
    passive_mean = np.array([1.000, 1.000, 0.920], dtype=float)
    passive_std = np.array([0.000, 0.000, 0.160], dtype=float)
    active_mean = np.array([0.860, 0.200, 0.430], dtype=float)
    active_std = np.array([0.183, 0.400, 0.384], dtype=float)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.3

    x = np.arange(len(conditions))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        x - width / 2,
        passive_mean,
        width,
        yerr=passive_std,
        capsize=4,
        label="T-Maze Passive",
        color="#2b6cb0",
        edgecolor="black",
        linewidth=0.8,
    )
    ax.bar(
        x + width / 2,
        active_mean,
        width,
        yerr=active_std,
        capsize=4,
        label="T-Maze Active",
        color="#d97706",
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Mean Evaluation Return")
    ax.set_xlabel("Coupling Condition")
    ax.set_ylim(0.0, 1.2)
    ax.legend(frameon=True, loc="upper right")
    ax.set_title("Critic-to-Actor Coupling Ablation (T-Maze)")

    fig.tight_layout()
    png_path = out_dir / "coupling_ablation_tmaze_only.png"
    pdf_path = out_dir / "coupling_ablation_tmaze_only.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
