#!/usr/bin/env python3
"""
Generate thesis-ready spike activity figure from masters TMaze logs.

Outputs:
  - results/thesis_plots/spike_activity/spike_activity_statistics.png
  - results/thesis_plots/spike_activity/spike_activity_statistics.pdf
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    active_dir: str
    passive_dir: str
    ann2snn: bool = False


VARIANTS: List[Variant] = [
    Variant(
        key="snn_actor_ann_critic",
        label="SNN Actor +\nANN Critic",
        active_dir="tmaze_snn_actor_ann_critic_active",
        passive_dir="tmaze_snn_actor_ann_critic_passive",
    ),
    Variant(
        key="ann2snn_actor",
        label="ANN2SNN-\nActor",
        active_dir="tmaze_ann2snn_actor_active",
        passive_dir="tmaze_ann2snn_actor_passive",
        ann2snn=True,
    ),
    Variant(
        key="ann2snn_both",
        label="ANN2SNN-\nBoth",
        active_dir="tmaze_ann2snn_full_active",
        passive_dir="tmaze_ann2snn_full_passive",
        ann2snn=True,
    ),
    Variant(
        key="snn_timing_critic",
        label="SNN Timing\nCritic",
        active_dir="tmaze_snn_actor_snntiming_critic_active",
        passive_dir="tmaze_snn_actor_snntiming_critic_passive",
    ),
]


def _tail_mean(rows: List[Dict[str, str]], candidates: Sequence[str], tail_window: int) -> float:
    col = None
    if rows:
        keys = rows[0].keys()
        for c in candidates:
            if c in keys:
                col = c
                break
    if col is None:
        return float("nan")
    vals: List[float] = []
    for r in rows:
        try:
            v = float(r.get(col, ""))
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan")
    arr = np.asarray(vals[-tail_window:], dtype=float)
    return float(np.mean(arr))


def _read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _seed_metrics(csv_path: Path, tail_window: int) -> Tuple[float, float, float]:
    rows = _read_rows(csv_path)
    actor = _tail_mean(rows, ["spikes/actor_per_step", "eval/spikes_actor_per_step"], tail_window)
    critic = _tail_mean(rows, ["spikes/critic_per_step", "eval/spikes_critic_per_step"], tail_window)
    post_total = _tail_mean(rows, ["post_conversion/total_spikes"], tail_window)
    return actor, critic, post_total


def _agg(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def _collect_split(root: Path, split: str, tail_window: int) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for v in VARIANTS:
        model_dir = root / f"tmaze_{split}" / (v.active_dir if split == "active" else v.passive_dir)
        actor_seed: List[float] = []
        critic_seed: List[float] = []
        post_seed: List[float] = []
        for seed_dir in sorted(model_dir.glob("seed_*")):
            csv_path = seed_dir / "per_episode_metrics.csv"
            if not csv_path.exists():
                continue
            actor, critic, post_total = _seed_metrics(csv_path, tail_window)
            actor_seed.append(actor)
            critic_seed.append(critic)
            post_seed.append(post_total)
        actor_m, actor_s = _agg(actor_seed)
        critic_m, critic_s = _agg(critic_seed)
        post_m, _ = _agg(post_seed)
        out[v.key] = {
            "actor_mean": actor_m,
            "actor_std": actor_s,
            "critic_mean": critic_m,
            "critic_std": critic_s,
            "post_total_mean": post_m,
        }
    return out


def _draw_split(ax: plt.Axes, title: str, split_data: Dict[str, Dict[str, float]], y_max: float) -> None:
    x = np.arange(len(VARIANTS), dtype=float)
    width = 0.34

    actor_means = np.asarray([split_data[v.key]["actor_mean"] for v in VARIANTS], dtype=float)
    actor_stds = np.asarray([split_data[v.key]["actor_std"] for v in VARIANTS], dtype=float)
    critic_means = np.asarray([split_data[v.key]["critic_mean"] for v in VARIANTS], dtype=float)
    critic_stds = np.asarray([split_data[v.key]["critic_std"] for v in VARIANTS], dtype=float)

    # Keep ANN2SNN bars visually absent for actor/critic split (not logged as split rates).
    for i, v in enumerate(VARIANTS):
        if v.ann2snn:
            actor_means[i] = np.nan
            actor_stds[i] = np.nan
            critic_means[i] = np.nan
            critic_stds[i] = np.nan

    am = np.nan_to_num(actor_means, nan=0.0)
    asd = np.nan_to_num(actor_stds, nan=0.0)
    cm = np.nan_to_num(critic_means, nan=0.0)
    csd = np.nan_to_num(critic_stds, nan=0.0)

    ax.bar(
        x - width / 2,
        am,
        width=width,
        yerr=asd,
        capsize=3,
        color="#1f77b4",
        label="Actor firing rate (spikes/step)",
        alpha=0.9,
    )
    ax.bar(
        x + width / 2,
        cm,
        width=width,
        yerr=csd,
        capsize=3,
        color="#ff7f0e",
        label="Critic firing rate (spikes/step)",
        alpha=0.9,
    )

    for i, v in enumerate(VARIANTS):
        if v.ann2snn:
            post = split_data[v.key]["post_total_mean"]
            ax.text(i, y_max * 0.06, "split N/A", ha="center", va="bottom", fontsize=8, color="#444444")
            if np.isfinite(post):
                ax.text(i, y_max * 0.015, f"post total≈{post:.0f}", ha="center", va="bottom", fontsize=7, color="#666666")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([v.label for v in VARIANTS], fontsize=9)
    ax.set_xlabel("Model variant", fontsize=10)
    ax.set_ylim(0.0, y_max)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def main() -> None:
    p = argparse.ArgumentParser("Generate spike activity statistics figure from masters logs.")
    p.add_argument("--logs-root", type=str, default="results/logs/masters")
    p.add_argument("--out-dir", type=str, default="results/thesis_plots/spike_activity")
    p.add_argument("--tail-window", type=int, default=10, help="Last-K window for per-seed averages.")
    args = p.parse_args()

    logs_root = Path(args.logs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    active = _collect_split(logs_root, "active", int(args.tail_window))
    passive = _collect_split(logs_root, "passive", int(args.tail_window))

    y_candidates: List[float] = []
    for split_data in (active, passive):
        for v in VARIANTS:
            for k in ("actor_mean", "critic_mean"):
                vv = split_data[v.key][k]
                if np.isfinite(vv):
                    y_candidates.append(vv)
    y_max = max(y_candidates) if y_candidates else 1.0
    y_max = max(1.0, y_max * 1.22)

    fig = plt.figure(figsize=(13.5, 6.2), constrained_layout=True)
    gs = GridSpec(2, 2, height_ratios=[14, 3], figure=fig)
    ax_active = fig.add_subplot(gs[0, 0])
    ax_passive = fig.add_subplot(gs[0, 1], sharey=ax_active)
    ax_note = fig.add_subplot(gs[1, :])
    ax_note.axis("off")

    _draw_split(ax_active, "TMaze Active", active, y_max)
    _draw_split(ax_passive, "TMaze Passive", passive, y_max)
    ax_active.set_ylabel("Firing rate (spikes per decision step)", fontsize=10)

    handles, labels = ax_active.get_legend_handles_labels()
    ax_note.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10)
    ax_note.text(
        0.5,
        0.22,
        "ANN2SNN variants: actor/critic split firing-rate columns are unavailable in current logs; "
        "post-conversion total spikes are shown as in-panel text.",
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
    )
    fig.suptitle("Spike Activity Statistics Across Spiking Model Variants", fontsize=14, fontweight="bold")

    out_png = out_dir / "spike_activity_statistics.png"
    out_pdf = out_dir / "spike_activity_statistics.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
