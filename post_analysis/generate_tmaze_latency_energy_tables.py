#!/usr/bin/env python3
"""
Generate publication-ready TMaze latency/energy summary tables for five variants.

Outputs (under --out-dir):
  - tmaze_latency_energy_seed_metrics.csv
  - tmaze_latency_energy_summary.csv
  - tmaze_latency_energy_active.tex
  - tmaze_latency_energy_passive.tex

Method:
  - Read per-seed `per_episode_metrics.csv` from masters logs.
  - For each metric, compute seed value as mean of last `tail_window` non-NaN points.
  - Aggregate across seeds with mean, sample std, and 95% CI (Student-t).
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


# Two-sided 95% t critical values by dof (1..30).
T_CRIT_95: Dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    dir_active: str
    dir_passive: str
    actor_steps_expected: bool
    critic_steps_expected: bool
    ann2snn: bool = False


VARIANTS: List[VariantSpec] = [
    VariantSpec(
        key="ann_baseline",
        label="ANN Baseline",
        dir_active="tmaze_ann_baseline_active",
        dir_passive="tmaze_ann_baseline_passive",
        actor_steps_expected=False,
        critic_steps_expected=False,
    ),
    VariantSpec(
        key="snn_actor_ann_critic",
        label="SNN Actor + ANN Critic",
        dir_active="tmaze_snn_actor_ann_critic_active",
        dir_passive="tmaze_snn_actor_ann_critic_passive",
        actor_steps_expected=True,
        critic_steps_expected=False,
    ),
    VariantSpec(
        key="ann2snn_actor",
        label="ANN2SNN-Actor",
        dir_active="tmaze_ann2snn_actor_active",
        dir_passive="tmaze_ann2snn_actor_passive",
        actor_steps_expected=False,
        critic_steps_expected=False,
        ann2snn=True,
    ),
    VariantSpec(
        key="ann2snn_both",
        label="ANN2SNN-Both",
        dir_active="tmaze_ann2snn_full_active",
        dir_passive="tmaze_ann2snn_full_passive",
        actor_steps_expected=False,
        critic_steps_expected=False,
        ann2snn=True,
    ),
    VariantSpec(
        key="snn_timing_critic",
        label="SNN Timing Critic (Proposed)",
        dir_active="tmaze_snn_actor_snntiming_critic_active",
        dir_passive="tmaze_snn_actor_snntiming_critic_passive",
        actor_steps_expected=True,
        critic_steps_expected=True,
    ),
]


def _existing_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    return None


def _tail_mean(rows: List[Dict[str, str]], columns: Sequence[str], candidates: Sequence[str], tail_window: int) -> float:
    col = _existing_col(columns, candidates)
    if col is None:
        return float("nan")
    values: List[float] = []
    for r in rows:
        raw = r.get(col, "")
        try:
            v = float(raw)
        except Exception:
            continue
        if np.isfinite(v):
            values.append(v)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values[-tail_window:]))


def _t_critical_95(n: int) -> float:
    if n <= 1:
        return float("nan")
    dof = n - 1
    if dof in T_CRIT_95:
        return T_CRIT_95[dof]
    # Conservative fallback for large dof.
    return 1.96


def _stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "ci95_lo": float("nan"),
            "ci95_hi": float("nan"),
        }
    mean = float(np.mean(arr))
    if n == 1:
        return {
            "n": 1.0,
            "mean": mean,
            "std": 0.0,
            "ci95_lo": mean,
            "ci95_hi": mean,
        }
    std = float(np.std(arr, ddof=1))
    tcrit = _t_critical_95(n)
    half = float(tcrit * std / math.sqrt(n))
    return {
        "n": float(n),
        "mean": mean,
        "std": std,
        "ci95_lo": mean - half,
        "ci95_hi": mean + half,
    }


def _clip_nonnegative_ci(s: Dict[str, float]) -> Dict[str, float]:
    out = dict(s)
    if np.isfinite(out.get("ci95_lo", float("nan"))):
        out["ci95_lo"] = max(0.0, float(out["ci95_lo"]))
    if np.isfinite(out.get("ci95_hi", float("nan"))):
        out["ci95_hi"] = max(0.0, float(out["ci95_hi"]))
    return out


def _fmt_mean_std_ci(mean: float, std: float, lo: float, hi: float, digits: int = 3) -> str:
    if not (np.isfinite(mean) and np.isfinite(std) and np.isfinite(lo) and np.isfinite(hi)):
        return "N/A"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _escape_latex(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def _build_caption(split_label: str, caption_style: str) -> str:
    split_txt = _escape_latex(split_label)
    if caption_style == "thesis":
        return (
            "Latency and energy diagnostics for all model variants on TMaze "
            + split_txt
            + ". Actor and critic simulation timesteps (architecture-level, hardware-independent) are reported "
              "alongside wall-clock milliseconds per decision step and GPU energy per episode "
              "(hardware-dependent, GPU simulation only). Higher GPU latency for SNN variants relative to "
              "the ANN baseline reflects sequential LIF timestep simulation overhead, not an inherent property "
              "of spiking computation."
        )
    return (
        "TMaze " + split_txt + " latency and energy diagnostics "
        "(mean $\\pm$ std with 95\\% CI over seeds)."
    )


def _build_latex_table(
    split_label: str,
    frame: List[Dict[str, object]],
    *,
    caption_style: str,
) -> str:
    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{" + _build_caption(split_label, caption_style) + "}")
    lines.append("\\renewcommand{\\arraystretch}{1.2}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\hline")
    lines.append("\\textbf{Model Variant} & \\textbf{Actor steps} & \\textbf{Critic steps} & \\textbf{Latency (ms)} & \\textbf{Energy (J/ep)} \\\\")
    lines.append("\\hline")
    for r in frame:
        lines.append(
            f"{_escape_latex(str(r['model_variant']))} & "
            f"{r['actor_steps_mean_std_ci']} & "
            f"{r['critic_steps_mean_std_ci']} & "
            f"{r['latency_ms_mean_std_ci']} & "
            f"{r['energy_j_ep_mean_std_ci']} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _seed_rows_for_variant(
    split: str,
    variant: VariantSpec,
    model_dir: Path,
    tail_window: int,
) -> List[Dict[str, object]]:
    out_rows: List[Dict[str, object]] = []
    for seed_dir in sorted(model_dir.glob("seed_*")):
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except Exception:
            continue
        csv_path = seed_dir / "per_episode_metrics.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
            columns = list(reader.fieldnames or [])

        if variant.ann2snn:
            latency = _tail_mean(csv_rows, columns, ["post_conversion/mean_latency", "latency_mean_ms", "latency/eval_wall_clock_ms"], tail_window)
            energy = _tail_mean(csv_rows, columns, ["post_conversion/inference_energy", "inference_energy"], tail_window)
        else:
            latency = _tail_mean(csv_rows, columns, ["latency_mean_ms", "latency/eval_wall_clock_ms"], tail_window)
            energy = _tail_mean(csv_rows, columns, ["inference_energy"], tail_window)

        actor_steps = _tail_mean(csv_rows, columns, ["latency/actor_spike_timing_steps"], tail_window)
        critic_steps = _tail_mean(csv_rows, columns, ["latency/critic_spike_timing_steps", "latency/critic_eval_spike_timing_steps"], tail_window)

        if not variant.actor_steps_expected:
            actor_steps = float("nan")
        if not variant.critic_steps_expected:
            critic_steps = float("nan")

        out_rows.append(
            {
                "split": split,
                "variant_key": variant.key,
                "model_variant": variant.label,
                "seed": seed,
                "latency_ms": latency,
                "energy_j_ep": energy,
                "actor_steps": actor_steps,
                "critic_steps": critic_steps,
                "source_csv": str(csv_path),
            }
        )
    return out_rows


def main() -> None:
    p = argparse.ArgumentParser("Generate TMaze active/passive latency-energy tables.")
    p.add_argument("--logs-root", type=str, default="results/logs/masters")
    p.add_argument("--out-dir", type=str, default="results/thesis_plots/tmaze_latency_energy")
    p.add_argument("--tail-window", type=int, default=10, help="Use mean of last K non-NaN points per seed.")
    p.add_argument("--digits", type=int, default=3, help="Decimal places for mean/std/CI string columns.")
    p.add_argument(
        "--caption-style",
        type=str,
        choices=("compact", "thesis"),
        default="thesis",
        help="LaTeX caption style.",
    )
    args = p.parse_args()

    logs_root = Path(args.logs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_seed_rows: List[Dict[str, object]] = []
    for split in ("active", "passive"):
        split_root = logs_root / f"tmaze_{split}"
        for v in VARIANTS:
            model_dir_name = v.dir_active if split == "active" else v.dir_passive
            model_dir = split_root / model_dir_name
            all_seed_rows.extend(_seed_rows_for_variant(split, v, model_dir, int(args.tail_window)))

    if not all_seed_rows:
        raise RuntimeError("No seed metrics found. Check --logs-root and directory layout.")
    seed_csv = out_dir / "tmaze_latency_energy_seed_metrics.csv"
    seed_fields = list(all_seed_rows[0].keys())
    with seed_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=seed_fields)
        writer.writeheader()
        writer.writerows(all_seed_rows)

    summary_rows: List[Dict[str, object]] = []
    for split in ("active", "passive"):
        for v in VARIANTS:
            g = [r for r in all_seed_rows if r["split"] == split and r["variant_key"] == v.key]
            lat = _stats(float(r["latency_ms"]) for r in g)
            ene = _stats(float(r["energy_j_ep"]) for r in g)
            act = _stats(float(r["actor_steps"]) for r in g)
            cri = _stats(float(r["critic_steps"]) for r in g)
            lat = _clip_nonnegative_ci(lat)
            ene = _clip_nonnegative_ci(ene)
            act = _clip_nonnegative_ci(act)
            cri = _clip_nonnegative_ci(cri)

            summary_rows.append(
                {
                    "split": split,
                    "variant_key": v.key,
                    "model_variant": v.label,
                    "n_seeds": int(max(lat["n"], ene["n"], act["n"], cri["n"])),
                    "actor_steps_mean": act["mean"],
                    "actor_steps_std": act["std"],
                    "actor_steps_ci95_lo": act["ci95_lo"],
                    "actor_steps_ci95_hi": act["ci95_hi"],
                    "critic_steps_mean": cri["mean"],
                    "critic_steps_std": cri["std"],
                    "critic_steps_ci95_lo": cri["ci95_lo"],
                    "critic_steps_ci95_hi": cri["ci95_hi"],
                    "latency_ms_mean": lat["mean"],
                    "latency_ms_std": lat["std"],
                    "latency_ms_ci95_lo": lat["ci95_lo"],
                    "latency_ms_ci95_hi": lat["ci95_hi"],
                    "energy_j_ep_mean": ene["mean"],
                    "energy_j_ep_std": ene["std"],
                    "energy_j_ep_ci95_lo": ene["ci95_lo"],
                    "energy_j_ep_ci95_hi": ene["ci95_hi"],
                    "actor_steps_mean_std_ci": _fmt_mean_std_ci(
                        act["mean"], act["std"], act["ci95_lo"], act["ci95_hi"], digits=int(args.digits)
                    ),
                    "critic_steps_mean_std_ci": _fmt_mean_std_ci(
                        cri["mean"], cri["std"], cri["ci95_lo"], cri["ci95_hi"], digits=int(args.digits)
                    ),
                    "latency_ms_mean_std_ci": _fmt_mean_std_ci(
                        lat["mean"], lat["std"], lat["ci95_lo"], lat["ci95_hi"], digits=int(args.digits)
                    ),
                    "energy_j_ep_mean_std_ci": _fmt_mean_std_ci(
                        ene["mean"], ene["std"], ene["ci95_lo"], ene["ci95_hi"], digits=int(args.digits)
                    ),
                }
            )

    summary_csv = out_dir / "tmaze_latency_energy_summary.csv"
    summary_fields = list(summary_rows[0].keys())
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    active_df = [r for r in summary_rows if r["split"] == "active"]
    passive_df = [r for r in summary_rows if r["split"] == "passive"]

    tex_active = _build_latex_table("Active (Primary)", active_df, caption_style=str(args.caption_style))
    tex_passive = _build_latex_table("Passive (Robustness)", passive_df, caption_style=str(args.caption_style))
    tex_active_path = out_dir / "tmaze_latency_energy_active.tex"
    tex_passive_path = out_dir / "tmaze_latency_energy_passive.tex"
    tex_active_path.write_text(tex_active, encoding="utf-8")
    tex_passive_path.write_text(tex_passive, encoding="utf-8")

    print(f"Saved seed metrics CSV: {seed_csv}")
    print(f"Saved summary CSV:      {summary_csv}")
    print(f"Saved active LaTeX:     {tex_active_path}")
    print(f"Saved passive LaTeX:    {tex_passive_path}")


if __name__ == "__main__":
    main()
