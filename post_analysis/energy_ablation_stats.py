#!/usr/bin/env python3
"""
Paired statistical analysis for energy ablation outputs.

Input CSV format is expected to come from:
  experiments/ablations.py --run-energy-benchmark
and contain columns including:
  - model_name
  - train_seed
  - metric columns like joules_per_episode, joules_per_1k_steps, raw_joules_per_1k_steps

Example:
  python3 post_analysis/energy_ablation_stats.py \
    --csv results/ablations/tmaze_eval_only_active_energy/energy_ablation_results.csv \
    --metric-col joules_per_1k_steps \
    --model-a ann_baseline \
    --model-b snn_timing_critic \
    --out-csv results/post_analysis/energy_ablation_stats.csv \
    --out-md results/post_analysis/energy_ablation_stats.md
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from scipy.stats import shapiro, ttest_rel, t, wilcoxon
except Exception:
    shapiro = None
    ttest_rel = None
    wilcoxon = None
    t = None


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float(v: str | None) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def _mean_std(arr: np.ndarray) -> tuple[float, float]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1 if arr.size > 1 else 0))


def _mean_ci95(arr: np.ndarray) -> tuple[float, float, float]:
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if n == 1:
        return mean, mean, mean
    std = float(np.std(arr, ddof=1))
    se = std / math.sqrt(n)
    if t is not None:
        tcrit = float(t.ppf(0.975, df=n - 1))
    else:
        tcrit = 1.96
    half = tcrit * se
    return mean, mean - half, mean + half


def _paired_test(diffs: np.ndarray, normality_alpha: float) -> tuple[str, float, float, float]:
    diffs = diffs[np.isfinite(diffs)]
    n = int(diffs.size)
    if n < 2:
        return "insufficient_n", float("nan"), float("nan"), float("nan")

    normality_p = float("nan")
    use_ttest = False
    if shapiro is not None and n >= 3:
        try:
            _, normality_p = shapiro(diffs)
            use_ttest = bool(normality_p >= normality_alpha)
        except Exception:
            use_ttest = False

    if use_ttest and ttest_rel is not None:
        stat, p_val = ttest_rel(diffs, np.zeros_like(diffs), alternative="two-sided")
        return "paired_ttest", float(stat), float(p_val), normality_p

    if np.allclose(diffs, 0.0):
        return "wilcoxon", 0.0, 1.0, normality_p

    if wilcoxon is not None:
        try:
            stat, p_val = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox", correction=False, mode="auto")
            return "wilcoxon", float(stat), float(p_val), normality_p
        except Exception:
            pass

    return "no_test_available", float("nan"), float("nan"), normality_p


def _format_mean_std(arr: np.ndarray) -> str:
    m, s = _mean_std(arr)
    if not np.isfinite(m):
        return "nan"
    return f"{m:.6g} ± {s:.6g}"


def _pair_by_seed(
    rows: Iterable[dict[str, str]],
    model_a: str,
    model_b: str,
    metric_col: str,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    per_seed_a: dict[int, list[float]] = {}
    per_seed_b: dict[int, list[float]] = {}

    for r in rows:
        name = str(r.get("model_name", "")).strip()
        seed_raw = r.get("train_seed", "")
        try:
            seed = int(seed_raw)
        except Exception:
            continue
        v = _to_float(r.get(metric_col))
        if not np.isfinite(v):
            continue
        if name == model_a:
            per_seed_a.setdefault(seed, []).append(v)
        elif name == model_b:
            per_seed_b.setdefault(seed, []).append(v)

    seeds = sorted(set(per_seed_a).intersection(per_seed_b))
    a_vals = []
    b_vals = []
    used = []
    for s in seeds:
        a_arr = np.asarray(per_seed_a[s], dtype=float)
        b_arr = np.asarray(per_seed_b[s], dtype=float)
        if a_arr.size == 0 or b_arr.size == 0:
            continue
        a_vals.append(float(np.mean(a_arr)))
        b_vals.append(float(np.mean(b_arr)))
        used.append(s)

    return np.asarray(a_vals, dtype=float), np.asarray(b_vals, dtype=float), used


def _write_csv_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _write_md(path: Path, row: dict[str, object], model_a: str, model_b: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Energy Ablation Paired Statistics",
        "",
        f"- Model A: `{model_a}`",
        f"- Model B: `{model_b}`",
        "",
        "| Metric | n | A (mean ± std) | B (mean ± std) | A mean [95% CI] | B mean [95% CI] | Δ(B-A) mean | Test | p-value |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
        (
            f"| {row['metric_col']} | {row['n']} | {row['a_mean_std']} | {row['b_mean_std']} | "
            f"{float(row['a_mean']):.6g} [{float(row['a_ci95_lo']):.6g}, {float(row['a_ci95_hi']):.6g}] | "
            f"{float(row['b_mean']):.6g} [{float(row['b_ci95_lo']):.6g}, {float(row['b_ci95_hi']):.6g}] | "
            f"{float(row['delta_mean_b_minus_a']):.6g} | {row['test']} | {float(row['p_value']):.6g} |"
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser("Paired stats for energy ablation CSV")
    p.add_argument("--csv", type=str, required=True, help="Path to energy_ablation_results.csv")
    p.add_argument("--metric-col", type=str, default="joules_per_1k_steps", help="Metric column to analyze.")
    p.add_argument("--model-a", type=str, default="ann_baseline")
    p.add_argument("--model-b", type=str, default="snn_timing_critic")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance threshold.")
    p.add_argument("--normality-alpha", type=float, default=0.05, help="Shapiro threshold.")
    p.add_argument("--out-csv", type=str, default=None)
    p.add_argument("--out-md", type=str, default=None)
    args = p.parse_args()

    rows = _read_rows(Path(args.csv))
    a, b, seeds = _pair_by_seed(rows, args.model_a, args.model_b, args.metric_col)
    if a.size == 0 or b.size == 0:
        raise RuntimeError("No paired seed values found. Check model names/metric column/train_seed values.")

    diffs = b - a
    test_name, stat, p_val, normality_p = _paired_test(diffs, args.normality_alpha)
    a_mean, a_lo, a_hi = _mean_ci95(a)
    b_mean, b_lo, b_hi = _mean_ci95(b)

    out_row: dict[str, object] = {
        "metric_col": args.metric_col,
        "model_a": args.model_a,
        "model_b": args.model_b,
        "n": int(len(seeds)),
        "paired_seeds": ",".join(str(s) for s in seeds),
        "a_mean_std": _format_mean_std(a),
        "b_mean_std": _format_mean_std(b),
        "a_mean": a_mean,
        "a_ci95_lo": a_lo,
        "a_ci95_hi": a_hi,
        "b_mean": b_mean,
        "b_ci95_lo": b_lo,
        "b_ci95_hi": b_hi,
        "delta_mean_b_minus_a": float(np.mean(diffs)),
        "delta_std_b_minus_a": float(np.std(diffs, ddof=1 if diffs.size > 1 else 0)),
        "test": test_name,
        "test_stat": stat,
        "p_value": p_val,
        "normality_p": normality_p,
        "significant": bool(np.isfinite(p_val) and p_val < args.alpha),
    }

    print("\n=== Energy Ablation Paired Stats ===")
    print(f"metric: {args.metric_col}")
    print(f"n (paired seeds): {out_row['n']} [{out_row['paired_seeds']}]")
    print(f"A ({args.model_a}): {out_row['a_mean_std']} ; 95% CI [{a_lo:.6g}, {a_hi:.6g}]")
    print(f"B ({args.model_b}): {out_row['b_mean_std']} ; 95% CI [{b_lo:.6g}, {b_hi:.6g}]")
    print(f"Δ(B-A): {out_row['delta_mean_b_minus_a']:.6g} ± {out_row['delta_std_b_minus_a']:.6g}")
    print(f"test={test_name}, p={p_val:.6g}, significant={out_row['significant']}")

    if args.out_csv:
        _write_csv_row(Path(args.out_csv), out_row)
        print(f"saved CSV: {args.out_csv}")
    if args.out_md:
        _write_md(Path(args.out_md), out_row, args.model_a, args.model_b)
        print(f"saved MD:  {args.out_md}")


if __name__ == "__main__":
    main()

