#!/usr/bin/env python3
"""
Paired multi-seed statistical comparison between two experiment folders.

Example:
  python3 post_analysis/paired_seed_stats.py \
    --model-a-dir results/logs/poc_ann_baseline_no_fs \
    --model-b-dir results/logs/poc_snn_actor_snntiming_critic_no_fs \
    --model-a-name "ANN-PPO" \
    --model-b-name "Adaptive SNN" \
    --reward-threshold 475.0 \
    --out-csv results/post_analysis/paired_stats.csv \
    --out-md results/post_analysis/paired_stats.md
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon


MetricFrame = dict[str, np.ndarray]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    display_name: str
    extractor: Callable[[MetricFrame, argparse.Namespace], float]
    higher_is_better: bool


def _read_metric_csv(path: Path) -> MetricFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    cols: dict[str, list[float]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}
        for k in reader.fieldnames:
            cols[k] = []

        for row in reader:
            for k in cols:
                raw = row.get(k, "")
                try:
                    v = float(raw) if raw is not None and raw != "" else float("nan")
                except Exception:
                    v = float("nan")
                cols[k].append(v)

    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def _first_existing_column(frame: MetricFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in frame:
            return c
    return None


def _last_finite(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    mask = np.isfinite(values)
    if not np.any(mask):
        return float("nan")
    return float(values[mask][-1])


def _tail_mean(values: np.ndarray, tail_window: int) -> float:
    if values.size == 0:
        return float("nan")
    mask = np.isfinite(values)
    if not np.any(mask):
        return float("nan")
    clean = values[mask]
    w = max(1, int(tail_window))
    return float(np.mean(clean[-w:]))


def _extract_cumulative_reward(frame: MetricFrame, _: argparse.Namespace) -> float:
    col = _first_existing_column(frame, ["total_cumulative_train_reward"])
    if col is not None:
        return _last_finite(frame[col])
    test_col = _first_existing_column(frame, ["test_reward", "eval_reward", "post_conversion_ft/eval_reward"])
    if test_col is None:
        return float("nan")
    vals = frame[test_col]
    vals = vals[np.isfinite(vals)]
    return float(np.sum(vals)) if vals.size else float("nan")


def _extract_final_eval_reward(frame: MetricFrame, _: argparse.Namespace) -> float:
    col = _first_existing_column(frame, ["test_reward", "eval_reward", "post_conversion_ft/eval_reward"])
    if col is None:
        return float("nan")
    return _last_finite(frame[col])


def _extract_convergence_update(frame: MetricFrame, args: argparse.Namespace) -> float:
    update_col = _first_existing_column(frame, ["update"])
    if update_col is not None:
        update_values = frame[update_col]
    else:
        n_rows = max((len(v) for v in frame.values()), default=0)
        update_values = np.arange(1, n_rows + 1, dtype=np.float64)

    success_col = _first_existing_column(frame, ["eval/success_rate", "post_conversion_ft/success_rate"])
    if success_col is not None:
        success = frame[success_col]
        n = min(len(update_values), len(success))
        idx = np.where(np.isfinite(success[:n]) & (success[:n] >= float(args.success_threshold)))[0]
        if idx.size:
            return float(update_values[idx[0]])

    reward_col = _first_existing_column(frame, ["test_reward", "eval_reward", "post_conversion_ft/eval_reward"])
    if reward_col is not None:
        rewards = frame[reward_col]
        n = min(len(update_values), len(rewards))
        idx = np.where(np.isfinite(rewards[:n]) & (rewards[:n] >= float(args.reward_threshold)))[0]
        if idx.size:
            return float(update_values[idx[0]])

    return float("nan")


def _extract_latency(frame: MetricFrame, args: argparse.Namespace) -> float:
    col = _first_existing_column(
        frame,
        [
            "post_conversion/mean_latency",
            "latency_mean_ms",
            "latency/eval_wall_clock_ms",
            "latency/actor_spike_timing_steps",
            "latency/critic_spike_timing_steps",
            "latency/critic_eval_spike_timing_steps",
            "latency/eval_spike_timing_steps",
        ],
    )
    if col is None:
        return float("nan")
    return _tail_mean(frame[col], args.tail_window)


def _extract_spikes(frame: MetricFrame, args: argparse.Namespace) -> float:
    col = _first_existing_column(frame, ["spikes/per_step", "post_conversion/total_spikes", "spike_count_total", "spikes/total"])
    if col is None:
        return float("nan")
    return _tail_mean(frame[col], args.tail_window)


def _extract_energy(frame: MetricFrame, args: argparse.Namespace) -> float:
    col = _first_existing_column(frame, ["post_conversion/inference_energy", "inference_energy"])
    if col is None:
        return float("nan")
    return _tail_mean(frame[col], args.tail_window)


METRICS: list[MetricSpec] = [
    MetricSpec("cumulative_reward", "Cumulative Reward", _extract_cumulative_reward, True),
    MetricSpec("final_eval_reward", "Final Eval Reward", _extract_final_eval_reward, True),
    MetricSpec("convergence_update", "Convergence Update (lower better)", _extract_convergence_update, False),
    MetricSpec("latency", "Inference Latency", _extract_latency, False),
    MetricSpec("spikes", "Spike Count", _extract_spikes, False),
    MetricSpec("energy", "Inference Energy", _extract_energy, False),
]


def _seed_map(exp_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in sorted(exp_dir.glob("seed_*")):
        if not p.is_dir():
            continue
        try:
            seed = int(p.name.split("_", 1)[1])
        except Exception:
            continue
        csv_path = p / "per_episode_metrics.csv"
        if csv_path.exists():
            out[seed] = csv_path
    return out


def _mean_std(arr: np.ndarray) -> tuple[float, float]:
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1 if arr.size > 1 else 0))


def _format_mean_std(arr: np.ndarray) -> str:
    m, s = _mean_std(arr)
    if not np.isfinite(m):
        return "nan"
    return f"{m:.6g} ± {s:.6g}"


def _paired_test(diffs: np.ndarray, normality_alpha: float) -> tuple[str, float, float, float]:
    diffs = diffs[np.isfinite(diffs)]
    n = int(diffs.size)
    if n < 2:
        return "insufficient_n", float("nan"), float("nan"), float("nan")

    # Degenerate case: identical paired values -> no detectable difference.
    # Return a valid non-significant result instead of propagating NaNs from t-test.
    if np.allclose(diffs, 0.0):
        return "paired_ttest", 0.0, 1.0, float("nan")

    normality_p = float("nan")
    use_ttest = False
    if n >= 3:
        try:
            _, normality_p = shapiro(diffs)
            use_ttest = bool(normality_p >= normality_alpha)
        except Exception:
            use_ttest = False

    if use_ttest:
        t_stat, p_val = ttest_rel(diffs, np.zeros_like(diffs), alternative="two-sided")
        if not (np.isfinite(t_stat) and np.isfinite(p_val)):
            return "paired_ttest", 0.0, 1.0, normality_p
        return "paired_ttest", float(t_stat), float(p_val), normality_p

    try:
        w_stat, p_val = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox", correction=False, mode="auto")
        return "wilcoxon", float(w_stat), float(p_val), normality_p
    except ValueError:
        return "wilcoxon", float("nan"), 1.0, normality_p


def _run_comparison(args: argparse.Namespace) -> list[dict[str, object]]:
    a_map = _seed_map(Path(args.model_a_dir))
    b_map = _seed_map(Path(args.model_b_dir))

    seeds = sorted(set(a_map).intersection(b_map))
    if args.seeds:
        requested = {int(s.strip()) for s in args.seeds.split(",") if s.strip()}
        seeds = [s for s in seeds if s in requested]
    if len(seeds) == 0:
        raise RuntimeError("No matched seeds found between the two experiment directories.")

    selected_metric_keys = set(args.metrics.split(",")) if args.metrics else {m.key for m in METRICS}
    metric_specs = [m for m in METRICS if m.key in selected_metric_keys]
    if not metric_specs:
        raise RuntimeError("No valid metrics selected. Check --metrics.")

    a_frames = {s: _read_metric_csv(a_map[s]) for s in seeds}
    b_frames = {s: _read_metric_csv(b_map[s]) for s in seeds}

    rows: list[dict[str, object]] = []
    for metric in metric_specs:
        a_vals = []
        b_vals = []
        used_seeds = []
        for seed in seeds:
            a_val = metric.extractor(a_frames[seed], args)
            b_val = metric.extractor(b_frames[seed], args)
            if np.isfinite(a_val) and np.isfinite(b_val):
                a_vals.append(float(a_val))
                b_vals.append(float(b_val))
                used_seeds.append(seed)

        a_arr = np.asarray(a_vals, dtype=np.float64)
        b_arr = np.asarray(b_vals, dtype=np.float64)
        diffs = b_arr - a_arr

        test_name, stat, p_val, norm_p = _paired_test(diffs, normality_alpha=float(args.normality_alpha))
        sig = bool(np.isfinite(p_val) and (p_val < float(args.alpha)))

        a_mean, a_std = _mean_std(a_arr)
        b_mean, b_std = _mean_std(b_arr)

        row = {
            "metric_key": metric.key,
            "metric": metric.display_name,
            "n": int(len(used_seeds)),
            "seeds": ",".join(str(s) for s in used_seeds),
            f"{args.model_a_name}_mean": a_mean,
            f"{args.model_a_name}_std": a_std,
            f"{args.model_b_name}_mean": b_mean,
            f"{args.model_b_name}_std": b_std,
            f"{args.model_a_name}_mean_std": _format_mean_std(a_arr),
            f"{args.model_b_name}_mean_std": _format_mean_std(b_arr),
            "delta_mean_b_minus_a": float(np.mean(diffs)) if diffs.size else float("nan"),
            "test": test_name,
            "test_statistic": stat,
            "normality_p": norm_p,
            "p_value": p_val,
            "significant_p_lt_alpha": sig,
            "alpha": float(args.alpha),
            "normality_alpha": float(args.normality_alpha),
            "higher_is_better": metric.higher_is_better,
        }
        rows.append(row)

    return rows


def _write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, object]], out_path: Path, args: argparse.Namespace) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Paired Multi-Seed Statistical Comparison")
    lines.append("")
    lines.append(f"- Model A: `{args.model_a_name}` ({args.model_a_dir})")
    lines.append(f"- Model B: `{args.model_b_name}` ({args.model_b_dir})")
    lines.append(f"- Significance threshold: `p < {args.alpha}`")
    lines.append(
        f"- Test rule: Shapiro on paired differences (`p >= {args.normality_alpha}` -> paired two-sided t-test, else Wilcoxon signed-rank)"
    )
    lines.append("")
    lines.append("| Metric | n | A (mean ± std) | B (mean ± std) | Δ(B-A) | Test | p-value | Significant |")
    lines.append("|---|---:|---:|---:|---:|---|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['metric']} | {int(r['n'])} | {r[f'{args.model_a_name}_mean_std']} | "
            f"{r[f'{args.model_b_name}_mean_std']} | {float(r['delta_mean_b_minus_a']):.6g} | "
            f"{r['test']} | {float(r['p_value']):.6g} | {'yes' if bool(r['significant_p_lt_alpha']) else 'no'} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired seed statistical testing between two experiment folders.")
    parser.add_argument("--model-a-dir", type=str, required=True, help="Experiment directory containing seed_* folders for model A.")
    parser.add_argument("--model-b-dir", type=str, required=True, help="Experiment directory containing seed_* folders for model B.")
    parser.add_argument("--model-a-name", type=str, default="model_a", help="Display name for model A.")
    parser.add_argument("--model-b-name", type=str, default="model_b", help="Display name for model B.")
    parser.add_argument(
        "--metrics",
        type=str,
        default="cumulative_reward,final_eval_reward,convergence_update,latency,spikes,energy",
        help="Comma-separated metric keys.",
    )
    parser.add_argument("--seeds", type=str, default="", help="Optional comma-separated seed IDs to include.")
    parser.add_argument("--reward-threshold", type=float, default=475.0, help="Reward threshold used for convergence metric fallback.")
    parser.add_argument("--success-threshold", type=float, default=95.0, help="Success-rate threshold (%) for convergence metric.")
    parser.add_argument("--tail-window", type=int, default=10, help="Tail window size for latency/spikes/energy averages.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold.")
    parser.add_argument("--normality-alpha", type=float, default=0.05, help="Normality test threshold.")
    parser.add_argument(
        "--out-csv",
        type=str,
        default="results/post_analysis/paired_seed_stats.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="results/post_analysis/paired_seed_stats.md",
        help="Output markdown summary path.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    rows = _run_comparison(args)

    out_csv = Path(args.out_csv)
    _write_csv(rows, out_csv)
    _write_markdown(rows, Path(args.out_md), args)

    print(f"Saved CSV: {out_csv}")
    print(f"Saved Markdown: {args.out_md}")
    print("")
    for r in rows:
        p = float(r["p_value"])
        p_txt = "nan" if not math.isfinite(p) else f"{p:.6g}"
        print(
            f"{r['metric']}: "
            f"{r[f'{args.model_a_name}_mean_std']} vs {r[f'{args.model_b_name}_mean_std']} | "
            f"test={r['test']} p={p_txt} sig={'yes' if bool(r['significant_p_lt_alpha']) else 'no'}"
        )


if __name__ == "__main__":
    main()
