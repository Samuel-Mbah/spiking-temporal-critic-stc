import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _series_from_metrics(raw: dict, key: str) -> list[float]:
    events = raw.get(key, [])
    values = []
    for e in events:
        try:
            values.append(float(e.get("value")))
        except Exception:
            continue
    return values


def _safe_last(values: list[float]) -> float:
    return float(values[-1]) if values else float("nan")


def _safe_max(values: list[float]) -> float:
    return float(np.max(values)) if values else float("nan")


def _load_seed_metrics(seed_dir: Path) -> dict[str, float]:
    raw_path = seed_dir / "metrics_raw.json"
    if not raw_path.exists():
        return {}

    with open(raw_path, "r") as f:
        raw = json.load(f)

    eval_reward = _series_from_metrics(raw, "eval/reward")
    eval_success = _series_from_metrics(raw, "eval/success_rate")
    value_loss = _series_from_metrics(raw, "train/value_loss")
    approx_kl = _series_from_metrics(raw, "train/approx_kl")
    entropy = _series_from_metrics(raw, "train/entropy")

    return {
        "final_eval_reward": _safe_last(eval_reward),
        "best_eval_reward": _safe_max(eval_reward),
        "final_eval_success_rate": _safe_last(eval_success),
        "final_value_loss": _safe_last(value_loss),
        "final_approx_kl": _safe_last(approx_kl),
        "final_entropy": _safe_last(entropy),
    }


def _agg(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def _variant_row(variant_dir: Path, variant_name: str) -> dict[str, float | str]:
    seed_dirs = sorted([p for p in variant_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])
    per_seed = [_load_seed_metrics(sd) for sd in seed_dirs]
    per_seed = [m for m in per_seed if m]

    row: dict[str, float | str] = {"variant": variant_name, "num_seeds": len(per_seed)}
    metric_keys = [
        "final_eval_reward",
        "best_eval_reward",
        "final_eval_success_rate",
        "final_value_loss",
        "final_approx_kl",
        "final_entropy",
    ]
    for k in metric_keys:
        vals = [m.get(k, float("nan")) for m in per_seed]
        m, s = _agg(vals)
        row[f"{k}_mean"] = m
        row[f"{k}_std"] = s
    return row


def _write_csv(rows: list[dict[str, float | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize critic_informs_actor ablation results from metrics_raw.json"
    )
    parser.add_argument(
        "--ablation_root",
        type=str,
        default="results/logs/critic_informs_actor_ablation",
        help="Root folder created by experiments/critic_informs_actor_ablation.py",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="results/post_analysis/critic_informs_actor_ablation_summary.csv",
        help="Output CSV summary path",
    )
    args = parser.parse_args()

    root = Path(args.ablation_root)
    if not root.exists():
        raise FileNotFoundError(f"Ablation root not found: {root}")

    variants = [("off", root / "off"), ("on_detach", root / "on_detach"), ("on_no_detach", root / "on_no_detach")]
    rows = []
    for name, path in variants:
        if not path.exists():
            continue
        rows.append(_variant_row(path, name))

    if not rows:
        raise RuntimeError(f"No variant folders found under: {root}")

    out_csv = Path(args.out_csv)
    _write_csv(rows, out_csv)
    print(f"Saved ablation summary: {out_csv}")


if __name__ == "__main__":
    main()
