#!/usr/bin/env python3
"""
Generate a LaTeX table from latency_summary.csv.

Default input:
  results/post_analysis/latency_actor_critic_multiseed_gpu/latency_summary.csv

Default output:
  same folder / latency_table.tex
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


TASK_ORDER = ["cartpole", "partial_cartpole", "tmaze_active", "tmaze_passive"]
TASK_LABEL = {
    "cartpole": "CartPole",
    "partial_cartpole": "Partial CartPole",
    "tmaze_active": "TMaze Active",
    "tmaze_passive": "TMaze Passive",
}
AGENT_ORDER = ["ann_baseline", "snn_timing_critic"]
AGENT_LABEL = {
    "ann_baseline": "ANN-Baseline",
    "snn_timing_critic": "SNN-Timing Critic",
}


def _fmt_pm(mean: float, std: float, decimals: int) -> str:
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float(row: Dict[str, str], key: str) -> float:
    val = row.get(key, "")
    try:
        return float(val)
    except Exception:
        return float("nan")


def build_table(rows: List[Dict[str, str]], decimals: int = 4) -> str:
    index = {(r["task"], r["agent"]): r for r in rows}
    lines: List[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Actor-vs-critic latency benchmark across tasks (mean $\\pm$ std over seeds).}")
    lines.append("\\label{tab:actor_critic_latency_multitask}")
    lines.append("\\begin{tabular}{llccc}")
    lines.append("\\toprule")
    lines.append("Task & Agent & Actor Latency (ms) & Critic Latency (ms) & Critic spike-timing-steps \\\\")
    lines.append("\\midrule")

    for t_idx, task in enumerate(TASK_ORDER):
        for agent in AGENT_ORDER:
            row = index.get((task, agent))
            if row is None:
                continue
            a_m = _to_float(row, "actor_ms_mean")
            a_s = _to_float(row, "actor_ms_std")
            c_m = _to_float(row, "critic_ms_mean")
            c_s = _to_float(row, "critic_ms_std")
            s_m = _to_float(row, "critic_spike_timing_steps_mean")
            s_s = _to_float(row, "critic_spike_timing_steps_std")

            spike_str = "--"
            if agent == "snn_timing_critic":
                spike_str = _fmt_pm(s_m, s_s, decimals)

            lines.append(
                f"{TASK_LABEL.get(task, task)} & {AGENT_LABEL.get(agent, agent)} & "
                f"{_fmt_pm(a_m, a_s, decimals)} & {_fmt_pm(c_m, c_s, decimals)} & {spike_str} \\\\"
            )
        if t_idx < len(TASK_ORDER) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser("Generate LaTeX table from latency summary CSV")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="results/post_analysis/latency_actor_critic_multiseed_gpu/latency_summary.csv",
    )
    parser.add_argument(
        "--output-tex",
        type=str,
        default="",
        help="Optional output .tex path (default: sibling latency_table.tex)",
    )
    parser.add_argument("--decimals", type=int, default=4)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_tex = Path(args.output_tex) if args.output_tex else input_csv.with_name("latency_table.tex")

    rows = _read_rows(input_csv)
    latex = build_table(rows, decimals=max(0, int(args.decimals)))
    output_tex.write_text(latex, encoding="utf-8")
    print(f"Saved LaTeX table: {output_tex}")
    print("")
    print(latex)


if __name__ == "__main__":
    main()

