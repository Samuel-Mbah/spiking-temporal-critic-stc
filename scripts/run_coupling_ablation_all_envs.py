#!/usr/bin/env python3
"""
Run critic-to-actor coupling ablations for snn_actor_snntiming_critic across:
  - CartPole
  - PO-CartPole
  - T-Maze Passive
  - T-Maze Active

Each environment runs 3 conditions (off / on_detach / on_no_detach) over 5 seeds.
Then per-environment summaries are generated and merged into one cross-env table.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def build_active_tmaze_config(base_cfg_path: Path) -> Path:
    cfg = yaml.safe_load(base_cfg_path.read_text()) or {}
    cfg.setdefault("env", {}).setdefault("kwargs", {})["active"] = True

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_snn_actor_snntiming_critic_tmaze_active.yaml",
        delete=False,
        dir="/tmp",
        encoding="utf-8",
    )
    with tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
    return Path(tmp.name)


def write_combined_tables(summary_paths: dict[str, Path], out_dir: Path) -> None:
    row_order = ["off", "on_detach", "on_no_detach"]
    row_label = {
        "off": "Off (baseline)",
        "on_detach": "On (detached)",
        "on_no_detach": "On (no detach)",
    }

    mean_tbl = pd.DataFrame({"Coupling Condition": [row_label[r] for r in row_order]})
    meanstd_tbl = pd.DataFrame({"Coupling Condition": [row_label[r] for r in row_order]})

    for env_name, summary_csv in summary_paths.items():
        df = pd.read_csv(summary_csv).set_index("variant")
        mean_tbl[env_name] = [float(df.loc[r, "final_eval_reward_mean"]) for r in row_order]
        meanstd_tbl[env_name] = [
            f"{float(df.loc[r, 'final_eval_reward_mean']):.3f} ± {float(df.loc[r, 'final_eval_reward_std']):.3f}"
            for r in row_order
        ]

    out_dir.mkdir(parents=True, exist_ok=True)
    mean_csv = out_dir / "critic_informs_actor_coupling_table_mean.csv"
    meanstd_csv = out_dir / "critic_informs_actor_coupling_table_mean_std.csv"
    latex_path = out_dir / "critic_informs_actor_coupling_table.tex"

    mean_tbl.to_csv(mean_csv, index=False)
    meanstd_tbl.to_csv(meanstd_csv, index=False)

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"\textbf{Coupling Condition} & \textbf{CartPole} & \textbf{PO-CartPole} & \textbf{T-Maze Passive} & \textbf{T-Maze Active} \\",
        r"\hline",
    ]
    for _, row in meanstd_tbl.iterrows():
        lines.append(
            f"{row['Coupling Condition']} & "
            f"{row['CartPole']} & "
            f"{row['PO-CartPole']} & "
            f"{row['T-Maze Passive']} & "
            f"{row['T-Maze Active']} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    latex_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nSaved:")
    print(mean_csv)
    print(meanstd_csv)
    print(latex_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run coupling ablations for snn_actor_snntiming_critic on all environments and summarize."
    )
    parser.add_argument(
        "--python-exec",
        default=sys.executable,
        help="Python interpreter used to run project scripts.",
    )
    parser.add_argument(
        "--seeds",
        default="1,2,3,4,5",
        help="Comma-separated seed list passed to ablation runner.",
    )
    parser.add_argument(
        "--logs-root",
        default="results/logs",
        help="Base logs directory where ablation subfolders are created.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/post_analysis",
        help="Directory for combined output tables.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    args = parser.parse_args()

    py = args.python_exec
    logs_root = Path(args.logs_root)
    out_dir = Path(args.out_dir)

    ablation_script = REPO_ROOT / "post_analysis" / "critic_informs_actor_ablation.py"
    summary_script = REPO_ROOT / "post_analysis" / "summarize_critic_informs_actor_ablation.py"

    cfg_cart = REPO_ROOT / "configs" / "cartpole" / "snn_actor_snntiming_critic.yaml"
    cfg_poc = REPO_ROOT / "configs" / "poc" / "snn_actor_snntiming_critic.yaml"
    cfg_tmaze_passive = REPO_ROOT / "configs" / "tmaze" / "snn_actor_snntiming_critic.yaml"
    cfg_tmaze_active = build_active_tmaze_config(cfg_tmaze_passive)

    run_specs = [
        ("CartPole", cfg_cart, logs_root / "critic_informs_actor_ablation_cartpole"),
        ("PO-CartPole", cfg_poc, logs_root / "critic_informs_actor_ablation_poc"),
        ("T-Maze Passive", cfg_tmaze_passive, logs_root / "critic_informs_actor_ablation_tmaze_passive"),
        ("T-Maze Active", cfg_tmaze_active, logs_root / "critic_informs_actor_ablation_tmaze_active"),
    ]

    # 1) Launch all ablation runs.
    for _, cfg, out_root in run_specs:
        run(
            [
                py,
                str(ablation_script),
                "--config",
                str(cfg),
                "--seeds",
                args.seeds,
                "--output_root",
                str(out_root),
            ],
            dry_run=args.dry_run,
        )

    # 2) Summarize each environment.
    summary_paths: dict[str, Path] = {}
    for env_name, _, out_root in run_specs:
        summary_csv = out_dir / f"critic_informs_actor_ablation_summary_{env_name.lower().replace('-', '_').replace(' ', '_')}.csv"
        summary_paths[env_name] = summary_csv
        run(
            [
                py,
                str(summary_script),
                "--ablation_root",
                str(out_root),
                "--out_csv",
                str(summary_csv),
            ],
            dry_run=args.dry_run,
        )

    # 3) Build combined tables.
    if not args.dry_run:
        write_combined_tables(summary_paths, out_dir)


if __name__ == "__main__":
    main()
