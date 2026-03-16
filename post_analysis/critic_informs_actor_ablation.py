import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

MODE_TO_SCRIPT = {
    "ann": "ann_baseline.py",
    "snn_actor_ann_critic": "snn_actor_ann_critic.py",
    "snn_actor_snn_critic": "snn_actor_snntiming_critic.py",
    "snn_actor_snn_timing_critic": "snn_actor_snntiming_critic.py",
}


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p.resolve()
    candidate = (REPO_ROOT / path).resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Path not found: {path}")


def _parse_seeds(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _build_variants(base_config: dict) -> list[tuple[str, dict]]:
    variants = []

    off_cfg = copy.deepcopy(base_config)
    off_cfg.setdefault("model", {})
    off_cfg["model"]["critic_informs_actor"] = False
    off_cfg["model"]["detach_critic_for_actor"] = True
    off_cfg["model"]["normalize_critic_for_actor"] = True
    off_cfg["model"]["critic_actor_value_clip"] = 5.0
    variants.append(("off", off_cfg))

    on_detach_cfg = copy.deepcopy(base_config)
    on_detach_cfg.setdefault("model", {})
    on_detach_cfg["model"]["critic_informs_actor"] = True
    on_detach_cfg["model"]["detach_critic_for_actor"] = True
    on_detach_cfg["model"]["normalize_critic_for_actor"] = True
    on_detach_cfg["model"]["critic_actor_value_clip"] = 5.0
    variants.append(("on_detach", on_detach_cfg))

    on_nodetach_cfg = copy.deepcopy(base_config)
    on_nodetach_cfg.setdefault("model", {})
    on_nodetach_cfg["model"]["critic_informs_actor"] = True
    on_nodetach_cfg["model"]["detach_critic_for_actor"] = False
    on_nodetach_cfg["model"]["normalize_critic_for_actor"] = True
    on_nodetach_cfg["model"]["critic_actor_value_clip"] = 5.0
    variants.append(("on_no_detach", on_nodetach_cfg))

    return variants


def _pick_experiment_script(config: dict, explicit_script: str | None) -> str:
    if explicit_script:
        return explicit_script
    mode = str(config.get("model", {}).get("mode", "")).strip().lower()
    if mode in MODE_TO_SCRIPT:
        return MODE_TO_SCRIPT[mode]
    raise ValueError(
        f"Unsupported model.mode='{mode}'. Pass --script explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run critic_informs_actor ablation: off vs on_detach vs on_no_detach"
    )
    parser.add_argument("--config", type=str, required=True, help="Base YAML config path")
    parser.add_argument(
        "--seeds",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated seeds",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results/logs/critic_informs_actor_ablation",
        help="Root directory for per-variant/per-seed logs",
    )
    parser.add_argument(
        "--script",
        type=str,
        default=None,
        help="Optional experiment script override (e.g., snn_actor_snntiming_critic.py)",
    )
    parser.add_argument(
        "--python_exec",
        type=str,
        default=sys.executable,
        help="Python executable used to run experiment scripts",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    base_config = _load_yaml(config_path)
    seeds = _parse_seeds(args.seeds)
    script_name = _pick_experiment_script(base_config, args.script)
    script_path = REPO_ROOT / "experiments" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Experiment script not found: {script_path}")

    output_root = _resolve_path(args.output_root) if Path(args.output_root).exists() else (REPO_ROOT / args.output_root).resolve()
    generated_cfg_root = output_root / "generated_configs"
    variants = _build_variants(base_config)

    for variant_name, variant_cfg in variants:
        variant_cfg_path = generated_cfg_root / f"{variant_name}.yaml"
        _write_yaml(variant_cfg_path, variant_cfg)

        for seed in seeds:
            run_name = f"{variant_name}_seed_{seed}"
            log_dir = output_root / variant_name / f"seed_{seed}"
            cmd = [
                args.python_exec,
                str(script_path),
                "--config",
                str(variant_cfg_path),
                "--seed",
                str(seed),
                "--run-name",
                run_name,
                "--log-dir",
                str(log_dir),
            ]

            print(" ".join(cmd))
            if args.dry_run:
                continue
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
