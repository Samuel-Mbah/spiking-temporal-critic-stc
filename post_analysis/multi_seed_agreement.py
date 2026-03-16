import argparse
import csv
import os
import sys
from typing import Iterable

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from post_analysis.video_compare import (
    _build_env_kwargs,
    _infer_ann_dims_from_state,
    _infer_snn_dims_from_state,
    _load_actor_state,
    _load_yaml,
    _merge_dims,
    _resolve_agent_types,
    _resolve_dims_from_config,
    _resolve_frame_stack_from_input_dim,
    _resolve_path,
)
from src.training.agents import ActorType, CriticType, make_agent
from src.tools.video_comparison import collect_trajectory
from src.utils.checkpoint import load_checkpoint


def _parse_seeds(seeds_arg: str | None, seed_start: int, seed_end: int) -> list[int]:
    if seeds_arg:
        return [int(x.strip()) for x in seeds_arg.split(",") if x.strip()]
    return list(range(int(seed_start), int(seed_end) + 1))


def _build_agents_and_env_kwargs(args) -> tuple[str, object, object, dict, dict]:
    config_path = _resolve_path(args.config)
    ann_config_path = _resolve_path(args.ann_config) if args.ann_config else config_path
    snn_config_path = _resolve_path(args.snn_config) if args.snn_config else config_path

    base_config = _load_yaml(config_path)
    ann_config = _load_yaml(ann_config_path)
    snn_config = _load_yaml(snn_config_path)

    env_cfg = base_config.get("env", {})
    if not env_cfg:
        env_cfg = ann_config.get("env", {}) or snn_config.get("env", {})
    env_id = args.env_id or env_cfg.get("id", "tmaze-v0")

    ann_actor_state = _load_actor_state(args.ann_ckpt)
    snn_actor_state = _load_actor_state(args.snn_ckpt)

    ann_dims = _merge_dims(
        _resolve_dims_from_config(ann_config, env_id),
        _infer_ann_dims_from_state(ann_actor_state),
        label="ANN",
    )
    snn_dims = _merge_dims(
        _resolve_dims_from_config(snn_config, env_id),
        _infer_snn_dims_from_state(snn_actor_state),
        label="SNN",
    )

    ann_actor_type, ann_critic_type = _resolve_agent_types(ann_config, ActorType.ANN, CriticType.ANN)
    snn_actor_type, snn_critic_type = _resolve_agent_types(snn_config, ActorType.SNN_SPIKE, CriticType.SNN_TIMING)

    snn_cfg = snn_config.get("snn", {})
    snn_T = snn_cfg.get("T") or snn_cfg.get("actor_T") or 32
    snn_make_kwargs = {k: v for k, v in snn_cfg.items() if k != "T"}

    ann_agent = make_agent(
        actor_type=ann_actor_type,
        critic_type=ann_critic_type,
        in_dim=ann_dims[0],
        act_dim=ann_dims[1],
        hidden_dim=ann_dims[2],
        critic_informs_actor=bool(ann_config.get("model", {}).get("critic_informs_actor", False)),
        detach_critic_for_actor=bool(ann_config.get("model", {}).get("detach_critic_for_actor", True)),
        normalize_critic_for_actor=bool(ann_config.get("model", {}).get("normalize_critic_for_actor", True)),
        critic_actor_value_clip=ann_config.get("model", {}).get("critic_actor_value_clip", 5.0),
        critic_actor_norm_momentum=float(ann_config.get("model", {}).get("critic_actor_norm_momentum", 0.01)),
    )
    snn_agent = make_agent(
        actor_type=snn_actor_type,
        critic_type=snn_critic_type,
        in_dim=snn_dims[0],
        act_dim=snn_dims[1],
        hidden_dim=snn_dims[2],
        T=snn_T,
        critic_informs_actor=bool(snn_config.get("model", {}).get("critic_informs_actor", False)),
        detach_critic_for_actor=bool(snn_config.get("model", {}).get("detach_critic_for_actor", True)),
        normalize_critic_for_actor=bool(snn_config.get("model", {}).get("normalize_critic_for_actor", True)),
        critic_actor_value_clip=snn_config.get("model", {}).get("critic_actor_value_clip", 5.0),
        critic_actor_norm_momentum=float(snn_config.get("model", {}).get("critic_actor_norm_momentum", 0.01)),
        **snn_make_kwargs,
    )

    load_checkpoint(args.ann_ckpt, agent=ann_agent, map_location="cpu")
    load_checkpoint(args.snn_ckpt, agent=snn_agent, map_location="cpu")

    ann_env_cfg = ann_config.get("env", {}) or env_cfg
    snn_env_cfg = snn_config.get("env", {}) or env_cfg
    ann_frame_stack = _resolve_frame_stack_from_input_dim(
        env_id=env_id,
        env_cfg=ann_env_cfg,
        input_dim=ann_dims[0],
        config_frame_stack=ann_env_cfg.get("frame_stack"),
        user_override=args.frame_stack,
        label="ANN",
    )
    snn_frame_stack = _resolve_frame_stack_from_input_dim(
        env_id=env_id,
        env_cfg=snn_env_cfg,
        input_dim=snn_dims[0],
        config_frame_stack=snn_env_cfg.get("frame_stack"),
        user_override=args.frame_stack,
        label="SNN",
    )
    ann_env_kwargs = _build_env_kwargs(ann_env_cfg, ann_frame_stack)
    snn_env_kwargs = _build_env_kwargs(snn_env_cfg, snn_frame_stack)
    return env_id, ann_agent, snn_agent, ann_env_kwargs, snn_env_kwargs


def _row_for_seed(seed: int, ann_data: dict, snn_data: dict) -> dict[str, float]:
    ann_actions = ann_data.get("actions", [])
    snn_actions = snn_data.get("actions", [])
    n = min(len(ann_actions), len(snn_actions))
    matches = sum(int(ann_actions[i] == snn_actions[i]) for i in range(n))
    agreement = float(matches / n) if n > 0 else 0.0

    ann_return = float(ann_data.get("return", 0.0))
    snn_return = float(snn_data.get("return", 0.0))
    ann_env_steps = int(ann_data.get("steps", len(ann_actions)))
    snn_env_steps = int(snn_data.get("steps", len(snn_actions)))

    ann_env_steps_to_goal = float(ann_env_steps) if ann_return > 0.0 else float("nan")
    snn_env_steps_to_goal = float(snn_env_steps) if snn_return > 0.0 else float("nan")

    return {
        "seed": int(seed),
        "action_agreement": agreement,
        "return_gap_snn_minus_ann": float(snn_return - ann_return),
        "ann_env_steps_to_goal": ann_env_steps_to_goal,
        "snn_env_steps_to_goal": snn_env_steps_to_goal,
        "ann_return": ann_return,
        "snn_return": snn_return,
        "ann_env_steps": float(ann_env_steps),
        "snn_env_steps": float(snn_env_steps),
    }


def _aggregate_rows(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    keys = [k for k in rows[0].keys() if k != "seed"]
    agg = {"seed": -1}
    for k in keys:
        vals = np.asarray([r[k] for r in rows], dtype=np.float64)
        agg[k] = float(np.nanmean(vals))
    return agg


def _write_csv(rows: list[dict[str, float]], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary table to {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed ANN-vs-SNN agreement analysis")
    parser.add_argument("--config", type=str, required=True, help="Base config path")
    parser.add_argument("--ann_config", type=str, default=None, help="ANN config override")
    parser.add_argument("--snn_config", type=str, default=None, help="SNN config override")
    parser.add_argument("--env_id", type=str, default=None, help="Environment ID override")
    parser.add_argument("--ann_ckpt", type=str, required=True, help="ANN checkpoint path")
    parser.add_argument("--snn_ckpt", type=str, required=True, help="SNN checkpoint path")
    parser.add_argument("--frame_stack", type=int, default=None, help="Optional frame stack override")
    parser.add_argument("--max_steps", type=int, default=500, help="Max rollout steps per episode")
    parser.add_argument("--seed_start", type=int, default=1, help="Start seed (inclusive)")
    parser.add_argument("--seed_end", type=int, default=10, help="End seed (inclusive)")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds; overrides range")
    parser.add_argument(
        "--out_csv",
        type=str,
        default="results/post_analysis/tmaze_agreement_summary.csv",
        help="Output CSV summary table",
    )
    args = parser.parse_args()

    env_id, ann_agent, snn_agent, ann_env_kwargs, snn_env_kwargs = _build_agents_and_env_kwargs(args)
    seeds = _parse_seeds(args.seeds, args.seed_start, args.seed_end)

    rows = []
    for seed in seeds:
        ann_data = collect_trajectory(
            ann_agent,
            env_id=env_id,
            seed=seed,
            is_snn=False,
            max_steps=args.max_steps,
            **ann_env_kwargs,
        )
        snn_data = collect_trajectory(
            snn_agent,
            env_id=env_id,
            seed=seed,
            is_snn=True,
            max_steps=args.max_steps,
            **snn_env_kwargs,
        )
        rows.append(_row_for_seed(seed, ann_data, snn_data))

    mean_row = _aggregate_rows(rows)
    rows_with_mean = rows + [mean_row]
    _write_csv(rows_with_mean, args.out_csv)

    print("\nAggregate (mean across seeds):")
    print(f"- action agreement: {mean_row['action_agreement'] * 100:.2f}%")
    print(f"- return gap (SNN-ANN): {mean_row['return_gap_snn_minus_ann']:.4f}")
    print(
        f"- env steps-to-goal (ANN/SNN): "
        f"{mean_row['ann_env_steps_to_goal']:.2f}/{mean_row['snn_env_steps_to_goal']:.2f}"
    )


if __name__ == "__main__":
    main()
