"""
Evaluation-only ablations for T-Maze models.

Runs ANN baseline and SNN timing-critic checkpoints across:
- lengths: 2, 3, 4, 5
- modes: passive (active=False), active (active=True)
- frame stack: fixed to 4

No training is performed.
"""

import argparse
import csv
import importlib
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import yaml

# Repo path setup
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.training.agents import make_agent, resolve_cartpole_types
from src.training.envs import make_envs, set_global_seeds
from src.training.evaluate import evaluate, evaluate_snn
from src.utils.checkpoint import load_checkpoint
from src.tools.energy_benchmark import EnergyBenchmark, EnergyMetrics


@dataclass
class EvalRow:
    model_name: str
    mode: str
    checkpoint: str
    train_seed: int
    eval_seed: int
    length: int
    active: bool
    frame_stack: int
    episodes: int
    reward_mean: float
    reward_std: float
    length_mean: float
    length_std: float
    success_rate: float


@dataclass
class EnergyRow:
    model_name: str
    mode: str
    checkpoint: str
    train_seed: int
    eval_seed: int
    length: int
    active: bool
    frame_stack: int
    benchmark_episodes: int
    total_energy_joules: float
    dynamic_energy_joules: float
    joules_per_episode: float
    joules_per_1k_steps: float
    raw_joules_per_1k_steps: float
    avg_power_watts: float


def _register_custom_envs() -> None:
    for module in ("src.envs.t_maze",):
        importlib.import_module(module)


def _load_yaml(path: str) -> Dict[str, Any]:
    p = path if os.path.exists(path) else os.path.join(repo_root, path)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_checkpoint(checkpoint_arg: Optional[str], config: Dict[str, Any]) -> str:
    if checkpoint_arg:
        ckpt = checkpoint_arg if os.path.exists(checkpoint_arg) else os.path.join(repo_root, checkpoint_arg)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_arg}")
        return ckpt

    ckpt_dir = os.path.join(config.get("log_dir", ""), "checkpoints")
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(repo_root, ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    ckpts = [
        os.path.join(ckpt_dir, name)
        for name in os.listdir(ckpt_dir)
        if name.endswith(".pt")
    ]
    if not ckpts:
        raise FileNotFoundError(f"No .pt checkpoints found in: {ckpt_dir}")
    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1]

def _resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(repo_root, path)


def _is_tmaze_env(env_id: str) -> bool:
    e = str(env_id).lower()
    return ("tmaze" in e) or ("t-maze" in e) or ("t_maze" in e)


def _infer_act_dim(config: Dict[str, Any]) -> int:
    env_id = str((config.get("env", {}) or {}).get("id", ""))
    if _is_tmaze_env(env_id):
        return 4
    # CartPole variants use 2 actions.
    return 2


def _base_obs_dim_from_env_config(env_cfg: Dict[str, Any]) -> Optional[int]:
    """Best-effort base observation dimension before frame stacking."""
    partial_obs = env_cfg.get("partial_obs")
    if isinstance(partial_obs, dict):
        idx = partial_obs.get("indices")
        if isinstance(idx, list) and idx:
            return int(len(idx))

    env_id = str(env_cfg.get("id", ""))
    if _is_tmaze_env(env_id):
        return 4
    if "cartpole" in env_id.lower():
        return 4
    return None


def _resolve_effective_frame_stack(
    *,
    cli_frame_stack: Optional[int],
    env_cfg: Dict[str, Any],
    inferred_in_dim: Optional[int],
) -> int:
    """
    Resolve frame stack robustly so eval env observation width matches checkpoint input dim.
    Priority:
      1) explicit CLI override
      2) infer from checkpoint input dim and base obs dim
      3) config frame_stack
      4) default 1
    """
    if cli_frame_stack is not None:
        return int(cli_frame_stack)

    base_obs_dim = _base_obs_dim_from_env_config(env_cfg)
    if inferred_in_dim is not None and inferred_in_dim > 0 and base_obs_dim and base_obs_dim > 0:
        if inferred_in_dim % base_obs_dim == 0:
            return max(1, int(inferred_in_dim // base_obs_dim))

    configured_fs = env_cfg.get("frame_stack", None)
    return int(configured_fs or 1)


def _infer_dims_from_checkpoint(checkpoint_path: str) -> tuple[Optional[int], Optional[int]]:
    """
    Infer (in_dim, act_dim) from checkpoint actor_state when possible.
    Returns (None, None) if not inferable.
    """
    try:
        data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None, None
    actor_state = {}
    if isinstance(data, dict):
        actor_state = data.get("actor_state") or data.get("actor_state_dict") or {}
    if not isinstance(actor_state, dict) or not actor_state:
        return None, None

    in_dim = None
    act_dim = None

    # ANN-style keys
    w0 = actor_state.get("backbone.layers.0.weight")
    if torch.is_tensor(w0) and w0.ndim == 2:
        in_dim = int(w0.shape[1])
    ph = actor_state.get("policy_head.weight")
    if torch.is_tensor(ph) and ph.ndim == 2:
        act_dim = int(ph.shape[0])

    # SNN-style fallback
    if in_dim is None:
        for k, v in actor_state.items():
            if k.endswith(".linear.weight") and torch.is_tensor(v) and v.ndim == 2:
                in_dim = int(v.shape[1])
                break
    if act_dim is None:
        bo = actor_state.get("block_out.linear.weight")
        if torch.is_tensor(bo) and bo.ndim == 2:
            act_dim = int(bo.shape[0])

    return in_dim, act_dim


def _build_agent_from_config(
    config: Dict[str, Any],
    device: torch.device,
    *,
    in_dim_override: Optional[int] = None,
    act_dim_override: Optional[int] = None,
) -> torch.nn.Module:
    model_cfg = config.get("model", {}) or {}
    ppo_cfg = config.get("ppo", {}) or {}
    snn_cfg = config.get("snn", {}) or {}
    mode = str(model_cfg.get("mode", "ann"))
    actor_type, critic_type = resolve_cartpole_types(mode)

    act_dim = int(act_dim_override) if act_dim_override is not None else _infer_act_dim(config)
    in_dim = int(in_dim_override) if in_dim_override is not None else int(model_cfg.get("in_features", 4))
    agent = make_agent(
        actor_type=actor_type,
        critic_type=critic_type,
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        in_dim=in_dim,
        act_dim=int(act_dim),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        critic_informs_actor=bool(model_cfg.get("critic_informs_actor", False)),
        detach_critic_for_actor=bool(model_cfg.get("detach_critic_for_actor", True)),
        normalize_critic_for_actor=bool(model_cfg.get("normalize_critic_for_actor", True)),
        critic_actor_value_clip=model_cfg.get("critic_actor_value_clip", 5.0),
        critic_actor_norm_momentum=float(model_cfg.get("critic_actor_norm_momentum", 0.01)),
        **snn_cfg,
    ).to(device)
    agent.eval()
    return agent


def _get_eval_sticky(config: Dict[str, Any]) -> bool:
    sticky_cfg = config.get("sticky_action", {}) or {}
    train_cfg = config.get("training", {}) or {}
    return bool(sticky_cfg.get("eval", train_cfg.get("sticky_action", True)))


def _evaluate_condition(
    *,
    agent: torch.nn.Module,
    config: Dict[str, Any],
    seed: int,
    length: int,
    active: bool,
    frame_stack: int,
    episodes: int,
) -> Dict[str, float]:
    env_cfg = dict(config.get("env", {}) or {})
    env_kwargs = dict(env_cfg.get("kwargs", {}) or {})
    if _is_tmaze_env(str(env_cfg.get("id", "tmaze-v0"))):
        env_kwargs["length"] = int(length)
        env_kwargs["active"] = bool(active)

    _, env_eval = make_envs(
        seed=seed,
        env_id=str(env_cfg.get("id", "tmaze-v0")),
        n_envs=1,
        env_kwargs=env_kwargs,
        partial_obs=env_cfg.get("partial_obs"),
        frame_stack=frame_stack,
        frame_stack_flatten=True,
    )

    sticky_eval = _get_eval_sticky(config)
    reward_threshold = float((config.get("ppo", {}) or {}).get("reward_threshold", 0.95))

    rewards: List[float] = []
    lengths: List[int] = []
    successes = 0
    try:
        for ep in range(episodes):
            r, l = evaluate(env_eval, agent, sticky_action=sticky_eval, seed=seed + ep)
            rewards.append(float(r))
            lengths.append(int(l))
            if float(r) >= reward_threshold:
                successes += 1
    finally:
        if hasattr(env_eval, "close"):
            env_eval.close()

    return {
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "length_std": float(np.std(lengths)) if lengths else 0.0,
        "success_rate": (float(successes) / float(max(1, episodes))) * 100.0,
    }


def _benchmark_condition(
    *,
    agent: torch.nn.Module,
    model_name: str,
    config: Dict[str, Any],
    seed: int,
    length: int,
    active: bool,
    frame_stack: int,
    benchmark_episodes: int,
    warmup_runs: int,
    active_repeat: int,
) -> EnergyMetrics:
    env_cfg = dict(config.get("env", {}) or {})
    env_kwargs = dict(env_cfg.get("kwargs", {}) or {})
    if _is_tmaze_env(str(env_cfg.get("id", "tmaze-v0"))):
        env_kwargs["length"] = int(length)
        env_kwargs["active"] = bool(active)

    _, env_eval = make_envs(
        seed=seed,
        env_id=str(env_cfg.get("id", "tmaze-v0")),
        n_envs=1,
        env_kwargs=env_kwargs,
        partial_obs=env_cfg.get("partial_obs"),
        frame_stack=frame_stack,
        frame_stack_flatten=True,
    )
    sticky_eval = _get_eval_sticky(config)
    bench = EnergyBenchmark()
    try:
        model_type = "SNN" if "snn" in str(model_name).lower() else "ANN"
        return bench.benchmark_model(
            model=agent,
            episode_fn=lambda m: evaluate_snn(env_eval, m, sticky_action=sticky_eval),
            num_episodes=int(benchmark_episodes),
            model_type=model_type,
            prev_train_energy=0.0,
            warmup_runs=int(warmup_runs),
            active_repeat=int(active_repeat),
        )
    finally:
        if hasattr(env_eval, "close"):
            env_eval.close()


def _write_json(rows: Iterable[EvalRow], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = [asdict(r) for r in rows]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_csv(rows: Iterable[EvalRow], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _print_energy_table(rows: List[EnergyRow]) -> None:
    if not rows:
        return
    print("\n=== Energy Benchmark Ablation ===")
    header = "model                 tr_seed ev_seed length active  bench_ep   J/episode   J/1k_steps(raw)  J/1k_steps(dynamic)"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.model_name[:20]:<20} {r.train_seed:<7d} {r.eval_seed:<7d} {r.length:<6d} {str(r.active):<6} "
            f"{r.benchmark_episodes:<9d} {r.joules_per_episode:>11.4f} {r.raw_joules_per_1k_steps:>17.4f} {r.joules_per_1k_steps:>19.4f}"
        )


def _write_energy_summary(rows: List[EnergyRow], out_path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    by_model: Dict[str, List[EnergyRow]] = {}
    for r in rows:
        by_model.setdefault(r.model_name, []).append(r)
    fieldnames = [
        "model_name",
        "n",
        "joules_per_episode_mean",
        "joules_per_episode_std",
        "joules_per_1k_steps_raw_mean",
        "joules_per_1k_steps_raw_std",
        "joules_per_1k_steps_dynamic_mean",
        "joules_per_1k_steps_dynamic_std",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model, items in sorted(by_model.items()):
            j_ep = np.asarray([x.joules_per_episode for x in items], dtype=np.float64)
            j_1k_raw = np.asarray([x.raw_joules_per_1k_steps for x in items], dtype=np.float64)
            j_1k_dyn = np.asarray([x.joules_per_1k_steps for x in items], dtype=np.float64)
            writer.writerow(
                {
                    "model_name": model,
                    "n": int(len(items)),
                    "joules_per_episode_mean": float(np.mean(j_ep)),
                    "joules_per_episode_std": float(np.std(j_ep, ddof=0)),
                    "joules_per_1k_steps_raw_mean": float(np.mean(j_1k_raw)),
                    "joules_per_1k_steps_raw_std": float(np.std(j_1k_raw, ddof=0)),
                    "joules_per_1k_steps_dynamic_mean": float(np.mean(j_1k_dyn)),
                    "joules_per_1k_steps_dynamic_std": float(np.std(j_1k_dyn, ddof=0)),
                }
            )


def _print_table(rows: List[EvalRow]) -> None:
    print("\n=== T-Maze Evaluation Ablations (No Training) ===")
    header = "model                 tr_seed ev_seed mode     length active  R_mean   R_std   L_mean  L_std   success%"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.model_name[:20]:<20} {r.train_seed:<7d} {r.eval_seed:<7d} {r.mode[:8]:<8} {r.length:<6d} {str(r.active):<6} "
            f"{r.reward_mean:>7.3f} {r.reward_std:>7.3f} {r.length_mean:>7.3f} {r.length_std:>7.3f} {r.success_rate:>8.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser("TMaze evaluation-only ablations")
    parser.add_argument("--ann-config", type=str, default="configs/tmaze/ann_baseline.yaml")
    parser.add_argument("--snn-config", type=str, default="configs/tmaze/snn_actor_snn_timing_critic.yaml")
    parser.add_argument("--ann-checkpoint", type=str, default=None)
    parser.add_argument("--snn-checkpoint", type=str, default=None)
    parser.add_argument(
        "--ann-checkpoint-template",
        type=str,
        default=None,
        help="Checkpoint template with {seed}, e.g. results/.../seed_{seed}/checkpoints/checkpoint_best.pt",
    )
    parser.add_argument(
        "--snn-checkpoint-template",
        type=str,
        default=None,
        help="Checkpoint template with {seed}, e.g. results/.../seed_{seed}/checkpoints/checkpoint_best.pt",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Training seed IDs to evaluate (e.g. 1 2 3 4 5)")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--frame-stack", type=int, default=None, help="Override frame stack. Default: config env.frame_stack or 1.")
    parser.add_argument("--seed", type=int, default=42, help="Base eval seed. Multi-seed mode uses (base + train_seed).")
    parser.add_argument("--device", type=str, default=None, help="cpu|cuda|cuda:0 (default: auto)")
    parser.add_argument("--run-energy-benchmark", action="store_true", help="Run energy benchmarking in addition to reward ablations.")
    parser.add_argument("--energy-per-condition", action="store_true", help="If set, benchmark every (active,length) condition; otherwise benchmark one default condition per model.")
    parser.add_argument("--energy-episodes", type=int, default=None, help="Override number of episodes for energy benchmark.")
    parser.add_argument("--energy-warmup-runs", type=int, default=1, help="Warmup runs per measured benchmark episode.")
    parser.add_argument("--energy-active-repeat", type=int, default=1, help="How many episode_fn runs are included in one benchmark measurement.")
    parser.add_argument("--output-dir", type=str, default="results/ablations/tmaze_eval_only")
    args = parser.parse_args()

    _register_custom_envs()
    ann_cfg = _load_yaml(args.ann_config)
    snn_cfg = _load_yaml(args.snn_config)

    set_global_seeds(int(args.seed), deterministic_torch=True, cudnn_benchmark=False)

    if args.device:
        device = torch.device(args.device)
    else:
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(default_device)

    train_seeds = [0] if not args.seeds else [int(s) for s in args.seeds]
    rows: List[EvalRow] = []
    energy_rows: List[EnergyRow] = []
    for train_seed in train_seeds:
        if args.seeds:
            if not args.ann_checkpoint_template or not args.snn_checkpoint_template:
                raise ValueError("When using --seeds, provide both --ann-checkpoint-template and --snn-checkpoint-template.")
            ann_ckpt = _resolve_path(args.ann_checkpoint_template.format(seed=train_seed))
            snn_ckpt = _resolve_path(args.snn_checkpoint_template.format(seed=train_seed))
            if not os.path.exists(ann_ckpt):
                raise FileNotFoundError(f"ANN checkpoint not found for seed {train_seed}: {ann_ckpt}")
            if not os.path.exists(snn_ckpt):
                raise FileNotFoundError(f"SNN checkpoint not found for seed {train_seed}: {snn_ckpt}")
            eval_seed = int(args.seed) + int(train_seed)
        else:
            ann_ckpt = _resolve_checkpoint(args.ann_checkpoint, ann_cfg)
            snn_ckpt = _resolve_checkpoint(args.snn_checkpoint, snn_cfg)
            eval_seed = int(args.seed)

        ann_in_dim, ann_act_dim = _infer_dims_from_checkpoint(ann_ckpt)
        snn_in_dim, snn_act_dim = _infer_dims_from_checkpoint(snn_ckpt)

        ann_agent = _build_agent_from_config(
            ann_cfg, device, in_dim_override=ann_in_dim, act_dim_override=ann_act_dim
        )
        snn_agent = _build_agent_from_config(
            snn_cfg, device, in_dim_override=snn_in_dim, act_dim_override=snn_act_dim
        )
        load_checkpoint(ann_ckpt, agent=ann_agent, optimizer=None, logger=None, map_location=device)
        load_checkpoint(snn_ckpt, agent=snn_agent, optimizer=None, logger=None, map_location=device)

        models = [
            ("ann_baseline", ann_cfg, ann_ckpt, ann_agent),
            ("snn_timing_critic", snn_cfg, snn_ckpt, snn_agent),
        ]

        for model_name, config, ckpt, agent in models:
            mode = str((config.get("model", {}) or {}).get("mode", "unknown"))
            env_cfg_local = config.get("env", {}) or {}
            model_in_dim = ann_in_dim if model_name == "ann_baseline" else snn_in_dim
            effective_frame_stack = _resolve_effective_frame_stack(
                cli_frame_stack=args.frame_stack,
                env_cfg=env_cfg_local,
                inferred_in_dim=model_in_dim,
            )
            bench_cfg = config.get("benchmark", {}) or {}
            benchmark_episodes = int(args.energy_episodes) if args.energy_episodes is not None else int(bench_cfg.get("num_episodes_for_benchmark", 20))

            if args.run_energy_benchmark and not args.energy_per_condition:
                default_kwargs = dict((config.get("env", {}) or {}).get("kwargs", {}) or {})
                default_length = int(default_kwargs.get("length", args.lengths[0]))
                default_active = bool(default_kwargs.get("active", False))
                em = _benchmark_condition(
                    agent=agent,
                    model_name=model_name,
                    config=config,
                    seed=eval_seed,
                    length=default_length,
                    active=default_active,
                    frame_stack=effective_frame_stack,
                    benchmark_episodes=benchmark_episodes,
                    warmup_runs=int(args.energy_warmup_runs),
                    active_repeat=int(args.energy_active_repeat),
                )
                energy_rows.append(
                    EnergyRow(
                        model_name=model_name,
                        mode=mode,
                        checkpoint=ckpt,
                        train_seed=int(train_seed),
                        eval_seed=int(eval_seed),
                        length=default_length,
                        active=default_active,
                        frame_stack=effective_frame_stack,
                        benchmark_episodes=benchmark_episodes,
                        total_energy_joules=float(em.inference_energy_joules),
                        dynamic_energy_joules=float(em.dynamic_energy_joules),
                        joules_per_episode=float(em.energy_per_episode),
                        joules_per_1k_steps=float((em.dynamic_joules_per_env_step or 0.0) * 1000.0),
                        raw_joules_per_1k_steps=float((em.raw_joules_per_env_step or 0.0) * 1000.0),
                        avg_power_watts=float(em.avg_power_watts),
                    )
                )

            for active in (False, True):
                for length in args.lengths:
                    stats = _evaluate_condition(
                        agent=agent,
                        config=config,
                        seed=eval_seed,
                        length=int(length),
                        active=bool(active),
                        frame_stack=effective_frame_stack,
                        episodes=int(args.episodes),
                    )
                    rows.append(
                        EvalRow(
                            model_name=model_name,
                            mode=mode,
                            checkpoint=ckpt,
                            train_seed=int(train_seed),
                            eval_seed=int(eval_seed),
                            length=int(length),
                            active=bool(active),
                            frame_stack=effective_frame_stack,
                            episodes=int(args.episodes),
                            reward_mean=stats["reward_mean"],
                            reward_std=stats["reward_std"],
                            length_mean=stats["length_mean"],
                            length_std=stats["length_std"],
                            success_rate=stats["success_rate"],
                        )
                    )
                    if args.run_energy_benchmark and args.energy_per_condition:
                        em = _benchmark_condition(
                            agent=agent,
                            model_name=model_name,
                            config=config,
                            seed=eval_seed,
                            length=int(length),
                            active=bool(active),
                            frame_stack=effective_frame_stack,
                            benchmark_episodes=benchmark_episodes,
                            warmup_runs=int(args.energy_warmup_runs),
                            active_repeat=int(args.energy_active_repeat),
                        )
                        energy_rows.append(
                            EnergyRow(
                                model_name=model_name,
                                mode=mode,
                                checkpoint=ckpt,
                                train_seed=int(train_seed),
                                eval_seed=int(eval_seed),
                                length=int(length),
                                active=bool(active),
                                frame_stack=effective_frame_stack,
                                benchmark_episodes=benchmark_episodes,
                                total_energy_joules=float(em.inference_energy_joules),
                                dynamic_energy_joules=float(em.dynamic_energy_joules),
                                joules_per_episode=float(em.energy_per_episode),
                                joules_per_1k_steps=float((em.dynamic_joules_per_env_step or 0.0) * 1000.0),
                                raw_joules_per_1k_steps=float((em.raw_joules_per_env_step or 0.0) * 1000.0),
                                avg_power_watts=float(em.avg_power_watts),
                            )
                        )

    out_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(repo_root, args.output_dir)
    json_path = os.path.join(out_dir, "ablations_results.json")
    csv_path = os.path.join(out_dir, "ablations_results.csv")
    _write_json(rows, json_path)
    _write_csv(rows, csv_path)
    _print_table(rows)
    if energy_rows:
        energy_json = os.path.join(out_dir, "energy_ablation_results.json")
        energy_csv = os.path.join(out_dir, "energy_ablation_results.csv")
        energy_summary_csv = os.path.join(out_dir, "energy_ablation_summary.csv")
        _write_json(energy_rows, energy_json)
        _write_csv(energy_rows, energy_csv)
        _write_energy_summary(energy_rows, energy_summary_csv)
        _print_energy_table(energy_rows)
        print(f"Saved Energy JSON: {energy_json}")
        print(f"Saved Energy CSV:  {energy_csv}")
        print(f"Saved Energy Summary CSV: {energy_summary_csv}")
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
