#!/usr/bin/env python3
"""
Targeted multi-seed actor/critic latency benchmark.

Outputs:
  - per_seed_latency.csv
  - latency_summary.csv
  - latency_summary.md

Notes:
  - actor_ms / critic_ms are wall-clock per-step timings in milliseconds.
  - SNN models also report actor_spike_timing_steps / critic_spike_timing_steps.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.agents import make_agent, resolve_cartpole_types
from src.training.envs import make_envs, set_global_seeds
from src.training.evaluate import get_last_latency
from src.utils.checkpoint import load_checkpoint


@dataclass(frozen=True)
class TaskSpec:
    task: str
    env_id: str
    env_kwargs: Dict[str, Any]
    partial_obs: Optional[Dict[str, Any]]
    frame_stack: Optional[int]
    ann_root: str
    snn_root: str
    ann_mode: str
    snn_mode: str


@dataclass
class SeedLatencyRow:
    task: str
    agent: str
    seed: int
    checkpoint: str
    n_steps: int
    actor_ms: float
    critic_ms: float
    total_ms: float
    actor_spike_timing_steps: float
    critic_spike_timing_steps: float


TASKS: List[TaskSpec] = [
    TaskSpec(
        task="cartpole",
        env_id="CartPole-v1",
        env_kwargs={},
        partial_obs=None,
        frame_stack=None,
        ann_root="results/logs/cartpole/ann_baseline",
        snn_root="results/logs/cartpole/snn_actor_snn_timing_critic",
        ann_mode="ann",
        snn_mode="snn_actor_snn_timing_critic",
    ),
    TaskSpec(
        task="partial_cartpole",
        env_id="CartPole-v1",
        env_kwargs={},
        partial_obs={"indices": [0, 2]},
        frame_stack=None,
        ann_root="results/logs/partial_cartpole/poc_ann_baseline",
        snn_root="results/logs/partial_cartpole/poc_snn_actor_snn_timing_critic",
        ann_mode="ann",
        snn_mode="snn_actor_snn_timing_critic",
    ),
    TaskSpec(
        task="tmaze_active",
        env_id="tmaze-v0",
        env_kwargs={"length": 3, "active": True},
        partial_obs=None,
        frame_stack=None,
        ann_root="results/logs/tmaze_active/tmaze_ann_baseline_active",
        snn_root="results/logs/tmaze_active/tmaze_snn_actor_snn_timing_critic_active",
        ann_mode="ann",
        snn_mode="snn_actor_snn_timing_critic",
    ),
    TaskSpec(
        task="tmaze_passive",
        env_id="tmaze-v0",
        env_kwargs={"length": 3, "active": False},
        partial_obs=None,
        frame_stack=None,
        ann_root="results/logs/tmaze_passive/tmaze_ann_baseline_passive",
        snn_root="results/logs/tmaze_passive/tmaze_snn_actor_snn_timing_critic_passive",
        ann_mode="ann",
        snn_mode="snn_actor_snn_timing_critic",
    ),
]


def _register_custom_envs() -> None:
    import importlib

    importlib.import_module("src.envs.t_maze")


def _resolve(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(REPO_ROOT / p)


def _checkpoint_path(root: str, seed: int) -> str:
    return _resolve(f"{root}/seed_{seed}/checkpoints/checkpoint_best.pt")


def _infer_dims_from_checkpoint(checkpoint_path: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    try:
        data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None, None, None

    actor_state = {}
    if isinstance(data, dict):
        actor_state = data.get("actor_state") or data.get("actor_state_dict") or {}
    if not isinstance(actor_state, dict) or not actor_state:
        return None, None, None

    in_dim = None
    act_dim = None
    hidden_dim = None

    w0 = actor_state.get("backbone.layers.0.weight")
    if torch.is_tensor(w0) and w0.ndim == 2:
        in_dim = int(w0.shape[1])
        hidden_dim = int(w0.shape[0])
    ph = actor_state.get("policy_head.weight")
    if torch.is_tensor(ph) and ph.ndim == 2:
        act_dim = int(ph.shape[0])

    if in_dim is None:
        for k, v in actor_state.items():
            if k.endswith(".linear.weight") and torch.is_tensor(v) and v.ndim == 2:
                in_dim = int(v.shape[1])
                if hidden_dim is None:
                    hidden_dim = int(v.shape[0])
                break
    if act_dim is None:
        bo = actor_state.get("block_out.linear.weight")
        if torch.is_tensor(bo) and bo.ndim == 2:
            act_dim = int(bo.shape[0])

    return in_dim, act_dim, hidden_dim


def _build_agent(mode: str, in_dim: int, act_dim: int, hidden_dim: int, device: torch.device) -> torch.nn.Module:
    actor_type, critic_type = resolve_cartpole_types(mode)
    kwargs: Dict[str, Any] = {
        "actor_type": actor_type,
        "critic_type": critic_type,
        "hidden_dim": int(hidden_dim),
        "in_dim": in_dim,
        "act_dim": act_dim,
        "gamma": 0.99,
        "dropout": 0.0,
        "critic_informs_actor": False,
        "detach_critic_for_actor": True,
        "normalize_critic_for_actor": True,
        "critic_actor_value_clip": 5.0,
        "critic_actor_norm_momentum": 0.01,
    }
    if "snn" in mode:
        kwargs.update(
            {
                "T": 32,
                "beta": 0.95,
                "V_th": 1.0,
                "poisson_encode": False,
                "rate_scale": 1.0,
                "actor_T": 16,
                "critic_T": 32,
                "actor_V_th": 0.6,
                "critic_V_th": 1.0,
                "critic_spike_temp": 5.0,
                "actor_surrogate_slope": 5.0,
                "critic_cosh_alpha": 2.0,
                "critic_cosh_beta": 1.0,
                "critic_use_hard_no_spike": False,
                "Rmax": 500.0 if "tmaze" not in mode else 1.0,
                "Rmin": 0.0 if "tmaze" not in mode else -1.0,
            }
        )
    agent = make_agent(**kwargs).to(device)
    agent.eval()
    return agent


def _base_obs_dim(task: TaskSpec) -> int:
    if task.partial_obs and isinstance(task.partial_obs, dict):
        idx = task.partial_obs.get("indices")
        if isinstance(idx, list) and idx:
            return int(len(idx))
    if "tmaze" in task.env_id.lower():
        return 4
    return 4


def _effective_frame_stack(task: TaskSpec, in_dim: int) -> int:
    if task.frame_stack is not None:
        return int(task.frame_stack)
    base = _base_obs_dim(task)
    if base > 0 and in_dim > 0 and (in_dim % base == 0):
        return max(1, int(in_dim // base))
    return 1


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def _measure_checkpoint_latency(
    *,
    agent: torch.nn.Module,
    env_id: str,
    env_kwargs: Dict[str, Any],
    partial_obs: Optional[Dict[str, Any]],
    frame_stack: int,
    steps_per_seed: int,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    _, env = make_envs(
        seed=seed,
        env_id=env_id,
        n_envs=1,
        env_kwargs=env_kwargs,
        partial_obs=partial_obs,
        frame_stack=int(frame_stack),
        frame_stack_flatten=True,
    )

    actor_ms: List[float] = []
    critic_ms: List[float] = []
    total_ms: List[float] = []
    actor_steps: List[float] = []
    critic_steps: List[float] = []

    obs, _ = env.reset(seed=seed)
    done = False
    steps = 0
    try:
        while steps < int(steps_per_seed):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0)

            critic_module = getattr(agent, "critic", None)
            _sync_if_cuda(device)
            t0 = time.perf_counter()
            if hasattr(critic_module, "forward_detailed"):
                value, _ = critic_module.forward_detailed(obs_t)
            else:
                value = agent.critic_forward(obs_t)
            _sync_if_cuda(device)
            t1 = time.perf_counter()
            critic_dt = (t1 - t0) * 1000.0

            _sync_if_cuda(device)
            t2 = time.perf_counter()
            if getattr(agent, "critic_informs_actor", False):
                value_for_actor = agent._prepare_critic_value_for_actor(value)
                logits = agent.actor_forward(obs_t, critic_value=value_for_actor)
            else:
                logits = agent.actor_forward(obs_t)
            _sync_if_cuda(device)
            t3 = time.perf_counter()
            actor_dt = (t3 - t2) * 1000.0

            actor_ms.append(actor_dt)
            critic_ms.append(critic_dt)
            total_ms.append(actor_dt + critic_dt)

            actor_steps.append(float(get_last_latency(getattr(agent, "actor", None))))
            critic_steps.append(float(get_last_latency(getattr(agent, "critic", None))))

            action = int(torch.argmax(logits, dim=-1).item())
            obs, _, terminated, truncated, _ = env.step([action])
            done = bool(terminated[0] or truncated[0])
            steps += 1
            if done:
                obs, _ = env.reset()
                done = False
    finally:
        if hasattr(env, "close"):
            env.close()

    def _mean_or_nan(x: List[float]) -> float:
        arr = np.asarray(x, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else float("nan")

    return {
        "n_steps": int(steps),
        "actor_ms": _mean_or_nan(actor_ms),
        "critic_ms": _mean_or_nan(critic_ms),
        "total_ms": _mean_or_nan(total_ms),
        "actor_spike_timing_steps": _mean_or_nan(actor_steps),
        "critic_spike_timing_steps": _mean_or_nan(critic_steps),
    }


def _write_csv(rows: Iterable[Dict[str, Any]], out_path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(seed_rows: List[SeedLatencyRow]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    keyset = sorted({(r.task, r.agent) for r in seed_rows})
    for task, agent in keyset:
        group = [r for r in seed_rows if r.task == task and r.agent == agent]
        n = len(group)

        def _stats(values: List[float]) -> Tuple[float, float]:
            arr = np.asarray(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return float("nan"), float("nan")
            ddof = 1 if arr.size > 1 else 0
            return float(np.mean(arr)), float(np.std(arr, ddof=ddof))

        actor_m, actor_s = _stats([r.actor_ms for r in group])
        critic_m, critic_s = _stats([r.critic_ms for r in group])
        total_m, total_s = _stats([r.total_ms for r in group])
        actor_step_m, actor_step_s = _stats([r.actor_spike_timing_steps for r in group])
        critic_step_m, critic_step_s = _stats([r.critic_spike_timing_steps for r in group])

        out.append(
            {
                "task": task,
                "agent": agent,
                "n_seeds": n,
                "actor_ms_mean": actor_m,
                "actor_ms_std": actor_s,
                "critic_ms_mean": critic_m,
                "critic_ms_std": critic_s,
                "total_ms_mean": total_m,
                "total_ms_std": total_s,
                "actor_spike_timing_steps_mean": actor_step_m,
                "actor_spike_timing_steps_std": actor_step_s,
                "critic_spike_timing_steps_mean": critic_step_m,
                "critic_spike_timing_steps_std": critic_step_s,
            }
        )
    return out


def _format_pm(mean: float, std: float, fmt: str = ".6g") -> str:
    if not (np.isfinite(mean) and np.isfinite(std)):
        return "NA"
    return f"{format(mean, fmt)} ± {format(std, fmt)}"


def _write_markdown_summary(summary: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Targeted Actor/Critic Latency Benchmark")
    lines.append("")
    lines.append("| Task | Agent | n | Actor (ms) | Critic (ms) | Critic spike-timing steps (SNN only) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in summary:
        lines.append(
            "| "
            f"{r['task']} | {r['agent']} | {int(r['n_seeds'])} | "
            f"{_format_pm(r['actor_ms_mean'], r['actor_ms_std'])} | "
            f"{_format_pm(r['critic_ms_mean'], r['critic_ms_std'])} | "
            f"{_format_pm(r['critic_spike_timing_steps_mean'], r['critic_spike_timing_steps_std']) if 'snn' in str(r['agent']).lower() else '--'} |"
        )
    lines.append("")
    lines.append("`critic_spike_timing_steps` is an algorithmic SNN timing metric (steps), not milliseconds.")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Targeted actor/critic latency benchmark")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--steps-per-seed", type=int, default=2000)
    p.add_argument("--device", type=str, default=None, help="cpu|cuda|cuda:0 (default: auto)")
    p.add_argument(
        "--out-dir",
        type=str,
        default="results/post_analysis/latency_actor_critic_multiseed",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    _register_custom_envs()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_global_seeds(42, deterministic_torch=True, cudnn_benchmark=False)

    rows: List[SeedLatencyRow] = []
    for task in TASKS:
        for agent_label, root_dir, mode in (
            ("ann_baseline", task.ann_root, task.ann_mode),
            ("snn_timing_critic", task.snn_root, task.snn_mode),
        ):
            for seed in args.seeds:
                ckpt = _checkpoint_path(root_dir, int(seed))
                if not os.path.exists(ckpt):
                    print(f"[warn] missing checkpoint, skip: {ckpt}")
                    continue

                in_dim, act_dim, hidden_dim = _infer_dims_from_checkpoint(ckpt)
                if in_dim is None or act_dim is None or hidden_dim is None:
                    print(f"[warn] could not infer dims from checkpoint, skip: {ckpt}")
                    continue

                agent = _build_agent(
                    mode=mode,
                    in_dim=int(in_dim),
                    act_dim=int(act_dim),
                    hidden_dim=int(hidden_dim),
                    device=device,
                )
                load_checkpoint(ckpt, agent=agent, optimizer=None, logger=None, map_location=device)
                frame_stack = _effective_frame_stack(task, int(in_dim))

                m = _measure_checkpoint_latency(
                    agent=agent,
                    env_id=task.env_id,
                    env_kwargs=task.env_kwargs,
                    partial_obs=task.partial_obs,
                    frame_stack=frame_stack,
                    steps_per_seed=int(args.steps_per_seed),
                    device=device,
                    seed=1000 + int(seed),
                )
                rows.append(
                    SeedLatencyRow(
                        task=task.task,
                        agent=agent_label,
                        seed=int(seed),
                        checkpoint=ckpt,
                        n_steps=int(m["n_steps"]),
                        actor_ms=float(m["actor_ms"]),
                        critic_ms=float(m["critic_ms"]),
                        total_ms=float(m["total_ms"]),
                        actor_spike_timing_steps=float(m["actor_spike_timing_steps"]),
                        critic_spike_timing_steps=float(m["critic_spike_timing_steps"]),
                    )
                )
                print(
                    f"[ok] {task.task} {agent_label} seed={seed} "
                    f"actor_ms={m['actor_ms']:.6f} critic_ms={m['critic_ms']:.6f} "
                    f"critic_steps={m['critic_spike_timing_steps']:.6f}"
                )

    out_dir = Path(_resolve(args.out_dir))
    seed_csv = out_dir / "per_seed_latency.csv"
    summary_csv = out_dir / "latency_summary.csv"
    summary_md = out_dir / "latency_summary.md"

    _write_csv([asdict(r) for r in rows], seed_csv)
    summary = _summary_rows(rows)
    _write_csv(summary, summary_csv)
    _write_markdown_summary(summary, summary_md)

    print(f"Saved per-seed: {seed_csv}")
    print(f"Saved summary:  {summary_csv}")
    print(f"Saved markdown: {summary_md}")


if __name__ == "__main__":
    main()
