#!/usr/bin/env python3
"""
Generate multi-seed mechanistic timing-critic figures/tables for thesis use.

Outputs (under --out-dir):
- fig_5_1_tau_over_training_mean_sem.png/.pdf
- fig_5_2_tau_within_episode_heatmap.png/.pdf
- tau_vs_next_return_scatter.png/.pdf
- tau_training_curves.csv
- tau_eval_step_samples.csv
- tau_episode_returns.csv
- tau_next_return_correlations.csv
- tau_next_return_correlations.md
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, spearmanr

# Repo path setup
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.training.agents import make_agent, resolve_cartpole_types
from src.training.envs import make_envs, set_global_seeds
from src.training.evaluate import get_last_latency, get_spike_stats_safe
from src.utils.checkpoint import load_checkpoint


@dataclass(frozen=True)
class SplitSpec:
    name: str
    run_dir: Path
    active: bool


def _register_custom_envs() -> None:
    for module in ("src.envs.t_maze",):
        importlib.import_module(module)


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float_or_nan(v: Any) -> float:
    try:
        out = float(v)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _infer_dims_from_checkpoint(checkpoint_path: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        data = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except Exception:
        return None, None

    actor_state = {}
    if isinstance(data, dict):
        actor_state = data.get("actor_state") or data.get("actor_state_dict") or {}
    if not isinstance(actor_state, dict) or not actor_state:
        return None, None

    in_dim: Optional[int] = None
    act_dim: Optional[int] = None

    w0 = actor_state.get("backbone.layers.0.weight")
    if torch.is_tensor(w0) and w0.ndim == 2:
        in_dim = int(w0.shape[1])
    ph = actor_state.get("policy_head.weight")
    if torch.is_tensor(ph) and ph.ndim == 2:
        act_dim = int(ph.shape[0])

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


def _is_tmaze_env(env_id: str) -> bool:
    e = str(env_id).lower()
    return ("tmaze" in e) or ("t-maze" in e) or ("t_maze" in e)


def _base_obs_dim_from_env_config(env_cfg: Dict[str, Any]) -> Optional[int]:
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
    env_cfg: Dict[str, Any], in_dim: Optional[int], cli_frame_stack: Optional[int]
) -> int:
    if cli_frame_stack is not None:
        return int(cli_frame_stack)
    base_dim = _base_obs_dim_from_env_config(env_cfg)
    if in_dim is not None and base_dim and base_dim > 0 and in_dim % base_dim == 0:
        return max(1, int(in_dim // base_dim))
    cfg_fs = env_cfg.get("frame_stack", None)
    return int(cfg_fs or 1)


def _infer_act_dim(config: Dict[str, Any]) -> int:
    env_id = str((config.get("env", {}) or {}).get("id", ""))
    if _is_tmaze_env(env_id):
        return 4
    return 2


def _build_agent_from_config(
    config: Dict[str, Any],
    device: torch.device,
    in_dim_override: Optional[int],
    act_dim_override: Optional[int],
) -> torch.nn.Module:
    model_cfg = config.get("model", {}) or {}
    ppo_cfg = config.get("ppo", {}) or {}
    snn_cfg = config.get("snn", {}) or {}
    mode = str(model_cfg.get("mode", "ann"))
    actor_type, critic_type = resolve_cartpole_types(mode)

    in_dim = int(in_dim_override) if in_dim_override is not None else int(model_cfg.get("in_features", 4))
    act_dim = int(act_dim_override) if act_dim_override is not None else _infer_act_dim(config)

    agent = make_agent(
        actor_type=actor_type,
        critic_type=critic_type,
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        in_dim=int(in_dim),
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


def _tau_col_candidates() -> Sequence[str]:
    return (
        "eval/critic_tau_mean",
        "latency/critic_eval_spike_timing_steps",
        "latency/critic_spike_timing_steps",
    )


def _reward_col_candidates() -> Sequence[str]:
    return (
        "test_reward",
        "eval_reward",
        "post_conversion_ft/eval_reward",
    )


def _x_col_candidates() -> Sequence[str]:
    return (
        "total_timesteps",
        "update",
    )


def _extract_training_series(per_episode_csv: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _read_csv_rows(per_episode_csv)
    if not rows:
        return np.array([]), np.array([]), np.array([])

    keys = rows[0].keys()

    tau_col = next((c for c in _tau_col_candidates() if c in keys), None)
    rew_col = next((c for c in _reward_col_candidates() if c in keys), None)
    x_col = next((c for c in _x_col_candidates() if c in keys), None)

    xs: List[float] = []
    taus: List[float] = []
    rewards: List[float] = []

    for i, r in enumerate(rows):
        x = _float_or_nan(r.get(x_col, i)) if x_col is not None else float(i)
        tau = _float_or_nan(r.get(tau_col, "")) if tau_col is not None else float("nan")
        rew = _float_or_nan(r.get(rew_col, "")) if rew_col is not None else float("nan")
        xs.append(x)
        taus.append(tau)
        rewards.append(rew)

    return np.asarray(xs, dtype=float), np.asarray(taus, dtype=float), np.asarray(rewards, dtype=float)


def _sem(arr: np.ndarray) -> float:
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def _aggregate_tau_curves(seed_series: List[Tuple[int, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Interpolate each seed onto a common x-grid (union of finite x points).
    finite_x = []
    for _, xs, ys in seed_series:
        mask = np.isfinite(xs) & np.isfinite(ys)
        if np.any(mask):
            finite_x.append(np.unique(xs[mask]))
    if not finite_x:
        return np.array([]), np.array([]), np.array([])

    x_grid = np.unique(np.concatenate(finite_x))
    x_grid = np.sort(x_grid)

    y_interp_rows: List[np.ndarray] = []
    for _, xs, ys in seed_series:
        mask = np.isfinite(xs) & np.isfinite(ys)
        if np.sum(mask) < 2:
            continue
        x_s = xs[mask]
        y_s = ys[mask]
        order = np.argsort(x_s)
        x_s = x_s[order]
        y_s = y_s[order]
        x_u, idx = np.unique(x_s, return_index=True)
        y_u = y_s[idx]
        yi = np.interp(x_grid, x_u, y_u, left=np.nan, right=np.nan)
        # mask outside support
        yi[(x_grid < x_u[0]) | (x_grid > x_u[-1])] = np.nan
        y_interp_rows.append(yi)

    if not y_interp_rows:
        return np.array([]), np.array([]), np.array([])

    mat = np.vstack(y_interp_rows)
    mean = np.nanmean(mat, axis=0)
    sem = np.array([
        _sem(col[np.isfinite(col)]) if np.sum(np.isfinite(col)) > 0 else float("nan")
        for col in mat.T
    ])

    keep = np.isfinite(mean)
    return x_grid[keep], mean[keep], sem[keep]


def _collect_seed_dirs(run_dir: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for p in sorted(run_dir.glob("seed_*")):
        if not p.is_dir():
            continue
        try:
            s = int(p.name.split("_", 1)[1])
        except Exception:
            continue
        out.append((s, p))
    return out


def _sticky_eval_flag(config: Dict[str, Any]) -> bool:
    sticky_cfg = config.get("sticky_action", {}) or {}
    train_cfg = config.get("training", {}) or {}
    return bool(sticky_cfg.get("eval", train_cfg.get("sticky_action", True)))


def _run_checkpoint_eval_tau_samples(
    *,
    split: SplitSpec,
    config: Dict[str, Any],
    device: torch.device,
    eval_episodes: int,
    eval_seed_base: int,
    frame_stack_override: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    env_cfg = dict(config.get("env", {}) or {})
    env_id = str(env_cfg.get("id", "tmaze-v0"))
    env_kwargs = dict(env_cfg.get("kwargs", {}) or {})
    env_kwargs["active"] = bool(split.active)

    partial_obs = env_cfg.get("partial_obs", None)
    frame_stack_cfg = env_cfg.get("frame_stack", None)
    frame_stack_flatten = bool(env_cfg.get("frame_stack_flatten", True))

    sticky_eval = _sticky_eval_flag(config)

    step_rows: List[Dict[str, Any]] = []
    ep_rows: List[Dict[str, Any]] = []

    for seed, seed_dir in _collect_seed_dirs(split.run_dir):
        ckpt = seed_dir / "checkpoints" / "checkpoint_best.pt"
        if not ckpt.exists():
            continue

        in_dim, act_dim = _infer_dims_from_checkpoint(ckpt)
        fs_eff = _resolve_effective_frame_stack(env_cfg, in_dim, frame_stack_override)

        # Keep env + agent local per seed to avoid state bleed.
        env_train, env_eval = make_envs(
            seed=1000 + int(seed),
            env_id=env_id,
            n_envs=1,
            env_kwargs=env_kwargs,
            partial_obs=partial_obs,
            frame_stack=fs_eff if fs_eff > 1 else frame_stack_cfg,
            frame_stack_flatten=frame_stack_flatten,
        )

        try:
            agent = _build_agent_from_config(
                config=config,
                device=device,
                in_dim_override=in_dim,
                act_dim_override=act_dim,
            )
            load_checkpoint(str(ckpt), agent=agent, map_location=str(device))
            agent.eval()

            for ep in range(int(eval_episodes)):
                obs, _ = env_eval.reset(seed=int(eval_seed_base + seed * 1000 + ep))
                done = False
                prev_action: Optional[int] = None
                step_idx = 0
                ep_return = 0.0
                ep_tau_vals: List[float] = []

                prev_actor_spikes = float(get_spike_stats_safe(getattr(agent, "actor", None)).get("total_spikes", 0.0))

                while not done:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    logits, _ = agent(obs_t)
                    action = int(torch.argmax(logits, dim=-1).item())

                    # Match sticky-action eval behavior for spiking actors.
                    actor_stats = get_spike_stats_safe(getattr(agent, "actor", None))
                    actor_cum = float(actor_stats.get("total_spikes", 0.0))
                    actor_step_spikes = actor_cum - prev_actor_spikes if actor_cum >= prev_actor_spikes else actor_cum
                    prev_actor_spikes = actor_cum
                    if sticky_eval and actor_step_spikes == 0.0 and prev_action is not None:
                        action = prev_action

                    tau_val = float("nan")
                    if hasattr(agent, "critic") and hasattr(agent.critic, "forward_detailed"):
                        try:
                            _, tau_t = agent.critic.forward_detailed(obs_t)
                            tau_val = float(tau_t.mean().item())
                        except Exception:
                            tau_val = float(get_last_latency(getattr(agent, "critic", None)))
                    else:
                        tau_val = float(get_last_latency(getattr(agent, "critic", None)))

                    obs, reward, terminated, truncated, _ = env_eval.step([action])
                    reward_f = float(reward[0])
                    done = bool(terminated[0] or truncated[0])
                    ep_return += reward_f

                    step_rows.append(
                        {
                            "split": split.name,
                            "seed": int(seed),
                            "episode": int(ep),
                            "step": int(step_idx),
                            "tau": tau_val,
                            "reward_step": reward_f,
                        }
                    )
                    ep_tau_vals.append(tau_val)

                    prev_action = action
                    step_idx += 1

                ep_rows.append(
                    {
                        "split": split.name,
                        "seed": int(seed),
                        "episode": int(ep),
                        "episode_return": float(ep_return),
                        "episode_tau_mean": float(np.nanmean(ep_tau_vals)) if ep_tau_vals else float("nan"),
                        "episode_tau_std": float(np.nanstd(ep_tau_vals)) if ep_tau_vals else float("nan"),
                        "episode_len": int(step_idx),
                    }
                )
        finally:
            env_eval.close()
            env_train.close()

    return step_rows, ep_rows


def _compute_lagged_correlations(seed_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # seed_rows fields: split, seed, x, tau, reward
    out: List[Dict[str, Any]] = []

    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in seed_rows:
        grouped.setdefault((str(r["split"]), int(r["seed"])), []).append(r)

    pooled_by_split: Dict[str, List[Tuple[float, float]]] = {}

    for (split, seed), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda z: float(z["x"]))
        tau_raw = np.asarray([_float_or_nan(z["tau"]) for z in rows], dtype=float)
        rew_raw = np.asarray([_float_or_nan(z["reward"]) for z in rows], dtype=float)
        finite_eval = np.isfinite(tau_raw) & np.isfinite(rew_raw)
        tau = tau_raw[finite_eval]
        rew = rew_raw[finite_eval]
        if tau.size < 2 or rew.size < 2:
            continue
        x_tau = tau[:-1]
        y_next = rew[1:]
        mask = np.isfinite(x_tau) & np.isfinite(y_next)
        x_tau = x_tau[mask]
        y_next = y_next[mask]
        if x_tau.size < 3:
            continue

        try:
            p_r, p_p = pearsonr(x_tau, y_next)
        except Exception:
            p_r, p_p = float("nan"), float("nan")
        try:
            s_r, s_p = spearmanr(x_tau, y_next)
        except Exception:
            s_r, s_p = float("nan"), float("nan")

        out.append(
            {
                "scope": "per_seed",
                "split": split,
                "seed": seed,
                "n_pairs": int(x_tau.size),
                "pearson_r": float(p_r),
                "pearson_p": float(p_p),
                "spearman_rho": float(s_r),
                "spearman_p": float(s_p),
            }
        )

        pooled_by_split.setdefault(split, []).extend(list(zip(x_tau.tolist(), y_next.tolist())))

    for split, pairs in sorted(pooled_by_split.items()):
        if len(pairs) < 3:
            continue
        x_tau = np.asarray([p[0] for p in pairs], dtype=float)
        y_next = np.asarray([p[1] for p in pairs], dtype=float)

        try:
            p_r, p_p = pearsonr(x_tau, y_next)
        except Exception:
            p_r, p_p = float("nan"), float("nan")
        try:
            s_r, s_p = spearmanr(x_tau, y_next)
        except Exception:
            s_r, s_p = float("nan"), float("nan")

        out.append(
            {
                "scope": "pooled",
                "split": split,
                "seed": "all",
                "n_pairs": int(x_tau.size),
                "pearson_r": float(p_r),
                "pearson_p": float(p_p),
                "spearman_rho": float(s_r),
                "spearman_p": float(s_p),
            }
        )

    return out


def _write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _plot_tau_over_training(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    tau_max: Optional[float],
    out_png: Path,
    out_pdf: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    colors = {"tmaze_active": "#1f77b4", "tmaze_passive": "#d62728"}
    labels = {"tmaze_active": "T-Maze Active", "tmaze_passive": "T-Maze Passive"}

    for split_name, (x, m, s) in curves.items():
        if x.size == 0:
            continue
        c = colors.get(split_name, "#333333")
        ax.plot(x, m, color=c, linewidth=2.0, label=labels.get(split_name, split_name))
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.22, linewidth=0)

    ax.set_title("Figure 5.1 Equivalent: Critic Spike Time Compression (Mean ± SEM)")
    ax.set_xlabel("Training progress (total timesteps / update index)")
    ax.set_ylabel("Critic spike time $\\tau$ (timing steps)")
    if tau_max is not None and np.isfinite(float(tau_max)):
        ax.set_ylim(0.0, float(tau_max))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_tau_within_episode_heatmap(
    step_rows: List[Dict[str, Any]],
    tau_max: Optional[float],
    out_png: Path,
    out_pdf: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    split_order = ["tmaze_active", "tmaze_passive"]
    split_titles = {"tmaze_active": "T-Maze Active", "tmaze_passive": "T-Maze Passive"}

    for ax, split in zip(axes, split_order):
        cur = [r for r in step_rows if str(r.get("split")) == split and np.isfinite(_float_or_nan(r.get("tau")))]
        if not cur:
            ax.set_title(f"{split_titles.get(split, split)} (no data)")
            continue

        steps = np.asarray([int(r["step"]) for r in cur], dtype=float)
        taus = np.asarray([_float_or_nan(r["tau"]) for r in cur], dtype=float)

        max_step = max(5, int(np.nanmax(steps)) + 1)
        if tau_max is not None and np.isfinite(float(tau_max)):
            tau_lo, tau_hi = 0.0, float(tau_max)
        else:
            tau_lo = float(np.nanpercentile(taus, 1))
            tau_hi = float(np.nanpercentile(taus, 99))
            if not np.isfinite(tau_lo) or not np.isfinite(tau_hi) or tau_hi <= tau_lo:
                tau_lo, tau_hi = float(np.nanmin(taus)), float(np.nanmax(taus) + 1e-6)

        h = ax.hist2d(
            steps,
            taus,
            bins=[min(80, max_step), 40],
            range=[[0, max_step], [tau_lo, tau_hi]],
            cmap="magma",
        )
        ax.set_title(split_titles.get(split, split))
        ax.set_xlabel("Within-episode step index")
        ax.grid(False)
        cbar = plt.colorbar(h[3], ax=ax)
        cbar.set_label("Count")

    axes[0].set_ylabel("Critic spike time $\\tau$ (timing steps)")
    fig.suptitle("Figure 5.2 Equivalent: Aggregated Within-Episode $\\tau$ Distributions")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _collect_lag_pairs(
    seed_rows: List[Dict[str, Any]],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in seed_rows:
        grouped.setdefault((str(r["split"]), int(r["seed"])), []).append(r)

    by_split: Dict[str, List[Tuple[float, float]]] = {}
    for (split, _seed), rows in grouped.items():
        rows = sorted(rows, key=lambda z: float(z["x"]))
        tau_raw = np.asarray([_float_or_nan(z["tau"]) for z in rows], dtype=float)
        rew_raw = np.asarray([_float_or_nan(z["reward"]) for z in rows], dtype=float)
        finite_eval = np.isfinite(tau_raw) & np.isfinite(rew_raw)
        tau = tau_raw[finite_eval]
        rew = rew_raw[finite_eval]
        if tau.size < 2 or rew.size < 2:
            continue
        x_tau = tau[:-1]
        y_next = rew[1:]
        mask = np.isfinite(x_tau) & np.isfinite(y_next)
        x_tau = x_tau[mask]
        y_next = y_next[mask]
        if x_tau.size == 0:
            continue
        by_split.setdefault(split, []).extend(list(zip(x_tau.tolist(), y_next.tolist())))

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for split, pairs in by_split.items():
        x = np.asarray([p[0] for p in pairs], dtype=float)
        y = np.asarray([p[1] for p in pairs], dtype=float)
        out[split] = (x, y)
    return out


def _plot_tau_next_return_scatter(
    seed_rows: List[Dict[str, Any]],
    lag_rows: List[Dict[str, Any]],
    out_png: Path,
    out_pdf: Path,
) -> None:
    pairs_by_split = _collect_lag_pairs(seed_rows)
    pooled = [r for r in lag_rows if str(r.get("scope")) == "pooled"]
    pooled_by_split: Dict[str, Dict[str, Any]] = {str(r["split"]): r for r in pooled}
    split_order = ["tmaze_active", "tmaze_passive"]
    title_map = {"tmaze_active": "T-Maze Active", "tmaze_passive": "T-Maze Passive"}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    for ax, split in zip(axes, split_order):
        xy = pairs_by_split.get(split, None)
        if xy is None or xy[0].size == 0:
            ax.set_title(f"{title_map.get(split, split)} (no data)")
            ax.set_xlabel("Critic spike time $\\tau_t$")
            ax.grid(True, alpha=0.25, linestyle="--")
            continue
        x, y = xy
        ax.scatter(x, y, s=18, alpha=0.45, edgecolors="none")
        ax.set_title(title_map.get(split, split))
        ax.set_xlabel("Critic spike time $\\tau_t$")
        ax.grid(True, alpha=0.25, linestyle="--")

        # Fit line only when y has variance.
        if np.nanstd(y) > 0 and x.size >= 2:
            try:
                m, b = np.polyfit(x, y, 1)
                xx = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
                yy = m * xx + b
                ax.plot(xx, yy, linewidth=2.0)
            except Exception:
                pass

        stats = pooled_by_split.get(split)
        if stats is not None:
            ax.text(
                0.02,
                0.98,
                f"Pearson r={float(stats['pearson_r']):.3f} (p={float(stats['pearson_p']):.3g})\n"
                f"Spearman rho={float(stats['spearman_rho']):.3f} (p={float(stats['spearman_p']):.3g})\n"
                f"n={int(stats['n_pairs'])}",
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.8, "ec": "none"},
            )

    axes[0].set_ylabel("Subsequent return $R_{t+1}$")
    fig.suptitle("$\\tau_t$ vs Subsequent Return $R_{t+1}$ (Pooled Across Seeds)")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _write_corr_markdown(corr_rows: List[Dict[str, Any]], out_md: Path) -> None:
    lines = []
    lines.append("# Tau vs Subsequent Return Correlations")
    lines.append("")
    lines.append("Computed as lagged pairs within each seed: `tau_t` vs `return_(t+1)` from per-update training logs.")
    lines.append("")
    lines.append("| Scope | Split | Seed | n_pairs | Pearson r | Pearson p | Spearman rho | Spearman p |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in corr_rows:
        lines.append(
            f"| {r['scope']} | {r['split']} | {r['seed']} | {int(r['n_pairs'])} | "
            f"{float(r['pearson_r']):.6g} | {float(r['pearson_p']):.6g} | "
            f"{float(r['spearman_rho']):.6g} | {float(r['spearman_p']):.6g} |"
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser("Generate multi-seed T-Maze timing-critic mechanistic thesis figures.")
    p.add_argument("--logs-root", type=str, default="results/logs/masters")
    p.add_argument("--config", type=str, default="configs/tmaze/snn_actor_snn_timing_critic.yaml")
    p.add_argument("--eval-episodes-per-seed", type=int, default=50)
    p.add_argument("--eval-seed-base", type=int, default=4242)
    p.add_argument("--frame-stack", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=str, default="results/thesis plots/timing_critic_mechanistic")
    args = p.parse_args()

    _register_custom_envs()
    set_global_seeds(42, deterministic_torch=True, cudnn_benchmark=False)

    logs_root = Path(args.logs_root)
    cfg = _read_yaml(Path(args.config))
    requested_device = str(args.device).strip().lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        SplitSpec(
            name="tmaze_active",
            run_dir=logs_root / "tmaze_active" / "tmaze_snn_actor_snn_timing_critic_active",
            active=True,
        ),
        SplitSpec(
            name="tmaze_passive",
            run_dir=logs_root / "tmaze_passive" / "tmaze_snn_actor_snn_timing_critic_passive",
            active=False,
        ),
    ]

    training_curve_rows: List[Dict[str, Any]] = []
    lag_input_rows: List[Dict[str, Any]] = []
    curves: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    tau_max_cfg = _float_or_nan((cfg.get("snn", {}) or {}).get("critic_T", (cfg.get("snn", {}) or {}).get("T", float("nan"))))
    tau_max = tau_max_cfg if np.isfinite(tau_max_cfg) and tau_max_cfg > 0 else None

    for split in splits:
        seed_series: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for seed, seed_dir in _collect_seed_dirs(split.run_dir):
            per_csv = seed_dir / "per_episode_metrics.csv"
            if not per_csv.exists():
                continue
            xs, taus, rews = _extract_training_series(per_csv)
            if xs.size == 0:
                continue

            seed_series.append((seed, xs, taus))
            for x, tau, rw in zip(xs, taus, rews):
                training_curve_rows.append(
                    {
                        "split": split.name,
                        "seed": int(seed),
                        "x": float(x),
                        "tau": float(tau),
                        "reward": float(rw),
                    }
                )
                lag_input_rows.append(
                    {
                        "split": split.name,
                        "seed": int(seed),
                        "x": float(x),
                        "tau": float(tau),
                        "reward": float(rw),
                    }
                )

        curves[split.name] = _aggregate_tau_curves(seed_series)

    # Figure 5.1 equivalent
    _plot_tau_over_training(
        curves=curves,
        tau_max=tau_max,
        out_png=out_dir / "fig_5_1_tau_over_training_mean_sem.png",
        out_pdf=out_dir / "fig_5_1_tau_over_training_mean_sem.pdf",
    )

    # Figure 5.2 equivalent data from checkpoint evaluation
    all_step_rows: List[Dict[str, Any]] = []
    all_ep_rows: List[Dict[str, Any]] = []
    for split in splits:
        step_rows, ep_rows = _run_checkpoint_eval_tau_samples(
            split=split,
            config=cfg,
            device=device,
            eval_episodes=int(args.eval_episodes_per_seed),
            eval_seed_base=int(args.eval_seed_base),
            frame_stack_override=args.frame_stack,
        )
        all_step_rows.extend(step_rows)
        all_ep_rows.extend(ep_rows)

    _plot_tau_within_episode_heatmap(
        step_rows=all_step_rows,
        tau_max=tau_max,
        out_png=out_dir / "fig_5_2_tau_within_episode_heatmap.png",
        out_pdf=out_dir / "fig_5_2_tau_within_episode_heatmap.pdf",
    )

    # Lagged correlation tau_t vs return_{t+1}
    corr_rows = _compute_lagged_correlations(lag_input_rows)
    _plot_tau_next_return_scatter(
        seed_rows=lag_input_rows,
        lag_rows=corr_rows,
        out_png=out_dir / "tau_vs_next_return_scatter.png",
        out_pdf=out_dir / "tau_vs_next_return_scatter.pdf",
    )

    # Save data tables
    _write_csv(training_curve_rows, out_dir / "tau_training_curves.csv")
    _write_csv(all_step_rows, out_dir / "tau_eval_step_samples.csv")
    _write_csv(all_ep_rows, out_dir / "tau_episode_returns.csv")
    _write_csv(corr_rows, out_dir / "tau_next_return_correlations.csv")
    _write_corr_markdown(corr_rows, out_dir / "tau_next_return_correlations.md")

    print(f"Saved outputs under: {out_dir}")


if __name__ == "__main__":
    main()
