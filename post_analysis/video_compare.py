import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import yaml
import gymnasium as gym
import torch
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Keep matplotlib cache writable in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import src.envs  # Registers custom Gymnasium environments (e.g., tmaze-v0)
from src.utils.checkpoint import load_checkpoint
from src.training.agents import make_agent, ActorType, CriticType, resolve_cartpole_types
from src.tools.video_comparison import collect_trajectory, render_comparison_video
from src.models.ann import Actor
from src.models.ActorCritic import ActorCritic
from src.models.snn_spikeactor import SNNSpikeActor


def _resolve_path(path: str | None) -> str | None:
    if path is None:
        return None
    if os.path.exists(path):
        return path
    candidate = os.path.join(repo_root, path)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Path not found: {path}")


def _load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _load_actor_state(checkpoint_path: str) -> dict:
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor_state = data.get("actor_state") or data.get("actor_state_dict")
    if not isinstance(actor_state, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} has no actor_state")
    return actor_state


def _infer_ann_dims_from_state(actor_state: dict) -> tuple[int, int, int]:
    first = actor_state.get("backbone.layers.0.weight")
    head = actor_state.get("policy_head.weight")
    if first is None or head is None:
        raise ValueError("Cannot infer ANN dims from checkpoint actor_state keys")
    hidden_dim = int(first.shape[0])
    in_dim = int(first.shape[1])
    act_dim = int(head.shape[0])
    return in_dim, act_dim, hidden_dim


def _infer_snn_dims_from_state(actor_state: dict) -> tuple[int, int, int]:
    first = actor_state.get("block1.linear.weight")
    out = actor_state.get("block_out.linear.weight")
    if first is None or out is None:
        # ANN2SNN actor-style checkpoint: Actor(backbone=SNNSpikeActor, policy_head=Linear)
        first = actor_state.get("backbone.block1.linear.weight")
        out = actor_state.get("policy_head.weight")
    if first is None or out is None:
        raise ValueError("Cannot infer SNN dims from checkpoint actor_state keys")
    hidden_dim = int(first.shape[0])
    in_dim = int(first.shape[1])
    act_dim = int(out.shape[0])
    return in_dim, act_dim, hidden_dim


def _is_hybrid_converted_snn_actor_state(actor_state: dict) -> bool:
    return ("backbone.block1.linear.weight" in actor_state) and ("policy_head.weight" in actor_state)


def _build_hybrid_converted_snn_actor(
    *,
    in_dim: int,
    hidden_dim: int,
    act_dim: int,
    config: dict,
) -> Actor:
    snn_convert_cfg = config.get("snn_convert", {})
    backbone = SNNSpikeActor(
        in_dim=in_dim,
        hid_dim=hidden_dim,
        out_dim=hidden_dim,
        T=int(snn_convert_cfg.get("T", 32)),
        beta=float(snn_convert_cfg.get("beta", 1.0)),
        V_th=float(snn_convert_cfg.get("V_th", 1.0)),
        poisson_encode=bool(snn_convert_cfg.get("poisson_encode", False)),
        rate_scale=float(snn_convert_cfg.get("rate_scale", 1.0)),
        center_logits=bool(snn_convert_cfg.get("center_logits", True)),
    )
    return Actor(
        backbone=backbone,
        latent_dim=hidden_dim,
        action_dim=act_dim,
        critic_informs_actor=bool(config.get("model", {}).get("critic_informs_actor", False)),
    )


def _resolve_dims_from_config(config: dict, env_id: str) -> tuple[int | None, int | None, int | None]:
    model_cfg = config.get("model", {})
    in_dim = model_cfg.get("in_features")
    hidden_dim = model_cfg.get("hidden_dim")

    act_dim = model_cfg.get("act_dim") or model_cfg.get("action_dim")
    if act_dim is None and env_id:
        env_cfg = config.get("env", {})
        env_kwargs = env_cfg.get("kwargs", {})
        tmp_env = gym.make(env_id, **env_kwargs)
        if hasattr(tmp_env.action_space, "n"):
            act_dim = int(tmp_env.action_space.n)
        else:
            act_dim = int(tmp_env.action_space.shape[0])
        tmp_env.close()

    return (
        int(in_dim) if in_dim is not None else None,
        int(act_dim) if act_dim is not None else None,
        int(hidden_dim) if hidden_dim is not None else None,
    )


def _resolve_agent_types(config: dict, default_actor: ActorType, default_critic: CriticType) -> tuple[ActorType, CriticType]:
    mode = str(config.get("model", {}).get("mode", "")).strip()
    if not mode:
        return default_actor, default_critic
    try:
        return resolve_cartpole_types(mode)
    except ValueError:
        return default_actor, default_critic


def _merge_dims(
    config_dims: tuple[int | None, int | None, int | None],
    ckpt_dims: tuple[int, int, int],
    *,
    label: str,
) -> tuple[int, int, int]:
    in_dim_cfg, act_dim_cfg, hidden_dim_cfg = config_dims
    in_dim_ckpt, act_dim_ckpt, hidden_dim_ckpt = ckpt_dims
    if in_dim_cfg is not None and in_dim_cfg != in_dim_ckpt:
        print(f"[warn] {label} in_dim config={in_dim_cfg} differs from checkpoint={in_dim_ckpt}; using checkpoint")
    if act_dim_cfg is not None and act_dim_cfg != act_dim_ckpt:
        print(f"[warn] {label} act_dim config={act_dim_cfg} differs from checkpoint={act_dim_ckpt}; using checkpoint")
    if hidden_dim_cfg is not None and hidden_dim_cfg != hidden_dim_ckpt:
        print(f"[warn] {label} hidden_dim config={hidden_dim_cfg} differs from checkpoint={hidden_dim_ckpt}; using checkpoint")
    return (
        in_dim_ckpt if in_dim_cfg is None or in_dim_cfg != in_dim_ckpt else in_dim_cfg,
        act_dim_ckpt if act_dim_cfg is None or act_dim_cfg != act_dim_ckpt else act_dim_cfg,
        hidden_dim_ckpt if hidden_dim_cfg is None or hidden_dim_cfg != hidden_dim_ckpt else hidden_dim_cfg,
    )


def _build_env_kwargs(env_cfg: dict, frame_stack_override: int | None) -> dict:
    frame_stack = frame_stack_override if frame_stack_override is not None else env_cfg.get("frame_stack")
    return {
        "frame_stack": frame_stack,
        "frame_stack_flatten": env_cfg.get("frame_stack_flatten", True),
        "partial_obs": env_cfg.get("partial_obs"),
        "env_kwargs": env_cfg.get("kwargs", {}),
        "pad_video_tail": False,  # Keep trajectory metrics on true env decision steps.
    }


def _infer_base_obs_dim(env_id: str, env_kwargs: dict) -> int:
    env = gym.make(env_id, render_mode="rgb_array", **(env_kwargs or {}))
    try:
        obs, _ = env.reset(seed=0)
    finally:
        env.close()
    return int(torch.as_tensor(obs).numel())


def _resolve_frame_stack_from_input_dim(
    *,
    env_id: str,
    env_cfg: dict,
    input_dim: int,
    config_frame_stack: int | None,
    user_override: int | None,
    label: str,
) -> int | None:
    if user_override is not None:
        return user_override

    base_obs_dim = _infer_base_obs_dim(env_id, env_cfg.get("kwargs", {}))
    if input_dim == base_obs_dim:
        inferred = None
    elif base_obs_dim > 0 and input_dim % base_obs_dim == 0:
        inferred = input_dim // base_obs_dim
    else:
        inferred = config_frame_stack

    if inferred != config_frame_stack:
        print(
            f"[warn] {label} frame_stack config={config_frame_stack} "
            f"does not match input_dim={input_dim}; using inferred={inferred}"
        )
    return inferred


def _infer_value_ylim(env_id: str, ann_config: dict, snn_config: dict) -> tuple[float, float] | None:
    env_key = str(env_id).lower()
    if "tmaze" in env_key:
        return (0.0, 1.0)

    snn_cfg = snn_config.get("snn", {})
    rmin = snn_cfg.get("Rmin")
    rmax = snn_cfg.get("Rmax")
    if rmin is not None and rmax is not None:
        return (float(min(rmin, rmax)), float(max(rmin, rmax)))
    return None


def _compute_comparison_metrics(
    ann_data: dict,
    snn_data: dict,
    seed: int,
    snn_critic_T: int,
) -> dict[str, float]:
    ann_steps = int(ann_data.get("steps", len(ann_data.get("actions", []))))
    snn_steps = int(snn_data.get("steps", len(snn_data.get("actions", []))))
    n = min(ann_steps, snn_steps)
    ann_actions = ann_data.get("actions", [])
    snn_actions = snn_data.get("actions", [])
    matches = sum(int(ann_actions[i] == snn_actions[i]) for i in range(n))
    agreement = float(matches / n) if n > 0 else 0.0

    ann_values = np.asarray(ann_data.get("values", []), dtype=np.float64)
    snn_values = np.asarray(snn_data.get("values", []), dtype=np.float64)
    n_val = min(len(ann_values), len(snn_values))
    mean_abs_value_gap = float(np.mean(np.abs(ann_values[:n_val] - snn_values[:n_val]))) if n_val > 0 else float("nan")

    ann_return = float(ann_data.get("return", np.sum(ann_data.get("rewards", []))))
    snn_return = float(snn_data.get("return", np.sum(snn_data.get("rewards", []))))

    return {
        "seed": int(seed),
        "ann_env_steps": ann_steps,
        "snn_env_steps": snn_steps,
        "env_steps_gap": float(snn_steps - ann_steps),
        "snn_critic_T": int(snn_critic_T),
        "snn_critic_internal_steps": float(snn_steps * int(snn_critic_T)),
        "action_match_count": float(matches),
        "action_compared_steps": float(n),
        "action_agreement": agreement,
        "ann_return": ann_return,
        "snn_return": snn_return,
        "return_gap_snn_minus_ann": float(snn_return - ann_return),
        "ann_value_mean": float(np.mean(ann_values)) if ann_values.size else float("nan"),
        "snn_value_mean": float(np.mean(snn_values)) if snn_values.size else float("nan"),
        "mean_abs_value_gap": mean_abs_value_gap,
    }


def _print_metrics(metrics: dict[str, float]) -> None:
    print("\nQuantitative comparison:")
    print(
        f"- action agreement: {metrics['action_match_count']:.0f}/{metrics['action_compared_steps']:.0f} "
        f"({metrics['action_agreement'] * 100:.2f}%)"
    )
    print(f"- environment steps (ANN/SNN): {int(metrics['ann_env_steps'])}/{int(metrics['snn_env_steps'])}")
    print(
        f"- SNN critic internal timesteps: {int(metrics['snn_critic_internal_steps'])} "
        f"(critic_T={int(metrics['snn_critic_T'])})"
    )
    print(f"- episode return (ANN/SNN): {metrics['ann_return']:.3f}/{metrics['snn_return']:.3f}")
    print(f"- return gap (SNN-ANN): {metrics['return_gap_snn_minus_ann']:.3f}")
    print(f"- mean value (ANN/SNN): {metrics['ann_value_mean']:.4f}/{metrics['snn_value_mean']:.4f}")
    print(f"- mean |value gap|: {metrics['mean_abs_value_gap']:.4f}")


def _write_metrics_csv(metrics: dict[str, float], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(f"Saved quantitative metrics to {out_path}")


def _write_rows_csv(rows: list[dict[str, float]], out_path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-seed metrics to {out_path}")


def _parse_seeds(seed: int, seeds: str | None, seed_start: int, seed_end: int) -> list[int]:
    if seeds:
        parsed = [int(x.strip()) for x in seeds.split(",") if x.strip()]
        return parsed if parsed else [int(seed)]
    if seed_start is not None and seed_end is not None:
        return list(range(int(seed_start), int(seed_end) + 1))
    return [int(seed)]


def _bootstrap_ci(values: np.ndarray, confidence: float = 0.95, n_boot: int = 1000) -> tuple[float, float]:
    values = values.astype(np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        v = float(values[0])
        return v, v
    rng = np.random.default_rng(42)
    means = np.empty(n_boot, dtype=np.float64)
    n = values.size
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        means[i] = np.mean(sample)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def _summarize_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    numeric_keys = [k for k in rows[0].keys() if k != "seed"]
    summary: dict[str, float] = {"n_seeds": float(len(rows))}
    for key in numeric_keys:
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            summary[f"{key}_mean"] = float("nan")
            summary[f"{key}_std"] = float("nan")
            summary[f"{key}_ci95_low"] = float("nan")
            summary[f"{key}_ci95_high"] = float("nan")
            continue
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals, ddof=0))
        lo, hi = _bootstrap_ci(vals, confidence=0.95, n_boot=1000)
        summary[f"{key}_ci95_low"] = lo
        summary[f"{key}_ci95_high"] = hi
    return summary


def _file_sha1(path: str | None) -> str:
    if not path or not os.path.exists(path):
        return "missing"
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:12]


def _build_video_metadata(
    *,
    env_id: str,
    ann_cfg_path: str,
    snn_cfg_path: str,
    ann_ckpt: str,
    snn_ckpt: str,
    seed: int,
    snn_critic_T: int,
) -> dict[str, str]:
    return {
        "env": env_id,
        "seed": str(seed),
        "critic_T": str(snn_critic_T),
        "ann_cfg": os.path.basename(ann_cfg_path),
        "snn_cfg": os.path.basename(snn_cfg_path),
        "ann_cfg_sha": _file_sha1(ann_cfg_path),
        "snn_cfg_sha": _file_sha1(snn_cfg_path),
        "ann_ckpt": os.path.basename(ann_ckpt),
        "snn_ckpt": os.path.basename(snn_ckpt),
    }


def _make_triplet_montage(
    *,
    videos: list[str],
    out_path: str,
    crf: int = 20,
) -> bool:
    if len(videos) != 3 or not shutil.which("ffmpeg"):
        return False
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", videos[0],
        "-i", videos[1],
        "-i", videos[2],
        "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]",
        "-map", "[v]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(int(crf)),
        "-movflags", "+faststart",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Render ANN vs SNN Internal Dynamics Comparison Video")
    parser.add_argument("--config", type=str, required=True, help="Base config (env/video defaults)")
    parser.add_argument("--ann_config", type=str, default=None, help="ANN config override")
    parser.add_argument("--snn_config", type=str, default=None, help="SNN config override")
    parser.add_argument("--env_id", type=str, default=None, help="Environment ID override")
    parser.add_argument("--ann_ckpt", type=str, required=True, help="Path to best ANN checkpoint")
    parser.add_argument("--snn_ckpt", type=str, required=True, help="Path to best/finetuned SNN checkpoint")
    parser.add_argument("--out", type=str, default="tmaze_comparison.mp4", help="Output video path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the deterministic rollout")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds for multi-seed analysis")
    parser.add_argument("--seed_start", type=int, default=None, help="Start seed (inclusive) when running a range")
    parser.add_argument("--seed_end", type=int, default=None, help="End seed (inclusive) when running a range")
    parser.add_argument("--frame_stack", type=int, default=None, help="Number of frames to stack (if any)")
    parser.add_argument("--fps", type=int, default=5, help="Video FPS")
    parser.add_argument(
        "--end_hold_frames",
        type=int,
        default=20,
        help="Number of final frames to hold in rendered video (visual-only padding)",
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default="results/post_analysis/tmaze_comparison_metrics.csv",
        help="Per-seed metrics CSV path",
    )
    parser.add_argument(
        "--summary_out",
        type=str,
        default="results/post_analysis/tmaze_comparison_summary.csv",
        help="Summary CSV path (mean/std/95%% CI)",
    )
    parser.add_argument(
        "--montage_out",
        type=str,
        default=None,
        help="Optional best/median/worst montage output path",
    )
    parser.add_argument("--crf", type=int, default=20, help="H.264 CRF quality (lower is better quality)")
    parser.add_argument(
        "--show_raw_critic_vout",
        action="store_true",
        help="Overlay raw timing-critic vout(t) on internal-timestep plot (SNN only)",
    )

    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    ann_config_path = _resolve_path(args.ann_config) if args.ann_config else config_path
    snn_config_path = _resolve_path(args.snn_config) if args.snn_config else config_path

    config = _load_yaml(config_path)
    ann_config = _load_yaml(ann_config_path)
    snn_config = _load_yaml(snn_config_path)

    env_cfg = config.get("env", {})
    if not env_cfg:
        env_cfg = ann_config.get("env", {}) or snn_config.get("env", {})
    env_id = args.env_id or env_cfg.get("id", "TMaze-v0")

    ann_actor_state = _load_actor_state(args.ann_ckpt)
    snn_actor_state = _load_actor_state(args.snn_ckpt)

    ann_cfg_dims = _resolve_dims_from_config(ann_config, env_id)
    snn_cfg_dims = _resolve_dims_from_config(snn_config, env_id)

    ann_ckpt_dims = _infer_ann_dims_from_state(ann_actor_state)
    snn_ckpt_dims = _infer_snn_dims_from_state(snn_actor_state)

    ann_dims = _merge_dims(
        ann_cfg_dims,
        ann_ckpt_dims,
        label="ANN",
    )
    snn_dims = _merge_dims(
        snn_cfg_dims,
        snn_ckpt_dims,
        label="SNN",
    )

    ann_actor_type, ann_critic_type = _resolve_agent_types(ann_config, ActorType.ANN, CriticType.ANN)
    snn_actor_type, snn_critic_type = _resolve_agent_types(snn_config, ActorType.SNN_SPIKE, CriticType.SNN_TIMING)

    snn_cfg = snn_config.get("snn", {})
    snn_T = snn_cfg.get("T") or snn_cfg.get("actor_T") or 32
    snn_critic_T = int(snn_cfg.get("critic_T") or snn_T)

    ann_model_cfg = ann_config.get("model", {})
    snn_model_cfg = snn_config.get("model", {})
    ann_critic_informs_actor = bool(ann_model_cfg.get("critic_informs_actor", False))
    snn_critic_informs_actor = bool(snn_model_cfg.get("critic_informs_actor", False))
    ann_detach_critic_for_actor = bool(ann_model_cfg.get("detach_critic_for_actor", True))
    ann_normalize_critic_for_actor = bool(ann_model_cfg.get("normalize_critic_for_actor", True))
    ann_critic_actor_value_clip = ann_model_cfg.get("critic_actor_value_clip", 5.0)
    ann_critic_actor_norm_momentum = float(ann_model_cfg.get("critic_actor_norm_momentum", 0.01))
    snn_detach_critic_for_actor = bool(snn_model_cfg.get("detach_critic_for_actor", True))
    snn_normalize_critic_for_actor = bool(snn_model_cfg.get("normalize_critic_for_actor", True))
    snn_critic_actor_value_clip = snn_model_cfg.get("critic_actor_value_clip", 5.0)
    snn_critic_actor_norm_momentum = float(snn_model_cfg.get("critic_actor_norm_momentum", 0.01))

    # Compatibility guard: infer critic_informs_actor from checkpoint in_features when config disagrees.
    snn_cfg_in_dim = snn_cfg_dims[0]
    snn_ckpt_in_dim = snn_ckpt_dims[0]
    if snn_cfg_in_dim is not None:
        inferred_snn_critic_informs_actor = bool(snn_ckpt_in_dim == int(snn_cfg_in_dim) + 1)
        if inferred_snn_critic_informs_actor != snn_critic_informs_actor:
            print(
                f"[warn] SNN critic_informs_actor config={snn_critic_informs_actor} "
                f"differs from checkpoint-inferred={inferred_snn_critic_informs_actor}; using checkpoint setting"
            )
            snn_critic_informs_actor = inferred_snn_critic_informs_actor

    # Avoid duplicate keyword for make_agent(T=..., **kwargs)
    snn_make_kwargs = {k: v for k, v in snn_cfg.items() if k != "T"}

    print("Initializing models...")
    ann_agent = make_agent(
        actor_type=ann_actor_type,
        critic_type=ann_critic_type,
        in_dim=ann_dims[0],
        act_dim=ann_dims[1],
        hidden_dim=ann_dims[2],
        critic_informs_actor=ann_critic_informs_actor,
        detach_critic_for_actor=ann_detach_critic_for_actor,
        normalize_critic_for_actor=ann_normalize_critic_for_actor,
        critic_actor_value_clip=ann_critic_actor_value_clip,
        critic_actor_norm_momentum=ann_critic_actor_norm_momentum,
    )
    if _is_hybrid_converted_snn_actor_state(snn_actor_state):
        print("[info] Detected ANN2SNN actor checkpoint format (SNN backbone + ANN policy head).")
        # Build critic via existing factory, then replace actor with matching hybrid module.
        snn_agent_tmp = make_agent(
            actor_type=ActorType.ANN,
            critic_type=snn_critic_type,
            in_dim=snn_dims[0],
            act_dim=snn_dims[1],
            hidden_dim=snn_dims[2],
            T=snn_T,
            critic_informs_actor=snn_critic_informs_actor,
            detach_critic_for_actor=snn_detach_critic_for_actor,
            normalize_critic_for_actor=snn_normalize_critic_for_actor,
            critic_actor_value_clip=snn_critic_actor_value_clip,
            critic_actor_norm_momentum=snn_critic_actor_norm_momentum,
            **snn_make_kwargs,
        )
        hybrid_actor = _build_hybrid_converted_snn_actor(
            in_dim=snn_dims[0],
            hidden_dim=snn_dims[2],
            act_dim=snn_dims[1],
            config=snn_config,
        )
        snn_agent = ActorCritic(
            actor=hybrid_actor,
            critic=snn_agent_tmp.critic,
            critic_informs_actor=snn_critic_informs_actor,
        )
    else:
        snn_agent = make_agent(
            actor_type=snn_actor_type,
            critic_type=snn_critic_type,
            in_dim=snn_dims[0],
            act_dim=snn_dims[1],
            hidden_dim=snn_dims[2],
            T=snn_T,
            critic_informs_actor=snn_critic_informs_actor,
            detach_critic_for_actor=snn_detach_critic_for_actor,
            normalize_critic_for_actor=snn_normalize_critic_for_actor,
            critic_actor_value_clip=snn_critic_actor_value_clip,
            critic_actor_norm_momentum=snn_critic_actor_norm_momentum,
            **snn_make_kwargs,
        )

    print(f"Loading ANN from {args.ann_ckpt}...")
    load_checkpoint(args.ann_ckpt, agent=ann_agent, map_location="cpu")

    print(f"Loading SNN from {args.snn_ckpt}...")
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

    run_seeds = _parse_seeds(args.seed, args.seeds, args.seed_start, args.seed_end)
    rows: list[dict[str, float]] = []
    trajectories: dict[int, tuple[dict, dict]] = {}

    for seed in run_seeds:
        print(f"\nCollecting ANN trajectory (seed={seed})...")
        ann_data = collect_trajectory(ann_agent, env_id=env_id, seed=seed, is_snn=False, **ann_env_kwargs)
        print(f"Collecting SNN trajectory (seed={seed})...")
        snn_data = collect_trajectory(snn_agent, env_id=env_id, seed=seed, is_snn=True, **snn_env_kwargs)

        metrics = _compute_comparison_metrics(ann_data, snn_data, seed=seed, snn_critic_T=snn_critic_T)
        rows.append(metrics)
        trajectories[int(seed)] = (ann_data, snn_data)

    # Single-seed diagnostics
    if rows:
        _print_metrics(rows[0] if len(rows) == 1 else rows[int(np.argsort([r["action_agreement"] for r in rows])[len(rows) // 2])])
    if len(rows) == 1:
        _write_metrics_csv(rows[0], args.metrics_out)
    else:
        _write_rows_csv(rows, args.metrics_out)

    # Multi-seed summary artifact
    summary = _summarize_metrics(rows)
    _write_rows_csv([summary], args.summary_out)
    if len(rows) > 1:
        print("\nMulti-seed summary:")
        print(f"- n_seeds: {int(summary['n_seeds'])}")
        print(
            f"- action_agreement mean±std: {summary['action_agreement_mean']:.4f}±{summary['action_agreement_std']:.4f} "
            f"(95% CI [{summary['action_agreement_ci95_low']:.4f}, {summary['action_agreement_ci95_high']:.4f}])"
        )
        print(
            f"- return_gap mean±std: {summary['return_gap_snn_minus_ann_mean']:.4f}±{summary['return_gap_snn_minus_ann_std']:.4f} "
            f"(95% CI [{summary['return_gap_snn_minus_ann_ci95_low']:.4f}, {summary['return_gap_snn_minus_ann_ci95_high']:.4f}])"
        )
        print(
            f"- env_steps_to_goal ANN/SNN mean: {summary['ann_env_steps_mean']:.2f}/{summary['snn_env_steps_mean']:.2f}"
        )

    # Render main comparison video using median-by-agreement seed when multi-seed.
    metric_vals = np.asarray([r["action_agreement"] for r in rows], dtype=np.float64)
    chosen_idx = int(np.argsort(metric_vals)[len(rows) // 2]) if len(rows) > 1 else 0
    chosen_seed = int(rows[chosen_idx]["seed"])
    ann_data, snn_data = trajectories[chosen_seed]
    value_ylim = _infer_value_ylim(env_id, ann_config, snn_config)
    metadata = _build_video_metadata(
        env_id=env_id,
        ann_cfg_path=ann_config_path,
        snn_cfg_path=snn_config_path,
        ann_ckpt=args.ann_ckpt,
        snn_ckpt=args.snn_ckpt,
        seed=chosen_seed,
        snn_critic_T=snn_critic_T,
    )
    print(f"\nRendering main comparison video (seed={chosen_seed})...")
    render_comparison_video(
        ann_data,
        snn_data,
        output_path=args.out,
        fps=args.fps,
        value_ylim=value_ylim,
        snn_critic_T=snn_critic_T,
        show_raw_critic_vout=args.show_raw_critic_vout,
        end_hold_frames=args.end_hold_frames,
        metadata=metadata,
        use_production_codec=True,
        crf=args.crf,
    )

    # Failure-case montage: best / median / worst seeds by action agreement.
    if len(rows) >= 3:
        rank = np.argsort(metric_vals)
        worst_seed = int(rows[int(rank[0])]["seed"])
        median_seed = int(rows[int(rank[len(rank) // 2])]["seed"])
        best_seed = int(rows[int(rank[-1])]["seed"])
        montage_out = args.montage_out
        if not montage_out:
            stem, ext = os.path.splitext(args.out)
            montage_out = f"{stem}_montage{ext or '.mp4'}"
        with tempfile.TemporaryDirectory(prefix="tmaze_triplet_", dir="/tmp") as td:
            triplet = [("worst", worst_seed), ("median", median_seed), ("best", best_seed)]
            triplet_videos: list[str] = []
            for tag, seed in triplet:
                a, s = trajectories[seed]
                p = os.path.join(td, f"{tag}_seed{seed}.mp4")
                meta = _build_video_metadata(
                    env_id=env_id,
                    ann_cfg_path=ann_config_path,
                    snn_cfg_path=snn_config_path,
                    ann_ckpt=args.ann_ckpt,
                    snn_ckpt=args.snn_ckpt,
                    seed=seed,
                    snn_critic_T=snn_critic_T,
                )
                meta["rank"] = tag
                render_comparison_video(
                    a,
                    s,
                    output_path=p,
                    fps=args.fps,
                    value_ylim=value_ylim,
                    snn_critic_T=snn_critic_T,
                    show_raw_critic_vout=args.show_raw_critic_vout,
                    end_hold_frames=args.end_hold_frames,
                    metadata=meta,
                    use_production_codec=False,
                    crf=args.crf,
                )
                triplet_videos.append(p)
            ok = _make_triplet_montage(videos=triplet_videos, out_path=montage_out, crf=args.crf)
            if ok:
                print(f"Saved best/median/worst montage to {montage_out}")
            else:
                print("[warn] Montage generation skipped (ffmpeg unavailable or montage failed).")


if __name__ == "__main__":
    main()



# single video + metrics
# python3 post_analysis/video_compare.py \
#   --config configs/tmaze/ann_baseline.yaml \
#   --ann_config configs/tmaze/ann_baseline.yaml \
#   --snn_config configs/tmaze/snn_actor_snntiming_critic.yaml \
#   --ann_ckpt results/logs/tmaze_ann/checkpoints/checkpoint_best.pt \
#   --snn_ckpt results/logs/tmaze_snn_timing_critic/checkpoints/checkpoint_best.pt \
#   --out results/videos/tmaze_comparison.mp4 \
#   --fps 2 \
#   --metrics_out results/post_analysis/tmaze_single_seed_metrics.csv


# multiple seeds + metrics

# python3 post_analysis/video_compare.py \
#   --config configs/tmaze/ann_baseline.yaml \
#   --ann_config configs/tmaze/ann_baseline.yaml \
#   --snn_config configs/tmaze/snn_actor_snntiming_critic.yaml \
#   --ann_ckpt results/logs/tmaze_ann/checkpoints/checkpoint_best.pt \
#   --snn_ckpt results/logs/tmaze_snn_timing_critic/checkpoints/checkpoint_best.pt \
#   --out results/videos/tmaze_comparison.mp4 \
#   --fps 2 \
#   --metrics_out results/post_analysis/tmaze_single_seed_metrics.csv


# single video + metrics with raw critic vout overlay
# python3 post_analysis/video_compare.py \
#   --config configs/tmaze/ann_baseline.yaml \
#   --ann_config configs/tmaze/ann_baseline.yaml \
#   --snn_config configs/tmaze/snn_actor_snntiming_critic.yaml \
#   --ann_ckpt results/logs/tmaze_ann/checkpoints/checkpoint_best.pt \
#   --snn_ckpt results/logs/tmaze_snn_timing_critic/checkpoints/checkpoint_best.pt \
#   --out results/videos/tmaze_comparison_raw_vout.mp4 \
#   --fps 2 \
#   --show_raw_critic_vout
