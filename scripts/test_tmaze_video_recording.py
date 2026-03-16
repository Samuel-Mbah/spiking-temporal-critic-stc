#!/usr/bin/env python3
"""Quick TMaze video-recording smoke test.

Usage:
  python3 scripts/test_tmaze_video_recording.py

Optional args:
  --out-dir results/logs/tmaze_ann/videos
  --name ANN_TMaze_video_test
  --seed 42
  --max-steps 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# Registers `tmaze-v0`
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
ENVS_DIR = SRC_DIR / "envs"
for p in (REPO_ROOT, SRC_DIR, ENVS_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)
try:
    import src.envs.t_maze  # noqa: F401
except ModuleNotFoundError:
    import t_maze  # noqa: F401


def run_test(out_dir: Path, name: str, seed: int, max_steps: int) -> list[Path]:
    target_dir = out_dir / name
    target_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("tmaze-v0", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=str(target_dir),
        episode_trigger=lambda ep: ep == 0,
        name_prefix=name,
    )

    try:
        obs, _ = env.reset(seed=seed)
        for _ in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()

    mp4_files = sorted(target_dir.glob("*.mp4"))
    if not mp4_files:
        raise RuntimeError(
            "Video smoke test failed: no .mp4 files were produced. "
            f"Checked: {target_dir}"
        )
    return mp4_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TMaze RecordVideo smoke test")
    parser.add_argument("--out-dir", type=Path, default=Path("results/logs/tmaze_ann/videos"))
    parser.add_argument("--name", type=str, default="ANN_TMaze_video_test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = run_test(args.out_dir, args.name, args.seed, args.max_steps)
    print("TMaze video recording test passed.")
    print(f"Output dir: {(args.out_dir / args.name).resolve()}")
    print("Videos:")
    for f in files:
        print(f"- {f.name}")


if __name__ == "__main__":
    main()
