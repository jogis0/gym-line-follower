"""Inference-parity replay in sim: feed sim camera frames through the exact
inference_node.py path (preprocess_bgr -> 4-frame deque -> vecnorm.normalize_obs
-> model.predict) and dump the 84x84 grayscale frames the policy actually sees.

Why this script exists: if the policy completes a deterministic eval seed via
this code path, the inference path is byte-equivalent to training-eval and any
real-robot failure is a sim-to-real domain-gap problem (camera framing,
lighting, line contrast). If parity fails, the inference pipeline diverges
somewhere — most likely the BGR-vs-RGB grayscale-coefficient mismatch.

Run with the sim available (PyBullet headless is fine; pass --gui to watch):
    python testing/dump_sim_frames.py --dump-dir /tmp/lf_diag/sim --seed 1000

Outputs:
    <dump_dir>/sim_<idx:03>.png        — single 84x84 grayscale frame
    <dump_dir>/sim_stack_<idx:03>.png  — 4-frame deque tiled horizontally
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

# Make the ROS-side gymnasium_line_follower package importable so we can reuse
# preprocess_bgr, the numpy-compat model loader, and the action bounds from the
# same source the inference node uses. Layout:
#   ros2_ws/src/gymnasium_line_follower/                 ROS package
#     gymnasium_line_follower/inference_node.py          Python module
#     gym-line-follower/testing/dump_sim_frames.py       this script
_PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PKG_ROOT))

from gymnasium_line_follower.inference_node import (  # noqa: E402
    IMG_H,
    IMG_W,
    STACK_SIZE,
    VX_LIMIT,
    WZ_LIMIT,
    TurtleBot3InferenceNode,
    preprocess_bgr,
)

# Also need ppo_runtime + eval_framework (both live under gym-line-follower/).
_GYM_LF_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_GYM_LF_ROOT))
sys.path.insert(0, str(_GYM_LF_ROOT / "testing"))

import gym_line_follower  # noqa: E402, F401  — registers TurtleBot3LineFollower-v0
from ppo_runtime import (  # noqa: E402
    ENV_ID,
    ENV_KWARGS,
    DEFAULT_SEED,
    MODEL_STEPS,
    model_path_for,
    vecnorm_for,
)
from eval_framework import FixedTrackWrapper  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    default_model = model_path_for(DEFAULT_SEED, MODEL_STEPS)
    parser.add_argument("--model-path", type=Path, default=default_model)
    parser.add_argument("--vecnorm-path", type=Path, default=None,
                        help="Defaults to the sibling .pkl of --model-path.")
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000,
                        help="Eval track seed (default 1000 — first of EVAL_TRACK_SEEDS).")
    parser.add_argument("--steps", type=int, default=1500,
                        help="Max steps before giving up (env truncates earlier on timeout).")
    parser.add_argument("--dump-count", type=int, default=30)
    parser.add_argument("--dump-every", type=int, default=5)
    parser.add_argument("--gui", action="store_true",
                        help="Open the PyBullet GUI window (slow; headless by default).")
    return parser.parse_args()


def build_raw_env(seed: int, gui: bool) -> gym.Env:
    """Sim env that returns raw (h, w, 3) RGB frames — no observation pipeline.

    We deliberately skip apply_observation_pipeline so the parity script
    exercises preprocess_bgr exactly the way inference_node.py does on the
    real robot.
    """
    overrides = {**ENV_KWARGS, "randomize": False}
    env = gym.make(ENV_ID, gui=gui, **overrides)
    env = FixedTrackWrapper(env, [seed])
    return env


def main() -> int:
    args = parse_args()

    vecnorm_path = args.vecnorm_path or vecnorm_for(args.model_path)
    if not args.model_path.exists():
        raise SystemExit(f"Model not found: {args.model_path}")
    if not vecnorm_path.exists():
        raise SystemExit(f"VecNormalize stats not found: {vecnorm_path}")

    args.dump_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    print(f"Loading vecnorm: {vecnorm_path}")
    model, vecnorm = TurtleBot3InferenceNode._load_model(args.model_path, vecnorm_path)

    print(f"Building sim env (seed={args.seed}, gui={args.gui})")
    env = build_raw_env(args.seed, args.gui)
    raw_obs, _info = env.reset(seed=args.seed)

    # raw_obs: (h, w, 3) uint8 RGB. Mirror inference's BGR pipeline by feeding
    # the BGR-flipped frame into preprocess_bgr.
    def _to_gray(rgb: np.ndarray) -> np.ndarray:
        return preprocess_bgr(rgb[:, :, ::-1])

    frame_stack: deque = deque(maxlen=STACK_SIZE)
    first = _to_gray(raw_obs)
    for _ in range(STACK_SIZE):
        frame_stack.append(first)

    track_errs: list[float] = []
    progresses: list[float] = []
    raw_vxs: list[float] = []
    raw_wzs: list[float] = []
    pub_vxs: list[float] = []
    pub_wzs: list[float] = []
    dumped = 0
    termination = "max_steps"

    for step in range(args.steps):
        obs = np.stack(list(frame_stack), axis=0)[np.newaxis]  # (1,4,84,84)

        if dumped < args.dump_count and step % args.dump_every == 0:
            idx = dumped
            cv2.imwrite(str(args.dump_dir / f"sim_{idx:03d}.png"), frame_stack[-1])
            tiled = np.hstack(list(frame_stack))
            cv2.imwrite(str(args.dump_dir / f"sim_stack_{idx:03d}.png"), tiled)
            dumped += 1

        obs_n = vecnorm.normalize_obs(obs)
        action, _ = model.predict(obs_n, deterministic=True)
        raw_vx = float(action[0][0])
        raw_wz = float(action[0][1])

        # Same clip the inference node applies before publishing Twist.
        vx = float(np.clip(raw_vx, 0.10, VX_LIMIT))
        wz = float(np.clip(raw_wz, -WZ_LIMIT, WZ_LIMIT))

        raw_vxs.append(raw_vx)
        raw_wzs.append(raw_wz)
        pub_vxs.append(vx)
        pub_wzs.append(wz)

        raw_obs, _reward, terminated, truncated, info = env.step(np.array([vx, wz], dtype=np.float32))
        frame_stack.append(_to_gray(raw_obs))

        track_errs.append(float(info.get("track_err", float("nan"))))
        progresses.append(float(info.get("progress", float("nan"))))

        if terminated or truncated:
            termination = info.get("termination", "terminated" if terminated else "truncated")
            break

    env.close()

    track_errs_arr = np.array(track_errs, dtype=np.float64)
    print()
    print(f"Episode finished: termination={termination}, steps={len(track_errs)}")
    print(f"  final_progress       = {progresses[-1] if progresses else float('nan'):.3f}")
    print(f"  mean_abs_track_err   = {np.mean(np.abs(track_errs_arr)):.4f} m")
    print(f"  max_abs_track_err    = {np.max(np.abs(track_errs_arr)):.4f} m")
    print(f"  raw vx  mean={np.mean(raw_vxs):+.3f}  min={np.min(raw_vxs):+.3f}  max={np.max(raw_vxs):+.3f}")
    print(f"  raw wz  mean={np.mean(raw_wzs):+.3f}  min={np.min(raw_wzs):+.3f}  max={np.max(raw_wzs):+.3f}")
    print(f"  pub vx  mean={np.mean(pub_vxs):+.3f}  min={np.min(pub_vxs):+.3f}  max={np.max(pub_vxs):+.3f}")
    print(f"  pub wz  mean={np.mean(pub_wzs):+.3f}  min={np.min(pub_wzs):+.3f}  max={np.max(pub_wzs):+.3f}")
    print(f"  dumped {dumped} frames to {args.dump_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
