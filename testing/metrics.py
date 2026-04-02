"""
Reusable evaluation metric tracking for gym-line-follower test scripts.

Usage in any SB3 test script::

    from metrics import run_eval

    run_eval(
        venv, model, n_episodes=5,
        log_dir=Path(__file__).parent / "logs",
        prefix="ppo_turtlebot3",
        smooth_steering_k=0.05,
    )

Works with any SB3 algorithm (PPO, DDPG, SAC, TD3, …) and any wrapper chain
built on top of DummyVecEnv.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

class _EpisodeMetrics:
    """Collects per-step data for one episode."""

    def __init__(self, line_lost_threshold: float) -> None:
        self._threshold = line_lost_threshold
        self.lateral_errors: list[float] = []
        self.forward_speeds: list[float] = []
        self.angular_vels: list[float] = []
        self.line_lost_periods: list[tuple[int, int]] = []
        self._line_lost = False
        self._line_lost_start = 0
        # set by finalize()
        self.driven_distance_m = 0.0
        self.success = False
        self.total_reward = 0.0
        self.wall_time_s = 0.0

    def update(
        self,
        step_idx: int,
        track_err: float,
        forward_speed: float,
        angular_vel: float,
    ) -> None:
        self.lateral_errors.append(track_err)
        self.forward_speeds.append(forward_speed)
        self.angular_vels.append(angular_vel)

        lost = track_err > self._threshold
        if lost and not self._line_lost:
            self._line_lost = True
            self._line_lost_start = step_idx
        elif not lost and self._line_lost:
            self._line_lost = False
            self.line_lost_periods.append((self._line_lost_start, step_idx))

    def finalize(
        self,
        driven_distance_m: float,
        success: bool,
        total_reward: float,
        wall_time_s: float,
    ) -> None:
        self.driven_distance_m = driven_distance_m
        self.success = success
        self.total_reward = total_reward
        self.wall_time_s = wall_time_s
        # close any open line-lost period
        if self._line_lost:
            n = len(self.lateral_errors)
            self.line_lost_periods.append((self._line_lost_start, n))

    def compute(self, smooth_steering_k: float) -> dict:
        n = len(self.lateral_errors)

        lateral_mse = (
            sum(e * e for e in self.lateral_errors) / n if n > 0 else 0.0
        )

        steering_penalty = 0.0
        for i in range(1, len(self.angular_vels)):
            steering_penalty += smooth_steering_k * abs(
                self.angular_vels[i] - self.angular_vels[i - 1]
            )

        mean_speed = sum(self.forward_speeds) / n if n > 0 else 0.0
        if n > 1:
            variance = sum((v - mean_speed) ** 2 for v in self.forward_speeds) / n
            vel_std = math.sqrt(variance)
        else:
            vel_std = 0.0

        recovery_times = [end - start for start, end in self.line_lost_periods]
        mean_recovery = (
            sum(recovery_times) / len(recovery_times) if recovery_times else None
        )

        return {
            "steps": n,
            "total_reward": round(self.total_reward, 4),
            "wall_time_s": round(self.wall_time_s, 3),
            "driven_distance_m": round(self.driven_distance_m, 4),
            "lateral_mse": round(lateral_mse, 6),
            "success": self.success,
            "line_lost_count": len(self.line_lost_periods),
            "recovery_times_steps": recovery_times,
            "mean_recovery_time_steps": (
                round(mean_recovery, 2) if mean_recovery is not None else None
            ),
            "total_steering_penalty": round(steering_penalty, 4),
            "mean_forward_speed_ms": round(mean_speed, 4),
            "velocity_std_ms": round(vel_std, 4),
        }


def _get_base_env(venv):
    """Traverse any SB3 VecWrapper chain and return the base LineFollowerEnv.

    SB3 VecWrapper subclasses (VecMonitor, VecNormalize, VecTransposeImage, …)
    all expose a `.venv` attribute.  DummyVecEnv/SubprocVecEnv do not, so the
    loop stops there.
    """
    v = venv
    while hasattr(v, "venv"):
        v = v.venv
    return v.envs[0].unwrapped


def _print_episode(episode: int, n_episodes: int, r: dict) -> None:
    recovery = r["mean_recovery_time_steps"]
    recovery_str = f"{recovery:.1f}" if recovery is not None else "n/a"
    print(
        f"Episode {episode}/{n_episodes} — "
        f"steps: {r['steps']}, reward: {r['total_reward']:.2f}, "
        f"time: {r['wall_time_s']:.1f}s"
    )
    print(
        f"  Performance : distance={r['driven_distance_m']:.3f} m"
    )
    print(
        f"  Precision   : lateral_MSE={r['lateral_mse']:.5f}, "
        f"success={r['success']}, "
        f"lost_line={r['line_lost_count']}x, "
        f"mean_recovery={recovery_str} steps"
    )
    print(
        f"  Smoothness  : steering_penalty={r['total_steering_penalty']:.3f}, "
        f"vel={r['mean_forward_speed_ms']:.3f}±{r['velocity_std_ms']:.3f} m/s"
    )


def _compute_summary(results: list[dict], model_path: str | None) -> dict:
    n = len(results)
    successes = [r for r in results if r["success"]]
    recoveries = [
        r["mean_recovery_time_steps"]
        for r in results
        if r["mean_recovery_time_steps"] is not None
    ]

    def mean(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    print("\n--- Summary ---")
    print(f"Success rate : {len(successes)}/{n} = {len(successes)/n:.0%}")
    print(f"Mean distance: {mean([r['driven_distance_m'] for r in results]):.3f} m")
    print(f"Mean MSE     : {mean([r['lateral_mse'] for r in results]):.5f}")
    print(f"Mean vel std : {mean([r['velocity_std_ms'] for r in results]):.4f} m/s")
    if recoveries:
        print(f"Mean recovery: {mean(recoveries):.1f} steps")

    return {
        "model_path": model_path,
        "total_episodes": n,
        "success_rate": round(len(successes) / n, 4),
        "mean_steps": mean([r["steps"] for r in results]),
        "mean_driven_distance_m": mean([r["driven_distance_m"] for r in results]),
        "mean_lateral_mse": mean([r["lateral_mse"] for r in results]),
        "mean_total_steering_penalty": mean(
            [r["total_steering_penalty"] for r in results]
        ),
        "mean_forward_speed_ms": mean([r["mean_forward_speed_ms"] for r in results]),
        "mean_velocity_std_ms": mean([r["velocity_std_ms"] for r in results]),
        "mean_recovery_time_steps": mean(recoveries) if recoveries else None,
    }


def _save_log(
    results: list[dict],
    summary: dict,
    log_dir: Path,
    prefix: str,
    model_path: str | None,
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(log_dir.glob(f"{prefix}_*.json"))
    run_num = len(existing) + 1
    log_path = log_dir / f"{prefix}_{run_num:03d}.json"

    payload = {
        "run": run_num,
        "prefix": prefix,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_path": model_path,
        "episodes": [{"episode": i + 1, **r} for i, r in enumerate(results)],
        "summary": summary,
    }
    with open(log_path, "w") as f:
        json.dump(payload, f, indent=2)
    return log_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_eval(
    venv,
    model,
    n_episodes: int,
    log_dir: Path | None = None,
    prefix: str = "eval",
    smooth_steering_k: float = 0.05,
    line_lost_threshold: float = 0.12,
    model_path: str | None = None,
) -> list[dict]:
    """Run *n_episodes* of deterministic evaluation and collect metrics.

    Parameters
    ----------
    venv:
        Any SB3 VecEnv wrapper chain wrapping a single ``DummyVecEnv``.
    model:
        SB3 model with a ``.predict(obs, deterministic=True)`` method.
    n_episodes:
        Number of episodes to evaluate.
    log_dir:
        Directory for JSON log files. Pass ``None`` to skip saving.
    prefix:
        Filename prefix used for the numbered log file, e.g. ``"ppo_turtlebot3"``
        produces ``ppo_turtlebot3_001.json``.
    smooth_steering_k:
        Coefficient used to compute the steering penalty metric.  Should match
        the value used during training.
    line_lost_threshold:
        ``track_err`` (metres) above which the robot is considered to have lost
        the line.  Default is 0.12 m (~40 % of ``max_track_err``).
    model_path:
        Optional string stored in the log for traceability.

    Returns
    -------
    list[dict]
        One dict of metrics per episode (same dicts printed to the terminal).
    """
    base_env = _get_base_env(venv)
    all_results: list[dict] = []

    for episode in range(1, n_episodes + 1):
        obs = venv.reset()
        ep = _EpisodeMetrics(line_lost_threshold)
        done = False
        step_idx = 0
        total_reward = 0.0
        t0 = time.perf_counter()

        last_progress = 0.0
        success = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            done = bool(dones[0])
            total_reward += float(reward[0])
            step_idx += 1

            if not done:
                # base_env is still in the current episode — safe to read
                track_err = float(
                    base_env.track.distance_from_point(base_env.follower_bot.pos[0])
                )
                vel = base_env.follower_bot.vel  # ((vx, vy), wz)
                forward_speed = math.sqrt(vel[0][0] ** 2 + vel[0][1] ** 2)
                ep.update(step_idx - 1, track_err, forward_speed, float(vel[1]))
                last_progress = float(base_env.track.progress)
            else:
                # DummyVecEnv auto-reset already happened inside venv.step(); infer
                # success from the terminal signals rather than base_env state.
                is_truncated = bool(infos[0].get("TimeLimit.truncated", False))
                is_penalty = float(reward[0]) < -50  # env returns -100 on track-error/divergence
                success = not is_truncated and not is_penalty

        ep.finalize(
            driven_distance_m=last_progress,  # from step N-1, < 1 step lag (~7 mm max)
            success=success,
            total_reward=total_reward,
            wall_time_s=time.perf_counter() - t0,
        )
        result = ep.compute(smooth_steering_k)
        all_results.append(result)
        _print_episode(episode, n_episodes, result)

    summary = _compute_summary(all_results, model_path)

    if log_dir is not None:
        log_path = _save_log(all_results, summary, Path(log_dir), prefix, model_path)
        print(f"Metrics saved → {log_path}")

    return all_results
