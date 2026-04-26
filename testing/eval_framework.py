"""Reproducible evaluation framework for line-follower policies.

Runs a fixed set of pre-seeded tracks deterministically, computes a rich set of
thesis-grade metrics (task success, tracking precision, speed, smoothness,
reward decomposition), and returns aggregates suitable for TensorBoard logging.

Used by:
  - testing/eval_callback.py    (periodic in-training eval)
  - testing/test_ppo_model.py   (manual post-training eval)

Algorithm-agnostic: any object exposing ``model.predict(obs, deterministic=True)``
works (PPO, SAC, DDPG, TD3, ...).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecMonitor,
    VecNormalize,
    sync_envs_normalization,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ppo_runtime import build_env  # noqa: E402

from gym_line_follower.track import Track  # noqa: E402


# Single source of truth for the 10 deterministic eval tracks. Same seeds for
# every checkpoint and every algorithm — direct A/B comparison.
EVAL_TRACK_SEEDS: list[int] = list(range(1000, 1010))

# Track-error threshold (m) above which the bot is considered to have lost the
# line. Used both for line-loss event counting and recovery-time measurement.
LINE_LOST_THRESHOLD = 0.12

# Track.generate kwargs must match those used by LineFollowerEnv.reset() so
# eval tracks have the same shape/scale as training tracks.
_TRACK_GEN_KWARGS = dict(
    approx_width=1.75,
    hw_ratio=0.7,
    spikeyness=0.1,
    irregularity=0.3,
    nb_checkpoints=500,
)


# ---------------------------------------------------------------------------
# FixedTrackWrapper
# ---------------------------------------------------------------------------

class FixedTrackWrapper(gym.Wrapper):
    """Round-robins through a fixed list of track seeds, one per reset.

    Sets ``base_env.preset_track`` to a freshly generated Track before each
    reset so the env reproduces the same track shape every time it sees a given
    seed. The current seed is exposed via ``self.current_track_seed``.
    """

    def __init__(self, env: gym.Env, seeds: list[int]):
        super().__init__(env)
        if not seeds:
            raise ValueError("FixedTrackWrapper requires at least one seed.")
        self._seeds = list(seeds)
        self._idx = 0
        self.current_track_seed: int | None = None

    def reset(self, **kwargs):
        seed = self._seeds[self._idx % len(self._seeds)]
        self._idx += 1
        base_env = self.env.unwrapped
        base_env.preset_track = Track.generate(
            seed=seed,
            render_params=base_env.track_render_params,
            **_TRACK_GEN_KWARGS,
        )
        self.current_track_seed = seed
        return super().reset(**kwargs)


# ---------------------------------------------------------------------------
# Env construction
# ---------------------------------------------------------------------------

def _make_eval_env_factory(seeds: list[int], gui: bool):
    def _factory():
        env = build_env(
            seed=seeds[0],
            gui=gui,
            with_oscillation_penalty=True,
            env_kwargs_override={"randomize": False},
            eager_reset=False,
        )
        env = FixedTrackWrapper(env, seeds)
        return env
    return _factory


def build_eval_env(seeds: list[int] = EVAL_TRACK_SEEDS, *, gui: bool = False,
                   vecnorm_path: Path | None = None) -> VecNormalize:
    """Construct the persistent eval VecEnv.

    Parameters
    ----------
    seeds:
        Track seeds to round-robin through. Default = ``EVAL_TRACK_SEEDS``.
    gui:
        Open the PyBullet GUI window (slow, only for visual inspection).
    vecnorm_path:
        If provided, load VecNormalize stats from disk (manual eval path).
        Otherwise build fresh stats which the caller must sync from a training
        env via ``sync_envs_normalization`` before each eval (callback path).
    """
    venv = DummyVecEnv([_make_eval_env_factory(seeds, gui)])
    if vecnorm_path is not None:
        venv = VecNormalize.load(str(vecnorm_path), venv)
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=False)
    venv.training = False
    venv.norm_reward = False
    venv = VecMonitor(venv)
    return venv


def _get_base_env(venv) -> gym.Env:
    """Walk the SB3 VecWrapper chain to the underlying LineFollowerEnv."""
    v = venv
    while hasattr(v, "venv"):
        v = v.venv
    return v.envs[0].unwrapped


def _find_fixed_track_wrapper(venv) -> FixedTrackWrapper:
    """Locate the FixedTrackWrapper in the wrapper chain (single-env eval)."""
    v = venv
    while hasattr(v, "venv"):
        v = v.venv
    e = v.envs[0]
    while isinstance(e, gym.Wrapper):
        if isinstance(e, FixedTrackWrapper):
            return e
        e = e.env
    raise RuntimeError("FixedTrackWrapper not found in eval env chain.")


# ---------------------------------------------------------------------------
# Per-episode metrics
# ---------------------------------------------------------------------------

class RichEpisodeMetrics:
    """Collects per-step data and computes a thesis-grade metrics dict."""

    def __init__(self, line_lost_threshold: float = LINE_LOST_THRESHOLD) -> None:
        self._threshold = line_lost_threshold
        self.lateral_errors: list[float] = []
        self.forward_speeds: list[float] = []
        self.angular_vels: list[float] = []
        self.line_lost_periods: list[tuple[int, int]] = []
        self._line_lost = False
        self._line_lost_start = 0
        self.reward_components_sum: dict[str, float] = {}
        # filled by finalize()
        self.completed_distance_m = 0.0
        self.success = False
        self.total_reward = 0.0
        self.wall_time_s = 0.0
        self.termination = "unknown"
        self.track_seed: int | None = None

    def update(self, step_idx: int, track_err: float, forward_speed: float,
               angular_vel: float, reward_components: dict) -> None:
        self.lateral_errors.append(track_err)
        self.forward_speeds.append(forward_speed)
        self.angular_vels.append(angular_vel)
        for k, v in reward_components.items():
            self.reward_components_sum[k] = self.reward_components_sum.get(k, 0.0) + float(v)

        lost = track_err > self._threshold
        if lost and not self._line_lost:
            self._line_lost = True
            self._line_lost_start = step_idx
        elif not lost and self._line_lost:
            self._line_lost = False
            self.line_lost_periods.append((self._line_lost_start, step_idx))

    def finalize(self, *, completed_distance_m: float, success: bool,
                 total_reward: float, wall_time_s: float,
                 termination: str, track_seed: int) -> None:
        self.completed_distance_m = completed_distance_m
        self.success = success
        self.total_reward = total_reward
        self.wall_time_s = wall_time_s
        self.termination = termination
        self.track_seed = track_seed
        if self._line_lost:
            self.line_lost_periods.append((self._line_lost_start, len(self.lateral_errors)))

    def compute(self) -> dict:
        n = len(self.lateral_errors)
        if n == 0:
            return self._empty_metrics()

        abs_lat = [abs(e) for e in self.lateral_errors]
        lateral_mse = sum(e * e for e in self.lateral_errors) / n
        mean_abs_lateral = sum(abs_lat) / n
        max_lateral = max(abs_lat)

        mean_speed = sum(self.forward_speeds) / n
        mean_abs_wz = sum(abs(w) for w in self.angular_vels) / n

        if n > 1:
            mean_wz = sum(self.angular_vels) / n
            std_wz = math.sqrt(sum((w - mean_wz) ** 2 for w in self.angular_vels) / n)
            mean_angular_jerk = sum(
                abs(self.angular_vels[i] - self.angular_vels[i - 1]) for i in range(1, n)
            ) / (n - 1)
            mean_linear_jerk = sum(
                abs(self.forward_speeds[i] - self.forward_speeds[i - 1]) for i in range(1, n)
            ) / (n - 1)
            sign_changes = sum(
                1 for i in range(1, n)
                if self.angular_vels[i - 1] * self.angular_vels[i] < 0
            )
        else:
            std_wz = 0.0
            mean_angular_jerk = 0.0
            mean_linear_jerk = 0.0
            sign_changes = 0

        recovery_times = [end - start for start, end in self.line_lost_periods]
        mean_recovery = sum(recovery_times) / len(recovery_times) if recovery_times else None

        distance_per_step = self.completed_distance_m / n if n > 0 else 0.0

        return {
            "track_seed": self.track_seed,
            "termination": self.termination,
            "success": bool(self.success),
            "steps": n,
            "wall_time_s": round(self.wall_time_s, 3),
            "completed_distance_m": round(self.completed_distance_m, 4),
            "total_reward": round(self.total_reward, 4),
            # Precision
            "lateral_mse": round(lateral_mse, 6),
            "mean_abs_lateral_error": round(mean_abs_lateral, 5),
            "max_lateral_error": round(max_lateral, 5),
            # Speed
            "mean_forward_speed": round(mean_speed, 4),
            "distance_per_step": round(distance_per_step, 6),
            # Smoothness
            "mean_abs_wz": round(mean_abs_wz, 4),
            "std_wz": round(std_wz, 4),
            "mean_angular_jerk": round(mean_angular_jerk, 5),
            "mean_linear_jerk": round(mean_linear_jerk, 5),
            "wz_sign_changes": int(sign_changes),
            # Line loss
            "line_lost_count": len(self.line_lost_periods),
            "mean_recovery_time_steps": round(mean_recovery, 2) if mean_recovery is not None else None,
            # Reward decomposition (cumulative per-component over the episode)
            "reward_components": {k: round(v, 4) for k, v in self.reward_components_sum.items()},
        }

    def _empty_metrics(self) -> dict:
        return {
            "track_seed": self.track_seed,
            "termination": self.termination,
            "success": bool(self.success),
            "steps": 0,
            "wall_time_s": 0.0,
            "completed_distance_m": 0.0,
            "total_reward": 0.0,
            "lateral_mse": 0.0,
            "mean_abs_lateral_error": 0.0,
            "max_lateral_error": 0.0,
            "mean_forward_speed": 0.0,
            "distance_per_step": 0.0,
            "mean_abs_wz": 0.0,
            "std_wz": 0.0,
            "mean_angular_jerk": 0.0,
            "mean_linear_jerk": 0.0,
            "wz_sign_changes": 0,
            "line_lost_count": 0,
            "mean_recovery_time_steps": None,
            "reward_components": {},
        }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_TERMINATION_BUCKETS = ("completed", "penalty", "timeout", "line_lost")


def _mean(vals):
    return float(sum(vals) / len(vals)) if vals else 0.0


def _aggregate(per_episode: list[dict]) -> dict:
    if not per_episode:
        return {}
    n = len(per_episode)
    successes = [r for r in per_episode if r["success"]]
    completed_lap_times = [r["wall_time_s"] for r in per_episode if r["success"]]
    recoveries = [r["mean_recovery_time_steps"] for r in per_episode
                  if r["mean_recovery_time_steps"] is not None]

    term_counts = {b: 0 for b in _TERMINATION_BUCKETS}
    for r in per_episode:
        t = r["termination"]
        if t in term_counts:
            term_counts[t] += 1

    # Reward component means across episodes (any component a single ep didn't
    # see counts as 0 for that ep, since it's a sum).
    all_keys: set[str] = set()
    for r in per_episode:
        all_keys.update(r["reward_components"].keys())
    reward_means = {
        k: round(_mean([r["reward_components"].get(k, 0.0) for r in per_episode]), 4)
        for k in sorted(all_keys)
    }

    summary = {
        # Task success
        "success_rate": round(len(successes) / n, 4),
        "mean_completed_distance": round(_mean([r["completed_distance_m"] for r in per_episode]), 4),
        "term_completed_rate": round(term_counts["completed"] / n, 4),
        "term_penalty_rate": round(term_counts["penalty"] / n, 4),
        "term_timeout_rate": round(term_counts["timeout"] / n, 4),
        "term_line_lost_rate": round(term_counts["line_lost"] / n, 4),
        # Precision
        "mean_lateral_mse": round(_mean([r["lateral_mse"] for r in per_episode]), 6),
        "mean_abs_lateral_error": round(_mean([r["mean_abs_lateral_error"] for r in per_episode]), 5),
        "mean_max_lateral_error": round(_mean([r["max_lateral_error"] for r in per_episode]), 5),
        "mean_line_lost_count": round(_mean([r["line_lost_count"] for r in per_episode]), 3),
        "mean_recovery_time_steps": round(_mean(recoveries), 2) if recoveries else 0.0,
        # Speed
        "mean_forward_speed": round(_mean([r["mean_forward_speed"] for r in per_episode]), 4),
        "mean_lap_time_completed": round(_mean(completed_lap_times), 3) if completed_lap_times else 0.0,
        "mean_distance_per_step": round(_mean([r["distance_per_step"] for r in per_episode]), 6),
        # Smoothness
        "mean_abs_wz": round(_mean([r["mean_abs_wz"] for r in per_episode]), 4),
        "mean_std_wz": round(_mean([r["std_wz"] for r in per_episode]), 4),
        "mean_angular_jerk": round(_mean([r["mean_angular_jerk"] for r in per_episode]), 5),
        "mean_linear_jerk": round(_mean([r["mean_linear_jerk"] for r in per_episode]), 5),
        "mean_wz_sign_changes": round(_mean([r["wz_sign_changes"] for r in per_episode]), 2),
        # Reward
        "mean_total_reward": round(_mean([r["total_reward"] for r in per_episode]), 4),
    }
    for k, v in reward_means.items():
        summary[f"reward_component_{k}"] = v
    return summary


def _per_track_summary(per_episode: list[dict]) -> list[dict]:
    """Slim per-seed view so we can see *which* tracks fail."""
    return [
        {
            "seed": r["track_seed"],
            "success": r["success"],
            "termination": r["termination"],
            "completed_distance_m": r["completed_distance_m"],
            "lateral_mse": r["lateral_mse"],
            "max_lateral_error": r["max_lateral_error"],
            "total_reward": r["total_reward"],
            "steps": r["steps"],
        }
        for r in per_episode
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_evaluation(model, eval_venv, train_venv=None, *,
                   line_lost_threshold: float = LINE_LOST_THRESHOLD) -> dict:
    """Run one deterministic episode per seed in the wrapper's seed list.

    Parameters
    ----------
    model:
        Any object with ``model.predict(obs, deterministic=True) -> (action, _)``.
    eval_venv:
        VecEnv built by ``build_eval_env`` (single-env DummyVecEnv with
        FixedTrackWrapper inside and VecNormalize on top).
    train_venv:
        Optional training VecEnv. If provided, VecNormalize stats are synced
        from training before each eval (required when the eval venv was built
        without ``vecnorm_path``).
    line_lost_threshold:
        track_err (m) above which we consider the line lost.

    Returns a dict with ``summary`` (TB-loggable scalars), ``per_track`` (one
    row per seed), and ``per_episode`` (full RichEpisodeMetrics dicts).
    """
    if train_venv is not None:
        sync_envs_normalization(train_venv, eval_venv)

    fixed_wrapper = _find_fixed_track_wrapper(eval_venv)
    n_eps = len(fixed_wrapper._seeds)

    # Restart the round-robin so every eval starts at seeds[0]. Without this,
    # successive evals during training drift their first track because the
    # VecEnv auto-resets advanced the index past the end of the seed list on
    # the previous eval's last episode.
    fixed_wrapper._idx = 0
    obs = eval_venv.reset()  # consumes seeds[0]; auto-resets between episodes consume the rest

    per_episode: list[dict] = []
    for _ in range(n_eps):
        track_seed = fixed_wrapper.current_track_seed
        ep = RichEpisodeMetrics(line_lost_threshold)
        done = False
        step_idx = 0
        total_reward = 0.0
        last_progress = 0.0
        last_termination = "unknown"
        last_success = False
        t0 = time.perf_counter()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_venv.step(action)
            done = bool(dones[0])
            total_reward += float(reward[0])
            step_idx += 1

            info = infos[0]
            track_err = float(info.get("track_err", 0.0))
            forward_speed = float(info.get("forward_speed", 0.0))
            angular_vel = float(info.get("angular_vel", 0.0))
            reward_components = info.get("reward_components", {}) or {}
            ep.update(step_idx - 1, track_err, forward_speed, angular_vel, reward_components)

            if "progress" in info:
                last_progress = float(info["progress"])
            if done:
                last_termination = info.get("termination", "unknown")
                last_success = (last_termination == "completed")

        ep.finalize(
            completed_distance_m=last_progress,
            success=last_success,
            total_reward=total_reward,
            wall_time_s=time.perf_counter() - t0,
            termination=last_termination,
            track_seed=track_seed if track_seed is not None else -1,
        )
        per_episode.append(ep.compute())

    return {
        "summary": _aggregate(per_episode),
        "per_track": _per_track_summary(per_episode),
        "per_episode": per_episode,
    }
