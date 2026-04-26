"""Shared PPO config used by training and testing scripts.

Single source of truth for env kwargs, observation pipeline, oscillation
penalty, and model checkpoint paths. Edit values here and they propagate to:
  - models/turtlebot_ppo.py    (training)
  - testing/test_ppo_model.py  (eval with metrics)
  - testing/test_camera.py     (live visualisation)

Why this exists: silent train/eval mismatches on obs_lag, vx_min, or
smooth_steering produce policies that work in training and wiggle at test.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym

import gym_line_follower  # noqa: F401  registers TurtleBot3LineFollower-v0


ENV_ID = "TurtleBot3LineFollower-v0"
ENV_KWARGS: dict = dict(
    randomize=True,
    obsv_type="down_camera",
    vx_min=0.1,
    progress_reward=True,
    progress_reward_k=2.0,
    smooth_steering=False,           # use OscillationPenaltyWrapper instead
    domain_randomize_physics=False,
    sensor_noise=0.0,
    obs_lag=0,
)

# OscillationPenaltyWrapper coefficient. testing/metrics.py reads this (as its
# `smooth_steering_k` arg) so its reported steering penalty matches training.
OSCILLATION_PENALTY_K = 0.6

RESIZE_SHAPE = (84, 84)
FRAME_STACK_SIZE = 4

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
# Bump this when switching to a different checkpoint — both .zip and the
# matching VecNormalize .pkl are derived from it.
MODEL_STEPS = 2_500_000
MODEL_PATH = MODEL_DIR / f"ppo_turtlebot3_{MODEL_STEPS}_steps.zip"
VECNORM_PATH = MODEL_DIR / f"ppo_turtlebot3_vecnorm_{MODEL_STEPS}_steps.pkl"


def vecnorm_for(model_path: Path) -> Path:
    """Map a ppo_turtlebot3_<N>_steps.zip checkpoint to its sibling vecnorm .pkl."""
    return model_path.with_name(
        model_path.stem.replace("ppo_turtlebot3_", "ppo_turtlebot3_vecnorm_") + ".pkl"
    )


class OscillationPenaltyWrapper(gym.Wrapper):
    """Penalises rapid steering reversals; logs per-episode mean penalty."""

    def __init__(self, env, penalty_k: float = OSCILLATION_PENALTY_K):
        super().__init__(env)
        self._penalty_k = penalty_k

    def reset(self, **kwargs):
        self._prev_wz = 0.0
        self._ep_penalty_accum = 0.0
        self._ep_steps = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        wz = float(action[1])
        penalty = self._penalty_k * abs(wz - self._prev_wz)
        reward -= penalty
        self._prev_wz = wz
        self._ep_penalty_accum += penalty
        self._ep_steps += 1
        if terminated or truncated:
            info["mean_osc_penalty"] = self._ep_penalty_accum / max(self._ep_steps, 1)
        return obs, reward, terminated, truncated, info


def apply_observation_pipeline(env: gym.Env) -> gym.Env:
    # keep_dim=False so space stays 2D: (H,W) -> Resize (84,84) -> FrameStack (4,84,84).
    # With keep_dim=True, ResizeObservation's cv2.resize silently drops the C=1 dim
    # while the declared space keeps it, breaking FrameStackObservation's concat buffer.
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, shape=RESIZE_SHAPE)
    env = gym.wrappers.FrameStackObservation(env, stack_size=FRAME_STACK_SIZE)
    return env


def build_env(*, seed: int, gui: bool, with_oscillation_penalty: bool = True) -> gym.Env:
    """Build the env with all preprocessing, exactly as configured for PPO.

    `with_oscillation_penalty` only affects reward — actions and observations
    are unchanged either way. Default True (matches training); test scripts
    can pass False if they want raw env reward in their reports.
    """
    env = gym.make(ENV_ID, gui=gui, **ENV_KWARGS)
    env.reset(seed=seed)
    env = apply_observation_pipeline(env)
    if with_oscillation_penalty:
        env = OscillationPenaltyWrapper(env)
    return env
