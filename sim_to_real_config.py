"""Sim-to-real env-kwargs overrides shared by SAC/PPO/DQN training + test scripts.

Single source of truth for the values flipped on by the ``--sim-to-real`` CLI
flag. Each entry corresponds to a kwarg already accepted by ``LineFollowerEnv``
that is OFF in the runtime ``ENV_KWARGS`` dicts.
"""
from __future__ import annotations


SIM_TO_REAL_OVERRIDES: dict = dict(
    domain_randomize_physics=True,
    sensor_noise=5.0,
    obs_lag=2,
)
