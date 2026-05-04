"""Discrete steering wrapper over the cmd_vel action space.

Maps a small set of named steering choices ("hard_left" .. "hard_right") to
the underlying continuous (vx, wz) action expected by
``TurtleBot3LineFollowerEnv``. Used to turn the env into a discrete-action
problem so value-based methods like DQN can be applied.

vx is held constant at a cruise speed inside the env's [vx_min, vx_limit]
range; only wz varies across actions.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# (vx [m/s], wz [rad/s]) per discrete action index. wz spans the full
# turtlebot3 limit (+-2.84 rad/s). vx=0.15 sits above the vx_min=0.1 floor in
# ENV_KWARGS and below the 0.22 cap from turtlebot3_burger_config.json.
ACTION_TABLE: list[tuple[float, float]] = [
    (0.15, +2.84),  # 0: hard_left
    (0.15, +1.90),  # 1: turn_left
    (0.15, +0.95),  # 2: slight_left
    (0.15,  0.00),  # 3: straight
    (0.15, -0.95),  # 4: slight_right
    (0.15, -1.90),  # 5: turn_right
    (0.15, -2.84),  # 6: hard_right
]

ACTION_NAMES: list[str] = [
    "hard_left",
    "turn_left",
    "slight_left",
    "straight",
    "slight_right",
    "turn_right",
    "hard_right",
]


class DiscreteCmdVelWrapper(gym.ActionWrapper):
    """Discrete(N) -> Box(2) action translator for cmd_vel envs."""

    def __init__(self, env: gym.Env,
                 action_table: list[tuple[float, float]] = ACTION_TABLE):
        super().__init__(env)
        self._table = np.asarray(action_table, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self._table))

    def action(self, action):
        return self._table[int(action)].copy()
