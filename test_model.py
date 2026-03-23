from pathlib import Path

import gymnasium as gym
import gym_line_follower  # registers envs

from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage

MODEL_PATH = Path("models") / "ddpg_turtlebot3.zip"

def _make() -> gym.Env:
    env = gym.make(
        "TurtleBot3LineFollower-v0",
        gui=True,
        randomize=False,
        obsv_type="down_camera",
    )
    env.reset(seed=42)
    env = Monitor(env)
    return env

venv = DummyVecEnv([_make])
venv = VecTransposeImage(venv)
venv = VecMonitor(venv)

model = DDPG.load(str(MODEL_PATH), env=venv)

obs = venv.reset()
while True:
    action, _state = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = venv.step(action)
    if bool(dones[0]):
        obs = venv.reset()