from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

import gym_line_follower  # registers LineFollower-v0

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


def make_env(*, seed: int, gui: bool) -> gym.Env:
    env = gym.make(
        "LineFollower-v0",
        gui=gui,
        obsv_type="points_latch",
        randomize=True,
    )
    env.reset(seed=seed)

    # This env historically returned lists; SB3 works best with numpy arrays.
    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = gym.spaces.Box(
            low=np.asarray(obs_space.low, dtype=np.float32),
            high=np.asarray(obs_space.high, dtype=np.float32),
            shape=obs_space.shape,
            dtype=np.float32,
        )
    env = gym.wrappers.TransformObservation(
        env,
        lambda obs: np.asarray(obs, dtype=np.float32),
        observation_space=obs_space,
    )
    env = Monitor(env)
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train a new policy.")
    parser.add_argument("--eval", action="store_true", help="Evaluate an existing policy.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI window.")
    parser.add_argument("--model-path", type=Path, default=Path("model_files") / "sac_line_follower.zip")
    args = parser.parse_args()

    if not args.train and not args.eval:
        parser.error("Choose one: --train or --eval")

    # Vectorized env is the standard SB3 interface
    venv = DummyVecEnv([lambda: make_env(seed=args.seed, gui=args.gui)])
    venv = VecMonitor(venv)

    if args.train:
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        model = SAC(
            "MlpPolicy",
            venv,
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(Path("logs") / "sac_line_follower"),
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
        model.save(str(args.model_path))

    if args.eval:
        model = SAC.load(str(args.model_path), env=venv)
        obs = venv.reset()
        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv.step(action)
            if bool(dones[0]):
                obs = venv.reset()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
