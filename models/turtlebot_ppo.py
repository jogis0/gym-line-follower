from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

import gym_line_follower  # registers envs


def make_env(*, seed: int, gui: bool) -> gym.Env:
    env = gym.make(
        "TurtleBot3LineFollower-v0",
        gui=gui,
        randomize=True,
        obsv_type="points_latch",
        progress_reward=True,
        progress_reward_k=2.0,
        smooth_steering=True,
        smooth_steering_k=0.05,
        domain_randomize_physics=True,
        sensor_noise=0.005,
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
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train a new PPO policy.")
    parser.add_argument("--eval", action="store_true", help="Evaluate an existing PPO policy.")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI window.")
    parser.add_argument("--model-path", type=Path, default=Path("models") / "ppo_turtlebot3_200000_steps.zip")
    args = parser.parse_args()

    if not args.train and not args.eval:
        parser.error("Choose one: --train or --eval")

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

    vecnorm_path = args.model_path.parent / "ppo_turtlebot3_vecnorm.pkl"

    def _make() -> gym.Env:
        env = make_env(seed=args.seed, gui=args.gui)
        env = Monitor(env)
        return env

    venv = DummyVecEnv([_make])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    venv = VecMonitor(venv)

    if args.train:
        args.model_path.parent.mkdir(parents=True, exist_ok=True)

        model = PPO(
            policy="MlpPolicy",
            env=venv,
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(Path("logs") / "ppo_turtlebot3"),
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            ent_coef=0.01,
            learning_rate=lambda f: 3e-4 * f,
        )
        checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path=str(args.model_path.parent), name_prefix="ppo_turtlebot3")
        model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=checkpoint_callback)
        model.save(str(args.model_path))
        venv.save(str(vecnorm_path))

    if args.eval:
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training = False
        venv.norm_reward = False
        model = PPO.load(str(args.model_path), env=venv)
        obs = venv.reset()
        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv.step(action)
            if bool(dones[0]):
                obs = venv.reset()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
