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
        obsv_type="down_camera",
        vx_min=0.0,
        progress_reward=True,
        progress_reward_k=2.0,
        smooth_steering=True,
        smooth_steering_k=0.05,
        domain_randomize_physics=True,
        sensor_noise=0.0,
    )
    env.reset(seed=seed)

    # Convert RGB (H, W, 3) -> grayscale (H, W, 1) uint8.
    # keep_dim=True preserves the channel axis; SB3 CnnPolicy requires 3-D input.
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)

    # Brightness jitter to bridge sim-to-real lighting gap (training only).
    aug_space = env.observation_space
    env = gym.wrappers.TransformObservation(
        env,
        lambda obs: np.clip(
            (obs.astype(np.float32) * np.random.uniform(0.7, 1.4)).astype(np.int16),
            0, 255,
        ).astype(np.uint8),
        observation_space=aug_space,
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
    parser.add_argument("--model-path", type=Path, default=Path("models") / "ppo_turtlebot3_300000_steps.zip")
    args = parser.parse_args()

    if not args.train and not args.eval:
        parser.error("Choose one: --train or --eval")

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    def _make() -> gym.Env:
        env = make_env(seed=args.seed, gui=args.gui)
        env = Monitor(env)
        return env

    venv = DummyVecEnv([_make] * 4)
    venv = VecMonitor(venv)

    if args.train:
        args.model_path.parent.mkdir(parents=True, exist_ok=True)

        model = PPO(
            policy="CnnPolicy",
            env=venv,
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(Path("logs") / "ppo_turtlebot3"),
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            ent_coef=0.01,
            learning_rate=lambda f: 3e-4 * f,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000,
            save_path=str(args.model_path.parent),
            name_prefix="ppo_turtlebot3",
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=checkpoint_callback)
        model.save(str(args.model_path))

    if args.eval:
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
