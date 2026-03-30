from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym

import gym_line_follower  # registers envs


def make_env(*, seed: int, gui: bool) -> gym.Env:
    # Downward-facing camera + cmd_vel action space
    env = gym.make(
        "TurtleBot3LineFollower-v0",
        gui=gui,
        randomize=True,
        obsv_type="down_camera",
        progress_reward=True,
        progress_reward_k=1.0,
        smooth_steering=True,
        smooth_steering_k=0.05,
    )
    env.reset(seed=seed)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train a new PPO policy.")
    parser.add_argument("--eval", action="store_true", help="Evaluate an existing PPO policy.")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI window.")
    parser.add_argument("--model-path", type=Path, default=Path("models") / "ppo_turtlebot3.zip")
    args = parser.parse_args()

    if not args.train and not args.eval:
        parser.error("Choose one: --train or --eval")

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage

    def _make() -> gym.Env:
        env = make_env(seed=args.seed, gui=args.gui)
        env = Monitor(env)
        return env

    venv = DummyVecEnv([_make])
    venv = VecTransposeImage(venv)  # HWC uint8 -> CHW for CnnPolicy
    venv = VecMonitor(venv)

    if args.train:
        args.model_path.parent.mkdir(parents=True, exist_ok=True)

        model = PPO(
            policy="CnnPolicy",
            env=venv,
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(Path("logs") / "ppo_turtlebot3"),
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
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
