import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import gym_line_follower

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize


def main():
    parser = argparse.ArgumentParser(description="Test a trained PPO model on the TurtleBot3 line follower.")
    parser.add_argument("--model-path", type=Path, default=Path("models") / "ppo_turtlebot3_200000_steps.zip",
                        help="Path to the trained .zip model file.")
    parser.add_argument("--vecnorm-path", type=Path, default=Path("models") / "ppo_turtlebot3_vecnorm.pkl",
                        help="Path to the VecNormalize statistics .pkl file.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gui", action="store_true", help="Disable PyBullet GUI window.")
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found at {args.model_path}. Train one first with models/turtlebot_ppo.py --train")
    if not args.vecnorm_path.exists():
        raise FileNotFoundError(f"VecNormalize stats not found at {args.vecnorm_path}.")

    gui = not args.no_gui

    def _make() -> gym.Env:
        env = gym.make(
            "TurtleBot3LineFollower-v0",
            gui=gui,
            randomize=True,
            obsv_type="points_latch",
            progress_reward=True,
            progress_reward_k=2.0,
            smooth_steering=True,
            smooth_steering_k=0.05,
        )
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

    venv = DummyVecEnv([_make])
    venv = VecNormalize.load(str(args.vecnorm_path), venv)
    venv.training = False
    venv.norm_reward = False
    venv = VecMonitor(venv)

    model = PPO.load(str(args.model_path), env=venv)
    print(f"Loaded model from {args.model_path}")

    for episode in range(1, args.episodes + 1):
        obs = venv.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            total_reward += float(reward[0])
            steps += 1
            done = bool(dones[0])

        ep_info = infos[0].get("episode", {})
        print(f"Episode {episode}/{args.episodes} — steps: {steps}, reward: {total_reward:.2f}"
              + (f", ep_len: {ep_info.get('l', '?')}" if ep_info else ""))

    venv.close()


if __name__ == "__main__":
    main()
