import argparse
from pathlib import Path

import gymnasium as gym
import gym_line_follower

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage


def main():
    parser = argparse.ArgumentParser(description="Test a trained PPO model on the TurtleBot3 line follower.")
    parser.add_argument("--model-path", type=Path, default=Path("models") / "ppo_turtlebot3.zip",
                        help="Path to the trained .zip model file.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gui", action="store_true", help="Disable PyBullet GUI window.")
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found at {args.model_path}. Train one first with models/turtlebot_ppo.py --train")

    gui = not args.no_gui

    def _make() -> gym.Env:
        env = gym.make(
            "TurtleBot3LineFollower-v0",
            gui=gui,
            randomize=True,
            obsv_type="down_camera", #Testuojant ijungti smooth_steering, progress_reward, domain_randomize_physics, kurie aprasyti @line_follower_env.py
        )
        env = Monitor(env)
        return env

    venv = DummyVecEnv([_make])
    venv = VecTransposeImage(venv)
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
