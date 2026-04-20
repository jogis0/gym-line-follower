import argparse
from pathlib import Path

import gymnasium as gym
import gym_line_follower

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from metrics import run_eval

LOG_DIR = Path(__file__).parent / "logs"
SMOOTH_STEERING_K = 0.05  # must match the value used during training


def main():
    parser = argparse.ArgumentParser(description="Test a trained PPO model on the TurtleBot3 line follower.")
    parser.add_argument("--model-path", type=Path, default=Path("models") / "ppo_turtlebot3_400000_steps.zip",
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
            obsv_type="down_camera",
            vx_min=0.0,
            progress_reward=True,
            progress_reward_k=2.0,
            smooth_steering=True,
            smooth_steering_k=SMOOTH_STEERING_K,
        )
        # Grayscale must match training (keep_dim=True → shape (H, W, 1) uint8).
        # No brightness augmentation at test time.
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
        env = Monitor(env)
        return env

    venv = DummyVecEnv([_make])
    venv = VecMonitor(venv)

    model = PPO.load(str(args.model_path), env=venv)
    print(f"Loaded model from {args.model_path}")

    run_eval(
        venv,
        model,
        n_episodes=args.episodes,
        log_dir=LOG_DIR,
        prefix="ppo_turtlebot3",
        smooth_steering_k=SMOOTH_STEERING_K,
        model_path=str(args.model_path),
    )

    venv.close()


if __name__ == "__main__":
    main()
