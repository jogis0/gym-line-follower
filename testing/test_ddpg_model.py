import argparse
from pathlib import Path

import gymnasium as gym
import gym_line_follower  # registers envs

from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage

from metrics import run_eval

LOG_DIR = Path(__file__).parent / "logs"


def main():
    parser = argparse.ArgumentParser(description="Test a trained DDPG model on the TurtleBot3 line follower.")
    parser.add_argument("--model-path", type=Path, default=Path("models") / "ddpg_turtlebot3.zip",
                        help="Path to the trained .zip model file.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gui", action="store_true", help="Disable PyBullet GUI window.")
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found at {args.model_path}.")

    gui = not args.no_gui

    def _make() -> gym.Env:
        env = gym.make(
            "TurtleBot3LineFollower-v0",
            gui=gui,
            randomize=False,
            obsv_type="down_camera",
        )
        env.reset(seed=args.seed)
        env = Monitor(env)
        return env

    venv = DummyVecEnv([_make])
    venv = VecTransposeImage(venv)
    venv = VecMonitor(venv)

    model = DDPG.load(str(args.model_path), env=venv)
    print(f"Loaded model from {args.model_path}")

    run_eval(
        venv,
        model,
        n_episodes=args.episodes,
        log_dir=LOG_DIR,
        prefix="ddpg_turtlebot3",
        model_path=str(args.model_path),
    )

    venv.close()


if __name__ == "__main__":
    main()
