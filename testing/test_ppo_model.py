import argparse
import sys
from pathlib import Path

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ppo_runtime import (  # noqa: E402
    MODEL_PATH,
    OSCILLATION_PENALTY_K,
    VECNORM_PATH,
    build_env,
    vecnorm_for,
)

from metrics import run_eval

LOG_DIR = Path(__file__).parent / "logs"


def main():
    parser = argparse.ArgumentParser(description="Test a trained PPO model on the TurtleBot3 line follower.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH,
                        help="Path to the trained .zip model file.")
    parser.add_argument("--vecnorm-path", type=Path, default=None,
                        help="Path to the matching VecNormalize stats. Auto-derived from --model-path if omitted.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gui", action="store_true", help="Disable PyBullet GUI window.")
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found at {args.model_path}. Train one first with models/turtlebot_ppo.py --train")

    vecnorm_path = args.vecnorm_path or (
        VECNORM_PATH if args.model_path == MODEL_PATH else vecnorm_for(args.model_path)
    )
    gui = not args.no_gui

    def _make() -> gym.Env:
        env = build_env(seed=args.seed, gui=gui, with_oscillation_penalty=False)
        env = Monitor(env)
        return env

    venv = DummyVecEnv([_make])
    venv = VecNormalize.load(str(vecnorm_path), venv)
    venv.training = False
    venv.norm_reward = False
    venv = VecMonitor(venv)

    model = PPO.load(str(args.model_path), env=venv)
    print(f"Loaded model from {args.model_path}")

    run_eval(
        venv,
        model,
        n_episodes=args.episodes,
        log_dir=LOG_DIR,
        prefix="ppo_turtlebot3",
        smooth_steering_k=OSCILLATION_PENALTY_K,
        model_path=str(args.model_path),
    )

    venv.close()


if __name__ == "__main__":
    main()
