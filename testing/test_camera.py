import argparse
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ppo_runtime import MODEL_PATH, VECNORM_PATH, build_env, vecnorm_for  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run a trained PPO model with live camera display.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--vecnorm-path", type=Path, default=None,
                        help="Path to the matching VecNormalize stats. Auto-derived from --model-path if omitted.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gui", action="store_true", help="Disable PyBullet GUI window.")
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found at {args.model_path}.")

    vecnorm_path = args.vecnorm_path or (
        VECNORM_PATH if args.model_path == MODEL_PATH else vecnorm_for(args.model_path)
    )
    gui = not args.no_gui

    def _make() -> gym.Env:
        env = build_env(seed=args.seed, gui=gui, with_oscillation_penalty=False)
        env = Monitor(env)
        return env

    venv = DummyVecEnv([_make])
    if vecnorm_path.exists():
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training = False
        venv.norm_reward = False
    venv = VecMonitor(venv)

    model = PPO.load(str(args.model_path), env=venv)
    print(f"Loaded model from {args.model_path}")

    # Access the unwrapped base env for raw camera images.
    base_env = venv.envs[0].unwrapped

    obs = venv.reset()

    plt.ion()
    fig, (ax_down, ax_pov) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Robot Camera Feed")

    down_img = base_env.follower_bot.get_down_camera_image()
    pov_img = base_env.follower_bot.get_pov_image()

    im_down = ax_down.imshow(down_img)
    ax_down.set_title("Down Camera")
    ax_down.axis("off")

    im_pov = ax_pov.imshow(pov_img)
    ax_pov.set_title("Forward POV")
    ax_pov.axis("off")

    plt.tight_layout()
    plt.pause(0.001)

    step = 0
    total_reward = 0.0

    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = venv.step(action)

        step += 1
        total_reward += float(reward[0])

        im_down.set_data(base_env.follower_bot.get_down_camera_image())
        im_pov.set_data(base_env.follower_bot.get_pov_image())
        fig.suptitle(f"Step {step} | Reward: {reward[0]:.3f} | Total: {total_reward:.1f}")
        fig.canvas.flush_events()
        plt.pause(0.001)

        if done[0]:
            step = 0
            total_reward = 0.0

    venv.close()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
