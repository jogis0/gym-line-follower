import argparse
from pathlib import Path

import gymnasium as gym
import gym_line_follower
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

SMOOTH_STEERING_K = 0.05


def main():
    parser = argparse.ArgumentParser(description="Run a trained PPO model with live camera display.")
    parser.add_argument("--model-path", type=Path, default=Path("models") / "ppo_turtlebot3_400000_steps.zip")
    parser.add_argument("--no-gui", action="store_true", help="Disable PyBullet GUI window.")
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found at {args.model_path}.")

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
            domain_randomize_physics=True,
            sensor_noise=0.0,
        )
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
        env = Monitor(env)
        return env

    venv = DummyVecEnv([_make])
    venv = VecMonitor(venv)

    model = PPO.load(str(args.model_path), env=venv)
    print(f"Loaded model from {args.model_path}")

    # Access the unwrapped base env for camera images
    base_env = venv.envs[0].env.env

    obs = venv.reset()

    plt.ion()
    fig, (ax_down, ax_pov) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Robot Camera Feed")

    down_img = base_env.unwrapped.follower_bot.get_down_camera_image()
    pov_img = base_env.unwrapped.follower_bot.get_pov_image()

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
        obs, reward, done, info = venv.step(action)

        step += 1
        total_reward += float(reward[0])

        im_down.set_data(base_env.unwrapped.follower_bot.get_down_camera_image())
        im_pov.set_data(base_env.unwrapped.follower_bot.get_pov_image())
        fig.suptitle(f"Step {step} | Reward: {reward[0]:.3f} | Total: {total_reward:.1f}")
        fig.canvas.flush_events()
        plt.pause(0.001)

        if done[0]:
            obs = venv.reset()
            step = 0
            total_reward = 0.0

    venv.close()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
