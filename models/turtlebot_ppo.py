from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

import gym_line_follower  # registers envs


class EpisodeMetricsWrapper(gym.Wrapper):
    """Tracks per-episode metrics and injects them into the final-step info dict."""

    def reset(self, **kwargs):
        self._wz_accum = 0.0
        self._steps = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._wz_accum += abs(float(action[1]))  # cmd_vel: action[1] is wz
        self._steps += 1
        if terminated or truncated:
            info["mean_abs_wz"] = self._wz_accum / max(self._steps, 1)
            if truncated:
                info["termination"] = "timeout"
            elif reward <= -50.0:
                info["termination"] = "penalty"
            else:
                info["termination"] = "completed"
        return obs, reward, terminated, truncated, info


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
        smooth_steering_k=0.15,
        domain_randomize_physics=True,
        sensor_noise=0.5,
        obs_lag=1,
    )
    env.reset(seed=seed)

    # keep_dim=False so space stays 2D: (120,160) -> Resize (84,84) -> FrameStack (4,84,84).
    # With keep_dim=True, ResizeObservation's cv2.resize silently drops the C=1 dim while the
    # declared space keeps it, breaking FrameStackObservation's concatenate buffer.
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, shape=(84, 84))

    # Brightness jitter to bridge sim-to-real lighting gap (training only).
    aug_space = env.observation_space  # (84, 84) after resize
    env = gym.wrappers.TransformObservation(
        env,
        lambda obs: np.clip(
            (obs.astype(np.float32) * np.random.uniform(0.7, 1.4)).astype(np.int16),
            0, 255,
        ).astype(np.uint8),
        observation_space=aug_space,
    )

    env = gym.wrappers.FrameStackObservation(env, stack_size=4)

    env = EpisodeMetricsWrapper(env)
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
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

    class CustomMetricsCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self._term_counts = {"completed": 0, "penalty": 0, "timeout": 0}
            self._wz_buf: list[float] = []
            self._n_episodes = 0

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                if "termination" in info:
                    self._term_counts[info["termination"]] += 1
                    self._n_episodes += 1
                if "mean_abs_wz" in info:
                    self._wz_buf.append(info["mean_abs_wz"])
            return True

        def _on_rollout_end(self) -> None:
            if self._n_episodes > 0:
                for k, v in self._term_counts.items():
                    self.logger.record(f"episode/term_{k}_rate", v / self._n_episodes)
                self._term_counts = {"completed": 0, "penalty": 0, "timeout": 0}
                self._n_episodes = 0
            if self._wz_buf:
                self.logger.record("episode/mean_abs_wz", float(np.mean(self._wz_buf)))
                self._wz_buf = []

    vecnorm_path = args.model_path.parent / "ppo_turtlebot3_vecnorm.pkl"

    def _make() -> gym.Env:
        env = make_env(seed=args.seed, gui=args.gui)
        env = Monitor(env)
        return env

    from stable_baselines3.common.env_checker import check_env
    check_env(make_env(seed=args.seed, gui=False))
    venv = DummyVecEnv([_make] * 8)
    # norm_obs=False: CnnPolicy divides images by 255 internally; norm_reward stabilises wide reward range
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_obs=10.0)
    venv = VecMonitor(venv)

    if args.train:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        args.model_path.parent.mkdir(parents=True, exist_ok=True)

        model = PPO(
            policy="CnnPolicy",
            env=venv,
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(Path("logs") / "ppo_turtlebot3"),
            n_steps=1024,
            batch_size=256,
            n_epochs=4,
            ent_coef=0.01,
            learning_rate=3e-4,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000 // 8,  # per-env steps; 8 envs → saves every 100k total timesteps
            save_path=str(args.model_path.parent),
            name_prefix="ppo_turtlebot3",
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=[checkpoint_callback, CustomMetricsCallback()])
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
