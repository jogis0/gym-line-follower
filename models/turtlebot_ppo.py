from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ppo_runtime import MODEL_PATH, build_env, vecnorm_for  # noqa: E402


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
    env = build_env(seed=seed, gui=gui, with_oscillation_penalty=True)
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
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Resume training from a checkpoint .zip. Loads matching vecnorm pkl alongside.")
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
            self._osc_buf: list[float] = []
            self._n_episodes = 0

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                if "termination" in info:
                    self._term_counts[info["termination"]] += 1
                    self._n_episodes += 1
                if "mean_abs_wz" in info:
                    self._wz_buf.append(info["mean_abs_wz"])
                if "mean_osc_penalty" in info:
                    self._osc_buf.append(info["mean_osc_penalty"])
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
            if self._osc_buf:
                self.logger.record("episode/mean_osc_penalty", float(np.mean(self._osc_buf)))
                self._osc_buf = []

    class VecNormalizeCheckpointCallback(BaseCallback):
        """Saves VecNormalize stats alongside each model checkpoint so mid-run checkpoints are loadable."""

        def __init__(self, save_freq: int, save_path: Path, venv_ref):
            super().__init__()
            self._save_freq = save_freq
            self._save_path = save_path
            self._venv_ref = venv_ref

        def _on_step(self) -> bool:
            if self.n_calls % self._save_freq == 0:
                path = self._save_path / f"ppo_turtlebot3_vecnorm_{self.num_timesteps}_steps.pkl"
                self._venv_ref.save(str(path))
            return True

    vecnorm_path = args.model_path.parent / "ppo_turtlebot3_vecnorm.pkl"

    def _make() -> gym.Env:
        env = make_env(seed=args.seed, gui=args.gui)
        env = Monitor(env)
        return env

    from stable_baselines3.common.env_checker import check_env
    check_env(make_env(seed=args.seed, gui=False))
    venv = DummyVecEnv([_make] * 8)
    # norm_obs=False: CnnPolicy divides images by 255 internally; norm_reward stabilises wide reward range
    if args.resume_from is not None:
        vecnorm_resume = vecnorm_for(args.resume_from)
        if not vecnorm_resume.exists():
            parser.error(f"VecNormalize stats not found at {vecnorm_resume}")
        venv = VecNormalize.load(str(vecnorm_resume), venv)
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_obs=10.0)
    venv = VecMonitor(venv)

    if args.train:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        args.model_path.parent.mkdir(parents=True, exist_ok=True)

        if args.resume_from is not None:
            print(f"Resuming from {args.resume_from}")
            model = PPO.load(str(args.resume_from), env=venv,
                             tensorboard_log=str(Path("logs") / "ppo_turtlebot3"))
        else:
            model = PPO(
                policy="CnnPolicy",
                env=venv,
                seed=args.seed,
                verbose=1,
                tensorboard_log=str(Path("logs") / "ppo_turtlebot3"),
                n_steps=1024,
                batch_size=256,
                n_epochs=4,
                ent_coef=0.02,
                learning_rate=2e-4,
                clip_range=0.2,
            )
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000 // 8,  # per-env steps; 8 envs → saves every 100k total timesteps
            save_path=str(args.model_path.parent),
            name_prefix="ppo_turtlebot3",
        )
        vecnorm_ckpt_callback = VecNormalizeCheckpointCallback(
            save_freq=100_000 // 8,
            save_path=args.model_path.parent,
            venv_ref=venv,
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=True,
                    callback=[checkpoint_callback, vecnorm_ckpt_callback, CustomMetricsCallback()],
                    reset_num_timesteps=args.resume_from is None)
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
