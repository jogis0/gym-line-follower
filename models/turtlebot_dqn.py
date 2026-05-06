from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dqn_runtime import (  # noqa: E402
    CHECKPOINT_FREQUENCY,
    DEFAULT_SEED,
    DQN_HYPERPARAMETERS,
    EVAL_FREQUENCY,
    N_ENVS,
    RunConfig,
    best_model_path_for,
    build_env,
    eval_log_dir,
    model_path_for,
    replay_buffer_for,
    run_dir,
    tb_log_dir,
    vecnorm_for,
)
from gym_line_follower.wrappers import ACTION_TABLE  # noqa: E402
from sim_to_real_config import SIM_TO_REAL_OVERRIDES  # noqa: E402


class EpisodeMetricsWrapper(gym.Wrapper):
    """Tracks per-episode metrics and injects them into the final-step info dict.

    Sits outside DiscreteCmdVelWrapper so the action it sees is a discrete
    index; the table lookup recovers wz for logging parity with continuous
    runtimes.
    """

    def reset(self, **kwargs):
        self._wz_accum = 0.0
        self._steps = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        wz = float(ACTION_TABLE[int(action)][1])
        self._wz_accum += abs(wz)
        self._steps += 1
        if terminated or truncated:
            info["mean_abs_wz"] = self._wz_accum / max(self._steps, 1)
        return obs, reward, terminated, truncated, info


def make_env(*, seed: int, gui: bool, sim_to_real: bool = False) -> gym.Env:
    overrides = dict(SIM_TO_REAL_OVERRIDES) if sim_to_real else None
    env = build_env(seed=seed, gui=gui, env_kwargs_override=overrides)
    env = EpisodeMetricsWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train a new DQN policy.")
    parser.add_argument("--eval", action="store_true", help="Evaluate an existing DQN policy.")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI window.")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Override model output path. Default: models/DQN/seed_<seed>/dqn_turtlebot3.zip")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Resume training from a checkpoint .zip. Loads matching vecnorm and replay-buffer pkls alongside.")
    parser.add_argument("--sim-to-real", action="store_true",
                        help="Enable sim-to-real overrides (domain_randomize_physics, sensor_noise, obs_lag).")
    args = parser.parse_args()

    if not args.train and not args.eval:
        parser.error("Choose one: --train or --eval")

    run_path = run_dir(args.seed)
    run_path.mkdir(parents=True, exist_ok=True)
    if args.model_path is None:
        args.model_path = model_path_for(args.seed)

    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

    from testing.eval_framework import EVAL_TRACK_SEEDS, build_eval_env
    from testing.eval_callback import PeriodicEvalCallback

    class CustomMetricsCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self._term_counts = {"completed": 0, "penalty": 0, "timeout": 0, "line_lost": 0}
            self._wz_buf: list[float] = []
            self._n_episodes = 0

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                if "termination" in info:
                    bucket = info["termination"]
                    if bucket in self._term_counts:
                        self._term_counts[bucket] += 1
                    self._n_episodes += 1
                if "mean_abs_wz" in info:
                    self._wz_buf.append(info["mean_abs_wz"])
            return True

        def _on_rollout_end(self) -> None:
            if self._n_episodes > 0:
                for k, v in self._term_counts.items():
                    self.logger.record(f"episode/term_{k}_rate", v / self._n_episodes)
                self._term_counts = {"completed": 0, "penalty": 0, "timeout": 0, "line_lost": 0}
                self._n_episodes = 0
            if self._wz_buf:
                self.logger.record("episode/mean_abs_wz", float(np.mean(self._wz_buf)))
                self._wz_buf = []

    class VecNormalizeCheckpointCallback(BaseCallback):
        """Saves VecNormalize stats alongside each model checkpoint so mid-run checkpoints are loadable."""

        def __init__(self, save_freq: int, save_path: Path, venv_ref):
            super().__init__()
            self._save_freq = save_freq
            self._save_path = save_path
            self._venv_ref = venv_ref

        def _on_step(self) -> bool:
            if self.n_calls % self._save_freq == 0:
                path = self._save_path / f"dqn_turtlebot3_vecnorm_{self.num_timesteps}_steps.pkl"
                self._venv_ref.save(str(path))
            return True

    class ReplayBufferCheckpointCallback(BaseCallback):
        """Saves DQN replay buffer alongside each checkpoint so mid-run resumes restore full off-policy state."""

        def __init__(self, save_freq: int, save_path: Path):
            super().__init__()
            self._save_freq = save_freq
            self._save_path = save_path

        def _on_step(self) -> bool:
            if self.n_calls % self._save_freq == 0:
                path = self._save_path / f"dqn_turtlebot3_replay_buffer_{self.num_timesteps}_steps.pkl"
                self.model.save_replay_buffer(str(path))
            return True

    vecnorm_path = vecnorm_for(args.model_path)
    replay_buffer_path = replay_buffer_for(args.model_path)

    def _make() -> gym.Env:
        env = make_env(seed=args.seed, gui=args.gui, sim_to_real=args.sim_to_real)
        env = Monitor(env)
        return env

    from stable_baselines3.common.env_checker import check_env
    check_env(make_env(seed=args.seed, gui=False, sim_to_real=args.sim_to_real))
    venv = DummyVecEnv([_make] * N_ENVS)
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

        RunConfig.capture(
            seed=args.seed,
            total_timesteps=args.timesteps,
            eval_track_seeds=EVAL_TRACK_SEEDS,
            resumed_from=str(args.resume_from) if args.resume_from else None,
            sim_to_real=args.sim_to_real,
        ).save(run_path / "run_config.json")

        if args.resume_from is not None:
            print(f"Resuming from {args.resume_from}")
            model = DQN.load(str(args.resume_from), env=venv,
                             tensorboard_log=str(tb_log_dir(args.seed)))
            buffer_resume = replay_buffer_for(args.resume_from)
            if buffer_resume.exists():
                print(f"Loading replay buffer from {buffer_resume}")
                model.load_replay_buffer(str(buffer_resume))
            else:
                print(f"WARNING: replay buffer not found at {buffer_resume}; "
                      "continuing with empty buffer (will re-warm during learning_starts).")
        else:
            model = DQN(
                env=venv,
                seed=args.seed,
                verbose=1,
                tensorboard_log=str(tb_log_dir(args.seed)),
                **DQN_HYPERPARAMETERS,
            )
        save_freq_per_env = max(1, CHECKPOINT_FREQUENCY // N_ENVS)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq_per_env,
            save_path=str(run_path),
            name_prefix="dqn_turtlebot3",
        )
        vecnorm_ckpt_callback = VecNormalizeCheckpointCallback(
            save_freq=save_freq_per_env,
            save_path=run_path,
            venv_ref=venv,
        )
        replay_ckpt_callback = ReplayBufferCheckpointCallback(
            save_freq=save_freq_per_env,
            save_path=run_path,
        )
        eval_venv = build_eval_env(EVAL_TRACK_SEEDS, gui=False,
                                   build_env_fn=build_env,
                                   env_kwargs_extra=SIM_TO_REAL_OVERRIDES if args.sim_to_real else None)
        best_model_path = best_model_path_for(args.seed)
        best_vecnorm_path = vecnorm_for(best_model_path)
        eval_callback = PeriodicEvalCallback(
            eval_venv=eval_venv,
            eval_freq=EVAL_FREQUENCY,
            log_dir=eval_log_dir(args.seed),
            best_model_save_path=best_model_path,
            best_vecnorm_save_path=best_vecnorm_path,
        )
        try:
            model.learn(
                total_timesteps=args.timesteps,
                progress_bar=True,
                callback=[checkpoint_callback, vecnorm_ckpt_callback,
                          replay_ckpt_callback, CustomMetricsCallback(), eval_callback],
                reset_num_timesteps=args.resume_from is None,
            )
            model.save(str(args.model_path))
            venv.save(str(vecnorm_path))
            model.save_replay_buffer(str(replay_buffer_path))
        finally:
            eval_venv.close()

    if args.eval:
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training = False
        venv.norm_reward = False
        model = DQN.load(str(args.model_path), env=venv)
        obs = venv.reset()
        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv.step(action)
            if bool(dones[0]):
                obs = venv.reset()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
