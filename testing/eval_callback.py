"""SB3 callback for periodic in-training evaluation.

Fires every ``eval_freq`` total timesteps (computed across all training envs),
runs the full eval framework, logs every summary scalar to TensorBoard under
``eval/``, plus per-track success flags under ``eval/track_<seed>/``, and
saves the full result dict (including per-episode rows) as JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from testing.eval_framework import run_evaluation


class PeriodicEvalCallback(BaseCallback):
    """Run a deterministic evaluation every N total timesteps."""

    def __init__(self, eval_venv, *, eval_freq: int, log_dir: Path,
                 prefix: str = "eval", verbose: int = 1):
        super().__init__(verbose=verbose)
        self.eval_venv = eval_venv
        self.eval_freq = eval_freq
        self.log_dir = Path(log_dir)
        self.prefix = prefix
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        # SB3 calls this once per rollout step per env; n_calls is per-env
        # tick count. With N training envs, total timesteps = n_calls * N.
        n_envs = self.training_env.num_envs
        per_env_freq = max(1, self.eval_freq // n_envs)
        if self.n_calls % per_env_freq != 0:
            return True
        # Avoid double-firing if a checkpoint and eval land on the same tick
        # and SB3's accounting wobbles by 1.
        if self.num_timesteps == self._last_eval_step:
            return True
        self._last_eval_step = self.num_timesteps

        if self.verbose:
            print(f"[eval] running evaluation at step {self.num_timesteps:,}")

        results = run_evaluation(self.model, self.eval_venv, self.training_env)

        for k, v in results["summary"].items():
            self.logger.record(f"{self.prefix}/{k}", float(v))
        for row in results["per_track"]:
            seed = row["seed"]
            self.logger.record(f"{self.prefix}/track_{seed}/success", float(row["success"]))
            self.logger.record(f"{self.prefix}/track_{seed}/lateral_mse", float(row["lateral_mse"]))
        # Force flush so eval scalars share the timestep with the rollout that triggered them.
        self.logger.dump(self.num_timesteps)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.log_dir / f"{self.prefix}_{self.num_timesteps:09d}.json"
        with open(out_path, "w") as f:
            json.dump({"step": self.num_timesteps, **results}, f, indent=2)
        if self.verbose:
            sr = results["summary"].get("success_rate", 0.0)
            print(f"[eval] success_rate={sr:.0%}  →  {out_path}")
        return True
