"""Shared DQN config used by training and testing scripts.

Mirror of sac_runtime.py for the DQN algorithm. DQN requires a discrete
action space, so build_env applies DiscreteCmdVelWrapper before the
observation pipeline; the underlying env still receives Box(2) cmd_vel
actions.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym

import gym_line_follower  # noqa: F401  registers TurtleBot3LineFollower-v0
from gym_line_follower.wrappers import (
    ACTION_NAMES,
    ACTION_TABLE,
    DiscreteCmdVelWrapper,
)


ALGORITHM = "DQN"
ENV_ID = "TurtleBot3LineFollower-v0"
ENV_KWARGS: dict = dict(
    randomize=True,
    obsv_type="down_camera",
    vx_min=0.1,
    progress_reward=True,
    progress_reward_k=2.0,
    smooth_steering=False,
    domain_randomize_physics=False,
    sensor_noise=0.0,
    obs_lag=0,
)

RESIZE_SHAPE = (84, 84)
FRAME_STACK_SIZE = 4

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_BASE_DIR = PROJECT_ROOT / "models" / ALGORITHM
LOGS_BASE_DIR = PROJECT_ROOT / "logs" / "dqn_turtlebot3"

MODEL_STEPS = 2_500_000
DEFAULT_SEED = 42

# DQN tends to be unstable above 2e-4 on images; tau=1.0 with a 1k hard-sync
# interval is the classic Atari-DQN pattern. optimize_memory_usage halves
# the replay-buffer RAM cost (4x84x84 uint8 stacks would otherwise need ~3GB
# at 100k transitions including next_obs).
DQN_HYPERPARAMETERS: dict = dict(
    policy="CnnPolicy",
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1_000,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    max_grad_norm=10.0,
    optimize_memory_usage=True,
)
N_ENVS = 1
EVAL_FREQUENCY = 50_000
CHECKPOINT_FREQUENCY = 50_000


def run_dir(seed: int) -> Path:
    """Per-run artifact directory — checkpoints, vecnorm/buffer pkls, run_config.json live here."""
    return MODEL_BASE_DIR / f"seed_{seed}"


def model_path_for(seed: int, steps: int | None = None) -> Path:
    """Resolve checkpoint path for (seed, steps). steps=None → final-model alias."""
    if steps is None:
        return run_dir(seed) / "dqn_turtlebot3.zip"
    return run_dir(seed) / f"dqn_turtlebot3_{steps}_steps.zip"


def best_model_path_for(seed: int) -> Path:
    """Path for the best-by-eval model snapshot saved during training."""
    return run_dir(seed) / "dqn_turtlebot3_best.zip"


def vecnorm_for(model_path: Path) -> Path:
    """Map a dqn_turtlebot3<...>.zip checkpoint to its sibling vecnorm .pkl."""
    return model_path.with_name(
        model_path.stem.replace("dqn_turtlebot3", "dqn_turtlebot3_vecnorm") + ".pkl"
    )


def replay_buffer_for(model_path: Path) -> Path:
    """Map a dqn_turtlebot3<...>.zip checkpoint to its sibling replay-buffer .pkl."""
    return model_path.with_name(
        model_path.stem.replace("dqn_turtlebot3", "dqn_turtlebot3_replay_buffer") + ".pkl"
    )


def tb_log_dir(seed: int) -> Path:
    return LOGS_BASE_DIR / f"seed_{seed}"


def eval_log_dir(seed: int) -> Path:
    return tb_log_dir(seed) / "eval"


MODEL_PATH = model_path_for(DEFAULT_SEED, MODEL_STEPS)
VECNORM_PATH = vecnorm_for(MODEL_PATH)


def apply_observation_pipeline(env: gym.Env) -> gym.Env:
    # keep_dim=False so space stays 2D: (H,W) -> Resize (84,84) -> FrameStack (4,84,84).
    # Mirrors sac_runtime / recurrent_ppo_runtime so all algorithms share the
    # same image-preprocessing contract.
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, shape=RESIZE_SHAPE)
    env = gym.wrappers.FrameStackObservation(env, stack_size=FRAME_STACK_SIZE)
    return env


def build_env(*, seed: int, gui: bool,
              env_kwargs_override: dict | None = None,
              eager_reset: bool = True,
              **_unused) -> gym.Env:
    """Build the env with all preprocessing, exactly as configured for DQN.

    DiscreteCmdVelWrapper sits innermost (next to the env) so the agent sees
    Discrete(N) while the underlying env still receives Box(2) cmd_vel.

    `env_kwargs_override` shallow-merges into ENV_KWARGS (eval passes
    ``{"randomize": False}`` for deterministic tracks).

    `eager_reset=False` skips the eager ``env.reset(seed=seed)`` call so an
    outer wrapper can install a preset_track before the first reset.

    Extra kwargs (e.g. ``with_oscillation_penalty``) are accepted and ignored
    so build_eval_env can pass the same kwargs to all algorithm runtimes.
    """
    kwargs = dict(ENV_KWARGS)
    if env_kwargs_override:
        kwargs.update(env_kwargs_override)
    env = gym.make(ENV_ID, gui=gui, **kwargs)
    if eager_reset:
        env.reset(seed=seed)
    env = DiscreteCmdVelWrapper(env)
    env = apply_observation_pipeline(env)
    return env


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


@dataclass
class RunConfig:
    """Snapshot of all knobs needed to reproduce a DQN training run."""

    algorithm: str
    seed: int
    total_timesteps: int
    env_id: str
    env_kwargs: dict
    action_table: list
    action_names: list
    resize_shape: tuple
    frame_stack_size: int
    n_envs: int
    hyperparameters: dict
    eval_frequency: int
    checkpoint_frequency: int
    eval_track_seeds: list
    started_at: str
    git_commit: str | None = None
    resumed_from: str | None = None

    @classmethod
    def capture(cls, *, seed: int, total_timesteps: int,
                eval_track_seeds: list,
                resumed_from: str | None = None) -> "RunConfig":
        return cls(
            algorithm=ALGORITHM,
            seed=seed,
            total_timesteps=total_timesteps,
            env_id=ENV_ID,
            env_kwargs=dict(ENV_KWARGS),
            action_table=[list(a) for a in ACTION_TABLE],
            action_names=list(ACTION_NAMES),
            resize_shape=tuple(RESIZE_SHAPE),
            frame_stack_size=FRAME_STACK_SIZE,
            n_envs=N_ENVS,
            hyperparameters=dict(DQN_HYPERPARAMETERS),
            eval_frequency=EVAL_FREQUENCY,
            checkpoint_frequency=CHECKPOINT_FREQUENCY,
            eval_track_seeds=list(eval_track_seeds),
            started_at=datetime.now().isoformat(timespec="seconds"),
            git_commit=_git_commit(),
            resumed_from=resumed_from,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
