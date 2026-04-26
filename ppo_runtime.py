"""Shared PPO config used by training and testing scripts.

Single source of truth for env kwargs, observation pipeline, oscillation
penalty, and model checkpoint paths. Edit values here and they propagate to:
  - models/turtlebot_ppo.py    (training)
  - testing/test_ppo_model.py  (eval with metrics)
  - testing/test_camera.py     (live visualisation)

Why this exists: silent train/eval mismatches on obs_lag, vx_min, or
smooth_steering produce policies that work in training and wiggle at test.

Note: legacy checkpoints from before the seeded-path migration live at
``models/ppo_turtlebot3_*.zip``. New runs land under ``models/PPO/seed_<seed>/``.
No automated migration; move old files manually if desired.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym

import gym_line_follower  # noqa: F401  registers TurtleBot3LineFollower-v0


ALGORITHM = "PPO"
ENV_ID = "TurtleBot3LineFollower-v0"
ENV_KWARGS: dict = dict(
    randomize=True,
    obsv_type="down_camera",
    vx_min=0.1,
    progress_reward=True,
    progress_reward_k=2.0,
    smooth_steering=False,           # use OscillationPenaltyWrapper instead
    domain_randomize_physics=False,
    sensor_noise=0.0,
    obs_lag=0,
)

# OscillationPenaltyWrapper coefficient. testing/eval_framework.py reads this so
# the eval-side steering penalty matches training.
OSCILLATION_PENALTY_K = 0.6

RESIZE_SHAPE = (84, 84)
FRAME_STACK_SIZE = 4

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_BASE_DIR = PROJECT_ROOT / "models" / ALGORITHM
LOGS_BASE_DIR = PROJECT_ROOT / "logs" / "ppo_turtlebot3"

# Bump this when switching to a different checkpoint — both .zip and the
# matching VecNormalize .pkl are derived from it.
MODEL_STEPS = 2_500_000
DEFAULT_SEED = 42

# Hyperparameters used by training. Mirrored into RunConfig at run start.
PPO_HYPERPARAMETERS: dict = dict(
    policy="CnnPolicy",
    n_steps=1024,
    batch_size=256,
    n_epochs=4,
    ent_coef=0.02,
    learning_rate=2e-4,
    clip_range=0.2,
)
N_ENVS = 8
EVAL_FREQUENCY = 50_000
CHECKPOINT_FREQUENCY = 50_000


def run_dir(seed: int) -> Path:
    """Per-run artifact directory — checkpoints, vecnorm pkls, run_config.json live here."""
    return MODEL_BASE_DIR / f"seed_{seed}"


def model_path_for(seed: int, steps: int | None = None) -> Path:
    """Resolve checkpoint path for (seed, steps). steps=None → final-model alias."""
    if steps is None:
        return run_dir(seed) / "ppo_turtlebot3.zip"
    return run_dir(seed) / f"ppo_turtlebot3_{steps}_steps.zip"


def vecnorm_for(model_path: Path) -> Path:
    """Map a ppo_turtlebot3<...>.zip checkpoint to its sibling vecnorm .pkl."""
    return model_path.with_name(
        model_path.stem.replace("ppo_turtlebot3", "ppo_turtlebot3_vecnorm") + ".pkl"
    )


def tb_log_dir(seed: int) -> Path:
    return LOGS_BASE_DIR / f"seed_{seed}"


def eval_log_dir(seed: int) -> Path:
    return tb_log_dir(seed) / "eval"


# Backwards-compatible defaults (resolve to seed_42 paths). Imported by callers
# that don't yet pass an explicit seed.
MODEL_PATH = model_path_for(DEFAULT_SEED, MODEL_STEPS)
VECNORM_PATH = vecnorm_for(MODEL_PATH)


class OscillationPenaltyWrapper(gym.Wrapper):
    """Penalises rapid steering reversals; logs per-episode mean penalty."""

    def __init__(self, env, penalty_k: float = OSCILLATION_PENALTY_K):
        super().__init__(env)
        self._penalty_k = penalty_k

    def reset(self, **kwargs):
        self._prev_wz = 0.0
        self._ep_penalty_accum = 0.0
        self._ep_steps = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        wz = float(action[1])
        penalty = self._penalty_k * abs(wz - self._prev_wz)
        reward -= penalty
        self._prev_wz = wz
        self._ep_penalty_accum += penalty
        self._ep_steps += 1
        # Inject per-step contribution into env's reward_components dict so eval
        # framework sees a complete breakdown including the wrapper-level penalty.
        info.setdefault("reward_components", {})["smoothness"] = -penalty
        if terminated or truncated:
            info["mean_osc_penalty"] = self._ep_penalty_accum / max(self._ep_steps, 1)
        return obs, reward, terminated, truncated, info


def apply_observation_pipeline(env: gym.Env) -> gym.Env:
    # keep_dim=False so space stays 2D: (H,W) -> Resize (84,84) -> FrameStack (4,84,84).
    # With keep_dim=True, ResizeObservation's cv2.resize silently drops the C=1 dim
    # while the declared space keeps it, breaking FrameStackObservation's concat buffer.
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, shape=RESIZE_SHAPE)
    env = gym.wrappers.FrameStackObservation(env, stack_size=FRAME_STACK_SIZE)
    return env


def build_env(*, seed: int, gui: bool, with_oscillation_penalty: bool = True,
              env_kwargs_override: dict | None = None,
              eager_reset: bool = True) -> gym.Env:
    """Build the env with all preprocessing, exactly as configured for PPO.

    `with_oscillation_penalty` only affects reward — actions and observations
    are unchanged either way. Default True (matches training); test scripts
    can pass False if they want raw env reward in their reports.

    `env_kwargs_override` shallow-merges into ENV_KWARGS (eval passes
    ``{"randomize": False}`` for deterministic tracks).

    `eager_reset=False` skips the eager ``env.reset(seed=seed)`` call so an
    outer wrapper can install a preset_track before the first reset.
    """
    kwargs = dict(ENV_KWARGS)
    if env_kwargs_override:
        kwargs.update(env_kwargs_override)
    env = gym.make(ENV_ID, gui=gui, **kwargs)
    if eager_reset:
        env.reset(seed=seed)
    env = apply_observation_pipeline(env)
    if with_oscillation_penalty:
        env = OscillationPenaltyWrapper(env)
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
    """Snapshot of all knobs needed to reproduce a training run.

    Saved as JSON at run start so any past run can be re-described without
    digging through git history. Loaded by test_ppo_model.py for traceability.
    """

    algorithm: str
    seed: int
    total_timesteps: int
    env_id: str
    env_kwargs: dict
    oscillation_penalty_k: float
    resize_shape: tuple
    frame_stack_size: int
    n_envs: int
    hyperparameters: dict
    eval_frequency: int
    checkpoint_frequency: int
    eval_track_seeds: list[int]
    started_at: str
    git_commit: str | None = None
    resumed_from: str | None = None

    @classmethod
    def capture(cls, *, seed: int, total_timesteps: int,
                eval_track_seeds: list[int],
                resumed_from: str | None = None) -> "RunConfig":
        return cls(
            algorithm=ALGORITHM,
            seed=seed,
            total_timesteps=total_timesteps,
            env_id=ENV_ID,
            env_kwargs=dict(ENV_KWARGS),
            oscillation_penalty_k=OSCILLATION_PENALTY_K,
            resize_shape=tuple(RESIZE_SHAPE),
            frame_stack_size=FRAME_STACK_SIZE,
            n_envs=N_ENVS,
            hyperparameters=dict(PPO_HYPERPARAMETERS),
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
