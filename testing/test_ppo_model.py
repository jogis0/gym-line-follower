import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ppo_runtime import (  # noqa: E402
    DEFAULT_SEED,
    RunConfig,
    eval_log_dir,
    model_path_for,
    vecnorm_for,
)

from testing.eval_framework import EVAL_TRACK_SEEDS, build_eval_env, run_evaluation


def _print_summary(summary: dict) -> None:
    print("\n--- Eval Summary ---")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:32s} {v:.5f}")
        else:
            print(f"  {k:32s} {v}")


def _print_per_track(per_track: list[dict]) -> None:
    print("\n--- Per-Track Results ---")
    print(f"  {'seed':<6} {'success':<8} {'termination':<12} {'distance':<10} {'lat_mse':<10}")
    for row in per_track:
        print(f"  {row['seed']:<6} {str(row['success']):<8} {row['termination']:<12} "
              f"{row['completed_distance_m']:<10.4f} {row['lateral_mse']:<10.6f}")


def _print_run_config(model_path: Path) -> None:
    cfg_path = model_path.parent / "run_config.json"
    if not cfg_path.exists():
        return
    try:
        cfg = RunConfig.load(cfg_path)
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"[warn] failed to read {cfg_path}: {e}")
        return
    print(f"\n--- Loaded Run Config ({cfg_path}) ---")
    print(f"  algorithm        : {cfg.algorithm}")
    print(f"  seed             : {cfg.seed}")
    print(f"  total_timesteps  : {cfg.total_timesteps}")
    print(f"  hyperparameters  : {cfg.hyperparameters}")
    print(f"  oscillation_k    : {cfg.oscillation_penalty_k}")
    print(f"  started_at       : {cfg.started_at}")
    if cfg.git_commit:
        print(f"  git_commit       : {cfg.git_commit[:12]}")
    if cfg.resumed_from:
        print(f"  resumed_from     : {cfg.resumed_from}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the deterministic 10-track evaluation on a trained PPO model.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Training seed used to locate the model directory.")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Path to the .zip model. Default: models/PPO/seed_<seed>/ppo_turtlebot3.zip")
    parser.add_argument("--vecnorm-path", type=Path, default=None,
                        help="Path to the matching VecNormalize stats. Auto-derived from --model-path.")
    parser.add_argument("--no-gui", action="store_true", help="Disable PyBullet GUI window.")
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = model_path_for(args.seed)

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {args.model_path}. "
            f"Train one first with: python models/turtlebot_ppo.py --train --seed {args.seed}"
        )

    if args.vecnorm_path is None:
        args.vecnorm_path = vecnorm_for(args.model_path)
    if not args.vecnorm_path.exists():
        raise FileNotFoundError(f"VecNormalize stats not found at {args.vecnorm_path}")

    _print_run_config(args.model_path)

    eval_venv = build_eval_env(EVAL_TRACK_SEEDS, gui=not args.no_gui,
                               vecnorm_path=args.vecnorm_path)
    model = PPO.load(str(args.model_path), env=eval_venv)
    print(f"\nLoaded model from {args.model_path}")
    print(f"Evaluating on {len(EVAL_TRACK_SEEDS)} fixed tracks: {EVAL_TRACK_SEEDS}")

    try:
        results = run_evaluation(model, eval_venv)
    finally:
        eval_venv.close()

    _print_summary(results["summary"])
    _print_per_track(results["per_track"])

    out_dir = eval_log_dir(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"manual_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model_path": str(args.model_path),
            "vecnorm_path": str(args.vecnorm_path),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
