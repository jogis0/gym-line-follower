"""Match the sim down-camera photometry to the real camera.

Cheapest sim-to-real fix for the line-follower: capture frames from the real
robot's /image_raw, render frames from the PyBullet down-camera, segment line
vs. background with Otsu, and tune four config knobs until the per-class
mean/std/contrast and grayscale histograms overlap.

The four knobs live in turtlebot3_burger_config.json and are read each frame
by LineFollowerBot.get_down_camera_image / _augment_image, so changes apply
without rebuilding the track:

    light_ambient_coeff, light_diffuse_coeff,
    image_brightness_factor, image_contrast_factor.

Usage:
    # 1) Bot on the line with the real camera publishing /image_raw
    python tools/photometric_match.py --num-frames 60

    # 2) Re-tune from already-captured PNGs
    python tools/photometric_match.py --skip-capture

    # 3) Just print stats / overlay histograms — no tuning
    python tools/photometric_match.py --skip-capture --skip-optimize
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_GYM_LF_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_GYM_LF_ROOT))

ENV_ID = "TurtleBot3LineFollower-v0"
ENV_KWARGS = dict(
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

CFG_PATH = _GYM_LF_ROOT / "gym_line_follower" / "turtlebot3_burger_config.json"

KNOB_NAMES = (
    "light_ambient_coeff",
    "light_diffuse_coeff",
    "image_brightness_factor",
    "image_contrast_factor",
)
KNOB_BOUNDS = {
    "light_ambient_coeff": (0.30, 0.90),
    "light_diffuse_coeff": (0.15, 0.70),
    "image_brightness_factor": (0.70, 1.30),
    "image_contrast_factor": (0.70, 1.30),
}

HIST_BINS = 64
TARGET_SIZE = (160, 120)  # (w, h) — sim native; real frames resized to match.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/photometric_match"))
    p.add_argument("--camera-topic", type=str, default="/image_raw")
    p.add_argument("--num-frames", type=int, default=60,
                   help="Real frames to capture from --camera-topic.")
    p.add_argument("--capture-timeout", type=float, default=15.0,
                   help="Seconds to wait while capturing real frames before giving up.")
    p.add_argument("--num-sim-frames", type=int, default=6,
                   help="Sim resets per optimizer evaluation (more = slower, more stable).")
    p.add_argument("--rounds", type=int, default=5,
                   help="Coordinate-descent rounds. Each round scans every knob over a 5-point grid.")
    p.add_argument("--skip-capture", action="store_true",
                   help="Reuse PNGs already in <out-dir>/real/.")
    p.add_argument("--skip-optimize", action="store_true",
                   help="Print stats / save histogram and exit; do not tune.")
    p.add_argument("--no-write", action="store_true",
                   help="Print recommended config but don't modify the JSON.")
    p.add_argument("--seed-base", type=int, default=4242)
    p.add_argument("--gui", action="store_true",
                   help="Open the PyBullet GUI window (slow).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Real-camera capture (rclpy)
# ---------------------------------------------------------------------------

def _ros_image_to_bgr(msg) -> np.ndarray | None:
    """Decode a sensor_msgs/Image to a BGR uint8 array without cv_bridge.

    cv_bridge's C extension is compiled against NumPy 1.x and segfaults on
    NumPy 2.x.  We decode from raw bytes instead — safe for all encodings the
    TurtleBot3 / Raspberry Pi camera typically publishes.
    """
    enc = msg.encoding.lower()
    h, w = msg.height, msg.width
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    if enc in ("rgb8",):
        rgb = data.reshape((h, w, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if enc in ("bgr8",):
        return data.reshape((h, w, 3)).copy()
    if enc in ("rgba8",):
        rgba = data.reshape((h, w, 4))
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    if enc in ("bgra8",):
        bgra = data.reshape((h, w, 4))
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    if enc in ("mono8", "8uc1"):
        gray = data.reshape((h, w))
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if enc in ("yuv422", "yuv422_yuy2", "yuyv"):
        yuv = data.reshape((h, w, 2))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUYV)
    if enc in ("bayer_rggb8",):
        return cv2.cvtColor(data.reshape((h, w)), cv2.COLOR_BayerBG2BGR)
    if enc in ("bayer_bggr8",):
        return cv2.cvtColor(data.reshape((h, w)), cv2.COLOR_BayerRG2BGR)
    if enc in ("bayer_gbrg8",):
        return cv2.cvtColor(data.reshape((h, w)), cv2.COLOR_BayerGR2BGR)
    if enc in ("bayer_grbg8",):
        return cv2.cvtColor(data.reshape((h, w)), cv2.COLOR_BayerGB2BGR)
    return None


def capture_real_frames(topic: str, out_dir: Path, num_frames: int, timeout: float) -> int:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("real_*.png"):
        old.unlink()

    saved = {"n": 0}

    rclpy.init()
    try:
        class _Cap(Node):
            def __init__(self):
                super().__init__("photometric_match_capture")
                self.create_subscription(Image, topic, self._cb, 10)

            def _cb(self, msg: Image):
                if saved["n"] >= num_frames:
                    return
                bgr = _ros_image_to_bgr(msg)
                if bgr is None:
                    self.get_logger().warn(
                        f"Unsupported image encoding '{msg.encoding}' — "
                        f"expected rgb8/bgr8/mono8/yuv422/bayer_*."
                    )
                    return
                cv2.imwrite(str(out_dir / f"real_{saved['n']:03d}.png"), bgr)
                saved["n"] += 1

        node = _Cap()
        try:
            deadline = time.monotonic() + timeout
            while saved["n"] < num_frames and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.2)
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()

    if saved["n"] == 0:
        raise SystemExit(
            f"No frames received on {topic} within {timeout:.1f}s — "
            f"is the camera node publishing? (try `ros2 topic hz {topic}`)"
        )
    return saved["n"]


def load_real_frames(real_dir: Path) -> list[np.ndarray]:
    paths = sorted(real_dir.glob("real_*.png"))
    if not paths:
        raise SystemExit(f"No real frames in {real_dir}; rerun without --skip-capture.")
    return [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]


# ---------------------------------------------------------------------------
# Sim env + per-eval render
# ---------------------------------------------------------------------------

def build_sim_env(gui: bool):
    import gymnasium as gym
    import gym_line_follower  # noqa: F401  registers TurtleBot3LineFollower-v0

    return gym.make(ENV_ID, gui=gui, **ENV_KWARGS)


def render_with_knobs(bot, knobs: dict[str, float]) -> list[np.ndarray]:
    """Override config knobs on an already-reset bot and render one frame.

    No env.reset() is called — the bot stays at its current (start) position.
    This avoids the expensive track rebuild that reset() triggers, which is
    the dominant cost on low-power hardware.
    """
    for k, v in knobs.items():
        bot.config[k] = float(v)
    rgb = bot.get_down_camera_image()
    return [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)]


# ---------------------------------------------------------------------------
# Stats + loss
# ---------------------------------------------------------------------------

def _to_gray(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if (gray.shape[1], gray.shape[0]) != TARGET_SIZE:
        gray = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return gray


def compute_stats(frames: list[np.ndarray]) -> dict:
    line_means: list[float] = []
    line_stds: list[float] = []
    bg_means: list[float] = []
    bg_stds: list[float] = []
    hist_acc = np.zeros(HIST_BINS, dtype=np.float64)
    valid = 0

    for bgr in frames:
        gray = _to_gray(bgr)
        # Otsu — track is white-on-dark, so mask>0 is the line class.
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        line_pix = gray[mask > 0]
        bg_pix = gray[mask == 0]
        if line_pix.size == 0 or bg_pix.size == 0:
            # Frame has no contrast (all line or all bg) — skip; happens on
            # blown-out exposures.
            continue
        line_means.append(float(line_pix.mean()))
        line_stds.append(float(line_pix.std()))
        bg_means.append(float(bg_pix.mean()))
        bg_stds.append(float(bg_pix.std()))
        hist, _ = np.histogram(gray, bins=HIST_BINS, range=(0, 256))
        hist_acc += hist.astype(np.float64) / float(gray.size)
        valid += 1

    if valid == 0:
        raise SystemExit("All frames had zero line/bg contrast under Otsu — check exposure.")

    L = float(np.mean(line_means))
    B = float(np.mean(bg_means))
    return {
        "line_mean": L,
        "line_std": float(np.mean(line_stds)),
        "bg_mean": B,
        "bg_std": float(np.mean(bg_stds)),
        "contrast": (L - B) / (L + B) if (L + B) > 0 else 0.0,
        "hist": hist_acc / valid,
        "n": valid,
    }


def chi2_dist(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-9
    return float(0.5 * np.sum(((p - q) ** 2) / (p + q + eps)))


def loss_fn(real: dict, sim: dict) -> float:
    dlm = abs(real["line_mean"] - sim["line_mean"])
    dbm = abs(real["bg_mean"] - sim["bg_mean"])
    dls = abs(real["line_std"] - sim["line_std"])
    dbs = abs(real["bg_std"] - sim["bg_std"])
    dc = abs(real["contrast"] - sim["contrast"])
    dh = chi2_dist(real["hist"], sim["hist"])
    return dlm + dbm + 0.5 * dls + 0.5 * dbs + 100.0 * dc + 50.0 * dh


# ---------------------------------------------------------------------------
# Coordinate-descent optimizer (scipy-free; SB3 stack here doesn't ship scipy)
# ---------------------------------------------------------------------------

def optimize(bot, real_stats: dict, rounds: int) -> tuple[dict, float, list]:
    keys = list(KNOB_NAMES)
    knobs = {k: 0.5 * (KNOB_BOUNDS[k][0] + KNOB_BOUNDS[k][1]) for k in keys}

    def evaluate(candidate: dict) -> float:
        frames = render_with_knobs(bot, candidate)
        return loss_fn(real_stats, compute_stats(frames))

    best_loss = evaluate(knobs)
    history = [(best_loss, dict(knobs))]
    print(f"[opt] start  loss={best_loss:.4f}  {knobs}")

    for r in range(rounds):
        # Each round: scan each knob over a 5-point grid spanning the bounds,
        # narrowing around the current best with a span that halves per round.
        span_factor = 0.5 ** r
        improved_in_round = False
        for k in keys:
            lo, hi = KNOB_BOUNDS[k]
            full = hi - lo
            half_span = 0.5 * full * span_factor
            center = knobs[k]
            grid = np.linspace(
                max(lo, center - half_span),
                min(hi, center + half_span),
                5,
            )
            best_v = knobs[k]
            for v in grid:
                trial = dict(knobs)
                trial[k] = float(v)
                L = evaluate(trial)
                history.append((L, dict(trial)))
                if L < best_loss - 1e-6:
                    best_loss = L
                    best_v = float(v)
                    improved_in_round = True
            knobs[k] = best_v
        print(f"[opt] round {r + 1}/{rounds}  loss={best_loss:.4f}  {knobs}")
        if not improved_in_round:
            print("[opt] no improvement — stopping early.")
            break

    return knobs, best_loss, history


# ---------------------------------------------------------------------------
# Reporting + write-back
# ---------------------------------------------------------------------------

def print_stats_table(real: dict, before: dict, after: dict | None = None) -> None:
    cols = ("line_mean", "bg_mean", "line_std", "bg_std", "contrast")
    rows = [("real", real), ("sim_before", before)]
    if after is not None:
        rows.append(("sim_after", after))
    w = 14
    header = "stat".ljust(w) + "".join(c.rjust(w) for c in cols)
    print(header)
    print("-" * len(header))
    for name, s in rows:
        line = name.ljust(w) + "".join(f"{s[c]:.3f}".rjust(w) for c in cols)
        print(line)


def save_hist_plot(out: Path, real: dict, before: dict, after: dict | None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not available — skipping {out}.")
        return
    centers = np.linspace(0, 255, HIST_BINS)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(centers, real["hist"], label="real", linewidth=2)
    ax.plot(centers, before["hist"], label="sim (before)", linestyle="--")
    if after is not None:
        ax.plot(centers, after["hist"], label="sim (after)", linewidth=2)
    ax.set_xlabel("grayscale value")
    ax.set_ylabel("normalized frequency")
    ax.set_title("Down-camera photometric match")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved histogram overlay -> {out}")


def write_back(knobs: dict[str, float], no_write: bool) -> None:
    if not CFG_PATH.exists():
        raise SystemExit(f"Config not found: {CFG_PATH}")
    cfg = json.loads(CFG_PATH.read_text())
    diffs = []
    for k, v in knobs.items():
        spec = cfg.get(k)
        if not isinstance(spec, dict) or "default" not in spec:
            print(f"WARN: {k} has no randomized spec in config; skipping.")
            continue
        old = spec["default"]
        new = round(float(v), 4)
        spec["default"] = new
        diffs.append((k, old, new))

    if not diffs:
        print("Nothing to write.")
        return

    print("\nRecommended config changes:")
    for k, old, new in diffs:
        print(f"  {k}: {old} -> {new}")
    if no_write:
        print("--no-write: not modifying file.")
        return
    backup = CFG_PATH.with_suffix(f".json.bak.{int(time.time())}")
    shutil.copy2(CFG_PATH, backup)
    CFG_PATH.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"Wrote {CFG_PATH.name} (backup: {backup.name})")


def maybe_suggest_opacity(real: dict, after: dict, gap_threshold: float = 15.0) -> None:
    real_gap = real["line_mean"] - real["bg_mean"]
    sim_gap = after["line_mean"] - after["bg_mean"]
    delta = real_gap - sim_gap
    if abs(delta) < gap_threshold:
        return
    direction = "up" if delta > 0 else "down"
    print(
        f"\nResidual line-vs-background gap: real={real_gap:+.1f}  "
        f"sim={sim_gap:+.1f}  delta={delta:+.1f}."
    )
    print(
        f"  Lighting alone can't close it. Nudge `line_opacity.default` {direction} "
        f"in gym_line_follower/track_render_config.json and rerun."
    )


def _spec_default(key: str) -> float:
    cfg = json.loads(CFG_PATH.read_text())
    spec = cfg[key]
    return float(spec["default"]) if isinstance(spec, dict) else float(spec)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    real_dir = args.out_dir / "real"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_capture:
        print(f"Capturing up to {args.num_frames} frames from {args.camera_topic} "
              f"(timeout {args.capture_timeout:.0f}s) ...")
        n = capture_real_frames(
            args.camera_topic, real_dir, args.num_frames, args.capture_timeout
        )
        print(f"Captured {n} real frames -> {real_dir}")

    real_frames = load_real_frames(real_dir)
    real_stats = compute_stats(real_frames)
    print(f"\nReal stats over {real_stats['n']} frames:")
    print_stats_table(real_stats, real_stats)  # second column same; just to see numbers

    print(f"\nBuilding sim env (gui={args.gui}) ...")
    env = build_sim_env(gui=args.gui)
    try:
        # Single reset — track rebuild happens here only (the expensive step).
        env.reset(seed=args.seed_base)
        bot = env.unwrapped.follower_bot

        baseline_knobs = {k: _spec_default(k) for k in KNOB_NAMES}
        baseline_stats = compute_stats(render_with_knobs(bot, baseline_knobs))
        print("\n--- Baseline sim (current config defaults) ---")
        print_stats_table(real_stats, baseline_stats)

        if args.skip_optimize:
            save_hist_plot(args.out_dir / "hist.png", real_stats, baseline_stats, None)
            return 0

        print(f"\nOptimizing over {len(KNOB_NAMES)} knobs, {args.rounds} rounds ...")
        best_knobs, best_loss, _hist = optimize(bot, real_stats, args.rounds)
        tuned_stats = compute_stats(render_with_knobs(bot, best_knobs))
    finally:
        env.close()

    print(f"\nBest loss: {best_loss:.4f}")
    print("\n--- Result ---")
    print_stats_table(real_stats, baseline_stats, tuned_stats)
    save_hist_plot(args.out_dir / "hist.png", real_stats, baseline_stats, tuned_stats)
    write_back(best_knobs, no_write=args.no_write)
    maybe_suggest_opacity(real_stats, tuned_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
