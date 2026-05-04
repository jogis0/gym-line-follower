"""Summarize Stable-Baselines3 TensorBoard scalars at fixed step intervals.

Loads scalar tags from one or more run directories (each containing
events.out.tfevents.* files), samples every metric at each interval boundary
using the most recent value at or before the target step, and prints a Markdown
report to stdout. With two or more runs, also emits a delta section for
metrics common to all runs.

eval/* tags are excluded by default (they duplicate the JSON snapshots).
"""

import argparse
import bisect
import sys
from collections import Counter, namedtuple
from datetime import datetime, timedelta
from pathlib import Path

try:
    from tensorboard.backend.event_processing import event_accumulator
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    sys.exit("tensorboard not installed. Run: pip install tensorboard")


RunData = namedtuple(
    "RunData",
    ["path", "series", "wall_first", "wall_last", "max_step"],
)


def load_run(run_dir: Path, include_eval: bool):
    tfevents = list(run_dir.glob("events.out.tfevents.*"))
    if not tfevents:
        return None

    ea = EventAccumulator(
        str(run_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()

    series = {}
    wall_first = float("inf")
    wall_last = 0.0
    max_step = 0

    for tag in ea.Tags().get("scalars", []):
        if not include_eval and tag.startswith("eval/"):
            continue
        events = ea.Scalars(tag)
        if not events:
            continue
        by_step = {}
        for e in events:
            cur = by_step.get(e.step)
            if cur is None or e.wall_time >= cur[0]:
                by_step[e.step] = (e.wall_time, e.value)
            if e.wall_time < wall_first:
                wall_first = e.wall_time
            if e.wall_time > wall_last:
                wall_last = e.wall_time
            if e.step > max_step:
                max_step = e.step
        series[tag] = sorted((s, v) for s, (_, v) in by_step.items())

    if wall_first == float("inf"):
        wall_first = 0.0

    return RunData(run_dir, series, wall_first, wall_last, max_step)


def sample_at(steps, values, target):
    idx = bisect.bisect_right(steps, target) - 1
    if idx < 0:
        return None
    return values[idx]


def fmt_value(v):
    if v is None:
        return "-"
    return f"{v:.4g}"


def fmt_step_label(n, interval):
    if interval % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def sort_tags(tags):
    def key(tag):
        ns, _, name = tag.partition("/")
        if not name:
            return ("", ns)
        return (ns, name)
    return sorted(tags, key=key)


def fmt_duration(seconds):
    if seconds <= 0:
        return "0:00:00"
    return str(timedelta(seconds=int(seconds)))


def fmt_iso(ts):
    if ts <= 0:
        return "n/a"
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def format_table(label, run, interval, include_eval):
    lines = []
    lines.append(f"## Run: {label}")
    lines.append(f"- path: {run.path.as_posix()}")
    lines.append(f"- total steps: {run.max_step}")
    lines.append(
        f"- wall time: {fmt_iso(run.wall_first)} to {fmt_iso(run.wall_last)} "
        f"({fmt_duration(run.wall_last - run.wall_first)})"
    )
    lines.append(
        f"- scalar tags retained: {len(run.series)} "
        f"(eval included: {'yes' if include_eval else 'no'})"
    )
    lines.append("")

    n_targets = run.max_step // interval
    if n_targets <= 0:
        lines.append(f"_no full interval reached, max_step={run.max_step}_")
        return "\n".join(lines)

    targets = [i * interval for i in range(1, n_targets + 1)]
    tags = sort_tags(run.series.keys())

    cached = {}
    for tag in tags:
        steps = [s for s, _ in run.series[tag]]
        values = [v for _, v in run.series[tag]]
        cached[tag] = (steps, values)

    header = ["step"] + tags
    sep = ["---"] * len(header)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for t in targets:
        row = [fmt_step_label(t, interval)]
        for tag in tags:
            steps, values = cached[tag]
            row.append(fmt_value(sample_at(steps, values, t)))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def format_delta_section(runs, interval):
    common = set.intersection(*(set(r.series.keys()) for _, r in runs))
    if not common:
        return "## Deltas\n\n_no scalar tags common to all runs_"

    min_max_step = min(r.max_step for _, r in runs)
    n_targets = min_max_step // interval
    if n_targets <= 0:
        return (
            f"## Deltas\n\n_no full interval reached across all runs "
            f"(min max_step={min_max_step})_"
        )
    targets = [i * interval for i in range(1, n_targets + 1)]

    cached = {}
    for label, run in runs:
        for tag in common:
            steps = [s for s, _ in run.series[tag]]
            values = [v for _, v in run.series[tag]]
            cached[(label, tag)] = (steps, values)

    out = ["## Deltas (metrics common to all runs)", ""]
    show_delta = len(runs) == 2
    labels = [lbl for lbl, _ in runs]

    for tag in sort_tags(common):
        out.append(f"### {tag}")
        header = ["step"] + labels
        if show_delta:
            header.append(f"Δ ({labels[1]} - {labels[0]})")
        sep = ["---"] * len(header)
        out.append("| " + " | ".join(header) + " |")
        out.append("| " + " | ".join(sep) + " |")
        for t in targets:
            row = [fmt_step_label(t, interval)]
            sampled = []
            for label in labels:
                steps, values = cached[(label, tag)]
                sampled.append(sample_at(steps, values, t))
            for v in sampled:
                row.append(fmt_value(v))
            if show_delta:
                a, b = sampled[0], sampled[1]
                if a is None or b is None:
                    row.append("-")
                else:
                    row.append(f"{(b - a):+.4g}")
            out.append("| " + " | ".join(row) + " |")
        out.append("")

    return "\n".join(out).rstrip()


def default_labels(run_dirs):
    basenames = [Path(p).name for p in run_dirs]
    counts = Counter(basenames)
    labels = []
    for p in run_dirs:
        p = Path(p)
        if counts[p.name] > 1:
            labels.append(f"{p.parent.name}/{p.name}")
        else:
            labels.append(p.name)
    return labels


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Summarize SB3 TensorBoard scalars at fixed step intervals."
    )
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--interval", type=int, default=100_000)
    p.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override column label for a run dir. Repeatable.",
    )
    p.add_argument(
        "--include-eval",
        action="store_true",
        help="Include eval/* tags (default: excluded).",
    )
    args = p.parse_args(argv)

    if args.interval <= 0:
        p.error("--interval must be positive")

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    overrides = {}
    for spec in args.label:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            p.error(f"--label expects NAME=PATH, got: {spec!r}")
        overrides[Path(path).resolve()] = name

    auto_labels = default_labels(args.run_dirs)

    loaded = []
    for d, auto_label in zip(args.run_dirs, auto_labels):
        label = overrides.get(d.resolve(), auto_label)
        if not d.exists():
            print(f"[warn] {d}: directory does not exist", file=sys.stderr)
            continue
        try:
            rd = load_run(d, args.include_eval)
        except Exception as exc:
            print(f"[warn] {d}: failed to load ({exc})", file=sys.stderr)
            continue
        if rd is None:
            print(f"[warn] {d}: no events.out.tfevents.* found", file=sys.stderr)
            continue
        if not rd.series:
            print(f"[warn] {d}: no scalar tags retained", file=sys.stderr)
            continue
        loaded.append((label, rd))

    if not loaded:
        sys.exit("no runs loaded")

    sections = [
        format_table(label, run, args.interval, args.include_eval)
        for label, run in loaded
    ]
    if len(loaded) >= 2:
        sections.append(format_delta_section(loaded, args.interval))

    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
