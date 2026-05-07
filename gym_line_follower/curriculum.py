"""Curriculum scheduling + theme resolution for visual domain randomization.

Used by ``LineFollowerEnv.reset()`` to scale per-episode randomization toward
defaults early in training, ramping to full strength as a training callback
pushes ``CURRICULUM.progress`` from 0 to 1.

Why scale post-randomize values rather than the underlying ranges: the
``RandomizerDict`` already drew a sample; interpolating that sample toward the
default is mathematically equivalent to sampling from a shrunk range without
having to mutate the dict's ``original`` definition.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from gym_line_follower.randomizer_dict import RandomizerDict


@dataclass
class CurriculumState:
    progress: float = 1.0


CURRICULUM = CurriculumState()


THEME_MAP: dict[str, tuple[str, str]] = {
    "white_on_foam_dark": ("white", "foam_dark"),
    "white_on_black":     ("white", "black"),
    "white_on_concrete":  ("white", "concrete"),
    "black_on_white":     ("black", "white"),
    "black_on_gray":      ("black", "gray"),
    "black_on_wood":      ("black", "wood"),
    "black_on_wood_2":    ("black", "wood_2"),
    "red_on_white":       ("red", "white"),
    "blue_on_white":      ("blue", "white"),
}


def apply_curriculum(rdict: RandomizerDict, scale: float, rng: random.Random) -> None:
    """Pull each randomized value toward its declared default by ``1 - scale``.

    scale = 1.0 → no change (full random sample retained)
    scale = 0.0 → values reset to their defaults

    Numeric (range) entries: linear interpolation default + scale*(current - default).
    Categorical (choice) entries: with probability (1 - scale), revert to default.

    The passed-in ``rng`` is used for the categorical Bernoulli draw so we don't
    perturb the global ``random`` state that ``RandomizerDict.randomize`` reseeds.
    """
    if scale >= 1.0:
        return
    s = max(0.0, float(scale))
    for key, original in rdict.original.items():
        if not isinstance(original, dict) or "default" not in original:
            continue
        default = original["default"]
        if "range" in original:
            current = rdict.get(key, default)
            if isinstance(current, (int, float)) and isinstance(default, (int, float)):
                interp = default + s * (current - default)
                rdict[key] = type(default)(interp) if isinstance(default, int) else interp
        elif "choice" in original:
            if rng.random() > s:
                rdict[key] = default


def resolve_theme(rdict: RandomizerDict) -> None:
    """If ``rdict`` carries a ``line_bg_theme`` key from THEME_MAP, write the
    corresponding ``line_color`` and ``background`` entries so the track
    renderer (which reads those two keys directly) picks up the pair."""
    theme = rdict.get("line_bg_theme")
    if theme and theme in THEME_MAP:
        line_c, bg_c = THEME_MAP[theme]
        rdict["line_color"] = line_c
        rdict["background"] = bg_c
