"""Adapter that lets RecurrentPPO use the algorithm-agnostic eval framework.

run_evaluation() calls model.predict(obs, deterministic=True) — the standard
SB3 signature. RecurrentPPO additionally needs the prior LSTM hidden state and
an episode_start flag passed through. This wrapper hides that bookkeeping.

Usage:
    wrapped = LstmEvalWrapper(model)
    results = run_evaluation(wrapped, eval_venv)

run_evaluation calls wrapped.reset_state() at the start of each episode (see
eval_framework.run_evaluation), which clears the LSTM state and re-flags the
next predict() call as the start of a new episode.
"""
from __future__ import annotations

import numpy as np


class LstmEvalWrapper:
    """Wraps a RecurrentPPO model so it can be driven by predict(obs, deterministic=...)."""

    def __init__(self, model):
        self._model = model
        self._lstm_states = None
        self._episode_starts = np.ones((1,), dtype=bool)

    def predict(self, obs, deterministic: bool = True):
        action, self._lstm_states = self._model.predict(
            obs,
            state=self._lstm_states,
            episode_start=self._episode_starts,
            deterministic=deterministic,
        )
        # After the first step of an episode, subsequent steps should not be
        # flagged as starts. reset_state() flips this back to True at episode
        # boundaries.
        self._episode_starts = np.zeros((1,), dtype=bool)
        return action, self._lstm_states

    def reset_state(self) -> None:
        """Clear LSTM hidden state and mark the next predict() as an episode start."""
        self._lstm_states = None
        self._episode_starts = np.ones((1,), dtype=bool)

    def __getattr__(self, name):
        # Delegate anything we don't override (save, get_parameters, policy, ...)
        # to the wrapped model so the wrapper is a drop-in replacement. Note:
        # __getattr__ is only invoked when normal lookup fails, so our own
        # attributes (_model, predict, reset_state, ...) take precedence.
        return getattr(self._model, name)
