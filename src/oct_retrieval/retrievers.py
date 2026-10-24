"""History-selection strategies. Each ranks the visits *before* the current visit of the same eye.

A retriever sees only the parsed question (never the gold question type), the current visit
and its history, and returns history positions ordered from most to least relevant.
Ties are broken by recency (most recent first) so every ranking is deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .questions import QuestionSpec


@dataclass
class History:
    """Chronologically ordered prior visits of one eye (position 0 = baseline)."""
    visit_ids: list[str]
    cst: np.ndarray
    bcva: np.ndarray
    n_images: np.ndarray

    def __len__(self):
        return len(self.visit_ids)

    def values(self, target: str) -> np.ndarray:
        return self.cst if target == "cst" else self.bcva


def _norm(x: np.ndarray) -> np.ndarray:
    """Min-max scale finite values to [0, 1]; NaN stays NaN; a constant vector maps to 0."""
    lo, hi = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x) if not np.isfinite(hi - lo) or hi == lo else (x - lo) / (hi - lo)


def _order(scores: np.ndarray) -> list[int]:
    """Sort by score desc, NaN last, ties -> more recent first."""
    n = len(scores)
    s = np.where(np.isnan(scores), -np.inf, scores)
    return sorted(range(n), key=lambda i: (-s[i], -i))


def temporal_score(spec: QuestionSpec, hist: History) -> np.ndarray:
    pos = np.arange(len(hist), dtype=float)
    if spec.intent == "baseline":
        return 1.0 - _norm(pos)
    if spec.intent == "previous":
        return _norm(pos)
    return np.zeros(len(hist))  # no temporal preference expressed


def clinical_score(spec: QuestionSpec, hist: History, current: dict) -> np.ndarray:
    if spec.target is None:
        return np.zeros(len(hist))
    v = hist.values(spec.target).astype(float)
    if np.all(np.isnan(v)):
        return np.full(len(hist), np.nan)
    if spec.intent == "extreme":
        return _norm(v) if spec.direction == "max" else 1.0 - _norm(v)
    # No extremum requested: clinical relevance = closeness to the current measurement.
    cur = current.get(spec.target, np.nan)
    if np.isnan(cur):
        return np.zeros(len(hist))
    return 1.0 - _norm(np.abs(v - cur))


class Retriever:
    name = "base"

    def rank(self, spec: QuestionSpec, hist: History, current: dict, rng: np.random.Generator) -> list[int]:
        raise NotImplementedError


class LatestPrior(Retriever):
    """Most recent prior visits first (also equals a fixed last-K window)."""
    name = "latest_prior"

    def rank(self, spec, hist, current, rng):
        return list(range(len(hist) - 1, -1, -1))


class EarliestFirst(Retriever):
    """Always compare with baseline (then the next-oldest visits)."""
    name = "earliest_first"

    def rank(self, spec, hist, current, rng):
        return list(range(len(hist)))


class RandomHistory(Retriever):
    name = "random"

    def rank(self, spec, hist, current, rng):
        return [int(i) for i in rng.permutation(len(hist))]


class FullHistory(LatestPrior):
    """Passes every prior visit to the downstream model. Ranked latest-first; cost = whole history."""
    name = "full_history"


class TemporalOnly(Retriever):
    name = "temporal_only"

    def rank(self, spec, hist, current, rng):
        return _order(temporal_score(spec, hist))


class ClinicalOnly(Retriever):
    name = "clinical_only"

    def rank(self, spec, hist, current, rng):
        return _order(clinical_score(spec, hist, current))


class Combined(Retriever):
    """alpha * clinical + (1 - alpha) * temporal, both min-max scaled within the history."""
    name = "combined"

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def rank(self, spec, hist, current, rng):
        c = clinical_score(spec, hist, current)
        t = temporal_score(spec, hist)
        return _order(self.alpha * np.nan_to_num(c, nan=-1.0) + (1 - self.alpha) * t)


class RuleRouter(Retriever):
    """Deterministic routing on the parsed intent: temporal anchors use visit order,
    value questions use the recorded measurement."""
    name = "rule_router"

    def rank(self, spec, hist, current, rng):
        if spec.intent in ("baseline", "previous"):
            return _order(temporal_score(spec, hist))
        if spec.intent in ("extreme", "similar"):
            return _order(clinical_score(spec, hist, current))
        return LatestPrior().rank(spec, hist, current, rng)  # unparsed question: fall back to recency
