"""Phase 1.3 retrieval: execute a QueryIntent over one eye's visit memory, with explicit outcomes.

Independent of ground_truth_v2 by design (no import). Operations are general (any target, direction,
threshold, offset); none is specific to a benchmark question type.

Outcomes (RetrievalResult.status):
  retrieved             ranked prior visits (position 0 = best)
  no_qualifying_visit   intent understood and computable, but no prior visit satisfies it
  insufficient_data     intent understood, but required inputs are missing (e.g. no weeks, current value missing)
  unavailable_variable  the question needs a variable that is not in this memory
  unsupported_image     the question needs OCT image findings
  clarification_needed  the question was not understood reliably
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import intent as I
from .retrievers import _order
from .side_tables import HELDOUT_COLUMNS

RETRIEVED = "retrieved"
NO_QUALIFYING = "no_qualifying_visit"
INSUFFICIENT = "insufficient_data"
UNAVAILABLE = "unavailable_variable"
UNSUPPORTED = "unsupported_image"
CLARIFY = "clarification_needed"
ABSTAIN_STATUSES = (NO_QUALIFYING, INSUFFICIENT, UNAVAILABLE, UNSUPPORTED, CLARIFY)

MEMORY_FIELDS = ("visit_id", "visit_index", "cst", "bcva", "week", "n_images", "trial")


@dataclass
class VisitMemory:
    """All visits of one eye in chronological order. Only MEMORY_FIELDS, plus explicitly declared extras."""
    visit_ids: list[str]
    visit_index: np.ndarray
    cst: np.ndarray
    bcva: np.ndarray
    week: np.ndarray
    n_images: np.ndarray
    trial: str
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    def values(self, var: str) -> np.ndarray | None:
        if var in ("cst", "bcva", "week"):
            return getattr(self, var)
        return self.extra.get(var)


@dataclass(frozen=True)
class RetrievalResult:
    status: str
    ranked: tuple[int, ...] = ()       # positions in the history (0 = baseline)
    reason: str = ""

    @property
    def abstained(self) -> bool:
        return self.status != RETRIEVED


def build_memories(episodes: pd.DataFrame, declared_extra: pd.DataFrame | None = None) -> dict[int, VisitMemory]:
    """One VisitMemory per eye. Refuses held-out columns unless passed explicitly via `declared_extra`."""
    leaked = [c for c in HELDOUT_COLUMNS if c in episodes.columns]
    if leaked:
        raise ValueError(f"Held-out columns in episodes passed to the retriever: {leaked}")
    ep = episodes
    extra_cols: list[str] = []
    if declared_extra is not None:
        extra_cols = [c for c in declared_extra.columns if c != "visit_id"]
        ep = episodes.merge(declared_extra, on="visit_id", how="left", validate="one_to_one")
    out = {}
    for eye_id, g in ep.groupby("eye_id"):
        g = g.sort_values("visit_index")
        out[eye_id] = VisitMemory(
            visit_ids=list(g.visit_id), visit_index=g.visit_index.to_numpy(int), cst=g.cst.to_numpy(float),
            bcva=g.bcva.to_numpy(float), week=g.week.to_numpy(float), n_images=g.n_images.to_numpy(int),
            trial=str(g.trial.iloc[0]), extra={c: g[c].to_numpy(float) for c in extra_cols})
    return out


def _qualifying_first(mask: np.ndarray, order_desc_recent: bool) -> tuple[int, ...]:
    """Qualifying positions first (by recency or by earliness), then the rest by recency."""
    idx = np.arange(len(mask))
    q = sorted(idx[mask], key=lambda i: -i if order_desc_recent else i)
    rest = sorted(idx[~mask], key=lambda i: -i)
    return tuple(int(i) for i in q + rest)


def _cmp(v: np.ndarray, cur: float, comparator: str) -> np.ndarray:
    ok = ~np.isnan(v)
    with np.errstate(invalid="ignore"):
        res = {"gt": v > cur, "ge": v >= cur, "lt": v < cur, "le": v <= cur}[comparator]
    return ok & res


def retrieve(q: I.QueryIntent, mem: VisitMemory, t: int) -> RetrievalResult:
    """Rank visits 0..t-1 of `mem` for the question asked at visit t (t itself is never a candidate)."""
    if q.task == I.UNSUPPORTED_IMAGE:
        return RetrievalResult(UNSUPPORTED, reason=q.reason)
    if q.task == I.UNAVAILABLE:
        return RetrievalResult(UNAVAILABLE, reason=q.reason)
    if q.task != I.RETRIEVE or q.operation is None:
        return RetrievalResult(CLARIFY, reason=q.reason or "intent not recognised")
    if t < 1:
        return RetrievalResult(INSUFFICIENT, reason="no prior visits")
    n = t
    pos = np.arange(n)
    op = q.operation

    if op == I.ANCHOR_BASELINE:
        return RetrievalResult(RETRIEVED, tuple(range(n)))
    if op == I.ANCHOR_PREVIOUS:
        return RetrievalResult(RETRIEVED, tuple(range(n - 1, -1, -1)))

    if op == I.TIME_OFFSET:
        weeks = mem.week[:n + 1]
        if np.isnan(weeks[n]) or np.all(np.isnan(weeks[:n])):
            return RetrievalResult(INSUFFICIENT, reason="visit timing (weeks) not available for this eye")
        target_week = weeks[n] - q.offset_weeks
        if target_week < 0:
            return RetrievalResult(INSUFFICIENT, reason="history does not extend back to the requested time")
        dist = np.abs(weeks[:n] - target_week)
        if np.nanmin(dist) > q.tolerance_weeks:
            return RetrievalResult(NO_QUALIFYING, reason=f"no prior visit within ±{q.tolerance_weeks} weeks of the target")
        return RetrievalResult(RETRIEVED, tuple(_order(-dist)))

    v_all = mem.values(q.target) if q.target else None
    if v_all is None:
        return RetrievalResult(UNAVAILABLE, reason=f"'{q.target}' is not in memory")
    v, cur = v_all[:n].astype(float), float(v_all[n])

    if op == I.LAST_EVENT:
        mask = v == 1
        if not mask.any():
            return RetrievalResult(NO_QUALIFYING, reason="no prior event recorded")
        return RetrievalResult(RETRIEVED, _qualifying_first(mask, True))

    if op == I.EXTREMUM:
        if np.all(np.isnan(v)):
            return RetrievalResult(INSUFFICIENT, reason="no prior measurement")
        return RetrievalResult(RETRIEVED, tuple(_order(v if q.direction == "max" else -v)))

    if op == I.CLOSEST_TO_CURRENT:
        if np.isnan(cur) or np.all(np.isnan(v)):
            return RetrievalResult(INSUFFICIENT, reason="current or prior measurement missing")
        return RetrievalResult(RETRIEVED, tuple(_order(-np.abs(v - cur))))

    if op == I.CONSECUTIVE_CHANGE:
        change = np.r_[np.nan, np.diff(v)]          # change at i relative to i-1; NaN if either missing
        if q.direction == "decrease":
            change = -change
        if np.all(np.isnan(change)):
            return RetrievalResult(INSUFFICIENT, reason="no consecutive pair of measurements")
        if np.nanmax(change) <= 0:
            return RetrievalResult(NO_QUALIFYING, reason=f"{q.target} never {q.direction}d between consecutive visits")
        return RetrievalResult(RETRIEVED, tuple(_order(change)))

    if op == I.THRESHOLD_FROM_BASELINE:
        if np.isnan(v[0]):
            return RetrievalResult(INSUFFICIENT, reason="baseline measurement missing")
        delta = v - v[0] if q.direction == "increase" else v[0] - v
        mask = ~np.isnan(delta) & (delta >= q.threshold) & (pos >= 1)
        if not mask.any():
            return RetrievalResult(NO_QUALIFYING, reason="threshold never reached")
        return RetrievalResult(RETRIEVED, _qualifying_first(mask, False))

    if op == I.COMPARE_TO_CURRENT:
        if np.isnan(cur):
            return RetrievalResult(INSUFFICIENT, reason="current measurement missing")
        mask = _cmp(v, cur, q.comparator)
        if not mask.any():
            return RetrievalResult(NO_QUALIFYING, reason="no prior visit satisfies the comparison")
        return RetrievalResult(RETRIEVED, _qualifying_first(mask, q.recency == "last"))

    return RetrievalResult(CLARIFY, reason=f"operation {op!r} not implemented")


def answer(question: str, mem: VisitMemory, t: int,
           available_variables: tuple[str, ...] = I.MEMORY_VARIABLES) -> tuple[I.QueryIntent, RetrievalResult]:
    """Parse then retrieve. The single entry point used by the v3 runner."""
    q = I.parse_intent(question, available_variables)
    return q, retrieve(q, mem, t)
