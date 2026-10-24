"""v2 answer key (docs/PHASE1_V2_QUESTION_SPEC.md). Independent of retrievers.py by design.

answer_key() returns:
  INELIGIBLE (None)  -> the question is not posed at this visit (no instance)
  IE ([])            -> insufficient evidence: posed, but no prior visit satisfies the definition
  [i, ...]           -> positions (in history order) of every acceptable visit; ties all included
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .side_tables import HELDOUT_COLUMNS

INELIGIBLE = None
IE: list[int] = []
ONE_LINE_LETTERS = 5          # ETDRS: 5 letters per line (verified convention, spec §1.1)
WEEK_TOLERANCE = 4            # half of Prime's largest scheduled gap (spec §3)
MIN_HISTORY = 2

TYPES_V2 = {
    # name: (role, tier, cohort)
    "largest_cst_rise": ("primary", "B", "all"),
    "largest_bcva_drop": ("primary", "B", "all"),
    "first_vision_gain_1line": ("primary", "B", "all"),
    "one_year_ago": ("primary", "B", "Prime"),
    "six_months_ago": ("primary", "B", "Prime"),
    "max_leakage": ("primary", "A", "Prime"),
    "last_thicker_than_now": ("secondary", "B", "all"),
    "last_vision_as_good": ("secondary", "B", "all"),
}
PRIMARY_TYPES = [q for q, (role, _, _) in TYPES_V2.items() if role == "primary"]
SECONDARY_TYPES = [q for q, (role, _, _) in TYPES_V2.items() if role == "secondary"]
WEEK_OFFSET = {"one_year_ago": 52, "six_months_ago": 26}


def _argmax_set(v: np.ndarray) -> list[int]:
    ok = ~np.isnan(v)
    if not ok.any():
        return IE
    return [int(i) for i in np.flatnonzero(ok & (v == np.nanmax(v)))]


def answer_key(qtype: str, hist: pd.DataFrame, current: pd.Series):
    if len(hist) < MIN_HISTORY:
        return INELIGIBLE
    prime = current.trial == "Prime_FULL"
    cst = hist.cst.to_numpy(float)
    bcva = hist.bcva.to_numpy(float)

    if qtype in ("largest_cst_rise", "largest_bcva_drop"):
        v = cst if qtype == "largest_cst_rise" else -bcva          # drop in BCVA == rise in -BCVA
        change = np.r_[np.nan, np.diff(v)]                         # change at i vs i-1; NaN if either missing
        if np.all(np.isnan(change)):
            return INELIGIBLE
        if np.nanmax(change) <= 0:
            return IE
        return _argmax_set(change)

    if qtype == "first_vision_gain_1line":
        if np.isnan(bcva[0]):
            return INELIGIBLE
        hits = np.flatnonzero(bcva[1:] >= bcva[0] + ONE_LINE_LETTERS) + 1
        return [int(hits[0])] if len(hits) else IE

    if qtype in WEEK_OFFSET:
        if not prime or np.isnan(current.week) or current.week < WEEK_OFFSET[qtype]:
            return INELIGIBLE
        dist = np.abs(hist.week.to_numpy(float) - (current.week - WEEK_OFFSET[qtype]))
        if np.nanmin(dist) > WEEK_TOLERANCE:
            return IE
        return [int(i) for i in np.flatnonzero(dist == np.nanmin(dist))]

    if qtype == "max_leakage":
        if not prime:
            return INELIGIBLE
        leak = hist.leakage_index.to_numpy(float)
        return INELIGIBLE if np.all(np.isnan(leak)) else _argmax_set(leak)

    if qtype == "last_thicker_than_now":
        if np.isnan(current.cst):
            return INELIGIBLE
        hits = np.flatnonzero(cst > current.cst)
        return [int(hits[-1])] if len(hits) else IE

    if qtype == "last_vision_as_good":
        if np.isnan(current.bcva):
            return INELIGIBLE
        hits = np.flatnonzero(bcva >= current.bcva)
        return [int(hits[-1])] if len(hits) else IE

    raise ValueError(f"Unknown v2 question type {qtype!r}")


def generate_instances_v2(episodes: pd.DataFrame, heldout: pd.DataFrame,
                          paraphrases: dict[str, list[str]]) -> pd.DataFrame:
    """One row per (current visit, v2 type) where the type is eligible.

    `episodes` is the retriever-visible memory and must not carry held-out columns; they are joined
    here only for the answer key. Question text rotates through the given paraphrases per type.
    """
    leaked = [c for c in HELDOUT_COLUMNS if c in episodes.columns]
    if leaked:
        raise ValueError(f"Held-out columns present in retriever-visible episodes: {leaked}")
    keyed = episodes.merge(heldout[["visit_id", "leakage_index"]], on="visit_id", how="left", validate="one_to_one")
    rows, counter = [], {q: 0 for q in TYPES_V2}
    for eye_id, eye in keyed.sort_values(["eye_id", "visit_index"]).groupby("eye_id", sort=True):
        eye = eye.reset_index(drop=True)
        for t in range(MIN_HISTORY, len(eye)):
            hist, cur = eye.iloc[:t], eye.iloc[t]
            for qtype, (role, tier, cohort) in TYPES_V2.items():
                rel = answer_key(qtype, hist, cur)
                if rel is INELIGIBLE:
                    continue
                texts = paraphrases[qtype]
                rows.append(dict(
                    instance_id=f"{cur.visit_id}__{qtype}", eye_id=eye_id, patient_id=cur.patient_id,
                    split=cur.split, trial=cur.trial, current_visit_id=cur.visit_id, current_visit_index=t,
                    qtype=qtype, role=role, tier=tier, cohort=cohort,
                    question=texts[counter[qtype] % len(texts)], n_history=t,
                    answerable=len(rel) > 0, relevant_visit_ids=";".join(hist.visit_id.iloc[rel]),
                    n_relevant=len(rel), multi_answer=len(rel) > 1,
                ))
                counter[qtype] += 1
    return pd.DataFrame(rows)


def last_injection_lookup(episodes: pd.DataFrame, heldout: pd.DataFrame) -> pd.DataFrame:
    """Structured-lookup task (spec §6), NOT a retrieval benchmark: most recent prior injected visit."""
    keyed = episodes[episodes.trial == "Prime_FULL"].merge(heldout[["visit_id", "injection"]], on="visit_id")
    rows = []
    for eye_id, eye in keyed.sort_values(["eye_id", "visit_index"]).groupby("eye_id"):
        eye = eye.reset_index(drop=True)
        for t in range(MIN_HISTORY, len(eye)):
            hits = np.flatnonzero(eye.injection.to_numpy(float)[:t] == 1)
            rows.append(dict(eye_id=eye_id, patient_id=eye.patient_id[t], split=eye.split[t],
                             current_visit_id=eye.visit_id[t],
                             answer_visit_id=eye.visit_id[hits[-1]] if len(hits) else "",
                             latest_prior_is_answer=bool(len(hits)) and hits[-1] == t - 1))
    return pd.DataFrame(rows)
