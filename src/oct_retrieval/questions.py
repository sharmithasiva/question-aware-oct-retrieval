"""Question types, a rule-based question parser, and metadata-derived ground truth.

Every question type here has ground truth that follows deterministically from recorded
CST/BCVA and visit order. No clinical labels are invented. "Edema" is mapped to CST as a proxy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGETS = ("cst", "bcva")


@dataclass(frozen=True)
class QuestionSpec:
    """Structured reading of a question (component 1 of the architecture)."""
    target: str | None      # 'cst' | 'bcva' | None
    intent: str             # 'baseline' | 'previous' | 'extreme' | 'similar' | 'unknown'
    direction: str | None   # 'max' | 'min' for intent == 'extreme', else None
    evidence: str           # 'metadata' | 'image'


# qtype -> (gold spec, templates). Templates take {target_phrase} where useful.
QUESTION_TYPES: dict[str, tuple[QuestionSpec, list[str]]] = {
    "baseline_cst": (QuestionSpec("cst", "baseline", None, "metadata"), [
        "How has the central subfield thickness changed since baseline?",
        "Compared with the first visit, is the CST higher or lower now?",
        "Has retinal thickness improved relative to the initial visit?",
    ]),
    "baseline_bcva": (QuestionSpec("bcva", "baseline", None, "metadata"), [
        "How has visual acuity changed since baseline?",
        "Compared with the first visit, is BCVA better or worse now?",
        "Has vision improved relative to the initial visit?",
    ]),
    "previous_cst": (QuestionSpec("cst", "previous", None, "metadata"), [
        "Has the CST changed since the previous visit?",
        "Compared with the last visit, is the retina thicker or thinner?",
        "How does central thickness compare with the most recent prior visit?",
    ]),
    "worst_edema": (QuestionSpec("cst", "extreme", "max", "metadata"), [
        "Compared with when the edema was worst, has it improved?",
        "Which visit had the highest recorded CST, and how does today compare?",
        "Is the retina less swollen than at its peak thickness?",
    ]),
    "best_anatomy": (QuestionSpec("cst", "extreme", "min", "metadata"), [
        "Compared with the visit with the lowest CST, has thickness increased?",
        "How does current thickness compare with when the retina was thinnest?",
        "Is the edema back above its best recorded level?",
    ]),
    "worst_vision": (QuestionSpec("bcva", "extreme", "min", "metadata"), [
        "Compared with when vision was worst, has it recovered?",
        "Which visit had the lowest BCVA, and is acuity better now?",
        "How does visual acuity compare with its poorest recorded value?",
    ]),
    "best_vision": (QuestionSpec("bcva", "extreme", "max", "metadata"), [
        "Compared with the best vision recorded, has acuity dropped?",
        "Which visit had the highest BCVA, and how does today compare?",
        "Is visual acuity as good as at its peak?",
    ]),
    "similar_cst": (QuestionSpec("cst", "similar", None, "metadata"), [
        "When was the CST most similar to today's measurement?",
        "Find the prior visit with the closest central thickness to now.",
        "At which earlier visit was retinal thickness comparable to the current scan?",
    ]),
}

# Paraphrases not used to generate benchmark instances; used only to check the parser.
# Written by the same author as the parser lexicon, so accuracy here is optimistic.
HELDOUT_PARAPHRASES: list[tuple[str, QuestionSpec]] = [
    ("Is the macular oedema better than at its most severe?", QuestionSpec("cst", "extreme", "max", "metadata")),
    ("Since the screening visit, has the thickness gone down?", QuestionSpec("cst", "baseline", None, "metadata")),
    ("Has the patient's vision deteriorated since last time?", QuestionSpec("bcva", "previous", None, "metadata")),
    ("Is the BCVA now below its maximum?", QuestionSpec("bcva", "extreme", "max", "metadata")),
    ("Which earlier visit had a central subfield thickness closest to the current one?",
     QuestionSpec("cst", "similar", None, "metadata")),
    ("Compared with the thickest the retina has been, is it improving?",
     QuestionSpec("cst", "extreme", "max", "metadata")),
    ("How many letters has the patient gained from the start of treatment?",
     QuestionSpec("bcva", "baseline", None, "metadata")),
    ("How has the retinal structure changed?", QuestionSpec(None, "unknown", None, "image")),
    # Needs biomarker/image evidence: should be routed away from metadata retrieval.
    ("Has the intraretinal fluid resolved compared with the earlier scans?",
     QuestionSpec(None, "unknown", None, "image")),
]

_BCVA_WORDS = r"\b(bcva|vision|visual acuity|acuity|letters?|see|seeing)\b"
_CST_WORDS = r"\b(cst|central subfield|thickness|thick|thicker|thinner|thickest|thinnest|edema|oedema|swelling|swollen|retina)\b"
_IMAGE_WORDS = r"\b(structure|structural|appearance|look like|layers?|fluid|irf|srf|hyperreflective|image|images)\b"
_SIMILAR = r"\b(similar|closest|comparable|most like|resembl\w*)\b"
_BASELINE = r"\b(baseline|first visit|initial|screening|start of|beginning)\b"
_PREVIOUS = r"\b(previous|last visit|last time|prior visit|most recent)\b"
_WORST = r"\b(worst|most severe|poorest)\b"
_BEST = r"\b(best)\b"
_MAX = r"\b(highest|peak|maximum|max|thickest|greatest|most swollen)\b"
_MIN = r"\b(lowest|minimum|min|thinnest|least)\b"


def parse_question(text: str) -> QuestionSpec:
    """Deterministic keyword parser for clinical target, temporal intent and evidence type."""
    t = text.lower()
    has = lambda pat: re.search(pat, t) is not None  # noqa: E731
    target = "bcva" if has(_BCVA_WORDS) else "cst" if has(_CST_WORDS) else None

    if target is None:
        evidence = "image" if has(_IMAGE_WORDS) else "metadata"
        return QuestionSpec(None, "unknown", None, evidence)
    if has(r"\b(fluid|irf|srf)\b") and not has(r"\b(cst|thickness|bcva|vision|acuity)\b"):
        # Fluid questions need biomarker or image evidence; CST is not a fluid label.
        return QuestionSpec(None, "unknown", None, "image")

    if has(_SIMILAR):
        return QuestionSpec(target, "similar", None, "metadata")
    if has(_WORST) or has(_BEST) or has(_MAX) or has(_MIN):
        if has(_WORST):
            direction = "max" if target == "cst" else "min"
        elif has(_BEST):
            # "best recorded level" of edema = thinnest; best vision = highest BCVA
            direction = "min" if target == "cst" else "max"
        elif has(_MAX):
            direction = "max"
        else:
            direction = "min"
        return QuestionSpec(target, "extreme", direction, "metadata")
    if has(_BASELINE):
        return QuestionSpec(target, "baseline", None, "metadata")
    if has(_PREVIOUS):
        return QuestionSpec(target, "previous", None, "metadata")
    return QuestionSpec(target, "unknown", None, "metadata")


def relevant_indices(gold: QuestionSpec, hist_values: np.ndarray, current_value: float) -> list[int]:
    """Ground-truth relevant positions within a chronologically ordered history (all visits before current).

    hist_values holds the gold target measurement for each history visit (NaN if missing).
    Ties are all relevant. Returns [] if ground truth is undefined.
    """
    n = len(hist_values)
    if n == 0:
        return []
    if gold.intent == "baseline":
        return [0]
    if gold.intent == "previous":
        return [n - 1]
    valid = ~np.isnan(hist_values)
    if not valid.any():
        return []
    if gold.intent == "extreme":
        best = np.nanmax(hist_values) if gold.direction == "max" else np.nanmin(hist_values)
        return [int(i) for i in np.flatnonzero(valid & (hist_values == best))]
    if gold.intent == "similar":
        if np.isnan(current_value):
            return []
        dist = np.abs(hist_values - current_value)
        return [int(i) for i in np.flatnonzero(valid & (dist == np.nanmin(dist)))]
    raise ValueError(f"No ground truth for intent {gold.intent!r}")


def generate_instances(episodes: pd.DataFrame, min_history: int = 2) -> pd.DataFrame:
    """One instance per (eye, current visit, question type) where ground truth is defined.

    The current visit is excluded from its history. Templates rotate deterministically.
    """
    rows = []
    counter = 0
    for eye_id, eye in episodes.sort_values(["eye_id", "visit_index"]).groupby("eye_id", sort=True):
        eye = eye.reset_index(drop=True)
        for t in range(min_history, len(eye)):
            hist = eye.iloc[:t]
            cur = eye.iloc[t]
            for qtype, (gold, templates) in QUESTION_TYPES.items():
                vals = hist[gold.target].to_numpy(dtype=float)
                rel = relevant_indices(gold, vals, float(cur[gold.target]))
                if not rel:
                    continue
                rows.append(dict(
                    instance_id=f"{cur.visit_id}__{qtype}",
                    eye_id=eye_id, patient_id=cur.patient_id, split=cur.get("split"), trial=cur.trial,
                    current_visit_id=cur.visit_id, current_visit_index=int(t),
                    qtype=qtype, question=templates[counter % len(templates)],
                    n_history=t, relevant_visit_ids=";".join(hist.visit_id.iloc[rel]),
                    latest_prior_is_relevant=(t - 1) in rel,
                    baseline_is_relevant=0 in rel,
                ))
                counter += 1
    return pd.DataFrame(rows)
