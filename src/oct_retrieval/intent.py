"""Phase 1.3 structured question intent and a transparent, deterministic parser.

`parse_intent(text)` returns a QueryIntent whose `task` says what the system should do:
  retrieve              -> rank prior visits with a named operation
  unavailable_variable  -> the question needs a variable that is not in the retriever's memory
  unsupported_image     -> the question needs OCT image findings (no metadata answer exists)
  clarify               -> intent could not be identified reliably (never silently mapped to "latest")

Rules run in a fixed order of stages; each stage matches phrase-level patterns, not single words.
Nothing here reads ground truth, held-out values, or benchmark identifiers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# --- task / operation vocabulary ---------------------------------------------------------------
RETRIEVE, UNAVAILABLE, UNSUPPORTED_IMAGE, CLARIFY = "retrieve", "unavailable_variable", "unsupported_image", "clarify"

ANCHOR_BASELINE = "anchor_baseline"
ANCHOR_PREVIOUS = "anchor_previous"
EXTREMUM = "extremum"
CLOSEST_TO_CURRENT = "closest_to_current"
CONSECUTIVE_CHANGE = "max_consecutive_change"
THRESHOLD_FROM_BASELINE = "first_threshold_crossing"
COMPARE_TO_CURRENT = "compare_to_current"
TIME_OFFSET = "time_offset"
LAST_EVENT = "last_event"

MEMORY_VARIABLES = ("cst", "bcva")                       # always in memory
OUT_OF_MEMORY_VARIABLES = ("leakage_index", "injection", "drss_level")
TEMPORAL_TOLERANCE_WEEKS = 4                             # system parameter; same value as the frozen spec rule
WEEKS_PER_YEAR = 52
ETDRS_LETTERS_PER_LINE = 5


@dataclass(frozen=True)
class QueryIntent:
    task: str
    operation: str | None = None
    target: str | None = None          # 'cst' | 'bcva' | out-of-memory variable name
    direction: str | None = None       # extremum: 'max'/'min'; change/threshold: 'increase'/'decrease'
    comparator: str | None = None      # compare_to_current: 'gt' | 'ge' | 'lt' | 'le'
    recency: str | None = None         # compare_to_current: 'last' | 'first'
    offset_weeks: int | None = None
    tolerance_weeks: int | None = None
    threshold: float | None = None     # in the target's native units (ETDRS letters, micrometres)
    missing_variable: str | None = None
    reason: str = ""


# --- lexicons (phrase-level) ------------------------------------------------------------------
_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20}
_NUMBER = r"(\d+(?:\.\d+)?|" + "|".join(_NUM) + r")"

CST_STRONG = r"\bcst\b|\bcentral subfield\b|\b(central|retinal|macular|oct|centre|center)( retinal)? thickness\b"
CST_WEAK = r"\bthick\w*|\bthin(ner|nest|ning|ned)?\b|\bswell\w*|\bswollen\b|\bo?edema\b|\boedema\b"
BCVA_TERMS = (r"\bbcva\b|\bvisual acuity\b|\bacuity\b|\bva\b|\bvision\b|\beyesight\b|\betdrs\b|\bletters?\b"
              r"|\bline(s)? on the chart\b|\bread\b[^.?!]*\blines?\b")

VARIABLE_TERMS = {
    "leakage_index": r"\bleak\w*|\bangiogra\w*|\bfluorescein\b|\bfa\b|\bdye\b",
    "injection": r"\binject\w*|\bintravitreal\b|\banti-?vegf\b|\bshots?\b|\bdoses?\b",
    "drss_level": r"\bdrss\b|\b(retinopathy|dr) (severity|grade|level|stage)\b|\bseverity (scale|grade|level)\b"
                  r"|\bretinopathy\b[^.?!]*\bgrade\b",
}
# Provenance: OLIVES biomarker column names (Load: BIOMARKER_COLS), dev "unsupported" paraphrases, the project
# brief ("when did the scan look most like today"), and v1 held-out paraphrases. Terms recognised only from the
# locked set were deliberately NOT added (see docs/RETRIEVAL_IMPROVEMENT_PLAN.md, contamination disclosure).
IMAGE_TERMS = r"\b(" + "|".join([
    r"fluid", r"irf", r"srf", r"ellipsoid", r"ez", r"rpe", r"pigment epithel\w*", r"ped", r"hyperreflective",
    r"foci", r"hrf", r"dril", r"disorgani[sz]ation", r"epiretinal", r"vitreo\w*", r"vitreous", r"vmt",
    r"atroph\w*", r"contour", r"structur\w*", r"layers?", r"appearance", r"look(s|ed)?( most)? (like|similar)",
    r"\w+-looking", r"hemorrhag\w*", r"haemorrhag\w*", r"shrm", r"preretinal",
]) + r")\b"

BASELINE_REF = (r"\b(from|than|over|above|below|vs\.?|versus|relative to|compared (with|to)|since|to)\s+(at\s+)?"
                r"(the\s+|my\s+|this\s+|their\s+)?(baseline|first (visit|exam|appointment|scan)"
                r"|initial( (visit|exam|bcva|vision|score|value|measurement|acuity))?"
                r"|start( of treatment)?|beginning|screening( visit)?)\b")
BASELINE_ANCHOR = r"\bbaseline\b|\bfirst (visit|exam|appointment|scan)\b|\binitial (visit|exam)\b|\bscreening\b|\bstart of treatment\b"
PREVIOUS_ANCHOR = (r"\bprevious\b|\blast (visit|time|check|exam|appointment|scan)\b|\bprior (visit|exam|check)\b"
                   r"|\bmost recent\b")
GOING_BACK = r"\bgoing back(wards)? from (today|now|this visit)\b"
RECENCY_LAST = r"\b(last|most recent|latest)\b"
FIRST_OP = r"\b(first|earliest|soonest)\b|\bhow soon\b"

TIME_CUE = (r"\bago\b|\bbefore\b|\bearlier\b|\bprior\b|\bback\b|\blook-?back\b|\bpreviously\b|\bfrom then\b"
            r"|\bthis time last year\b|\b(over|in) the (last|past)\b")
DURATION = _NUMBER + r"\s*-?\s*(years?|yrs?|months?|mos?|weeks?|wks?)\b"

CHANGE_UP = r"\b(jump\w*|ris(e|es|ing)|rose|increas\w*|gain\w*|thicken\w*|grew|grow\w*)\b"
CHANGE_DOWN = (r"\b(drop\w*|declin\w*|fall(s|ing)?|fell|loss|lost|los(e|es|ing)|deteriorat\w*|decreas\w*|reduc\w*"
               r"|thinn\w*)\b")
MAGNITUDE = r"\b(biggest|largest|steepest|sharpest|greatest|most|max(imum)?|hardest|worst|highest|top)\b"
CONSECUTIVE = (r"visit[- ]to[- ]visit|visit[- ]over[- ]visit"
               r"|\bconsecutive\b|\bsequential\b|\bsuccessive\b|\binterval\b|single[- ]step"
               r"|\b(than|from|since|compared (with|to)|relative to|vs\.?) (the |my )?((visit|one|exam|check|appointment|scan)"
               r" )?(right |just |immediately )?(before( it)?|preceding|prior( one)?|previous( visit| check| one| exam"
               r"| appointment)?)\b|\bpreceding (visit|exam|check|appointment)\b|\bthe prior one\b")

CURRENT_REF = r"\btoday'?s?\b|\bnow\b|\bcurrent(ly)?\b|\bthis visit\b|\bthis (good|thick|high)\b"
CMP_GE = (r"at least as (good|high|thick)|\bas (good|high|thick) as\b|greater than or equal|equal to or"
          r" (better|greater|higher)|\b(matched|matches|match) or (beat|beats|exceeded|exceeds)\b"
          r"|at least (this|that) (good|thick|high)|≥")
CMP_LE = r"\bat most\b|less than or equal|equal to or (worse|lower|less)"
CMP_UP = r"\b(thicker|higher|greater|above|exceed\w*|beat|larger|bigger)\b"
CMP_DOWN = r"\b(thinner|lower|below|smaller)\b"

SIMILAR = r"\b(similar|closest|comparable|most like|resembl\w*|nearest)\b"
EXT_MAX = r"\b(highest|peak|maximum|max|thickest|greatest|most swollen|top)\b"
EXT_MIN = r"\b(lowest|minimum|min|thinnest|smallest|least)\b"
EXT_WORST = r"\b(worst|most severe|poorest)\b"
EXT_BEST = r"\bbest\b"


def _normalise(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[‒-―−]", " - ", t)          # dashes
    t = t.replace("’", "'").replace(">=", " ≥ ").replace("<=", " at most ")
    t = re.sub(r"\bbest[- ]corrected (visual acuity|vision|acuity)\b", "bcva", t)
    t = re.sub(r"\bbest[- ]corrected\b", "", t)
    t = re.sub(r"\bat least\b(?! (as|this|that) )", " ≥ ", t)   # "at least 5" is a threshold, not "least"
    t = re.sub(r"\bhalf (a |of a )?year\b|\bhalf-year\b", "26 weeks", t)
    return re.sub(r"\s+", " ", t).strip()


def _has(pat: str, t: str) -> bool:
    return re.search(pat, t) is not None


def _num(tok: str) -> float:
    return float(_NUM[tok]) if tok in _NUM else float(tok)


def _duration_weeks(t: str) -> int | None:
    found = [(m, _num(m.group(1)), m.group(2)) for m in re.finditer(DURATION, t)]
    weeks = set()
    for _, n, unit in found:
        if unit.startswith(("y",)):
            weeks.add(round(n * WEEKS_PER_YEAR))
        elif unit.startswith("mo"):
            weeks.add(round(n * WEEKS_PER_YEAR / 12))
        else:
            weeks.add(round(n))
    if not weeks and _has(r"\b(last|this time last|a) year\b|\bannual\b", t) and _has(TIME_CUE + r"|\blast year\b", t):
        weeks.add(WEEKS_PER_YEAR)
    return weeks.pop() if len(weeks) == 1 else (-1 if weeks else None)


def _target(t: str) -> str | None | str:
    cst = _has(CST_STRONG, t) or _has(CST_WEAK, t)
    bcva = _has(BCVA_TERMS, t)
    if cst and bcva:
        return "ambiguous"
    return "cst" if cst else "bcva" if bcva else None


def _threshold(t: str, target: str) -> float | None:
    if target == "bcva":
        m = re.search(_NUMBER + r"\s*(more\s+)?-?\s*(lines?|letters?)\b", t)
        if m:
            n = _num(m.group(1))
            return n * ETDRS_LETTERS_PER_LINE if m.group(3).startswith("line") else n
    if target == "cst":
        m = re.search(r"(\d+(?:\.\d+)?)\s*(µm|um|microns?|micrometres?|micrometers?)\b", t)
        if m:
            return float(m.group(1))
    return None


def _direction_words(t: str, target: str) -> str | None:
    up = _has(r"\b(improv\w*|gain\w*|better|up|increas\w*|more|exceed\w*|ris\w*|rose|higher)\b", t)
    down = _has(r"\b(worse|loss|lost|drop\w*|declin\w*|fall\w*|fell|decreas\w*|reduc\w*|lower|less)\b", t)
    if up == down:
        return None
    return "increase" if up else "decrease"


def _compare_to_current(t: str, target: str) -> QueryIntent | None:
    if not _has(CURRENT_REF, t):
        return None
    if _has(CMP_GE, t):
        comparator = "ge" if target == "bcva" or _has(r"\b(high|thick)\b", t) else "le"
    elif _has(CMP_LE, t):
        comparator = "le" if target == "bcva" else "ge"
    else:
        up, down = _has(CMP_UP, t), _has(CMP_DOWN, t)
        better, worse = _has(r"\bbetter\b", t), _has(r"\bworse\b", t)
        if target == "bcva":
            up, down = up or better, down or worse
        else:  # for thickness, "worse" = thicker, "better" = thinner
            up, down = up or worse, down or better
        if up == down:
            return None
        comparator = "gt" if up else "lt"
    if _has(GOING_BACK, t) or _has(RECENCY_LAST, re.sub(GOING_BACK, " ", t)):
        recency = "last"
    elif _has(FIRST_OP, t):
        recency = "first"
    else:
        return None
    return QueryIntent(RETRIEVE, COMPARE_TO_CURRENT, target, comparator=comparator, recency=recency)


def parse_intent(text: str, available_variables: tuple[str, ...] = MEMORY_VARIABLES) -> QueryIntent:
    t = _normalise(text)

    # 1. Time offset ("about a year ago", "26 weeks prior"): needs no clinical target.
    weeks = _duration_weeks(t)
    if weeks is not None and _has(TIME_CUE, t):
        if weeks <= 0:
            return QueryIntent(CLARIFY, reason="conflicting or invalid time expressions")
        return QueryIntent(RETRIEVE, TIME_OFFSET, offset_weeks=weeks, tolerance_weeks=TEMPORAL_TOLERANCE_WEEKS)

    # 2. Variables that exist in the dataset but are not in the retriever's memory unless declared.
    for var, pat in VARIABLE_TERMS.items():
        if _has(pat, t):
            if var not in available_variables:
                return QueryIntent(UNAVAILABLE, target=var, missing_variable=var,
                                   reason=f"requires '{var}', which is not in the retriever memory")
            if var == "injection":
                if _has(RECENCY_LAST, t):
                    return QueryIntent(RETRIEVE, LAST_EVENT, var)
                return QueryIntent(CLARIFY, reason="injection question without a recognised operation")
            mx, mn = _has(EXT_MAX + "|" + EXT_WORST + r"|\bmost\b", t), _has(EXT_MIN, t)
            if mx != mn:
                return QueryIntent(RETRIEVE, EXTREMUM, var, direction="max" if mx else "min")
            return QueryIntent(CLARIFY, reason=f"'{var}' question without a recognised operation")

    # 3. Image-only findings take priority over weak thickness words ("choroid thickest").
    if _has(IMAGE_TERMS, t):
        if _has(CST_STRONG, t) or _has(r"\bbcva\b|\bvisual acuity\b|\betdrs\b", t):
            return QueryIntent(CLARIFY, reason="mixes image findings with a metadata measurement")
        return QueryIntent(UNSUPPORTED_IMAGE, reason="requires OCT image findings; no metadata answer")

    target = _target(t)
    if target == "ambiguous":
        return QueryIntent(CLARIFY, reason="mentions both CST and BCVA")

    # 4. Largest change between consecutive visits.
    up, down = _has(CHANGE_UP, t), _has(CHANGE_DOWN, t)
    if target and (up or down) and _has(MAGNITUDE, t):
        if not _has(CONSECUTIVE, t):
            return QueryIntent(CLARIFY, target=target, reason="largest change requested without a reference "
                                                               "(previous visit? baseline?)")
        if up == down:
            return QueryIntent(CLARIFY, target=target, reason="change direction is ambiguous")
        return QueryIntent(RETRIEVE, CONSECUTIVE_CHANGE, target, direction="increase" if up else "decrease")

    # 5. First crossing of a threshold relative to baseline.
    no_ref = re.sub(BASELINE_REF, " ", t)
    if target and _has(BASELINE_REF, t) and _has(FIRST_OP, no_ref):
        thr = _threshold(t, target)
        direction = _direction_words(no_ref, target)
        if direction is None:  # "N letters above/below baseline": the reference preposition carries the direction
            prep = re.search(BASELINE_REF, t).group(1)
            direction = {"above": "increase", "below": "decrease"}.get(prep)
        if thr is None or direction is None:
            return QueryIntent(CLARIFY, target=target, reason="threshold or direction of change not specified")
        return QueryIntent(RETRIEVE, THRESHOLD_FROM_BASELINE, target, direction=direction, threshold=thr)

    # 6. Most recent (or first) visit satisfying a comparison with the current value.
    if target:
        cmp = _compare_to_current(t, target)
        if cmp is not None:
            return cmp

    # 7. Similarity to the current value.
    if target and _has(SIMILAR, t):
        return QueryIntent(RETRIEVE, CLOSEST_TO_CURRENT, target)

    # 8. Extremum of a recorded measurement.
    if target:
        mx, mn = _has(EXT_MAX, t), _has(EXT_MIN, t)
        if _has(EXT_WORST, t):
            mx, mn = (mx or target == "cst"), (mn or target == "bcva")
        if _has(EXT_BEST, t):
            mx, mn = (mx or target == "bcva"), (mn or target == "cst")
        if mx and mn:
            return QueryIntent(CLARIFY, target=target, reason="both maximum and minimum requested")
        if mx or mn:
            return QueryIntent(RETRIEVE, EXTREMUM, target, direction="max" if mx else "min")

    # 9-10. Temporal anchors.
    base, prev = _has(BASELINE_ANCHOR, t), _has(PREVIOUS_ANCHOR, t)
    if base and prev:
        return QueryIntent(CLARIFY, target=target, reason="refers to both baseline and previous visit")
    if base:
        return QueryIntent(RETRIEVE, ANCHOR_BASELINE, target)
    if prev:
        return QueryIntent(RETRIEVE, ANCHOR_PREVIOUS, target)

    return QueryIntent(CLARIFY, target=target, reason="intent not recognised")
