"""INTERPRETATION CHOICES

1. "Earlier visits" means visits[0:current_pos] (positions < current_pos). Baseline (pos 0) is a candidate
   for every type unless the definition says i >= 1 (types 1, 2, 3).
2. The common "at least 2 earlier visits" check (current_pos >= 2) runs first, before any type-specific check.
3. Types 1/2: only changes where both i and i-1 are earlier visits (1 <= i < current_pos) are considered; the
   change into the current visit is not a candidate.
4. Type 3: positions with missing BCVA are skipped; the search range is 1 <= i < current_pos.
5. Types 4/5/6: the cohort check (TREX -> insufficient_data) runs before the other insufficient-data checks.
6. Types 4/5: the threshold is week >= 52 (or 26), so week exactly 52 gives target 0. Distance <= 4 qualifies.
7. Numeric ties and the <= 4 week threshold use an absolute tolerance of 1e-9 to absorb floating-point noise.
8. Types 7/8: earlier visits with missing CST/BCVA are skipped; "latest" means the highest position.
9. Type 6: leakage values are compared as given (no tolerance beyond 1e-9); missing leakage is skipped.
10. Visits are indexed by list position; the "pos" field is assumed to match the list index.
11. An unknown qtype raises ValueError.
"""

EPS = 1e-9


def _earlier(case):
    return case["visits"][:case["current_pos"]]


def _can_pose(case):
    return case["current_pos"] >= 2


def _argmax(pairs):
    """pairs: list of (pos, value). Returns (max_value, set of positions tied at max)."""
    best = None
    for _, v in pairs:
        if best is None or v > best:
            best = v
    return best, {p for p, v in pairs if abs(v - best) <= EPS}


def largest_cst_rise(case):
    if not _can_pose(case):
        return ("insufficient_data", set())
    ev = _earlier(case)
    pairs = []
    for i in range(1, len(ev)):
        a, b = ev[i - 1]["cst"], ev[i]["cst"]
        if a is not None and b is not None:
            pairs.append((i, b - a))
    if not pairs:
        return ("insufficient_data", set())
    best, pos = _argmax(pairs)
    if best <= EPS:
        return ("no_qualifying_visit", set())
    return ("visits", pos)


def largest_bcva_drop(case):
    if not _can_pose(case):
        return ("insufficient_data", set())
    ev = _earlier(case)
    pairs = []
    for i in range(1, len(ev)):
        a, b = ev[i - 1]["bcva"], ev[i]["bcva"]
        if a is not None and b is not None:
            pairs.append((i, a - b))
    if not pairs:
        return ("insufficient_data", set())
    best, pos = _argmax(pairs)
    if best <= EPS:
        return ("no_qualifying_visit", set())
    return ("visits", pos)


def first_vision_gain_1line(case):
    if not _can_pose(case):
        return ("insufficient_data", set())
    ev = _earlier(case)
    base = ev[0]["bcva"]
    if base is None:
        return ("insufficient_data", set())
    for i in range(1, len(ev)):
        b = ev[i]["bcva"]
        if b is not None and b >= base + 5 - EPS:
            return ("visits", {i})
    return ("no_qualifying_visit", set())


def _weeks_ago(case, offset):
    if not _can_pose(case):
        return ("insufficient_data", set())
    if case["cohort"] != "Prime":
        return ("insufficient_data", set())
    cur_week = case["visits"][case["current_pos"]]["week"]
    if cur_week is None or cur_week < offset:
        return ("insufficient_data", set())
    target = cur_week - offset
    ev = _earlier(case)
    dists = [(i, abs(v["week"] - target)) for i, v in enumerate(ev) if v["week"] is not None]
    if not dists:
        return ("insufficient_data", set())
    best = min(d for _, d in dists)
    if best > 4 + EPS:
        return ("no_qualifying_visit", set())
    return ("visits", {i for i, d in dists if abs(d - best) <= EPS})


def one_year_ago(case):
    return _weeks_ago(case, 52)


def six_months_ago(case):
    return _weeks_ago(case, 26)


def max_leakage(case):
    if not _can_pose(case):
        return ("insufficient_data", set())
    if case["cohort"] != "Prime":
        return ("insufficient_data", set())
    ev = _earlier(case)
    pairs = [(i, v["leakage"]) for i, v in enumerate(ev) if v["leakage"] is not None]
    if not pairs:
        return ("insufficient_data", set())
    _, pos = _argmax(pairs)
    return ("visits", pos)


def last_thicker_than_now(case):
    if not _can_pose(case):
        return ("insufficient_data", set())
    cur = case["visits"][case["current_pos"]]["cst"]
    if cur is None:
        return ("insufficient_data", set())
    ev = _earlier(case)
    for i in range(len(ev) - 1, -1, -1):
        c = ev[i]["cst"]
        if c is not None and c > cur + EPS:
            return ("visits", {i})
    return ("no_qualifying_visit", set())


def last_vision_as_good(case):
    if not _can_pose(case):
        return ("insufficient_data", set())
    cur = case["visits"][case["current_pos"]]["bcva"]
    if cur is None:
        return ("insufficient_data", set())
    ev = _earlier(case)
    for i in range(len(ev) - 1, -1, -1):
        b = ev[i]["bcva"]
        if b is not None and b >= cur - EPS:
            return ("visits", {i})
    return ("no_qualifying_visit", set())


_DISPATCH = {
    "largest_cst_rise": largest_cst_rise,
    "largest_bcva_drop": largest_bcva_drop,
    "first_vision_gain_1line": first_vision_gain_1line,
    "one_year_ago": one_year_ago,
    "six_months_ago": six_months_ago,
    "max_leakage": max_leakage,
    "last_thicker_than_now": last_thicker_than_now,
    "last_vision_as_good": last_vision_as_good,
}


def reference(case):
    fn = _DISPATCH.get(case["qtype"])
    if fn is None:
        raise ValueError("unknown qtype: %r" % (case["qtype"],))
    return fn(case)
