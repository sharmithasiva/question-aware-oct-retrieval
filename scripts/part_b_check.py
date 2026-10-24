"""fresh_v1 Part B: retrieval correctness on controlled histories (no natural language).

  python scripts/part_b_check.py

Compares three implementations of the 8 operations against each other and against hand-specified fixtures:
  retriever  = frozen src/oct_retrieval/intent_retrieval.retrieve (structured intents; parser bypassed)
  answer key = frozen src/oct_retrieval/ground_truth_v2.answer_key
  reference  = tests/reference/reference_ops.py (independent agent, written from definitions only)
Fixtures: tests/fixtures/retrieval_cases_fresh_v1.jsonl (independent agent, hand-derived expected answers).
Also runs seeded random property cases, real train/val histories, mutation tests and metamorphic tests.
Written after the code freeze; imports frozen modules read-only; mutants exist only in memory.
Never touches sealed/. Writes results/fresh_v1_partB/.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import oct_retrieval  # noqa: E402,F401
from oct_retrieval import intent as I  # noqa: E402
from oct_retrieval import intent_retrieval as R  # noqa: E402
from oct_retrieval.ground_truth_v2 import answer_key  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "retrieval_cases_fresh_v1.jsonl"
REFERENCE = ROOT / "tests" / "reference" / "reference_ops.py"
OUT = ROOT / "results" / "fresh_v1_partB"
SEED = 42
N_RANDOM = 2000
TYPES = ["largest_cst_rise", "largest_bcva_drop", "first_vision_gain_1line", "one_year_ago", "six_months_ago",
         "max_leakage", "last_thicker_than_now", "last_vision_as_good"]

# Structured intents (frozen type definitions; PROPOSED_CLARIFICATION_01 option (a)). max_leakage is tested as a
# declared-variable lookup here: Part B checks the operation, not the restricted-input routing (that is Part A).
STRUCT = {
    "largest_cst_rise": I.QueryIntent(I.RETRIEVE, I.CONSECUTIVE_CHANGE, "cst", direction="increase"),
    "largest_bcva_drop": I.QueryIntent(I.RETRIEVE, I.CONSECUTIVE_CHANGE, "bcva", direction="decrease"),
    "first_vision_gain_1line": I.QueryIntent(I.RETRIEVE, I.THRESHOLD_FROM_BASELINE, "bcva", direction="increase",
                                             threshold=5.0),
    "one_year_ago": I.QueryIntent(I.RETRIEVE, I.TIME_OFFSET, offset_weeks=52, tolerance_weeks=4),
    "six_months_ago": I.QueryIntent(I.RETRIEVE, I.TIME_OFFSET, offset_weeks=26, tolerance_weeks=4),
    "max_leakage": I.QueryIntent(I.RETRIEVE, I.EXTREMUM, "leakage_index", direction="max"),
    "last_thicker_than_now": I.QueryIntent(I.RETRIEVE, I.COMPARE_TO_CURRENT, "cst", comparator="gt", recency="last"),
    "last_vision_as_good": I.QueryIntent(I.RETRIEVE, I.COMPARE_TO_CURRENT, "bcva", comparator="ge", recency="last"),
}
STATUS = {R.RETRIEVED: "visits", R.NO_QUALIFYING: "no_qualifying_visit", R.INSUFFICIENT: "insufficient_data"}


def load_reference():
    spec = importlib.util.spec_from_file_location("reference_ops", REFERENCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _arr(case, key):
    return np.array([np.nan if v.get(key) is None else float(v[key]) for v in case["visits"]], float)


# ------------------------------------------------------------------ the three implementations, one interface
def run_retriever(case, retrieve=R.retrieve, extra_visits=None):
    visits = case["visits"] + (extra_visits or [])
    c = dict(case, visits=visits)
    mem = R.VisitMemory([f"p{i}" for i in range(len(visits))], np.arange(len(visits)), _arr(c, "cst"),
                        _arr(c, "bcva"), _arr(c, "week"), np.full(len(visits), 49),
                        "Prime_FULL" if case["cohort"] == "Prime" else "TREX DME",
                        {"leakage_index": _arr(c, "leakage")} if case["qtype"] == "max_leakage" else {})
    res = retrieve(STRUCT[case["qtype"]], mem, case["current_pos"])
    return STATUS.get(res.status, f"other:{res.status}"), list(res.ranked)


def run_key(case):
    t = case["current_pos"]
    df = pd.DataFrame(dict(
        visit_id=[f"p{i}" for i in range(len(case["visits"]))], visit_index=range(len(case["visits"])),
        trial="Prime_FULL" if case["cohort"] == "Prime" else "TREX DME",
        cst=_arr(case, "cst"), bcva=_arr(case, "bcva"), week=_arr(case, "week"), leakage_index=_arr(case, "leakage")))
    rel = answer_key(case["qtype"], df.iloc[:t], df.iloc[t])
    if rel is None:
        return "insufficient_data", set()
    return ("visits", set(rel)) if rel else ("no_qualifying_visit", set())


def run_ref(ref, case):
    out, pos = ref.reference(case)
    return out, set(pos)


def retriever_matches(r_out, ranked, exp_out, exp_set) -> bool:
    """Retriever returns a ranking: correct iff same outcome and its top-|expected| block equals the expected set."""
    if r_out != exp_out:
        return False
    return exp_out != "visits" or set(ranked[:len(exp_set)]) == exp_set


# ------------------------------------------------------------------------------------- random property cases
def random_case(rng, qtype):
    n = int(rng.choice([2, 3, 3, 4, 5, 6, 7, 8, 10, 12, 16]))
    prime = qtype in ("one_year_ago", "six_months_ago", "max_leakage") and rng.random() < 0.85 or rng.random() < 0.5
    weeks, w = [], 0
    for _ in range(n):
        weeks.append(w)
        w += int(rng.choice([4, 4, 4, 8, 8, 3, 5, 12, 2]))
    nan = lambda p=0.1: rng.random() < p  # noqa: E731
    visits = []
    for i in range(n):
        visits.append(dict(
            pos=i, week=(None if not prime or nan(0.03) else weeks[i]),
            cst=None if nan() else float(rng.choice([250, 260, 270, 280, 300, 320, 350, 400])),
            bcva=None if nan() else float(rng.choice([55, 58, 60, 63, 65, 68, 70, 75])),
            leakage=None if (not prime or nan(0.15)) else float(rng.choice([0.2, 0.5, 0.8, 1.0, 1.5, 2.0]))))
    return dict(case_id=f"R_{qtype}", qtype=qtype, cohort="Prime" if prime else "TREX", visits=visits,
                current_pos=n - 1)


def real_cases():
    from oct_retrieval.episodes import build_all
    from oct_retrieval.side_tables import build_heldout_prime
    ep, _ = build_all()
    held = build_heldout_prime(ep)[["visit_id", "leakage_index"]]
    dev = ep[ep.split.isin(["train", "val"])].merge(held, on="visit_id", how="left")
    for eye_id, g in dev.sort_values(["eye_id", "visit_index"]).groupby("eye_id"):
        visits = [dict(pos=int(r.visit_index), week=None if pd.isna(r.week) else float(r.week),
                       cst=None if pd.isna(r.cst) else float(r.cst), bcva=None if pd.isna(r.bcva) else float(r.bcva),
                       leakage=None if pd.isna(r.leakage_index) else float(r.leakage_index)) for r in g.itertuples()]
        cohort = "Prime" if g.trial.iloc[0] == "Prime_FULL" else "TREX"
        for t in range(1, len(visits)):
            for q in TYPES:
                yield dict(case_id=f"E{eye_id}_t{t}_{q}", qtype=q, cohort=cohort, visits=visits[:t + 1],
                           current_pos=t)


# ------------------------------------------------------------------------------------------------ mutants
MUTANTS = {
    "M1_strict_gt_becomes_ge": ('"gt": v > cur', '"gt": v >= cur'),
    "M2_tolerance_plus_one": ("if np.nanmin(dist) > q.tolerance_weeks:", "if np.nanmin(dist) > q.tolerance_weeks + 1:"),
    "M3_current_visit_is_candidate": ("v, cur = v_all[:n].astype(float), float(v_all[n])",
                                      "v, cur = v_all[:n + 1].astype(float), float(v_all[n])"),
    "M4_baseline_gets_a_change": ("change = np.r_[np.nan, np.diff(v)]",
                                  "change = np.r_[(v[1] - v[0]) if len(v) > 1 else np.nan, np.diff(v)]"),
    "M5_threshold_strict": ("(delta >= q.threshold)", "(delta > q.threshold)"),
    "M6_first_becomes_latest": ("_qualifying_first(mask, False)", "_qualifying_first(mask, True)"),
    "M7_target_week_off_by_one": ("target_week = weeks[n] - q.offset_weeks", "target_week = weeks[n] - q.offset_weeks + 1"),
    "M8_tie_break_reversed": None,   # patched _order (earliest-first among ties)
}


def mutant_retrieve(name):
    src = (ROOT / "src" / "oct_retrieval" / "intent_retrieval.py").read_text(encoding="utf-8")
    mod = types.ModuleType(f"oct_retrieval._mutant_{name}")
    mod.__package__ = "oct_retrieval"
    if MUTANTS[name] is not None:
        old, new = MUTANTS[name]
        assert src.count(old) == 1, f"mutation anchor not unique: {name}"
        src = src.replace(old, new)
    sys.modules[mod.__name__] = mod   # dataclasses need the module registered
    exec(compile(src, f"<mutant {name}>", "exec"), mod.__dict__)
    if name == "M8_tie_break_reversed":
        def _order_rev(scores):
            s = np.where(np.isnan(scores), -np.inf, scores)
            return sorted(range(len(scores)), key=lambda i: (-s[i], i))
        mod._order = _order_rev
    return mod.retrieve


# ------------------------------------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = load_reference()
    fixtures = [json.loads(x) for x in FIXTURES.read_text(encoding="utf-8").splitlines() if x.strip()]
    disagreements = []
    summary = {}

    # 1. fixtures vs hand-specified expected answers
    fx = defaultdict(Counter)
    for c in fixtures:
        exp = (c["expected_outcome"], set(c["expected_positions"]))
        r_out, ranked = run_retriever(c)
        k, rf = run_key(c), run_ref(ref, c)
        ok = dict(retriever=retriever_matches(r_out, ranked, *exp), answer_key=k == exp, reference=rf == exp)
        fx[c["qtype"]]["cases"] += 1
        for impl, good in ok.items():
            fx[c["qtype"]][impl] += good
        if not all(ok.values()):
            disagreements.append(dict(source="fixture", case_id=c["case_id"], qtype=c["qtype"],
                                      expected=[exp[0], sorted(exp[1])], retriever=[r_out, ranked[:4]],
                                      answer_key=[k[0], sorted(k[1])], reference=[rf[0], sorted(rf[1])],
                                      coverage=c.get("coverage"), rationale=c.get("rationale")))
    summary["fixtures"] = {q: dict(fx[q]) for q in TYPES}
    summary["fixture_coverage"] = {q: dict(Counter(t for c in fixtures if c["qtype"] == q for t in c["coverage"]))
                                   for q in TYPES}

    # 2. three-way agreement on random and real cases (no hand-specified answer: reference is the comparator)
    rng = np.random.default_rng(SEED)
    random_cases = [random_case(rng, q) for q in TYPES for _ in range(N_RANDOM)]
    for label, cases in (("random", random_cases), ("real_train_val", list(real_cases()))):
        agg = defaultdict(Counter)
        kept = defaultdict(list)
        for c in cases:
            rf = run_ref(ref, c)
            k = run_key(c)
            r_out, ranked = run_retriever(c)
            res = dict(retriever_vs_reference=retriever_matches(r_out, ranked, *rf), key_vs_reference=k == rf,
                       retriever_vs_key=retriever_matches(r_out, ranked, *k))
            agg[c["qtype"]]["cases"] += 1
            for name, good in res.items():
                agg[c["qtype"]][name] += good
            if not all(res.values()):
                kept[c["qtype"]].append(dict(source=label, case_id=c["case_id"], qtype=c["qtype"],
                                             n_visits=len(c["visits"]), cohort=c["cohort"], visits=c["visits"],
                                             reference=[rf[0], sorted(rf[1])], answer_key=[k[0], sorted(k[1])],
                                             retriever=[r_out, ranked[:4]]))
        for q, rows in kept.items():   # keep the 3 smallest disagreements per type
            disagreements += sorted(rows, key=lambda r: r["n_visits"])[:3]
        summary[label] = {q: dict(agg[q]) for q in TYPES}

    # 3. mutation tests. A mutant is "killed" only by NEW failures: cases the unmutated retriever gets right
    #    (vs fixture expectation / reference) and the mutant gets wrong or crashes on. Cases the unmutated retriever
    #    already fails (the min-history defect) are excluded, so they cannot inflate kill counts.
    fx_targets = [(c, (c["expected_outcome"], set(c["expected_positions"]))) for c in fixtures]
    rnd_targets = [(c, run_ref(ref, c)) for c in random_cases]
    fx_base_ok = [retriever_matches(*run_retriever(c), *e) for c, e in fx_targets]
    rnd_base_ok = [retriever_matches(*run_retriever(c), *e) for c, e in rnd_targets]
    mut = {}
    for name in MUTANTS:
        retrieve = mutant_retrieve(name)
        crashes = Counter()

        def fails(c, exp):
            try:
                return not retriever_matches(*run_retriever(c, retrieve), *exp)
            except Exception:   # a crashing mutant is also detected
                crashes["n"] += 1
                return True
        fx_new = sum(ok and fails(c, e) for (c, e), ok in zip(fx_targets, fx_base_ok))
        rnd_new = sum(ok and fails(c, e) for (c, e), ok in zip(rnd_targets, rnd_base_ok))
        mut[name] = dict(new_fixture_failures=fx_new, new_random_failures=rnd_new, crashes=crashes["n"],
                         killed=bool(fx_new or rnd_new))
    summary["mutation_baseline"] = dict(fixture_cases_scored=int(sum(fx_base_ok)),
                                        random_cases_scored=int(sum(rnd_base_ok)))
    summary["mutants"] = mut

    # 4. metamorphic tests (on random cases)
    mm = Counter()
    rng2 = np.random.default_rng(SEED + 1)
    for c in random_cases:
        base_r, base_k, base_f = run_retriever(c), run_key(c), run_ref(ref, c)
        shifted = dict(c, visits=[dict(v, cst=None if v["cst"] is None else v["cst"] + 37.0) for v in c["visits"]])
        mm["cst_shift_cases"] += 1
        mm["cst_shift_retriever_ok"] += run_retriever(shifted) == base_r
        mm["cst_shift_key_ok"] += run_key(shifted) == base_k
        mm["cst_shift_reference_ok"] += run_ref(ref, shifted) == base_f
        later = [dict(pos=len(c["visits"]) + j, week=None, cst=float(rng2.choice([200, 900])),
                      bcva=float(rng2.choice([20, 95])), leakage=9.9) for j in range(int(rng2.integers(1, 4)))]
        mm["append_after_current_cases"] += 1
        mm["append_after_current_retriever_ok"] += run_retriever(c, extra_visits=later) == base_r
    summary["metamorphic"] = dict(mm)
    summary["metamorphic_note"] = ("Renaming visit IDs is not tested separately: retriever, key adapter and reference "
                                   "all operate on positions, so it is a no-op for this harness.")

    (OUT / "partB_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "partB_disagreements.jsonl").write_text("".join(json.dumps(d) + "\n" for d in disagreements))
    print(json.dumps(summary, indent=2))
    print("disagreement records:", len(disagreements))


if __name__ == "__main__":
    main()
