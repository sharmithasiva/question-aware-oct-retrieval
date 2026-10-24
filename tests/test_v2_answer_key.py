"""Tests for the v2 answer key. Written before the implementation; histories are hand-built."""
import numpy as np
import pandas as pd
import pytest

from oct_retrieval.ground_truth_v2 import (IE, INELIGIBLE, PRIMARY_TYPES, SECONDARY_TYPES, TYPES_V2, answer_key,
                                           generate_instances_v2)

NAN = float("nan")


def eye(cst, bcva, trial="TREX DME", weeks=None, leak=None, subject="S"):
    n = len(cst)
    return pd.DataFrame(dict(
        visit_id=[f"{subject}_{i}" for i in range(n)], visit_index=range(n), trial=trial,
        cst=np.array(cst, float), bcva=np.array(bcva, float),
        week=np.array(weeks if weeks is not None else [NAN] * n, float),
        leakage_index=np.array(leak if leak is not None else [NAN] * n, float),
    ))


def key(qtype, e, t):
    return answer_key(qtype, e.iloc[:t], e.iloc[t])


def test_type_registry():
    assert set(PRIMARY_TYPES) == {"largest_cst_rise", "largest_bcva_drop", "first_vision_gain_1line",
                                  "one_year_ago", "six_months_ago", "max_leakage"}
    assert set(SECONDARY_TYPES) == {"last_thicker_than_now", "last_vision_as_good"}
    assert "last_injection" not in TYPES_V2  # structured lookup, not a retrieval benchmark


# --- P1 / P2: consecutive changes ---

def test_largest_cst_rise_basic_and_baseline_never_relevant():
    e = eye([300, 280, 330, 320, 340, 0], [70] * 6)
    assert key("largest_cst_rise", e, 5) == [2]  # +50 at index 2 beats +20 at index 4


def test_largest_cst_rise_ties_all_relevant():
    e = eye([300, 320, 300, 320, 0], [70] * 5)
    assert key("largest_cst_rise", e, 4) == [1, 3]


def test_largest_cst_rise_insufficient_when_never_rose():
    e = eye([400, 350, 300, 0], [70] * 4)
    assert key("largest_cst_rise", e, 3) == IE


def test_largest_cst_rise_skips_changes_with_missing():
    e = eye([300, NAN, 400, 410, 0], [70] * 5)
    # changes at 1 and 2 involve NaN -> undefined; only index 3 (+10) is defined
    assert key("largest_cst_rise", e, 4) == [3]


def test_largest_cst_rise_ineligible_without_any_defined_change():
    e = eye([300, NAN, 0], [70] * 3)
    assert key("largest_cst_rise", e, 2) == INELIGIBLE


def test_largest_bcva_drop():
    e = eye([300] * 5, [70, 60, 65, 50, 0])
    assert key("largest_bcva_drop", e, 4) == [3]  # 65->50 = 15 letters
    e2 = eye([300] * 4, [60, 65, 70, 0])
    assert key("largest_bcva_drop", e2, 3) == IE


# --- P3 ---

def test_first_vision_gain_one_line_boundary():
    e = eye([300] * 5, [60, 64, 65, 70, 0])
    assert key("first_vision_gain_1line", e, 4) == [2]  # +4 is not a line, +5 is
    e2 = eye([300] * 4, [60, 64, 63, 0])
    assert key("first_vision_gain_1line", e2, 3) == IE
    e3 = eye([300] * 3, [NAN, 70, 0])
    assert key("first_vision_gain_1line", e3, 2) == INELIGIBLE


# --- P4 / P5: Prime timing ---

def prime_weeks(weeks):
    return eye([300] * len(weeks), [80] * len(weeks), trial="Prime_FULL", weeks=weeks)


def test_one_year_ago_exact_and_tolerance_boundary():
    assert key("one_year_ago", prime_weeks([0, 4, 8, 60]), 3) == [2]  # target 8
    assert key("one_year_ago", prime_weeks([0, 4, 56]), 2) == [1]  # target 4 exact
    assert key("one_year_ago", prime_weeks([0, 8, 64]), 2) == [1]  # target 12: |8-12| = 4 -> eligible
    assert key("one_year_ago", prime_weeks([0, 7, 64]), 2) == IE  # |7-12| = 5 -> insufficient


def test_one_year_ago_ineligible_before_week_52_and_for_trex():
    assert key("one_year_ago", prime_weeks([0, 4, 48]), 2) == INELIGIBLE
    trex = eye([300] * 3, [80] * 3, trial="TREX DME", weeks=[0, 4, 60])
    assert key("one_year_ago", trex, 2) == INELIGIBLE


def test_six_months_ago_ties_between_visits():
    # current W60 -> target 34; W32 and W36 are both 2 weeks away -> both acceptable
    assert key("six_months_ago", prime_weeks([0, 28, 32, 36, 60]), 4) == [2, 3]
    assert key("six_months_ago", prime_weeks([0, 4, 24]), 2) == INELIGIBLE  # week < 26


# --- P6: held-out leakage ---

def test_max_leakage_ties_missing_and_cohort():
    e = eye([300] * 5, [80] * 5, trial="Prime_FULL", weeks=[0, 4, 8, 12, 16], leak=[1.2, NAN, 3.4, 3.4, 9.9])
    assert key("max_leakage", e, 4) == [2, 3]  # current visit's 9.9 is never read
    e_missing = eye([300] * 3, [80] * 3, trial="Prime_FULL", weeks=[0, 4, 8], leak=[NAN, NAN, 1.0])
    assert key("max_leakage", e_missing, 2) == INELIGIBLE
    trex = eye([300] * 3, [80] * 3, leak=[1.0, 2.0, 3.0])
    assert key("max_leakage", trex, 2) == INELIGIBLE


# --- S1 / S2 ---

def test_last_thicker_strict_and_last_vision_inclusive():
    e = eye([350, 300, 320, 310, 320], [70, 75, 72, 69, 72])
    assert key("last_thicker_than_now", e, 4) == [0]  # 320 is not > 320; 350 is
    assert key("last_vision_as_good", e, 4) == [2]  # 72 >= 72
    assert key("last_thicker_than_now", eye([300, 310, 400], [70] * 3), 2) == IE
    assert key("last_thicker_than_now", eye([300, 310, NAN], [70] * 3), 2) == INELIGIBLE


# --- instance generation ---

def _episodes():
    a = eye([300, 340, 320, 330, 310], [70, 66, 72, 71, 60], subject="A").assign(
        eye_id=1, patient_id=1, split="test", n_images=49)
    b = eye([300, 300, 300, 300], [80, 80, 80, 80], trial="Prime_FULL", weeks=[0, 26, 52, 56], subject="B",
            leak=[0.5, 0.2, 0.9, 0.1]).assign(eye_id=2, patient_id=2, split="train", n_images=49)
    return pd.concat([a, b], ignore_index=True)


def _paraphrases():
    return {q: [f"{q} question {i}" for i in range(3)] for q in TYPES_V2}


def test_instances_have_counts_and_exclude_current():
    ep = _episodes()
    heldout = ep[["visit_id", "leakage_index"]]
    inst = generate_instances_v2(ep.drop(columns="leakage_index"), heldout, _paraphrases())
    assert {"n_relevant", "multi_answer", "answerable", "role", "tier"} <= set(inst.columns)
    by_id = ep.set_index("visit_id")
    for r in inst[inst.answerable].itertuples():
        rel = r.relevant_visit_ids.split(";")
        assert r.n_relevant == len(rel) and r.multi_answer == (len(rel) > 1)
        for v in rel:
            assert v != r.current_visit_id
            assert by_id.loc[v, "visit_index"] < by_id.loc[r.current_visit_id, "visit_index"]
            assert by_id.loc[v, "eye_id"] == r.eye_id
    ie = inst[~inst.answerable]
    assert (ie.relevant_visit_ids == "").all() and (ie.n_relevant == 0).all()


def test_episodes_passed_to_generator_must_not_contain_heldout():
    ep = _episodes()
    with pytest.raises(ValueError):
        generate_instances_v2(ep, ep[["visit_id", "leakage_index"]], _paraphrases())
