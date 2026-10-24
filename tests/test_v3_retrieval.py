"""Phase 1.3 retrieval tests on hand-built memories, plus protocol and preservation checks."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oct_retrieval import config
from oct_retrieval import intent as I
from oct_retrieval import intent_retrieval as R

NAN = float("nan")


def mem(cst, bcva, weeks=None, trial="Prime_FULL", extra=None):
    n = len(cst)
    return R.VisitMemory([f"v{i}" for i in range(n)], np.arange(n), np.array(cst, float), np.array(bcva, float),
                         np.array(weeks if weeks is not None else [NAN] * n, float), np.full(n, 49), trial,
                         extra or {})


def intent(op, **kw):
    return I.QueryIntent(I.RETRIEVE, op, **kw)


# --- outcomes are distinct -------------------------------------------------------------------

def test_three_kinds_of_non_answer_are_distinct():
    m = mem([300, 280, 260, 250], [70] * 4)
    assert R.retrieve(I.QueryIntent(I.CLARIFY, reason="x"), m, 3).status == R.CLARIFY
    assert R.retrieve(I.QueryIntent(I.UNSUPPORTED_IMAGE), m, 3).status == R.UNSUPPORTED
    assert R.retrieve(I.QueryIntent(I.UNAVAILABLE, missing_variable="leakage_index"), m, 3).status == R.UNAVAILABLE
    rise = R.retrieve(intent(I.CONSECUTIVE_CHANGE, target="cst", direction="increase"), m, 3)
    assert rise.status == R.NO_QUALIFYING and rise.ranked == ()   # recognised, but CST never rose


def test_unrecognised_question_never_falls_back_to_latest():
    q, res = R.answer("Tell me something about this eye", mem([300, 290, 280], [70] * 3), 2)
    assert res.status == R.CLARIFY and res.ranked == ()


# --- consecutive change: ties, missing values ---------------------------------------------------

def test_consecutive_rise_ties_and_missing():
    m = mem([300, 320, NAN, 340, 320, 340, 0], [70] * 7)
    res = R.retrieve(intent(I.CONSECUTIVE_CHANGE, target="cst", direction="increase"), m, 6)
    # defined changes: i1 +20, i2 NaN, i3 NaN, i4 -20, i5 +20 -> tie i1/i5, recency first
    assert res.status == R.RETRIEVED and res.ranked[:2] == (5, 1)


def test_consecutive_bcva_drop_direction():
    m = mem([300] * 5, [70, 60, 65, 50, 0])
    res = R.retrieve(intent(I.CONSECUTIVE_CHANGE, target="bcva", direction="decrease"), m, 4)
    assert res.ranked[0] == 3


def test_consecutive_change_insufficient_without_pairs():
    m = mem([300, NAN, 0], [70] * 3)
    assert R.retrieve(intent(I.CONSECUTIVE_CHANGE, target="cst", direction="increase"), m, 2).status == R.INSUFFICIENT


# --- first threshold crossing ------------------------------------------------------------------

def test_first_threshold_crossing_boundary():
    m = mem([300] * 5, [60, 64, 65, 70, 0])
    res = R.retrieve(intent(I.THRESHOLD_FROM_BASELINE, target="bcva", direction="increase", threshold=5.0), m, 4)
    assert res.ranked[0] == 2           # +4 does not count, +5 does
    m2 = mem([300] * 4, [60, 64, 63, 0])
    assert R.retrieve(intent(I.THRESHOLD_FROM_BASELINE, target="bcva", direction="increase", threshold=5.0),
                      m2, 3).status == R.NO_QUALIFYING
    m3 = mem([300] * 3, [NAN, 70, 0])
    assert R.retrieve(intent(I.THRESHOLD_FROM_BASELINE, target="bcva", direction="increase", threshold=5.0),
                      m3, 2).status == R.INSUFFICIENT


# --- comparison with the current value ----------------------------------------------------------

def test_last_thicker_strict_and_last_vision_inclusive():
    m = mem([350, 300, 320, 310, 320], [70, 75, 72, 69, 72])
    gt = R.retrieve(intent(I.COMPARE_TO_CURRENT, target="cst", comparator="gt", recency="last"), m, 4)
    ge = R.retrieve(intent(I.COMPARE_TO_CURRENT, target="bcva", comparator="ge", recency="last"), m, 4)
    assert gt.ranked[0] == 0 and ge.ranked[0] == 2
    none = R.retrieve(intent(I.COMPARE_TO_CURRENT, target="cst", comparator="gt", recency="last"),
                      mem([300, 310, 400], [70] * 3), 2)
    assert none.status == R.NO_QUALIFYING
    missing = R.retrieve(intent(I.COMPARE_TO_CURRENT, target="cst", comparator="gt", recency="last"),
                         mem([300, 310, NAN], [70] * 3), 2)
    assert missing.status == R.INSUFFICIENT


# --- temporal boundaries and eligibility -------------------------------------------------------

T52 = dict(offset_weeks=52, tolerance_weeks=4)


def weeks_mem(weeks, trial="Prime_FULL"):
    return mem([300] * len(weeks), [80] * len(weeks), weeks=weeks, trial=trial)


def test_time_offset_tolerance_boundary():
    assert R.retrieve(intent(I.TIME_OFFSET, **T52), weeks_mem([0, 8, 64]), 2).ranked[0] == 1   # |8-12| = 4
    assert R.retrieve(intent(I.TIME_OFFSET, **T52), weeks_mem([0, 7, 64]), 2).status == R.NO_QUALIFYING


def test_time_offset_ties_and_eligibility():
    res = R.retrieve(intent(I.TIME_OFFSET, offset_weeks=26, tolerance_weeks=4), weeks_mem([0, 28, 32, 36, 60]), 4)
    assert set(res.ranked[:2]) == {2, 3}
    assert R.retrieve(intent(I.TIME_OFFSET, **T52), weeks_mem([0, 4, 48]), 2).status == R.INSUFFICIENT
    trex = weeks_mem([NAN, NAN, NAN], trial="TREX DME")
    assert R.retrieve(intent(I.TIME_OFFSET, **T52), trex, 2).status == R.INSUFFICIENT


# --- extremum / declared lookup ---------------------------------------------------------------

def test_extremum_nan_last_and_declared_variable():
    m = mem([300, NAN, 400, 350], [70] * 4, extra={"leakage_index": np.array([1.0, 3.0, 2.0, 9.0])})
    assert R.retrieve(intent(I.EXTREMUM, target="cst", direction="max"), m, 3).ranked == (2, 0, 1)
    assert R.retrieve(intent(I.EXTREMUM, target="leakage_index", direction="max"), m, 3).ranked[0] == 1
    m_default = mem([300, 310, 320], [70] * 3)
    assert R.retrieve(intent(I.EXTREMUM, target="leakage_index", direction="max"), m_default, 2).status == R.UNAVAILABLE


# --- leakage / protocol / preservation ------------------------------------------------------------

def test_memory_refuses_heldout_columns():
    ep = pd.DataFrame(dict(visit_id=["a"], eye_id=[1], visit_index=[0], cst=[300.0], bcva=[70.0], week=[0.0],
                           n_images=[49], trial=["Prime_FULL"], leakage_index=[1.0]))
    with pytest.raises(ValueError):
        R.build_memories(ep)


def test_real_memories_carry_only_declared_fields(real_data):
    mems = R.build_memories(real_data[0])
    m = next(iter(mems.values()))
    assert set(vars(m)) == {"visit_ids", "visit_index", "cst", "bcva", "week", "n_images", "trial", "extra"}
    assert all(mm.extra == {} for mm in mems.values())


def test_new_retriever_does_not_import_answer_key():
    import re
    for name in ("intent_retrieval.py", "intent.py"):
        src = (config.ROOT / "src" / "oct_retrieval" / name).read_text(encoding="utf-8")
        imports = [ln for ln in src.splitlines() if re.match(r"\s*(from|import)\s", ln)]
        assert not any("ground_truth_v2" in ln or "relevant_indices" in ln or "questions" in ln for ln in imports)


def test_dev_runner_never_touches_locked_or_test():
    src = (config.ROOT / "src" / "oct_retrieval" / "evaluate_v3_dev.py").read_text(encoding="utf-8")
    assert "locked_by_type" not in src and "LOCKED" not in src


def test_dev_episodes_exclude_test_patients(real_data):
    from oct_retrieval.evaluate_v3_dev import dev_episodes
    ep = real_data[0]
    dev = dev_episodes(ep)
    assert set(dev.split) == {"train", "val"}
    assert not set(dev.patient_id) & set(ep.loc[ep.split == "test", "patient_id"])
    assert (ep.groupby("patient_id").split.nunique() == 1).all()


FROZEN = config.ROOT / "results" / "v3_dev" / "frozen_artifacts.md5"


@pytest.mark.skipif(not FROZEN.exists(), reason="no frozen-artifact manifest")
def test_v1_v2_artifacts_unchanged():
    for line in FROZEN.read_text().splitlines():
        digest, path = line.split(" *", 1)
        assert hashlib.md5((config.ROOT / path).read_bytes()).hexdigest() == digest, path


def test_v1_snapshot_manifest_verifies():
    snap = config.ROOT / "results" / "v1_snapshot"
    for line in (snap / "MANIFEST.md5").read_text().splitlines():
        digest, name = line.split(" *", 1) if " *" in line else line.split("  ", 1)
        assert hashlib.md5((snap / name).read_bytes()).hexdigest() == digest, name
