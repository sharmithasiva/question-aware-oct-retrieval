"""Integration tests on the real OLIVES files: alignment, conventions, isolation, frozen paraphrases, v1 regression."""
import json

import numpy as np
import pandas as pd
import pytest

from oct_retrieval import config
from oct_retrieval.ground_truth_v2 import TYPES_V2, generate_instances_v2
from oct_retrieval.paraphrases import DEV, LOCKED, RAW, locked_by_type, read_jsonl, verify
from oct_retrieval.side_tables import HELDOUT_COLUMNS, build_heldout_prime, prime_alignment, trex_alignment

SNAPSHOT = config.RESULTS_DIR / "v1_snapshot"


@pytest.fixture(scope="module")
def episodes(real_data):
    return real_data[0]


@pytest.fixture(scope="module")
def heldout(episodes):
    return build_heldout_prime(episodes)


@pytest.fixture(scope="module")
def instances_v2(episodes, heldout):
    return generate_instances_v2(episodes, heldout, locked_by_type())


def test_side_table_alignment(episodes):
    assert prime_alignment(episodes) == {"visits": 660, "cst_match": 660, "bcva_match": 660}
    assert trex_alignment(episodes) == {"visits": 934, "cst_match": 933, "bcva_match": 933}


def test_bcva_is_etdrs_letter_score(episodes):
    b = episodes.bcva.dropna()
    assert (b % 1 == 0).all() and b.min() >= 0 and b.max() <= 100


def test_leakage_coverage_and_missing(heldout):
    assert len(heldout) == 660
    missing = set(heldout.loc[heldout.leakage_index.isna(), "visit_id"])
    assert missing == {"02-010_W4", "02-008_W4", "02-008_W8", "02-015_W16", "02-042_W8"}
    assert heldout.leakage_index.min() >= 0


def test_heldout_never_in_retriever_memory(episodes):
    banned = [c for c in episodes.columns if any(k in c.lower() for k in ("leak", "drss", "inject"))]
    assert banned == []
    assert not set(HELDOUT_COLUMNS) & set(episodes.columns)


def test_retriever_inputs_carry_no_heldout(episodes, instances_v2):
    from oct_retrieval.evaluate import eye_histories, run_retriever
    from oct_retrieval.retrievers import Retriever

    seen = []

    class Spy(Retriever):
        name = "spy"

        def rank(self, spec, hist, current, rng):
            seen.append((set(vars(hist)), set(current)))
            return list(range(len(hist)))

    sample = instances_v2[instances_v2.answerable & (instances_v2.qtype == "max_leakage")].head(50)
    run_retriever(Spy(), sample, eye_histories(episodes))
    assert seen
    for hist_fields, current_keys in seen:
        assert hist_fields == {"visit_ids", "cst", "bcva", "n_images"}
        assert current_keys == {"cst", "bcva", "visit_id"}


def test_paraphrases_frozen_and_disjoint():
    verify()
    raw, dev, locked = read_jsonl(RAW), read_jsonl(DEV), read_jsonl(LOCKED)
    as_set = lambda rows: {(r["qtype"], r["text"]) for r in rows}  # noqa: E731
    assert not as_set(dev) & as_set(locked)
    assert as_set(dev) | as_set(locked) == as_set(raw)
    by_type = locked_by_type()
    assert all(len(by_type[q]) == 5 for q in TYPES_V2)


def test_v2_counts_match_spec(instances_v2):
    spec = {  # (all eligible, all answerable) from docs/PHASE1_V2_QUESTION_SPEC.md §4
        "largest_cst_rise": (1402, 1113), "largest_bcva_drop": (1402, 1228),
        "first_vision_gain_1line": (1402, 988), "one_year_ago": (192, 191), "six_months_ago": (392, 371),
        "max_leakage": (580, 580), "last_thicker_than_now": (1401, 1348), "last_vision_as_good": (1401, 1141),
    }
    got = {q: (len(g), int(g.answerable.sum())) for q, g in instances_v2.groupby("qtype")}
    assert got == spec


def test_v2_instances_use_only_locked_text(instances_v2):
    locked = {t for texts in locked_by_type().values() for t in texts}
    assert set(instances_v2.question) <= locked


@pytest.mark.skipif(not (SNAPSHOT / "summary.csv").exists(), reason="v1 snapshot missing")
def test_v1_test_split_metrics_reproduce_snapshot(episodes):
    """The frozen v1 pipeline must give exactly the snapshotted test-split metrics."""
    from oct_retrieval.evaluate import eye_histories, eye_macro, run_retriever
    from oct_retrieval.questions import generate_instances
    from oct_retrieval.retrievers import (ClinicalOnly, Combined, EarliestFirst, LatestPrior, RandomHistory,
                                          RuleRouter, TemporalOnly)

    alpha = json.loads((SNAPSHOT / "run_config.json").read_text())["alpha"]
    inst = generate_instances(episodes)
    snap_inst = pd.read_csv(SNAPSHOT / "instances.csv")
    assert list(inst.instance_id) == list(snap_inst.instance_id)
    test = inst[inst.split == "test"]
    eyes = eye_histories(episodes)
    snap = pd.read_csv(SNAPSHOT / "summary.csv").query("split == 'test'").set_index("retriever")
    for r in [LatestPrior(), EarliestFirst(), RandomHistory(), TemporalOnly(), ClinicalOnly(), Combined(alpha),
              RuleRouter()]:
        m = eye_macro(run_retriever(r, test, eyes))
        for col in m.index:
            assert m[col] == pytest.approx(snap.loc[r.name, col], abs=1e-12), (r.name, col)
