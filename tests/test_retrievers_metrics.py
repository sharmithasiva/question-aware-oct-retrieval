import math

import numpy as np
import pytest

from oct_retrieval.metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from oct_retrieval.questions import QuestionSpec
from oct_retrieval.retrievers import (ClinicalOnly, Combined, EarliestFirst, History, LatestPrior, RandomHistory,
                                      RuleRouter, TemporalOnly)

NAN = float("nan")
HIST = History(["a", "b", "c", "d"], np.array([400.0, 250.0, NAN, 300.0]), np.array([60.0, 75.0, 70.0, 65.0]),
               np.array([49, 49, 49, 49]))
CUR = {"cst": 260.0, "bcva": 72.0}
RNG = np.random.default_rng(0)

WORST = QuestionSpec("cst", "extreme", "max", "metadata")
BASE = QuestionSpec("cst", "baseline", None, "metadata")
PREV = QuestionSpec("cst", "previous", None, "metadata")
SIM = QuestionSpec("cst", "similar", None, "metadata")
BEST_VA = QuestionSpec("bcva", "extreme", "max", "metadata")


def test_latest_and_earliest():
    assert LatestPrior().rank(WORST, HIST, CUR, RNG) == [3, 2, 1, 0]
    assert EarliestFirst().rank(WORST, HIST, CUR, RNG) == [0, 1, 2, 3]


def test_random_is_seeded_permutation():
    r1 = RandomHistory().rank(WORST, HIST, CUR, np.random.default_rng(1))
    r2 = RandomHistory().rank(WORST, HIST, CUR, np.random.default_rng(1))
    assert r1 == r2 and sorted(r1) == [0, 1, 2, 3]


def test_temporal_only():
    assert TemporalOnly().rank(BASE, HIST, CUR, RNG)[0] == 0
    assert TemporalOnly().rank(PREV, HIST, CUR, RNG)[0] == 3
    # no temporal preference -> recency tie-break
    assert TemporalOnly().rank(WORST, HIST, CUR, RNG) == [3, 2, 1, 0]


def test_clinical_only_extremes_and_nan_last():
    assert ClinicalOnly().rank(WORST, HIST, CUR, RNG) == [0, 3, 1, 2]
    assert ClinicalOnly().rank(BEST_VA, HIST, CUR, RNG)[0] == 1
    assert ClinicalOnly().rank(SIM, HIST, CUR, RNG)[0] == 1  # |250-260| smallest


@pytest.mark.parametrize("spec,top", [(BASE, 0), (PREV, 3), (WORST, 0), (SIM, 1), (BEST_VA, 1)])
def test_rule_router(spec, top):
    assert RuleRouter().rank(spec, HIST, CUR, RNG)[0] == top


def test_combined_extremes():
    assert Combined(0.0).rank(BASE, HIST, CUR, RNG)[0] == 0
    assert Combined(1.0).rank(WORST, HIST, CUR, RNG)[0] == 0


def test_unknown_question_falls_back_to_recency():
    spec = QuestionSpec(None, "unknown", None, "image")
    assert RuleRouter().rank(spec, HIST, CUR, RNG) == [3, 2, 1, 0]


def test_metrics_hand_computed():
    ranked = ["x", "a", "y", "b"]
    rel = {"a", "b"}
    assert hit_at_k(ranked, rel, 1) == 0 and hit_at_k(ranked, rel, 2) == 1
    assert recall_at_k(ranked, rel, 2) == 0.5 and recall_at_k(ranked, rel, 4) == 1
    assert reciprocal_rank(ranked, rel) == 0.5
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranked, rel, 2) == pytest.approx(expected)
    assert ndcg_at_k(["a"], {"a"}, 3) == 1.0
    assert reciprocal_rank(["x"], {"a"}) == 0.0
