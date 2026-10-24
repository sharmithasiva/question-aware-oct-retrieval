"""Tests for phase2/question_eval.py (no torch; synthetic rankings where needed)."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase2"))
sys.path.insert(0, str(ROOT / "src"))
import question_eval as QE  # noqa: E402
from oct_retrieval import questions as Q1  # noqa: E402

EP = pd.read_csv(ROOT / "data" / "processed" / "episodes.csv")


class Stub:
    """Systems stand-in returning fixed rankings."""
    def __init__(self, rankings):
        self.r = rankings

    def rank_all(self, q):
        return dict(self.r, _parsed="stub")


def qrow(cands=("a", "b", "c", "d", "e", "f"), qv="q"):
    return pd.DataFrame([dict(query_id="Q1", set="dev", tier="B", category="overall_appearance", question="x",
                              eye_id=1, query_visit_id=qv, t=len(cands), n_candidates=len(cands),
                              candidate_visit_ids=";".join(cands))])


def labels(grades, qid="Q1"):
    return pd.DataFrame([dict(query_id=qid, candidate_visit_id=v, grade=g) for v, g in grades.items()])


def test_missing_labels_are_not_evaluated_not_zero():
    qs = qrow()
    per, _ = QE.evaluate(qs, None, Stub({"S1_latest_prior": list("fedcba")}), {}, 2.0)
    assert set(per.status) == {"not_evaluated_missing_labels"}
    assert "hit1" not in per.columns or per.hit1.isna().all()
    s = QE.summarise(per, "B")
    assert (s.mrr == "NOT EVALUATED").all() and (s.n_evaluated == 0).all()


def test_partially_graded_query_is_not_evaluated():
    g = {v: 2.0 for v in "abcdef"}
    g["c"] = np.nan
    per, _ = QE.evaluate(qrow(), labels(g), Stub({"S1_latest_prior": list("fedcba")}), {}, 2.0)
    assert set(per.status) == {"not_evaluated_missing_labels"}


def test_no_relevant_candidate_is_excluded_and_counted():
    per, _ = QE.evaluate(qrow(), labels({v: 1.0 for v in "abcdef"}), Stub({"S1_latest_prior": list("fedcba")}), {}, 2.0)
    assert set(per.status) == {"no_relevant_candidate"}
    assert QE.summarise(per, "B").n_no_relevant_candidate.iloc[0] == 1


def test_duplicate_query_image_is_excluded():
    hashes = {"q": "h1", "c": "h1", "a": "h2"}
    per, _ = QE.evaluate(qrow(), labels({v: 2.0 for v in "abcdef"}), Stub({"S1_latest_prior": list("fedcba")}),
                         hashes, 2.0)
    assert set(per.status) == {"excluded_duplicate_image"}


def test_recency_tie_break_and_identical_scores():
    pos = {v: i for i, v in enumerate("abcd")}
    assert QE.recency_order({"a": .5, "b": .9, "c": .5, "d": .9}, pos) == ["d", "b", "c", "a"]


def test_metrics_known_values_and_abstention():
    g = {"a": 2.0, "b": 0.0, "c": 1.0, "d": 0.0, "e": 0.0, "f": 0.0}
    m = QE.score_ranking(list("bcadef"), g, 2.0)
    assert m["hit1"] == 0 and m["hit5"] == 1 and m["recall5"] == 1 and m["mrr"] == pytest.approx(1 / 3)
    ideal = QE.dcg([2, 1, 0, 0, 0])
    assert m["ndcg5"] == pytest.approx(QE.dcg([0, 1, 2, 0, 0]) / ideal)
    z = QE.score_ranking(None, g, 2.0)
    assert z["answered"] is False and z["mrr"] == 0


def test_random_expectation_matches_enumeration():
    g = {"a": 2.0, "b": 0.0, "c": 1.0, "d": 0.0, "e": 2.0, "f": 0.0, "h": 0.0}
    perms = list(itertools.permutations(g))
    mean = {k: np.mean([QE.score_ranking(list(p), g, 2.0)[k] for p in perms]) for k in QE.METRICS}
    exp = QE.random_expectation(g, 2.0)
    for k in QE.METRICS:
        assert exp[k] == pytest.approx(mean[k], abs=1e-9), k


def test_candidate_pool_has_no_leakage():
    for tier in "AB":
        qs = QE.build_query_set(EP, "dev", tier)
        QE.assert_no_candidate_leakage(qs, EP)
        assert (qs.n_candidates >= QE.MIN_CANDIDATES).all()
    bad = qs.copy()
    bad.loc[bad.index[0], "candidate_visit_ids"] += ";" + bad.query_visit_id.iloc[0]
    bad.loc[bad.index[0], "n_candidates"] += 1
    with pytest.raises(AssertionError):
        QE.assert_no_candidate_leakage(bad, EP)


def test_rankings_only_contain_candidates():
    qs = qrow()
    with pytest.raises(AssertionError):
        QE.evaluate(qs, labels({v: 2.0 for v in "abcdef"}), Stub({"S1_latest_prior": list("fedcbaZ")}), {}, 2.0)


def test_tier_a_labels_match_phase1_instances():
    qs = QE.build_query_set(EP, "dev", "A")
    lab = QE.tier_a_labels(qs[qs.category == "similar_cst"], EP)
    ours = lab[lab.grade == 1].groupby("query_id").candidate_visit_id.apply(lambda s: ";".join(sorted(s)))
    inst = Q1.generate_instances(EP[EP.eye_id.isin(QE.SETS["dev"]["eyes"])])
    inst = inst[inst.qtype == "similar_cst"].set_index("current_visit_id").relevant_visit_ids
    for qid, rel in ours.items():
        qv = qid.split("-similar_cst-")[1]
        assert ";".join(sorted(inst[qv].split(";"))) == rel


def test_human_labels_blank_is_missing_and_bad_grade_rejected(tmp_path):
    sheet = pd.DataFrame(dict(query_id=["Q1", "Q1"], query_visit_id=["q", "q"], candidate_code=["C01", "C02"],
                              grade=["2", ""], ungradable=["", ""], query_answerable=["", ""]))
    key = pd.DataFrame(dict(query_visit_id=["q", "q"], candidate_code=["C01", "C02"], candidate_visit_id=["a", "b"]))
    sheet.to_csv(tmp_path / "s.csv", index=False)
    key.to_csv(tmp_path / "k.csv", index=False)
    lab = QE.human_labels(tmp_path / "s.csv", tmp_path / "k.csv")
    assert lab.grade.iloc[0] == 2 and np.isnan(lab.grade.iloc[1])
    sheet.loc[1, "grade"] = "5"
    sheet.to_csv(tmp_path / "s.csv", index=False)
    with pytest.raises(ValueError):
        QE.human_labels(tmp_path / "s.csv", tmp_path / "k.csv")


def test_heldout_requires_freeze_flag():
    with pytest.raises(SystemExit):
        QE.main(["--set", "heldout", "--tier", "A"])


def test_dev_packet_sheet_has_no_grades_and_matches_key():
    sheet = ROOT / "phase2/annotations/tierB_dev_sheet.csv"
    key = ROOT / "phase2/annotations/keys/tierB_dev_key.csv"
    if not sheet.exists():
        pytest.skip("packet not generated")
    s = pd.read_csv(sheet, dtype=str, keep_default_na=False)
    assert (s.grade == "").all() and (s.label_source == "human").all()
    k = pd.read_csv(key)
    assert not k.identical_to_query_image.any()
    assert set(zip(s.query_visit_id, s.candidate_code)) == set(zip(k.query_visit_id, k.candidate_code))
