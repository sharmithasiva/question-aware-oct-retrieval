"""Tests for the v2 Tier B workflow: frozen rankings, blinded packet, sheet->key->ranking joins and metric handling.

Synthetic labels below exist only inside pytest's tmp_path as fixtures for the arithmetic; they are never written to
phase2/annotations and are not judgments.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase2"))
import evaluate_annotations as EA  # noqa: E402
import question_eval as QE  # noqa: E402

PACKET = ROOT / "phase2/annotations/v2/dev"
KEYS = ROOT / "phase2/annotations/keys/v2"
have_packet = pytest.mark.skipif(not (PACKET / "MANIFEST.json").exists() or not (KEYS / "dev_B1_key.csv").exists(),
                                 reason="v2 dev packet or key not present")


def mini(labels, gradable="yes", dup=False):
    """One query 'Q' with candidates a..e; labels: dict candidate -> label."""
    cands = list(labels)
    sheet = pd.DataFrame(dict(query_code="B1-Q01", candidate_code=[f"C{i + 1:02d}" for i in range(len(cands))],
                              label=list(labels.values()), query_gradable=gradable, packet="p",
                              annotator_id="T", annotator_role="r", rationale=""))
    key = pd.DataFrame(dict(packet="p", query_code="B1-Q01", candidate_code=sheet.candidate_code, query_id="Q",
                            query_visit_id="q", eye_id=1, candidate_visit_id=cands,
                            query_source_sha256="hq",
                            candidate_source_sha256=["hq" if (dup and i == 0) else f"h{i}" for i in range(len(cands))]))
    return sheet, key


def rk(order_by_system):
    return pd.DataFrame([dict(query_id="Q", system=s, rank=i + 1, candidate_visit_id=v)
                         for s, order in order_by_system.items() for i, v in enumerate(order)])


def systems(names):
    return pd.DataFrame(dict(query_id="Q", system=list(names), status="ranked"))


# --- joins ----------------------------------------------------------------------------------------------------

def test_join_requires_exact_coverage_and_rejects_duplicates():
    s, k = mini({"a": "2", "b": "0"})
    assert len(EA.join_sheet_key(s, k)) == 2
    with pytest.raises(ValueError):
        EA.join_sheet_key(s.iloc[:1], k)
    with pytest.raises(ValueError):
        EA.join_sheet_key(s, k.assign(candidate_code=["C01", "C09"]))
    with pytest.raises(ValueError):
        EA.join_sheet_key(s, k.assign(packet="other"))


def test_load_sheet_validates_labels(tmp_path):
    s, _ = mini({"a": "2", "b": "u"})
    s.to_csv(tmp_path / "s.csv", index=False)
    assert list(EA.load_sheet(tmp_path / "s.csv").label) == ["2", "U"]
    s.loc[0, "label"] = "3"
    s.to_csv(tmp_path / "s.csv", index=False)
    with pytest.raises(ValueError):
        EA.load_sheet(tmp_path / "s.csv")
    s, _ = mini({"a": "2", "b": "0"})
    pd.concat([s, s.iloc[:1]]).to_csv(tmp_path / "d.csv", index=False)
    with pytest.raises(ValueError):
        EA.load_sheet(tmp_path / "d.csv")


def test_check_pools_detects_leakage_gaps_and_bad_ranks():
    s, k = mini({"a": "2", "b": "0", "c": "1"})
    j = EA.join_sheet_key(s, k)
    EA.check_pools(j, rk({"S1": ["c", "b", "a"]}))
    with pytest.raises(ValueError):                       # extra (leaked) candidate
        EA.check_pools(j, rk({"S1": ["c", "b", "a", "z"]}))
    with pytest.raises(ValueError):                       # missing candidate
        EA.check_pools(j, rk({"S1": ["c", "b"]}))
    bad = rk({"S1": ["c", "b", "a"]})
    bad.loc[0, "rank"] = 5
    with pytest.raises(ValueError):
        EA.check_pools(j, bad)


# --- statuses and U/X handling --------------------------------------------------------------------------------

@pytest.mark.parametrize("labels,gradable,dup,expected", [
    ({"a": "2", "b": ""}, "yes", False, "not_evaluated_missing_labels"),
    ({"a": "", "b": ""}, "no", False, "excluded_query_ungradable"),
    ({"a": "2", "b": "0"}, "yes", True, "excluded_duplicate_image"),
    ({"a": "1", "b": "0"}, "yes", False, "no_relevant_candidate"),
    ({"a": "2", "b": "X", "c": "U"}, "yes", False, "excluded_too_few_gradable"),
    ({"a": "2", "b": "0"}, "yes", False, "evaluated"),
])
def test_query_status(labels, gradable, dup, expected):
    s, k = mini(labels, gradable, dup)
    assert EA.query_status(EA.join_sheet_key(s, k), "primary", 2.0)[0] == expected


def test_uncertain_and_ungradable_are_never_silently_zero_in_primary():
    s, k = mini({"a": "U", "b": "2", "c": "0", "d": "X"})
    j = EA.join_sheet_key(s, k)
    status, grade = EA.query_status(j, "primary", 2.0)
    assert status == "evaluated" and set(grade) == {"b", "c"}
    _, pess = EA.query_status(j, "pessimistic", 2.0)
    assert pess == {"a": 0.0, "b": 2.0, "c": 0.0, "d": 0.0}


def test_filtering_is_identical_for_every_system_and_metrics_follow():
    s, k = mini({"a": "U", "b": "2", "c": "0", "d": "X"})
    j = EA.join_sheet_key(s, k)
    r = rk({"S1_latest_prior": ["a", "c", "b", "d"], "S3": ["a", "b", "c", "d"]})
    per = EA.evaluate(j, r, systems(["S1_latest_prior", "S3"]), "primary", 2.0).set_index("system")
    # after removing U (a) and X (d): S1 -> [c, b] (relevant at rank 2), S3 -> [b, c] (rank 1)
    assert per.loc["S1_latest_prior", "mrr"] == pytest.approx(0.5)
    assert per.loc["S3", "hit1"] == 1.0 and per.loc["S3", "mrr"] == 1.0
    assert per.loc["random_expected", "hit1"] == pytest.approx(0.5)
    pes = EA.evaluate(j, r, systems(["S1_latest_prior", "S3"]), "pessimistic", 2.0).set_index("system")
    assert pes.loc["S3", "mrr"] == pytest.approx(0.5)           # U at rank 1 now counts as irrelevant


def test_multiple_relevant_and_graded_ndcg():
    s, k = mini({"a": "2", "b": "2", "c": "1", "d": "0", "e": "0", "f": "0"})
    j = EA.join_sheet_key(s, k)
    per = EA.evaluate(j, rk({"S": ["d", "a", "c", "b", "e", "f"]}), systems(["S"]), "primary", 2.0).set_index("system")
    assert per.loc["S", "recall5"] == 1.0 and per.loc["S", "mrr"] == 0.5
    exp = QE.dcg([0, 2, 1, 2, 0]) / QE.dcg([2, 2, 1, 0, 0])
    assert per.loc["S", "ndcg5"] == pytest.approx(exp)


def test_abstaining_system_scores_zero_with_zero_coverage():
    s, k = mini({"a": "2", "b": "0"})
    j = EA.join_sheet_key(s, k)
    per = EA.evaluate(j, rk({"S1_latest_prior": ["b", "a"]}), systems(["S1_latest_prior", "S2_metadata"]))
    prim, _ = EA.summarise(per)
    row = prim.set_index("system").loc["S2_metadata"]
    assert row.coverage == 0 and row.mrr == 0


def test_summary_marks_not_evaluated():
    s, k = mini({"a": "", "b": ""})
    per = EA.evaluate(EA.join_sheet_key(s, k), rk({"S": ["a", "b"]}), systems(["S"]))
    prim, eyes = EA.summarise(per)
    assert (prim.mrr == "NOT EVALUATED").all() and eyes.empty


def test_agreement_kappa():
    a, _ = mini({"a": "2", "b": "1", "c": "0", "d": "U"})
    assert EA.agreement(a, a)["weighted_kappa"] == pytest.approx(1.0)
    b = a.copy()
    b["label"] = ["0", "1", "2", "U"]
    ag = EA.agreement(a, b)
    assert ag["n_both_graded"] == 3 and ag["weighted_kappa"] < 0 and ag["pct_agree_grade2"] == pytest.approx(1 / 3, abs=1e-4)


# --- real frozen rankings and packet --------------------------------------------------------------------------

def test_frozen_rankings_hash_is_verified(tmp_path):
    rdir = EA.default_rankings_dir("dev")
    EA.load_rankings(rdir)
    t = tmp_path / rdir.name
    shutil.copytree(rdir, t)
    df = pd.read_csv(t / "rankings.csv")
    df.loc[0, "rank"], df.loc[1, "rank"] = df.loc[1, "rank"], df.loc[0, "rank"]
    df.to_csv(t / "rankings.csv", index=False)
    with pytest.raises(ValueError):
        EA.load_rankings(t)


@have_packet
@pytest.mark.parametrize("cat", ["B1", "B2"])
def test_real_packet_joins_frozen_rankings_and_is_blind(cat):
    sheet = EA.load_sheet(PACKET / cat / "sheet.csv")
    key = pd.read_csv(KEYS / f"dev_{cat}_key.csv")
    j = EA.join_sheet_key(sheet, key)
    rankings, sysdf, _ = EA.load_rankings(EA.default_rankings_dir("dev"))
    EA.check_pools(j, rankings[rankings.category == EA.CATEGORIES[cat]])
    assert (sheet.label == "").all(), "the committed packet must contain no grades"
    ep = pd.read_csv(ROOT / "data/processed/episodes.csv")
    text = (PACKET / cat / "index.html").read_text(encoding="utf8") + (PACKET / cat / "sheet.csv").read_text()
    assert not any(v in text for v in set(ep.visit_id) | set(ep.subject))
    assert not any(w in text for w in ("resnet", "dinov2", "latest_prior", "metadata_proxy", "rankings"))
    man = json.loads((PACKET / "MANIFEST.json").read_text())
    src = set(pd.read_csv(ROOT / "phase2/data/atlas_fullhist/manifest.csv").sha256)
    for rel, h in man["files_sha256"].items():
        if f"/{cat}/" in rel and rel.endswith(".webp"):
            assert EA.sha256(ROOT / rel) == h and h not in src


@have_packet
def test_b1_b2_codes_are_independent():
    k1, k2 = pd.read_csv(KEYS / "dev_B1_key.csv"), pd.read_csv(KEYS / "dev_B2_key.csv")
    m = k1.merge(k2, on=["query_visit_id", "candidate_visit_id"])
    assert len(m) == len(k1) == len(k2)
    assert (m.candidate_image_x != m.candidate_image_y).all()
    assert (m.query_code_x.str[3:] == m.query_code_y.str[3:]).mean() < 0.5


@have_packet
def test_end_to_end_and_never_overwrites(tmp_path):
    sheet = pd.read_csv(PACKET / "B1" / "sheet.csv", dtype=str, keep_default_na=False)
    rng = np.random.default_rng(0)                      # synthetic fixture labels, tmp only
    sheet["label"] = rng.choice(["0", "1", "2", "U", "X"], len(sheet), p=[.4, .25, .25, .05, .05])
    sheet["query_gradable"] = "yes"
    p = tmp_path / "fixture.csv"
    sheet.to_csv(p, index=False)
    args = ["--set", "dev", "--category", "B1", "--sheet", str(p), "--out-root", str(tmp_path / "out")]
    assert EA.main(args) == 0
    run = next((tmp_path / "out" / "dev").iterdir())
    prim = pd.read_csv(run / "summary_primary.csv").set_index("system")
    assert prim.loc["S1_latest_prior", "n_evaluated"] > 0
    assert "NOT EVALUATED" not in (run / "report.md").read_text(encoding="utf8").split("## Primary")[0]
    with pytest.raises(SystemExit):
        EA.main(args)
