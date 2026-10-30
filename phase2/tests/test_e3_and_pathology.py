"""E3 question-conditioned evaluation and the pathology probe (offline; saved embeddings or synthetic data)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase2"))
import e3_question_eval as E3  # noqa: E402
import large_eval as LE  # noqa: E402
import pathology_probe as PP  # noqa: E402

FULLHIST = ROOT / "phase2/data/atlas_fullhist"
EMB = {"resnet50": ROOT / "phase2/results/fullhist/resnet50/embeddings.npz"}


# --- E3 -------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def e3(tmp_path_factory):
    out = tmp_path_factory.mktemp("e3") / "run"
    E3.run(FULLHIST, EMB, out)
    return pd.read_csv(out / "summary.csv").set_index(["category", "system"]), out


def test_e3_reproduces_saved_tier_a_for_shared_keys(e3):
    summ, _ = e3
    ref = pd.read_csv(ROOT / "phase2/results/question_eval/dev/tierA/summary.csv").set_index(["category", "system"])
    pairs = {("cst_highest", "latest_prior"): ("worst_edema", "S1_latest_prior"),
             ("cst_highest", "image_resnet50"): ("worst_edema", "S3S4_image_resnet50"),
             ("cst_most_similar", "image_resnet50"): ("similar_cst", "S3S4_image_resnet50"),
             ("cst_largest_rise", "latest_prior"): ("largest_cst_rise", "S1_latest_prior"),
             ("cst_lowest", "metadata_only"): ("best_anatomy", "S2_metadata")}
    for mine, theirs in pairs.items():
        for m in ("hit1", "mrr", "ndcg5"):
            assert summ.loc[mine, m] == pytest.approx(float(ref.loc[theirs, m]), abs=1e-4), (mine, m)


def test_e3_reports_sources_sizes_and_recall1(e3):
    summ, out = e3
    g = summ.xs("metadata_only", level="system")
    assert (g.n_eyes == 3).all() and (g.n_patients == 3).all()
    assert (g.drop(index="image_similarity_B1", errors="ignore").n_queries > 50).all()
    assert "NOT a clinical edema grade" in g.loc["edema_worst_proxy", "gt_source"]
    assert g.loc["temporal_previous", "gt_source"].startswith("structural")
    per = pd.read_csv(out / "per_query.csv")
    ev = per[per.status == "evaluated"]
    assert np.allclose(ev.recall1, ev.hit1 / ev.n_relevant)
    md = (out / "summary.md").read_text()
    assert "Not evaluable (N/A)" in md and "pathology" in md and "mixed" in md
    with pytest.raises(SystemExit):
        E3.main(["--subset", str(FULLHIST), "--emb", f"resnet50={EMB['resnet50']}", "--out", str(out)])


def test_e3_b1_matches_frozen_evaluation_when_key_present(e3):
    summ, _ = e3
    if not LE.B1_KEY.exists():
        pytest.skip("B1 key not present (git-ignored)")
    ref = pd.read_csv(next((ROOT / "phase2/results/question_eval_v2/dev").glob("eval_B1_cda4ea5b_*"))
                      / "summary_primary.csv").set_index("system")
    for mine, theirs in (("image_resnet50", "S3S4_image_resnet50"), ("safe_hybrid_resnet50", "S5_hybrid_rrf_resnet50"),
                         ("question_aware_resnet50", "S5_hybrid_rrf_resnet50"), ("latest_prior", "S1_latest_prior")):
        assert summ.loc[("image_similarity_B1", mine), "ndcg5"] == pytest.approx(float(ref.loc[theirs, "ndcg5"]),
                                                                                 abs=1e-4)
    assert summ.loc[("image_similarity_B1", "metadata_only"), "coverage"] == 0
    per = pd.read_csv(e3[1] / "per_query.csv")
    assert "query_code" not in per.columns and not per.astype(str).apply(lambda c: c.str.match(r"B[12]-Q\d")).any().any()


# --- pathology probe ---------------------------------------------------------------------------------------------------

def test_metric_helpers():
    y = np.array([0, 0, 1, 1])
    assert PP.auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1 and PP.auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0
    assert PP.auroc(y, np.ones(4)) == 0.5
    assert PP.auprc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1
    assert PP.auprc(np.array([1, 0, 1]), np.array([0.9, 0.8, 0.7])) == pytest.approx((1 + 2 / 3) / 2)


def test_folds_never_split_a_patient():
    patients = np.repeat(np.arange(30), 4)
    folds = PP.patient_folds(patients)
    assert sorted(np.concatenate(folds)) == list(range(len(patients)))
    for f in folds:
        rest = np.setdiff1d(np.arange(len(patients)), f)
        assert not set(patients[f]) & set(patients[rest])


def synthetic(n_pat=12, per=10, signal=3.0, seed=0):
    rng = np.random.default_rng(seed)
    pats = np.repeat(np.arange(n_pat), per)
    y = (rng.random(len(pats)) < 0.4).astype(float)
    E = rng.normal(size=(len(pats), 32))
    E[:, 0] += signal * y
    scans = pd.DataFrame(dict(image_key=[f"k{i}" for i in range(len(pats))], visit_id=[f"v{p}_{i % 2}" for i, p in
                                                                                      enumerate(pats)],
                              eye_id=pats, patient_id=pats, scan_index=np.tile([12, 24, 36, 18, 30], len(pats) // 5),
                              irf=y, srf=np.zeros(len(pats)), drt_me=y, visit_cst=rng.normal(300, 30, len(pats))))
    for t in PP.TASKS:
        scans[f"visit_frac_{t}"] = scans.groupby("visit_id")[t].transform("mean")
    return scans, E


def test_probe_detects_real_signal_and_not_noise():
    scans, E = synthetic()
    summ, preds = PP.evaluate(scans, {"enc": E, "noise": np.random.default_rng(1).normal(size=E.shape)})
    s = summ.set_index(["task", "method"])
    assert s.loc[("irf", "probe_enc"), "auroc"] > 0.9 and s.loc[("irf", "knn_enc"), "auroc"] > 0.8
    assert abs(s.loc[("irf", "probe_noise"), "auroc"] - 0.5) < 0.2
    assert s.loc[("irf", "probe_enc"), "auroc_ci95_patient"] is not None
    assert summ[summ.task == "srf"].status.iloc[0].startswith("N/A")     # no positives: not evaluated
    assert preds.filter(like="irf__").notna().all().all()


def test_real_labels_join_by_exact_path_only():
    scans = PP.labelled_scans(ROOT / "phase2/data/atlas_subset")
    assert len(scans) == 48 and scans.eye_id.nunique() == 8
    assert (scans["Scan (n/49)"] - 1 == scans.scan_index).all()
    assert set(scans.irf.unique()) <= {0, 1} and scans.visit_cst.notna().all()


def test_report_marks_unrun_encoders_na_and_never_overwrites(tmp_path):
    import report_v3 as RV
    out = tmp_path / "rep"
    assert RV.main(["--e3", str(ROOT / "phase2/results/e3/fullhist_dev_v1"),
                    "--pathology", str(ROOT / "phase2/results/pathology/atlas_subset_pilot_v1"), "--out", str(out)]) == 0
    md = (out / "report.md").read_text()
    assert "| N/A |" in md and "C: OCT encoder-only" in md and "NOT a clinical edema grade" in md
    with pytest.raises(SystemExit):
        RV.main(["--e3", str(ROOT / "phase2/results/e3/fullhist_dev_v1"),
                 "--pathology", str(ROOT / "phase2/results/pathology/atlas_subset_pilot_v1"), "--out", str(out)])
