"""large_eval.py must reproduce the saved E2 metrics and the saved dev Tier A summary from the same embeddings."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase2"))
import large_eval as LE  # noqa: E402

FULLHIST = ROOT / "phase2/data/atlas_fullhist"
EMB = {e: ROOT / f"phase2/results/fullhist/{e}/embeddings.npz" for e in ("resnet50", "dinov2_vits14")}


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out = tmp_path_factory.mktemp("le") / "run"
    res = LE.run(FULLHIST, EMB, "central", out, do_b1=LE.B1_KEY.exists())
    return res, out


def test_e2_definition_matches_saved_metrics(result):
    res, _ = result
    saved = pd.read_csv(ROOT / "phase2/results/performance/metrics.csv")
    saved = saved[saved.experiment == "E2_73img_fullhist"].set_index(["encoder", "metric"]).value
    for enc in EMB:
        w = res["within_eye"][enc]
        assert w["image_top1_is_previous_rate"] == pytest.approx(saved[(enc, "image_top1_is_previous_visit_rate")], abs=1e-4)
        assert w["spearman_cos_vs_temporal_closeness_pooled"] == pytest.approx(
            saved[(enc, "spearman_cos_vs_temporal_closeness_pooled")], abs=1e-4)
        assert w["partial_spearman_cos_vs_cst_given_gap_pooled"] == pytest.approx(
            saved[(enc, "partial_spearman_cos_vs_cst_closeness_given_gap_pooled")], abs=1e-4)
    assert res["scan_level"]["resnet50"]["dropped_duplicates"] == 1          # the known 01-013 W8 / W12 pair


def test_tier_a_matches_saved_dev_summary_and_hybrid_is_safe(result):
    _, out = result
    got = pd.read_csv(out / "tierA_summary.csv").set_index(["category", "system"])
    ref = pd.read_csv(ROOT / "phase2/results/question_eval/dev/tierA/summary.csv").set_index(["category", "system"])
    for k, r in ref.iterrows():
        for m in ("hit1", "hit5", "mrr", "ndcg5"):
            assert float(got.loc[k, m]) == pytest.approx(float(r[m]), abs=1e-4), (k, m)
    h = got.xs("H_fuse_closest_resnet50", level="system")
    s2 = got.xs("S2_metadata", level="system")
    for cat in h.index:                                # identical to metadata wherever no fusion is allowed
        if cat != "similar_cst":
            assert float(h.loc[cat, "mrr"]) == pytest.approx(float(s2.loc[cat, "mrr"]))


def test_b1_posthoc_reproduces_frozen_evaluation(result):
    _, out = result
    if not LE.B1_KEY.exists():
        pytest.skip("B1 key not present (git-ignored)")
    got = pd.read_csv(out / "b1_posthoc_summary.csv").set_index("system")
    ref = next((ROOT / "phase2/results/question_eval_v2/dev").glob("eval_B1_cda4ea5b_*"))
    ref = pd.read_csv(ref / "summary_primary.csv").set_index("system")
    for s in ("S3S4_image_resnet50", "S5_hybrid_rrf_resnet50", "S3S4_image_dinov2_vits14", "S5_hybrid_rrf_dinov2_vits14"):
        for m in ("hit1", "mrr", "ndcg5"):
            assert float(got.loc[f"posthoc_{s}", m]) == pytest.approx(float(ref.loc[s, m]), abs=1e-4)


def test_refuses_to_overwrite(result):
    _, out = result
    with pytest.raises(SystemExit):
        LE.main(["--subset", str(FULLHIST), "--emb", f"resnet50={EMB['resnet50']}", "--out", str(out)])
