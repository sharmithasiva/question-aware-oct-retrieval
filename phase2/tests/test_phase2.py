"""Phase 2 tests (no torch needed). Run: python -m pytest -q phase2/tests"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase2"))
import fetch_atlas_subset as F  # noqa: E402
import oct_image as OI  # noqa: E402

SUBSET = ROOT / "phase2" / "data" / "atlas_subset"
EP = pd.read_csv(ROOT / "data" / "processed" / "episodes.csv")


def test_preprocess_shapes_and_scaling():
    g = np.tile(np.arange(256, dtype=np.uint8), (496, 2))[:, :504]
    r, x = OI.preprocess(g)
    assert r.shape == (224, 224) and 0 <= r.min() and r.max() <= 1
    assert x.shape == (3, 224, 224) and x.dtype == np.float32
    assert np.allclose(x[0] * OI.IMAGENET_STD[0] + OI.IMAGENET_MEAN[0], r, atol=1e-5)


def test_topk_excludes_query_and_orders():
    s = np.array([[1, .2, .9, .5], [.2, 1, .1, .3], [.9, .1, 1, .4], [.5, .3, .4, 1]])
    assert list(OI.topk(s, 0, 3)) == [2, 3, 1]


def test_neighbour_stats_perfect_clusters():
    m = pd.DataFrame({"eye_id": [1, 1, 2, 2], "visit_id": ["a", "a", "b", "b"], "trial": ["T"] * 4})
    s = np.array([[1, .9, 0, 0], [.9, 1, 0, 0], [0, 0, 1, .9], [0, 0, .9, 1]], float)
    ns = OI.neighbour_stats(s, m, k=1)
    assert ns.same_eye_at_k == 1 and ns.same_visit_at_1 == 1 and ns.same_eye_chance == pytest.approx(1 / 3)


def test_plan_refuses_test_and_excluded_eyes():
    test_eye = int(EP[EP.split == "test"].eye_id.iloc[0])
    with pytest.raises(ValueError):
        F.plan_visits(EP, [test_eye], 4)
    with pytest.raises(ValueError):
        F.plan_visits(EP, [79], 4)


def test_check_flags_mismatches():
    v = EP[EP.visit_id == "0210MOD_V1"].iloc[0]
    good = {"path": "/x", "eye_id": int(v.eye_id), "patient_id": int(v.patient_id), "eye": v.laterality,
            "visit": v.visit_label, "subject": v.subject, "bcva": v.bcva, "cst": v.cst}
    assert F.check(good, v, {"/x": v.visit_id}) == []
    assert set(F.check({**good, "eye": "OS", "cst": 1}, v, {"/x": v.visit_id})) == {"laterality", "cst"}
    assert F.check(good, v, {}) == ["path_in_metadata"]


FULLHIST = ROOT / "phase2" / "data" / "atlas_fullhist"


def test_plan_all_visits():
    v = F.plan_visits(EP, [25], None)
    assert list(v.visit_id) == list(EP[EP.eye_id == 25].sort_values("visit_index").visit_id)


def test_partial_spearman_removes_shared_driver():
    import image_vs_metadata as IVM
    z = np.arange(50, dtype=float)
    rng = np.random.default_rng(0)
    x, y = z + rng.normal(0, 1, 50), z + rng.normal(0, 1, 50)
    assert IVM.spearman(x, y) > 0.9 and abs(IVM.partial_spearman(x, y, z)) < 0.4


@pytest.mark.skipif(not (FULLHIST / "manifest.csv").exists(), reason="full-history subset not fetched")
def test_fullhist_complete_and_known_duplicate():
    m = OI.load_manifest(FULLHIST)
    assert (m.groupby("eye_id").visit_id.nunique() == EP[EP.eye_id.isin(m.eye_id)].groupby("eye_id").size()).all()
    dup = m[m.sha256.duplicated(keep=False)]
    # Documented data issue: the atlas serves identical images for 01-013 W8 and W12.
    assert sorted(dup.visit_id) == ["01-013_W12", "01-013_W8"]


@pytest.mark.parametrize("subset", [SUBSET, FULLHIST], ids=["subset", "fullhist"])
def test_manifest_associations_and_checksums(subset):
    if not (subset / "manifest.csv").exists():
        pytest.skip("not fetched")
    m = OI.load_manifest(subset)
    assert OI.verify_checksums(m, subset).all()
    vi = pd.read_csv(ROOT / "data" / "processed" / "visit_images.csv", usecols=["visit_id", "file_path"])
    j = m.merge(vi, left_on="archive_path", right_on="file_path", suffixes=("", "_vi"), validate="one_to_one")
    assert len(j) == len(m) and (j.visit_id == j.visit_id_vi).all()
    e = m.merge(EP, on="visit_id", suffixes=("", "_ep"))
    for c in ("eye_id", "patient_id", "laterality", "cst", "bcva", "split", "visit_index"):
        assert (e[c] == e[f"{c}_ep"]).all(), c
    assert not (m.split == "test").any() and 79 not in set(m.eye_id)
    for p in m.local_path:
        w, h = F.webp_size((subset / p).read_bytes()[:64])
        assert (w, h) == (504, 496)


@pytest.mark.skipif(not (ROOT / "phase2/results/fullhist/resnet50/embeddings.npz").exists(), reason="no saved results")
def test_consolidated_metrics_match_saved_outputs():
    import json
    import consolidate_metrics as CM
    assert CM.main() == 0
    c = json.loads((CM.OUT / "consistency.json").read_text())
    assert all(x["match"] for v in c.values() for x in v.values())
    m = pd.read_csv(CM.OUT / "metrics.csv")
    std = m[m.metric.isin(["Hit@1", "MRR", "nDCG@5"])]
    assert std.value.isna().all() and std.note.str.startswith("Not evaluated").all()
