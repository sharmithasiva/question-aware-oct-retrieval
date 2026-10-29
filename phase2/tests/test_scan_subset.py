"""Tests for the larger scan subset: offline plan and resumable fetch (fake atlas API; no network, no torch)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase2"))
import build_scan_manifest as B  # noqa: E402
import fetch_scan_subset as FS  # noqa: E402

EP = pd.read_csv(ROOT / "data/processed/episodes.csv")
VI = pd.read_csv(ROOT / "data/processed/visit_images.csv", usecols=["visit_id", "file_path"])
FULLHIST = ROOT / "phase2/data/atlas_fullhist"


@pytest.fixture(scope="module")
def plan():
    return B.build_plan(EP, VI)


def test_plan_is_train_only_whole_histories_and_in_range(plan):
    p, c = plan
    assert 3000 <= len(p) <= 5000 and (p.split == "train").all()
    assert not set(p.eye_id) & B.EXCLUDED_EYES and not p.image_key.duplicated().any()
    assert not p.archive_path.duplicated().any()
    ep = EP[EP.eye_id.isin(p.eye_id) & EP.qc_flags.isna()]
    assert set(p.visit_id) == set(ep.visit_id)                       # every visit of every kept eye
    assert (p.groupby("visit_id").size() == len(B.DEFAULT_SCANS)).all()
    assert c["planned_scans"] == len(p)


def test_plan_paths_match_verified_atlas_manifests(plan):
    p = plan[0].set_index("image_key")
    m = pd.read_csv(FULLHIST / "manifest.csv")
    assert (p.loc[m.image_key, "archive_path"].to_numpy() == m.archive_path.to_numpy()).all()


def test_max_scans_drops_whole_eyes_reproducibly():
    p1, c1 = B.build_plan(EP, VI, max_scans=1000)
    p2, _ = B.build_plan(EP, VI, max_scans=1000)
    assert p1.equals(p2) and len(p1) <= 1000
    full = B.build_plan(EP, VI)[0].groupby("eye_id").size()
    assert (p1.groupby("eye_id").size() == full[p1.eye_id.unique()].sort_index()).all()


class FakeAPI:
    """Atlas stand-in: serves fullhist images for its eyes; `bad` image keys get a wrong CST."""

    def __init__(self, plan, bad=(), dup=()):
        self.plan = plan.set_index("image_key")
        self.bad, self.dup = set(bad), set(dup)
        self.calls = 0
        self.png = (FULLHIST / "images" / "0232GOD_V1__scan24.webp").read_bytes()   # eye 11: not in the test plan

    def search(self, v):
        self.calls += 1
        g = self.plan[self.plan.visit_id == v.visit_id]
        return {"results": [{"id": k, "scan_index": int(r.scan_index)} for k, r in g.iterrows()]}

    def image(self, k):
        self.calls += 1
        r = self.plan.loc[k]
        return dict(id=k, path=r.archive_path, full_key=k, eye_id=int(r.eye_id), patient_id=int(r.patient_id),
                    eye=r.laterality, visit=r.visit_label, subject=r.subject, bcva=r.bcva,
                    cst=-1.0 if k in self.bad else r.cst, label_source="test")

    def bytes(self, k):
        self.calls += 1
        return self.png if k in self.dup else self.png + k.encode()   # unique bytes unless marked duplicate


def test_fetch_verifies_reuses_local_resumes_and_marks_duplicates(plan, tmp_path):
    p = plan[0]
    p = p[p.eye_id == 25].head(15)              # 3 visits x 5 scans; local: V1 scans 12/24/36 (E1), V2/V3 scan 24 (E2)
    bad, dup = p.image_key.iloc[1], list(p.image_key.iloc[[3, 6]])   # V1 scan18; V1 scan30 and V2 scan18
    api = FakeAPI(p, bad=[bad], dup=dup)
    man, rej = FS.fetch(p, tmp_path, api, local=FS.local_index(), log=lambda s: None)
    assert set(rej.image_key) == {bad} and "cst" in rej.failed.iloc[0]
    assert len(man) == 14 and (man.source.str.startswith("local_copy")).sum() == 5
    assert (man.set_index("image_key").loc[dup, "duplicate_of"] != "").sum() == 1
    for r in man.itertuples():
        assert (tmp_path / r.local_path).exists()
    calls = api.calls
    man2, rej2 = FS.fetch(p, tmp_path, api, local=FS.local_index(), log=lambda s: None)
    assert len(man2) == 14 and set(rej2.image_key) == {bad}
    assert api.calls - calls == 2                                     # only the rejected scan is retried


def test_fetch_respects_limit_and_refetches_corrupted_file(plan, tmp_path):
    p = plan[0]
    p = p[p.eye_id == 25].head(10)
    api = FakeAPI(p)
    man, _ = FS.fetch(p, tmp_path, api, limit=3, local={}, log=lambda s: None)
    assert len(man) == 3
    (tmp_path / man.local_path.iloc[0]).write_bytes(b"corrupt")
    man2, _ = FS.fetch(p, tmp_path, api, local={}, log=lambda s: None)
    assert len(man2) == 10 and man.image_key.iloc[0] in set(man2.image_key)
