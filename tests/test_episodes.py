import pandas as pd
import pytest

from oct_retrieval.episodes import assign_splits, build_episodes, build_visit_images
from oct_retrieval.load import parse_path, visit_number


def test_parse_trex_path():
    p = parse_path("/TREX DME/GILA/0201GOD/V1/OD/TREXJ_000000.tif")
    assert p == dict(trial="TREX DME", arm="GILA", subject="0201GOD", visit_label="V1", laterality="OD",
                     image_name="TREXJ_000000.tif")


def test_parse_prime_path_has_no_arm():
    p = parse_path("/Prime_FULL/01-001/W104/OS/27.png")
    assert p["arm"] is None and p["subject"] == "01-001" and p["visit_label"] == "W104" and p["laterality"] == "OS"


@pytest.mark.parametrize("bad", ["/Prime_FULL/01-001/W0/27.png", "/Other/a/b/c/d/e.tif", "/Prime_FULL/x/W0/OU/1.tif"])
def test_parse_rejects_unknown_layouts(bad):
    with pytest.raises(ValueError):
        parse_path(bad)


def test_visit_number():
    assert visit_number("V10") == 10 and visit_number("W104") == 104
    with pytest.raises(ValueError):
        visit_number("X1")


def test_episodes_one_row_per_visit_numeric_order(raw_clinical):
    ep = build_episodes(raw_clinical)
    assert len(ep) == 4 + 2 + 3
    eye1 = ep[ep.eye_id == 1].sort_values("visit_index")
    assert list(eye1.visit_label) == ["V1", "V2", "V3", "V10"]  # numeric, not lexicographic
    assert list(eye1.cst) == [500, 400, 350, 300]
    assert (ep.n_images == 3).all()
    assert ep.visit_id.is_unique


def test_episodes_flags_missing_and_keeps_week(raw_clinical):
    ep = build_episodes(raw_clinical).set_index("visit_id")
    assert "missing_cst_bcva" in ep.loc["01-001_W4", "qc_flags"]
    assert ep.loc["01-001_W12", "week"] == 12
    assert pd.isna(ep.loc["0201GOD_V2", "week"])  # TREX visit numbers are ordinal only


def test_inconsistent_cst_within_visit_raises(raw_clinical):
    bad = raw_clinical.copy()
    bad.loc[0, "CST"] = 999
    with pytest.raises(ValueError):
        build_episodes(bad)


def test_visit_images_preserve_every_path(raw_clinical):
    imgs = build_visit_images(raw_clinical)
    assert len(imgs) == len(raw_clinical)
    assert set(imgs.file_path) == set(raw_clinical.file_path)
    assert {"source_file", "source_row"} <= set(imgs.columns)


def test_split_is_patient_level(raw_clinical):
    ep = build_episodes(raw_clinical)
    ep["split"] = assign_splits(ep)
    assert (ep.groupby("patient_id").split.nunique() == 1).all()


# --- integration checks on the real OLIVES label files ---

def test_real_episode_counts(real_data):
    ep, images = real_data
    assert len(ep) == 1594
    assert ep.eye_id.nunique() == 96
    assert ep.patient_id.nunique() == 87
    assert len(images) == 78185 and images.file_path.is_unique
    assert ep.n_images.sum() == len(images)


def test_real_splits_no_leakage(real_data):
    ep, _ = real_data
    assert (ep.groupby("patient_id").split.nunique() == 1).all()
    assert (ep.groupby("eye_id").split.nunique() == 1).all()


def test_real_known_qc_issues(real_data):
    ep, _ = real_data
    flagged = ep.set_index("visit_id").qc_flags
    assert "missing_cst_bcva" in flagged["0247GOD_V3"]
    assert "laterality_conflict_biomarker_table" in flagged["02-016_W0"]
    assert ep.has_biomarkers.sum() == 192
    assert (ep[ep.has_biomarkers].groupby("eye_id").size() == 2).all()
