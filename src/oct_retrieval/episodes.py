"""Build the visit-level episodic memory table from image-level label rows."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config
from .load import BIOMARKER_COLS, load_biomarkers, load_clinical, visit_number

EXPECTED_IMAGES_PER_VISIT = 49


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def make_visit_id(subject: str, visit_label: str) -> str:
    return f"{subject}_{visit_label}"


def _single_value(s: pd.Series):
    vals = s.dropna().unique()
    if len(vals) > 1:
        raise ValueError(f"Expected one value per visit, got {vals!r}")
    return vals[0] if len(vals) else np.nan


def build_visit_images(clin: pd.DataFrame) -> pd.DataFrame:
    """One row per image with its visit id and provenance."""
    out = clin[["subject", "visit_label", "file_path", "image_name", "source_file", "source_row"]].copy()
    out.insert(0, "visit_id", [make_visit_id(s, v) for s, v in zip(out.subject, out.visit_label)])
    return out.drop(columns=["subject", "visit_label"]).sort_values(["visit_id", "source_row"]).reset_index(drop=True)


def biomarker_visit_table(bio: pd.DataFrame) -> pd.DataFrame:
    """Per-visit biomarker aggregates: fraction of labelled B-scans positive, and any-positive."""
    bio = bio.assign(visit_id=[make_visit_id(s, v) for s, v in zip(bio.subject, bio.visit_label)])
    g = bio.groupby("visit_id")
    frac = g[BIOMARKER_COLS].mean().rename(columns=lambda c: f"bm_frac_{slug(c)}")
    anyp = g[BIOMARKER_COLS].max().rename(columns=lambda c: f"bm_any_{slug(c)}")
    out = pd.concat([frac, anyp], axis=1)
    out["bm_n_scans"] = g.size()
    out["bm_laterality"] = g.laterality.agg(_single_value)
    out["bm_source_file"] = g.source_file.first()
    return out.reset_index()


def build_episodes(clin: pd.DataFrame, bio: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (eye, visit). CST/BCVA must be constant within a visit (checked)."""
    keys = ["Eye_ID", "visit_label"]
    g = clin.groupby(keys, sort=False)
    ep = g.agg(
        patient_id=("Patient_ID", _single_value),
        subject=("subject", _single_value),
        trial=("trial", _single_value),
        arm=("arm", lambda s: _single_value(s) if s.notna().any() else None),
        laterality=("laterality", _single_value),
        cst=("CST", _single_value),
        bcva=("BCVA", _single_value),
        n_images=("file_path", "size"),
        source_file=("source_file", "first"),
        source_row_min=("source_row", "min"),
        source_row_max=("source_row", "max"),
    ).reset_index().rename(columns={"Eye_ID": "eye_id"})
    ep["visit_id"] = [make_visit_id(s, v) for s, v in zip(ep.subject, ep.visit_label)]
    ep["visit_number"] = ep.visit_label.map(visit_number)
    # Prime visit labels are study weeks; TREX labels are ordinal visit numbers only.
    ep["week"] = np.where(ep.trial == "Prime_FULL", ep.visit_number, np.nan)
    ep = ep.sort_values(["eye_id", "visit_number"]).reset_index(drop=True)
    ep["visit_index"] = ep.groupby("eye_id").cumcount()
    ep["n_visits_eye"] = ep.groupby("eye_id").visit_id.transform("size")

    flags = [[] for _ in range(len(ep))]
    for i, r in enumerate(ep.itertuples()):
        if pd.isna(r.cst) or pd.isna(r.bcva):
            flags[i].append("missing_cst_bcva")
        if r.n_images != EXPECTED_IMAGES_PER_VISIT:
            flags[i].append("nonstandard_image_count")

    ep["has_biomarkers"] = False
    if bio is not None:
        bvt = biomarker_visit_table(bio)
        ep = ep.merge(bvt, on="visit_id", how="left", validate="one_to_one")
        ep["has_biomarkers"] = ep.bm_n_scans.notna()
        conflict = ep.has_biomarkers & (ep.bm_laterality != ep.laterality)
        for i in np.flatnonzero(conflict.to_numpy()):
            flags[i].append("laterality_conflict_biomarker_table")
    ep["qc_flags"] = [";".join(f) for f in flags]

    lead = ["visit_id", "patient_id", "eye_id", "subject", "trial", "arm", "laterality", "visit_label",
            "visit_number", "visit_index", "week", "n_visits_eye", "cst", "bcva", "n_images", "has_biomarkers",
            "qc_flags", "source_file", "source_row_min", "source_row_max"]
    return ep[lead + [c for c in ep.columns if c not in lead]]


def assign_splits(ep: pd.DataFrame, fractions=(0.6, 0.2, 0.2), seed: int = config.SEED) -> pd.Series:
    """Patient-level split, stratified by trial. Returns split name per episode row."""
    rng = np.random.default_rng(seed)
    patient_trial = ep.groupby("patient_id").trial.first()
    split_of = {}
    for trial in sorted(patient_trial.unique()):
        pids = np.array(sorted(patient_trial[patient_trial == trial].index))
        rng.shuffle(pids)
        n_train = round(fractions[0] * len(pids))
        n_val = round(fractions[1] * len(pids))
        for j, pid in enumerate(pids):
            split_of[pid] = "train" if j < n_train else "val" if j < n_train + n_val else "test"
    return ep.patient_id.map(split_of)


def build_all(clin_path=config.CLINICAL_XLSX, bio_path=config.BIOMARKER_CSV):
    clin = load_clinical(clin_path)
    bio = load_biomarkers(bio_path)
    ep = build_episodes(clin, bio)
    ep.insert(ep.columns.get_loc("trial") + 1, "split", assign_splits(ep))
    return ep, build_visit_images(clin)


def main():
    ep, images = build_all()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ep.to_csv(config.PROCESSED_DIR / "episodes.csv", index=False)
    images.to_csv(config.PROCESSED_DIR / "visit_images.csv", index=False)
    print(f"episodes: {len(ep)} visits, {ep.eye_id.nunique()} eyes, {ep.patient_id.nunique()} patients")
    print(f"visit_images: {len(images)} image rows")
    print(ep.groupby("split").agg(patients=("patient_id", "nunique"), eyes=("eye_id", "nunique"),
                                  visits=("visit_id", "size")))
    print("qc flags:", ep.qc_flags[ep.qc_flags != ""].value_counts().to_dict())


if __name__ == "__main__":
    main()
