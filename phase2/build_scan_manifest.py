"""Plan a larger, reproducible B-scan subset (3,000-5,000 scans) from LOCAL metadata only. No network.

Experiment `scan_subset_v1`: a NEW experiment. It does not touch the 169-image atlas_subset / atlas_fullhist
artifacts or any reported result.

Selection (deterministic, no random sampling unless --max-scans forces whole-eye subsampling):
  eyes     split == train only (val holds the protocol's held-out eyes, test is never used), minus
           fetch_atlas_subset.EXCLUDED_EYES (unresolved laterality / non-standard image counts)
  visits   every visit of every kept eye (the longitudinal structure is kept whole); visits with a qc_flag are
           excluded and counted
  scans    fixed B-scan positions (default 12, 18, 24, 30, 36 of the 49-scan volume; 24 = central). The position is
           read from the archive file name in visit_images.csv (TREXJ_000024.tif -> 24, 24.tif -> 24); this
           mapping was checked against both verified atlas manifests.
  --max-scans  if the plan is larger, whole eyes are dropped with a seeded RNG (never part of an eye), so no
           eye's history is split and correlated scans stay together.

Writes phase2/data/<name>/plan.csv (one row per planned scan, with archive_path = the visit_images.csv file_path
the fetcher must match) and plan_counts.json (inclusion / exclusion counts at every step).

Usage: python phase2/build_scan_manifest.py [--scans 12 18 24 30 36] [--max-scans 5000] [--name scan_subset_v1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase2"))
from fetch_atlas_subset import EXCLUDED_EYES  # noqa: E402

DEFAULT_SCANS = (12, 18, 24, 30, 36)
DEFAULT_NAME = "scan_subset_v1"
SEED = 20261029
PLAN_COLUMNS = ["image_key", "visit_id", "eye_id", "patient_id", "subject", "trial", "split", "laterality",
                "visit_label", "visit_index", "n_visits_eye", "week", "scan_index", "bcva", "cst", "archive_path"]


def scan_index_from_path(paths: pd.Series) -> pd.Series:
    return paths.str.extract(r"(\d+)\.\w+$")[0].astype(int)


def build_plan(episodes: pd.DataFrame, visit_images: pd.DataFrame, scans=DEFAULT_SCANS, splits=("train",),
               max_scans: int | None = None, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    c: dict = {"splits": list(splits), "scans": list(scans)}
    c["visits_total"] = len(episodes)
    ep = episodes[episodes.split.isin(splits)]
    c["visits_in_splits"] = len(ep)
    c["eyes_in_splits"] = int(ep.eye_id.nunique())
    excl = sorted(set(ep.eye_id) & EXCLUDED_EYES)
    ep = ep[~ep.eye_id.isin(EXCLUDED_EYES)]
    c["eyes_excluded_list"] = [int(e) for e in excl]
    qc = ep[ep.qc_flags.notna()]
    c["visits_excluded_qc_flag"] = {r.visit_id: r.qc_flags for r in qc.itertuples()}
    ep = ep[ep.qc_flags.isna()]

    vi = visit_images[visit_images.visit_id.isin(ep.visit_id)].copy()
    vi["scan_index"] = scan_index_from_path(vi.file_path)
    dup = vi.duplicated(["visit_id", "scan_index"], keep=False)
    c["visits_excluded_ambiguous_scan_index"] = sorted(vi[dup].visit_id.unique())
    vi = vi[~vi.visit_id.isin(c["visits_excluded_ambiguous_scan_index"])]
    vi = vi[vi.scan_index.isin(scans)]
    if vi.file_path.duplicated().any():
        raise ValueError("duplicate archive paths in visit_images.csv")

    plan = ep.merge(vi[["visit_id", "scan_index", "file_path"]], on="visit_id", how="left", validate="one_to_many")
    have = plan.groupby("visit_id").scan_index.apply(lambda s: set(s.dropna().astype(int)))
    c["visit_scan_slots_missing_in_metadata"] = int(sum(len(set(scans) - h) for h in have))
    plan = plan.dropna(subset=["file_path"]).rename(columns={"file_path": "archive_path"})
    plan["scan_index"] = plan.scan_index.astype(int)

    if max_scans is not None and len(plan) > max_scans:
        rng = np.random.default_rng(seed)
        eyes = list(rng.permutation(sorted(plan.eye_id.unique())))
        per_eye = plan.groupby("eye_id").size()
        keep, total = [], 0
        for e in eyes:
            if total + per_eye[e] <= max_scans:
                keep.append(e)
                total += per_eye[e]
        c["eyes_dropped_for_max_scans"] = sorted(int(e) for e in set(eyes) - set(keep))
        plan = plan[plan.eye_id.isin(keep)]

    plan["image_key"] = plan.visit_id + "__scan" + plan.scan_index.map("{:02d}".format)
    plan = plan.sort_values(["trial", "eye_id", "visit_index", "scan_index"]).reset_index(drop=True)[PLAN_COLUMNS]
    assert not plan.image_key.duplicated().any()
    c.update(planned_scans=len(plan), planned_visits=int(plan.visit_id.nunique()),
             planned_eyes=int(plan.eye_id.nunique()), planned_patients=int(plan.patient_id.nunique()),
             patients_with_two_eyes=int((plan.groupby("patient_id").eye_id.nunique() > 1).sum()),
             by_trial={t: dict(eyes=int(g.eye_id.nunique()), visits=int(g.visit_id.nunique()), scans=len(g))
                       for t, g in plan.groupby("trial")},
             scans_per_position={int(k): int(v) for k, v in plan.scan_index.value_counts().sort_index().items()},
             visits_with_missing_cst_or_bcva=int(plan.drop_duplicates("visit_id")[["cst", "bcva"]].isna()
                                                 .any(axis=1).sum()))
    return plan, c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=DEFAULT_NAME)
    ap.add_argument("--scans", type=int, nargs="+", default=list(DEFAULT_SCANS))
    ap.add_argument("--max-scans", type=int, default=5000)
    a = ap.parse_args(argv)
    if any(not 0 <= s <= 48 for s in a.scans):
        ap.error("scan positions must be within 0..48")
    out = ROOT / "phase2" / "data" / a.name
    episodes = pd.read_csv(ROOT / "data/processed/episodes.csv")
    vi = pd.read_csv(ROOT / "data/processed/visit_images.csv", usecols=["visit_id", "file_path"])
    plan, counts = build_plan(episodes, vi, a.scans, max_scans=a.max_scans)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.csv").exists():
        old = pd.read_csv(out / "plan.csv")
        if not old.equals(pd.read_csv(pd.io.common.StringIO(plan.to_csv(index=False)))):
            raise SystemExit(f"refusing to overwrite a different plan at {out / 'plan.csv'}; use --name")
    plan.to_csv(out / "plan.csv", index=False)
    counts.update(name=a.name, seed=SEED, max_scans=a.max_scans,
                  source="data/processed/episodes.csv + data/processed/visit_images.csv (local metadata only)")
    (out / "plan_counts.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps({k: counts[k] for k in ("planned_scans", "planned_visits", "planned_eyes", "planned_patients",
                                             "by_trial")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
