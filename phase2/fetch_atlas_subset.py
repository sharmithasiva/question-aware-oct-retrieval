"""Fetch a small, metadata-verified subset of OLIVES B-scans from the Retina OCT atlas (retinaoct.com).

Phase 2 prototype. Standard library only. Read-only with respect to all Phase 1 files.

Source: https://retinaoct.com (third-party index of OLIVES, CC BY 4.0; images served as downsampled WebP
derivatives of the source TIFF/PNG). Endpoints used, as found in the atlas page script (API_V = "2"):
  /api/search?q=<subject>&visit=<label>&eye=<OD|OS>&facets=0&per_page=100&v=2
  /api/image/<id>?v=2            -> includes the archive-relative `path`, e.g. /TREX DME/TREX/0209TOD/V1/OD/TREXB_000019.tif
  /img/<full_key>                -> WebP image

An image is kept only if ALL checks pass against data/processed/{episodes,visit_images}.csv:
  path_in_metadata   atlas `path` is exactly one `file_path` in visit_images.csv, belonging to the expected visit_id
  eye_id / patient_id / laterality / visit label / subject match episodes.csv
  bcva / cst         atlas values equal the visit's CST and BCVA (both are constant within a visit)
Rejected candidates are written to rejected.csv with the failing check. Nothing is inferred or repaired.

Usage (from the repository root):
  python phase2/fetch_atlas_subset.py                  # default subset, ~96 images, ~5 MB
  python phase2/fetch_atlas_subset.py --dry-run        # list planned visits only, no network
  python phase2/fetch_atlas_subset.py --eyes 25 11 61 --all-visits --scans 24       --out phase2/data/atlas_fullhist                 # full histories, central scan only (73 images)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import struct
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://retinaoct.com"
API_V = "2"
UA = "oct-retrieval-phase2/0.1 (research prototype; bounded fetch)"

# Train/val eyes only (never test), both trials, long and regular histories. Excluded on purpose:
# eye 79 (unresolved OD/OS), eyes with non-standard image counts or index gaps (29, 32, 81, 8, 45, 46).
DEFAULT_EYES = (
    25,  # TREX DME, train, 0210MOD OD, 27 visits
    30,  # TREX DME, train, 0222MOS OS, 26 visits
    11,  # TREX DME, train, 0232GOD OD (GILA arm), 25 visits
    24,  # TREX DME, val,   0206MOD OD, 26 visits
    61,  # Prime,    train, 01-013 OD, 21 visits
    67,  # Prime,    train, 01-027 OS, 21 visits
    89,  # Prime,    train, 02-036 OD, 21 visits
    72,  # Prime,    val,   01-040 OD, 21 visits
)
EXCLUDED_EYES = {79, 29, 32, 81, 8, 45, 46}
DEFAULT_VISITS_PER_EYE = 4          # evenly spaced over the eye's history, first and last included
DEFAULT_SCANS = (12, 24, 36)        # B-scan indices within the 49-scan volume (24 = central)


def get_json(path: str, params: dict | None = None) -> dict:
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf8"))


def get_bytes(path: str) -> bytes:
    req = urllib.request.Request(BASE + path, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        if r.headers.get("Content-Type", "") != "image/webp":
            raise ValueError(f"unexpected content type {r.headers.get('Content-Type')!r}")
        return r.read()


def webp_size(b: bytes) -> tuple[int, int]:
    """Width and height from a WebP header (lossy, lossless or extended)."""
    if b[:4] != b"RIFF" or b[8:12] != b"WEBP":
        raise ValueError("not a WebP file")
    chunk = b[12:16]
    if chunk == b"VP8 ":
        w, h = struct.unpack("<HH", b[26:30])
        return w & 0x3FFF, h & 0x3FFF
    if chunk == b"VP8L":
        x = int.from_bytes(b[21:25], "little")
        return (x & 0x3FFF) + 1, ((x >> 14) & 0x3FFF) + 1
    if chunk == b"VP8X":
        return int.from_bytes(b[24:27], "little") + 1, int.from_bytes(b[27:30], "little") + 1
    raise ValueError(f"unknown WebP chunk {chunk!r}")


def plan_visits(episodes: pd.DataFrame, eyes, n_visits: int | None) -> pd.DataFrame:
    """n_visits evenly spaced visits per eye (first and last included); None = every visit."""
    rows = []
    for eye in eyes:
        if eye in EXCLUDED_EYES:
            raise ValueError(f"eye {eye} is excluded (see EXCLUDED_EYES)")
        g = episodes[episodes.eye_id == eye].sort_values("visit_index")
        if g.empty:
            raise ValueError(f"eye {eye} not in episodes.csv")
        if (g.split == "test").any():
            raise ValueError(f"eye {eye} is a test-split eye; refusing")
        pick = (list(range(len(g))) if n_visits is None
                else sorted(set(np.round(np.linspace(0, len(g) - 1, n_visits)).astype(int))))
        rows.append(g.iloc[pick])
    return pd.concat(rows, ignore_index=True)


def check(rec: dict, visit: pd.Series, path_to_visit: dict) -> list[str]:
    """Return the names of failed checks (empty list = verified)."""
    fails = []
    if path_to_visit.get(rec.get("path")) != visit.visit_id:
        fails.append("path_in_metadata")
    if rec.get("eye_id") != int(visit.eye_id):
        fails.append("eye_id")
    if rec.get("patient_id") != int(visit.patient_id):
        fails.append("patient_id")
    if rec.get("eye") != visit.laterality:
        fails.append("laterality")
    if rec.get("visit") != visit.visit_label:
        fails.append("visit_label")
    if rec.get("subject") != visit.subject:
        fails.append("subject")
    for k in ("bcva", "cst"):
        a, m = rec.get(k), visit[k]
        if a is None or pd.isna(m) or float(a) != float(m):
            fails.append(k)
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "phase2" / "data" / "atlas_subset")
    ap.add_argument("--eyes", type=int, nargs="+", default=list(DEFAULT_EYES))
    ap.add_argument("--visits-per-eye", type=int, default=DEFAULT_VISITS_PER_EYE)
    ap.add_argument("--all-visits", action="store_true", help="take every visit of each eye (full history)")
    ap.add_argument("--scans", type=int, nargs="+", default=list(DEFAULT_SCANS))
    ap.add_argument("--max-images", type=int, default=100, help="hard cap on downloaded images")
    ap.add_argument("--sleep", type=float, default=0.2, help="seconds between requests")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    episodes = pd.read_csv(ROOT / "data" / "processed" / "episodes.csv")
    vi = pd.read_csv(ROOT / "data" / "processed" / "visit_images.csv", usecols=["visit_id", "file_path"])
    path_to_visit = dict(zip(vi.file_path, vi.visit_id))
    visits = plan_visits(episodes, a.eyes, None if a.all_visits else a.visits_per_eye)
    n_planned = len(visits) * len(a.scans)
    print(f"planned: {len(visits)} visits x {len(a.scans)} scans = {n_planned} images", file=sys.stderr)
    if n_planned > a.max_images:
        print(f"refusing: {n_planned} > --max-images {a.max_images}", file=sys.stderr)
        return 2
    if a.dry_run:
        print(visits[["eye_id", "visit_id", "trial", "split", "laterality", "visit_index", "week", "bcva", "cst"]]
              .to_string(index=False))
        return 0

    img_dir = a.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    kept, rejected = [], []
    for _, v in visits.iterrows():
        res = get_json("/api/search", {"q": v.subject, "visit": v.visit_label, "eye": v.laterality,
                                       "facets": "0", "per_page": "100", "v": API_V})
        time.sleep(a.sleep)
        by_scan = {}
        for r in res.get("results", []):
            by_scan.setdefault(r["scan_index"], []).append(r)
        for s in a.scans:
            cands = by_scan.get(s, [])
            if len(cands) != 1:
                rejected.append(dict(visit_id=v.visit_id, scan_index=s, atlas_id="",
                                     failed=f"search returned {len(cands)} records for this scan_index"))
                continue
            rec = get_json(f"/api/image/{cands[0]['id']}", {"v": API_V})
            time.sleep(a.sleep)
            fails = check(rec, v, path_to_visit)
            if fails:
                rejected.append(dict(visit_id=v.visit_id, scan_index=s, atlas_id=rec.get("id"),
                                     failed=";".join(fails)))
                continue
            b = get_bytes("/img/" + rec["full_key"])
            time.sleep(a.sleep)
            w, h = webp_size(b)
            rel = Path("images") / f"{v.visit_id}__scan{s:02d}.webp"
            (a.out / rel).write_bytes(b)
            kept.append(dict(
                image_key=f"{v.visit_id}__scan{s:02d}", local_path=rel.as_posix(), atlas_id=rec["id"],
                archive_path=rec["path"], atlas_full_key=rec["full_key"], visit_id=v.visit_id,
                eye_id=int(v.eye_id), patient_id=int(v.patient_id), subject=v.subject, trial=v.trial,
                split=v.split, laterality=v.laterality, visit_label=v.visit_label, visit_index=int(v.visit_index),
                n_visits_eye=int(v.n_visits_eye), week=v.week, scan_index=s, bcva=v.bcva, cst=v.cst,
                atlas_label_source=rec.get("label_source"), width=w, height=h, bytes=len(b),
                sha256=hashlib.sha256(b).hexdigest(), verified="path,eye_id,patient_id,laterality,visit,subject,bcva,cst"))
            print(f"ok  {kept[-1]['image_key']}  {w}x{h}", file=sys.stderr)

    man = pd.DataFrame(kept)
    man.to_csv(a.out / "manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(rejected, columns=["visit_id", "scan_index", "atlas_id", "failed"]).to_csv(
        a.out / "rejected.csv", index=False)
    prov = dict(
        source="Retina OCT atlas (third-party index of OLIVES)", base_url=BASE, api_version=API_V,
        fetched_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        image_format="WebP derivative (downsampled by the atlas; not the original TIFF/PNG)",
        dataset="OLIVES (Prabhushankar et al., NeurIPS Datasets and Benchmarks 2022), CC BY 4.0, "
                "doi:10.5281/zenodo.7105232",
        eyes=a.eyes, visits_per_eye="all" if a.all_visits else a.visits_per_eye, scans=a.scans,
        n_kept=len(kept), n_rejected=len(rejected), total_bytes=int(man.bytes.sum()) if len(man) else 0,
        verification="each kept image: atlas path == visit_images.csv file_path of the expected visit; "
                     "eye_id, patient_id, laterality, visit label, subject, BCVA and CST equal episodes.csv",
        caveat="Pixel content is as served by the atlas. It has not been compared byte-for-byte with the "
               "official Zenodo archive.")
    (a.out / "provenance.json").write_text(json.dumps(prov, indent=2))
    with open(a.out / "SHA256SUMS", "w", newline="\n") as f:
        for r in kept:
            f.write(f"{r['sha256']}  {r['local_path']}\n")
    print(f"kept {len(kept)}, rejected {len(rejected)} -> {a.out}", file=sys.stderr)
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
