"""Fetch the B-scans listed in phase2/data/<name>/plan.csv (from build_scan_manifest.py), resumably.

Reuses fetch_atlas_subset.py (same atlas endpoints, same verification `check`, same WebP header parsing); does not
modify it. Scans already present in the verified 169-image subsets are COPIED (SHA-256 re-checked against their
manifest), not downloaded again. Those subsets are only read.

Per planned scan, exactly one outcome is recorded:
  kept      manifest.csv  (same columns as the atlas_subset manifest, plus `source` and `duplicate_of`)
  rejected  rejected.csv  (failed verification, search returned 0 or >1 records, or a network error; errors are
            retried on the next run because they are not in the manifest)
Byte-identical images (e.g. the known 01-013 W8 / W12 pair) are kept but marked in `duplicate_of` with the first
image_key of that SHA-256, so evaluations can exclude them; nothing is silently dropped.

Images go to phase2/data/<name>/images/ (git-ignored; ~50 KB each). manifest.csv, rejected.csv, provenance.json and
fetch_counts.json are small and can be committed.

Usage (repository root; needs network access to retinaoct.com):
  python phase2/fetch_scan_subset.py --dry-run          # counts only, no network
  python phase2/fetch_scan_subset.py                    # fetch everything planned (resumable; Ctrl-C safe)
  python phase2/fetch_scan_subset.py --limit 200        # fetch at most 200 new images this run
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase2"))
import fetch_atlas_subset as F  # noqa: E402
from build_scan_manifest import DEFAULT_NAME  # noqa: E402

LOCAL_SUBSETS = (ROOT / "phase2/data/atlas_subset", ROOT / "phase2/data/atlas_fullhist")
MANIFEST_COLUMNS = ["image_key", "local_path", "atlas_id", "archive_path", "atlas_full_key", "visit_id", "eye_id",
                    "patient_id", "subject", "trial", "split", "laterality", "visit_label", "visit_index",
                    "n_visits_eye", "week", "scan_index", "bcva", "cst", "atlas_label_source", "width", "height",
                    "bytes", "sha256", "verified", "source", "duplicate_of"]
REJECT_COLUMNS = ["image_key", "visit_id", "scan_index", "atlas_id", "failed"]
VERIFIED = "path,eye_id,patient_id,laterality,visit,subject,bcva,cst"
HARD_CAP = 6000


class AtlasAPI:
    """Thin wrapper so tests can substitute a fake."""

    def __init__(self, sleep: float):
        self.sleep = sleep

    def search(self, v) -> dict:
        r = F.get_json("/api/search", {"q": v.subject, "visit": v.visit_label, "eye": v.laterality, "facets": "0",
                                       "per_page": "100", "v": F.API_V})
        time.sleep(self.sleep)
        return r

    def image(self, atlas_id) -> dict:
        r = F.get_json(f"/api/image/{atlas_id}", {"v": F.API_V})
        time.sleep(self.sleep)
        return r

    def bytes(self, full_key: str) -> bytes:
        b = F.get_bytes("/img/" + full_key)
        time.sleep(self.sleep)
        return b


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def local_index() -> dict[str, tuple[Path, pd.Series]]:
    """image_key -> (file, manifest row) for every image of the existing verified subsets."""
    out = {}
    for d in LOCAL_SUBSETS:
        if (d / "manifest.csv").exists():
            for _, r in pd.read_csv(d / "manifest.csv").iterrows():
                out.setdefault(r.image_key, (d / r.local_path, r))
    return out


def kept_row(p: pd.Series, rel: str, b: bytes, rec: dict, source: str) -> dict:
    w, h = F.webp_size(b)
    return dict(image_key=p.image_key, local_path=rel, atlas_id=rec.get("id"), archive_path=rec.get("path"),
                atlas_full_key=rec.get("full_key"), visit_id=p.visit_id, eye_id=int(p.eye_id),
                patient_id=int(p.patient_id), subject=p.subject, trial=p.trial, split=p.split,
                laterality=p.laterality, visit_label=p.visit_label, visit_index=int(p.visit_index),
                n_visits_eye=int(p.n_visits_eye), week=p.week, scan_index=int(p.scan_index), bcva=p.bcva, cst=p.cst,
                atlas_label_source=rec.get("label_source"), width=w, height=h, bytes=len(b), sha256=sha(b),
                verified=VERIFIED, source=source, duplicate_of="")


def fetch(plan: pd.DataFrame, out: Path, api, limit: int | None = None, local: dict | None = None,
          log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch every planned scan not yet in out/manifest.csv. Returns (manifest, rejected of this run)."""
    (out / "images").mkdir(parents=True, exist_ok=True)
    man_p = out / "manifest.csv"
    kept = pd.read_csv(man_p).to_dict("records") if man_p.exists() else []
    done = {r["image_key"] for r in kept if (out / r["local_path"]).exists()
            and sha((out / r["local_path"]).read_bytes()) == r["sha256"]}
    kept = [r for r in kept if r["image_key"] in done]          # drop rows whose file is missing or corrupted
    local = local_index() if local is None else local
    rejected, n_new = [], 0
    todo = plan[~plan.image_key.isin(done)]

    def save():
        m = pd.DataFrame(kept, columns=MANIFEST_COLUMNS)
        first = m.drop_duplicates("sha256").set_index("sha256").image_key
        m["duplicate_of"] = [first[s] if first[s] != k else "" for k, s in zip(m.image_key, m.sha256)]
        m = m.sort_values(["trial", "eye_id", "visit_index", "scan_index"]).reset_index(drop=True)
        m.to_csv(man_p, index=False)
        return m

    for vid, g in todo.groupby("visit_id", sort=False):
        v = g.iloc[0]
        need_net = []
        for _, p in g.iterrows():
            rel = f"images/{p.image_key}.webp"
            if p.image_key in local:
                src, r = local[p.image_key]
                b = src.read_bytes()
                if r.archive_path == p.archive_path and sha(b) == r.sha256:
                    (out / rel).write_bytes(b)
                    kept.append(kept_row(p, rel, b, dict(id=r.atlas_id, path=r.archive_path,
                                                         full_key=r.atlas_full_key,
                                                         label_source=r.atlas_label_source),
                                         f"local_copy:{src.parent.parent.name}"))
                    continue
            need_net.append(p)
        if need_net and limit is not None and n_new >= limit:
            break
        if need_net:
            try:
                res = api.search(v)
            except Exception as e:  # network errors are recorded, then retried on the next run
                rejected += [dict(image_key=p.image_key, visit_id=vid, scan_index=p.scan_index, atlas_id="",
                                  failed=f"error:search:{type(e).__name__}") for p in need_net]
                continue
            by_scan: dict[int, list] = {}
            for r in res.get("results", []):
                by_scan.setdefault(int(r["scan_index"]), []).append(r)
            for p in need_net:
                if limit is not None and n_new >= limit:
                    break
                cands = by_scan.get(int(p.scan_index), [])
                base = dict(image_key=p.image_key, visit_id=vid, scan_index=int(p.scan_index))
                if len(cands) != 1:
                    rejected.append(dict(base, atlas_id="", failed=f"search returned {len(cands)} records"))
                    continue
                try:
                    rec = api.image(cands[0]["id"])
                    fails = F.check(rec, v, {p.archive_path: vid})
                    if fails:
                        rejected.append(dict(base, atlas_id=rec.get("id"), failed=";".join(fails)))
                        continue
                    b = api.bytes(rec["full_key"])
                    rel = f"images/{p.image_key}.webp"
                    row = kept_row(p, rel, b, rec, "atlas_download")
                except Exception as e:
                    rejected.append(dict(base, atlas_id=cands[0].get("id", ""), failed=f"error:{type(e).__name__}"))
                    continue
                (out / rel).write_bytes(b)
                kept.append(row)
                n_new += 1
        save()
        log(f"{vid}: total kept {len(kept)}, new downloads {n_new}, rejected this run {len(rejected)}")
    return save(), pd.DataFrame(rejected, columns=REJECT_COLUMNS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=DEFAULT_NAME)
    ap.add_argument("--limit", type=int, help="maximum NEW downloads in this run")
    ap.add_argument("--sleep", type=float, default=0.2, help="seconds between requests")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    out = ROOT / "phase2/data" / a.name
    plan = pd.read_csv(out / "plan.csv")
    if len(plan) > HARD_CAP:
        raise SystemExit(f"plan has {len(plan)} scans > hard cap {HARD_CAP}")
    if (plan.split != "train").any():
        raise SystemExit("plan contains non-train eyes; refusing (held-out / test eyes are not fetched here)")
    local = local_index()
    n_local = int(plan.image_key.isin(local).sum())
    if a.dry_run:
        print(json.dumps(dict(planned=len(plan), reusable_local=n_local, to_download=len(plan) - n_local,
                              approx_mb=round((len(plan) - n_local) * 0.05, 1)), indent=2))
        return 0
    man, rej = fetch(plan, out, AtlasAPI(a.sleep), a.limit, local, log=lambda s: print(s, file=sys.stderr))
    # rejected.csv accumulates over runs; a later success supersedes an earlier error for the same image.
    rp = out / "rejected.csv"
    allrej = pd.concat([pd.read_csv(rp) if rp.exists() else pd.DataFrame(columns=REJECT_COLUMNS), rej])
    allrej = allrej[~allrej.image_key.isin(man.image_key)].drop_duplicates("image_key", keep="last")
    allrej.to_csv(rp, index=False)
    with open(out / "SHA256SUMS", "w", newline="\n") as f:
        for r in man.itertuples():
            f.write(f"{r.sha256}  {r.local_path}\n")
    counts = dict(planned=len(plan), kept=len(man), rejected=len(allrej),
                  not_attempted=len(plan) - len(man) - len(allrej),
                  kept_by_source=man.source.value_counts().to_dict(),
                  rejected_by_reason=allrej.failed.value_counts().to_dict(),
                  duplicate_images=int((man.duplicate_of != "").sum()),
                  kept_eyes=int(man.eye_id.nunique()), kept_visits=int(man.visit_id.nunique()),
                  total_bytes=int(man.bytes.sum()))
    (out / "fetch_counts.json").write_text(json.dumps(counts, indent=2))
    prov = dict(source="Retina OCT atlas (third-party index of OLIVES)", base_url=F.BASE, api_version=F.API_V,
                updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                image_format="WebP derivative (downsampled by the atlas; not the original TIFF/PNG)",
                dataset="OLIVES (Prabhushankar et al., NeurIPS Datasets and Benchmarks 2022), CC BY 4.0, "
                        "doi:10.5281/zenodo.7105232",
                plan="plan.csv (phase2/build_scan_manifest.py)", local_reuse=[str(d.relative_to(ROOT)) for d in
                                                                              LOCAL_SUBSETS],
                verification="as fetch_atlas_subset.check: atlas path == planned visit_images.csv file_path; eye, "
                             "patient, laterality, visit label, subject, BCVA and CST equal episodes.csv",
                caveat="Pixel content is as served by the atlas; not compared byte-for-byte with the Zenodo archive.")
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
