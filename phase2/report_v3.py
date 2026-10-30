"""Consolidated Phase 3 report from SAVED outputs only (E3 summary, pathology-probe summary, dataset tables).

Writes <out>/report.md with: dataset and annotation coverage, question-category table, encoder comparison, ablation
A-E and the pathology table. A system whose embeddings were not produced is shown as N/A, never estimated.

Usage: python phase2/report_v3.py --e3 phase2/results/e3/fullhist_dev_v1 \
           --pathology phase2/results/pathology/atlas_subset_pilot_v1 --out phase2/results/report_v3/local_pilot_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase2"))
import pathology_probe as PP  # noqa: E402

ABLATION = [("A: Metadata-only", "metadata_only"), ("B: ResNet-only", "image_resnet50"),
            ("C: OCT encoder-only", "image_retfound_oct"), ("D: Hybrid ResNet (question-aware)", "question_aware_resnet50"),
            ("E: Hybrid OCT (question-aware)", "question_aware_retfound_oct")]


def fmt(x) -> str:
    return "N/A" if x is None or (isinstance(x, float) and pd.isna(x)) else (f"{x:.3f}" if isinstance(x, float) else str(x))


def table(rows: list[list], header: list[str]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows])


def coverage() -> str:
    ep = pd.read_csv(ROOT / "data/processed/episodes.csv")
    plan = pd.read_csv(ROOT / "phase2/data/scan_subset_v1/plan.csv")
    rows = [["Visits / eyes / patients (all splits)", f"{len(ep)} / {ep.eye_id.nunique()} / {ep.patient_id.nunique()}",
             "OLIVES clinical table"],
            ["CST and BCVA", f"{ep.cst.notna().sum()} / {ep.bcva.notna().sum()} visits", "every visit (structured)"],
            ["Finding labels (16 biomarkers, per B-scan)", f"{int(ep.has_biomarkers.sum())} visits = first + last visit "
             f"of {ep[ep.has_biomarkers].eye_id.nunique()} eyes", "OLIVES; none for intermediate visits"],
            ["Clinical edema-severity grade", "none", "CST is a thickness measurement, not a grade"]]
    for name, path in [("atlas_subset (E1, local)", "phase2/data/atlas_subset"),
                       ("atlas_fullhist (E2, local)", "phase2/data/atlas_fullhist")]:
        s = PP.labelled_scans(ROOT / path)
        rows.append([f"Labelled B-scans in {name}", f"{len(s)} scans, {s.eye_id.nunique()} eyes", "exact path join"])
    lab = pd.read_csv(PP.LABELS).rename(columns={"Path (Trial/Arm/Folder/Visit/Eye/Image Name)": "archive_path"})
    n = plan.merge(lab[["archive_path"]], on="archive_path")
    rows.append(["Labelled B-scans in scan_subset_v1 (planned)", f"{len(n)} scans, {plan[plan.archive_path.isin(n.archive_path)].eye_id.nunique()} eyes",
                 "available once fetched + embedded"])
    rows.append(["Visual-similarity judgments (B1)", "18 queries, 264 judgments, 3 dev eyes",
                 "one researcher_non_expert annotator"])
    rows.append(["Fluid-similarity judgments (B2)", "none", "packet exists; not graded"])
    return table(rows, ["Source", "Coverage", "Notes"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e3", type=Path, required=True)
    ap.add_argument("--pathology", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    e3 = pd.read_csv(a.e3 / "summary.csv")
    e3cfg = json.loads((a.e3 / "config.json").read_text())
    pp = pd.read_csv(a.pathology / "summary.csv")
    ppcfg = json.loads((a.pathology / "config.json").read_text())
    prim = {c: ("ndcg5" if c == "image_similarity_B1" else "mrr") for c in e3.category.unique()}
    get = {(r.category, r.system): r for r in e3.itertuples()}

    def val(cat, sys_, m=None):
        r = get.get((cat, sys_))
        return None if r is None else getattr(r, m or prim[cat])

    cats = list(dict.fromkeys(e3.category))
    encs = sorted({s.removeprefix("image_") for s in e3.system if s.startswith("image_")})
    parts = ["# Phase 3 consolidated results", "",
             f"E3: `{a.e3}` (eyes {e3cfg['eyes']}). Pathology: `{a.pathology}`. Every number is read from those "
             "saved outputs. Exploratory: the unit of independence is the eye/patient, and no CI is reported with "
             "fewer than 5.", "", "## 1. Dataset and annotation coverage", "", coverage(), ""]

    rows = []
    for c in cats:
        r = get[(c, "metadata_only")]
        rows.append([c, r.gt_source, r.n_queries, r.n_eyes, r.n_patients, prim[c].upper(), val(c, "latest_prior"),
                     val(c, "metadata_only"), val(c, "image_resnet50"), val(c, "safe_hybrid_resnet50"),
                     val(c, "question_aware_resnet50"), val(c, "random_expected")])
    parts += ["## 2. Question categories (primary metric: MRR, or nDCG@5 for B1)", "",
              table(rows, ["Category", "Ground truth", "Queries", "Eyes", "Patients", "Metric", "latest_prior",
                           "metadata-only", "ResNet image-only", "safe hybrid", "question-aware", "random"]), "",
              "Full Recall@1 / Recall@5 / Hit@1 / MRR / nDCG@5 per system and category: `summary.md` in the E3 "
              "folder.", ""]

    rows = [[c] + [val(c, f"image_{e}") for e in encs + (["retfound_oct"] if "retfound_oct" not in encs else [])]
            for c in cats]
    hdr = ["Category"] + [f"image-only {e}" for e in encs] + (["image-only retfound_oct"] if "retfound_oct" not in encs
                                                             else [])
    parts += ["## 3. Encoder comparison (E3 image-only, same queries)", "", table(rows, hdr), ""]
    ev = pp[pp.status == "evaluated"]
    rows = []
    for task, g in pp.groupby("task", sort=False):
        if g.status.iloc[0] != "evaluated":
            rows.append([task, g.n_pos.iloc[0], g.n_patients.iloc[0], "N/A (insufficient labels)"] + [""] * 4)
            continue
        m = {r.method: r for r in g.itertuples()}
        for meth in ["position_only", "metadata_cst"] + [f"{k}_{e}" for e in encs for k in ("knn", "probe", "combined")]:
            if meth in m:
                r = m[meth]
                rows.append([task, r.n_pos, r.n_patients, meth, r.auroc, r.auroc_ci95_patient, r.auprc,
                             r.prevalence])
    if "retfound_oct" not in encs:
        rows.append(["all", "", "", "knn/probe_retfound_oct", "N/A", "", "N/A", ""])
    parts += ["## 4. Pathology probe (per-B-scan OLIVES labels, folds grouped by patient)", "",
              table(rows, ["Task", "Positives", "Patients", "Method", "AUROC", "AUROC 95% CI (patient)", "AUPRC",
                           "Prevalence (AUPRC chance)"]), "",
              f"Subset `{ppcfg['subset']}`; splits {ppcfg['splits']}.", ""]

    rows = [[c] + [val(c, s) for _, s in ABLATION] for c in cats]
    parts += ["## 5. Ablation (primary metric per category)", "", table(rows, ["Category"] + [n for n, _ in ABLATION]),
              "", "N/A = the system was not run (no embeddings for that encoder), not a zero.", ""]
    a.out.mkdir(parents=True)
    (a.out / "report.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    print((a.out / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
