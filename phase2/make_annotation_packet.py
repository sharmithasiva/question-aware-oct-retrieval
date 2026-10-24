"""Build a BLINDED Tier B annotation packet (protocol phase2/QUESTION_EVAL_PROTOCOL.md §4).

For every Tier B query visit: candidates (all earlier visits of the same eye) are shuffled with a fixed seed and
given opaque codes C01, C02, ...; images are copied under those codes (the atlas WebPs carry no metadata chunks);
one contact sheet shows the query and the coded candidates. No visit labels, weeks, CST, BCVA or system scores are
written to anything the annotator sees. The code -> visit key goes to phase2/annotations/keys/ (annotator must not
open it). Grade cells are left EMPTY; the harness treats empty as missing, never as 0.

Usage: python phase2/make_annotation_packet.py --set dev
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase2"))
import oct_image as OI  # noqa: E402
import question_eval as QE  # noqa: E402

ANN = ROOT / "phase2" / "annotations"
SHEET_COLUMNS = [
    "annotation_row_id", "query_id", "set", "tier", "category", "question", "eye_id", "patient_id",
    "query_visit_id", "query_image", "candidate_code", "candidate_image", "presentation_order", "contact_sheet",
    "grade", "ungradable", "query_answerable", "missing_data_flags", "evidence_source", "rationale",
    "annotator_id", "annotator_role", "annotation_date", "adjudication_status", "label_source",
    "candidate_generated_by",
]


def seed_for(query_visit_id: str) -> int:
    return int(hashlib.sha256(f"tierB-packet-v0.1:{query_visit_id}".encode()).hexdigest()[:8], 16)


def contact_sheet(query_img: Path, coded: list[tuple[str, Path]], out: Path, thumb=(252, 248), cols=6) -> None:
    rows = int(np.ceil(len(coded) / cols))
    W, H = cols * (thumb[0] + 8) + 8, 520 + rows * (thumb[1] + 30) + 8
    sheet = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(sheet)
    q = Image.open(query_img).convert("L")
    sheet.paste(q, (8, 8))
    d.text((q.width + 20, 12), "QUERY (today)", fill=0)
    for i, (code, p) in enumerate(coded):
        x = 8 + (i % cols) * (thumb[0] + 8)
        y = 520 + (i // cols) * (thumb[1] + 30)
        sheet.paste(Image.open(p).convert("L").resize(thumb), (x, y + 20))
        d.text((x, y + 4), code, fill=0)
    sheet.save(out, optimize=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(QE.SETS), required=True)
    a = ap.parse_args(argv)
    cfg = QE.SETS[a.set]
    if not (cfg["subset"] / "manifest.csv").exists():
        raise SystemExit(f"no verified images for set {a.set!r} yet ({cfg['subset']})")
    episodes = pd.read_csv(ROOT / "data/processed/episodes.csv")
    qs = QE.build_query_set(episodes, a.set, "B")
    QE.assert_no_candidate_leakage(qs, episodes)
    man = OI.load_manifest(cfg["subset"])
    img = {r.visit_id: cfg["subset"] / r.local_path for r in man[man.scan_index == 24].itertuples()}
    sha = {r.visit_id: r.sha256 for r in man[man.scan_index == 24].itertuples()}

    pk = ANN / "packets" / a.set
    if pk.exists():
        shutil.rmtree(pk)
    (ANN / "keys").mkdir(parents=True, exist_ok=True)
    sheet, key = [], []
    for qv, g in qs.groupby("query_visit_id", sort=False):
        q0 = g.iloc[0]
        cands = q0.candidate_visit_ids.split(";")
        order = np.random.default_rng(seed_for(qv)).permutation(len(cands))
        coded = [(f"C{i + 1:02d}", cands[j]) for i, j in enumerate(order)]
        qdir = pk / qv
        qdir.mkdir(parents=True)
        shutil.copyfile(img[qv], qdir / "QUERY.webp")
        for code, v in coded:
            shutil.copyfile(img[v], qdir / f"{code}.webp")
            pos = cands.index(v)
            key.append(dict(query_visit_id=qv, candidate_code=code, candidate_visit_id=v, candidate_position=pos,
                            candidate_sha256=sha[v], identical_to_query_image=sha[v] == sha[qv]))
        contact_sheet(qdir / "QUERY.webp", [(c, qdir / f"{c}.webp") for c, _ in coded], qdir / "contact_sheet.png")
        for q in g.itertuples():
            for i, (code, v) in enumerate(coded):
                sheet.append(dict(
                    annotation_row_id=f"{q.query_id}::{code}", query_id=q.query_id, set=q.set, tier="B",
                    category=q.category, question=q.question, eye_id=q.eye_id, patient_id=q.patient_id,
                    query_visit_id=qv, query_image=f"packets/{a.set}/{qv}/QUERY.webp", candidate_code=code,
                    candidate_image=f"packets/{a.set}/{qv}/{code}.webp", presentation_order=i + 1,
                    contact_sheet=f"packets/{a.set}/{qv}/contact_sheet.png", grade="", ungradable="",
                    query_answerable="", missing_data_flags="" if v in img else "candidate_image_missing",
                    evidence_source="", rationale="", annotator_id="", annotator_role="", annotation_date="",
                    adjudication_status="single", label_source="human",
                    candidate_generated_by=f"programmatic: all earlier visits of the same eye; shuffled seed={seed_for(qv)}"))
    s = pd.DataFrame(sheet, columns=SHEET_COLUMNS)
    s.to_csv(ANN / f"tierB_{a.set}_sheet.csv", index=False)
    pd.DataFrame(key).to_csv(ANN / "keys" / f"tierB_{a.set}_key.csv", index=False)
    print(f"{qs.query_visit_id.nunique()} query visits, {qs.query_id.nunique()} queries, {len(s)} sheet rows, "
          f"{len(key)} coded candidates -> {pk}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
