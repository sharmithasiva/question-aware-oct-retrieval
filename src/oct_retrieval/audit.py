"""Print the data-audit numbers quoted in docs/DATA_AUDIT.md.

Usage:  python -m oct_retrieval.audit
"""
from __future__ import annotations

import hashlib

import pandas as pd

from . import config
from .load import BIOMARKER_COLS, load_biomarkers, load_clinical


def md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    ml = config.LABELS_DIR.parent / "ml_centric_labels"
    for name in ("Clinical_Data_Images.xlsx", "Biomarker_Clinical_Data_Images.csv"):
        if (ml / name).exists():
            print(f"{name}: full_labels == ml_centric_labels ? {md5(config.LABELS_DIR / name) == md5(ml / name)}")

    clin, bio = load_clinical(), load_biomarkers()
    print(f"\nclinical rows {len(clin)}, biomarker rows {len(bio)}")
    print("rows by trial:", clin.trial.value_counts().to_dict())
    print("patients", clin.Patient_ID.nunique(), "| eyes", clin.Eye_ID.nunique(),
          "| subject folders", clin.subject.nunique())
    print("eyes per patient:", clin.groupby("Patient_ID").Eye_ID.nunique().value_counts().to_dict())

    g = clin.groupby(["Eye_ID", "visit_label"])
    print(f"visits {g.ngroups}; CST constant within visit: {(g.CST.nunique() <= 1).all()}; "
          f"BCVA constant: {(g.BCVA.nunique() <= 1).all()}")
    sizes = g.size()
    print("images/visit:", sizes.value_counts().to_dict())
    print("visits/eye:", clin.groupby("Eye_ID").visit_label.nunique().describe().round(1).to_dict())
    print("missing CST/BCVA visits:", clin[clin.CST.isna() | clin.BCVA.isna()]
          .groupby(["subject", "visit_label"]).size().to_dict())

    shared = bio.merge(clin[["file_path", "CST", "BCVA", "Eye_ID"]], on="file_path", suffixes=("", "_c"))
    print(f"\nbiomarker paths also in clinical table: {len(shared)}/{len(bio)}; "
          f"CST/BCVA/Eye agree: {((shared.CST == shared.CST_c) & (shared.BCVA == shared.BCVA_c) & (shared.Eye_ID == shared.Eye_ID_c)).all()}")
    lat_c = clin.groupby("Eye_ID").laterality.first()
    lat_b = bio.groupby("Eye_ID").laterality.first()
    conflict = (lat_b != lat_c.reindex(lat_b.index))
    print("laterality conflicts (Eye_ID):", {e: (lat_c[e], lat_b[e]) for e in lat_b[conflict].index})
    print("biomarker visits per eye:", bio.groupby("Eye_ID").visit_label.nunique().value_counts().to_dict())
    print("biomarker null cells:", bio[BIOMARKER_COLS].isna().sum()[lambda s: s > 0].to_dict())
    print("biomarker scan-level prevalence:", bio[BIOMARKER_COLS].mean().round(3).to_dict())


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
