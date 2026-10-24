"""Raw label loading and trial-specific path parsing."""
from __future__ import annotations

import pandas as pd

from . import config

BIOMARKER_PATH_COL = "Path (Trial/Arm/Folder/Visit/Eye/Image Name)"
BIOMARKER_COLS = [
    "Atrophy / thinning of retinal layers", "Disruption of EZ", "DRIL", "IR hemorrhages", "IR HRF",
    "Partially attached vitreous face", "Fully attached vitreous face", "Preretinal tissue/hemorrhage",
    "Vitreous debris", "VMT", "DRT/ME", "Fluid (IRF)", "Fluid (SRF)", "Disruption of RPE",
    "PED (serous)", "SHRM",
]
PATH_FIELDS = ["trial", "arm", "subject", "visit_label", "laterality", "image_name"]


def parse_path(path: str) -> dict:
    """Split an OLIVES image path. TREX DME has an arm folder; Prime_FULL does not."""
    parts = path.strip("/").split("/")
    if parts[0] == "TREX DME" and len(parts) == 6:
        trial, arm, subject, visit, lat, img = parts
    elif parts[0] == "Prime_FULL" and len(parts) == 5:
        trial, subject, visit, lat, img = parts
        arm = None
    else:
        raise ValueError(f"Unrecognised path layout: {path!r}")
    if lat not in ("OD", "OS"):
        raise ValueError(f"Unexpected laterality {lat!r} in {path!r}")
    return dict(trial=trial, arm=arm, subject=subject, visit_label=visit, laterality=lat, image_name=img)


def visit_number(label: str) -> int:
    """'V12' -> 12, 'W104' -> 104."""
    if label[:1] not in ("V", "W") or not label[1:].isdigit():
        raise ValueError(f"Unexpected visit label {label!r}")
    return int(label[1:])


def _with_parsed_paths(df: pd.DataFrame, path_col: str, source: str) -> pd.DataFrame:
    parsed = pd.DataFrame([parse_path(p) for p in df[path_col]], index=df.index)
    out = pd.concat([df.rename(columns={path_col: "file_path"}), parsed], axis=1)
    out["source_file"] = source
    # +2: 1-based and header row, so this is the row number a person sees in Excel / a text editor.
    out["source_row"] = df.index + 2
    return out


def load_clinical(path=config.CLINICAL_XLSX) -> pd.DataFrame:
    df = pd.read_excel(path)
    return _with_parsed_paths(df, "File_Path", f"full_labels/{path.name}")


def load_biomarkers(path=config.BIOMARKER_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"Scan (n/49)": "scan_index"})
    return _with_parsed_paths(df, BIOMARKER_PATH_COL, f"full_labels/{path.name}")
