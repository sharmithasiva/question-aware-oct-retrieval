"""Per-visit values from the study spreadsheets (OCT-DR.xlsx, OCT-DME.xlsx).

Values loaded here that are not in the image tables (leakage index, DRSS, injection) are HELD OUT:
they are written to `heldout_prime.csv` and read only by the v2 answer key, never by retrievers.
"""
from __future__ import annotations

import re

import pandas as pd

from . import config
from .episodes import make_visit_id

PRIME_SUBJECT = re.compile(r"^\d\d-\d\d\d$")
HELDOUT_COLUMNS = ("leakage_index", "drss_level", "injection")


def _prime_week_label(header: str) -> str | None:
    h = str(header).strip()
    if h == "Screen":
        return "W0"
    m = re.fullmatch(r"Week (\d+)", h)
    return f"W{m.group(1)}" if m else None


def load_prime_field(sheet: str, field: str, path=config.OCT_DR_XLSX) -> pd.DataFrame:
    """Long table (subject, visit_label, raw, value) for one field of a wide OCT-DR sheet.

    Row 0 holds 'Screen'/'Week n' spanning several columns; row 1 holds field names.
    Only rows whose first cell is a Prime subject id (e.g. '01-001') are read; legend rows are ignored.
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    weeks = raw.iloc[0].ffill().map(_prime_week_label)
    fields = raw.iloc[1].astype(str).str.strip()
    cols = [j for j in range(raw.shape[1]) if fields[j] == field and weeks[j] is not None]
    rows = []
    for i in range(2, raw.shape[0]):
        subject = str(raw.iat[i, 0]).strip()
        if not PRIME_SUBJECT.match(subject):
            continue
        for j in cols:
            rows.append((subject, weeks[j], raw.iat[i, j]))
    out = pd.DataFrame(rows, columns=["subject", "visit_label", "raw"])
    if out.duplicated(["subject", "visit_label"]).any():
        raise ValueError(f"Duplicate subject/week in sheet {sheet!r} field {field!r}")
    out["value"] = pd.to_numeric(out.raw, errors="coerce")
    return out


def load_prime_injection(path=config.OCT_DR_XLSX) -> pd.DataFrame:
    d = load_prime_field("Data", "Injection", path)
    d["value"] = d.raw.astype(str).str.strip().map({"Yes": 1.0, "No": 0.0})
    return d


def build_heldout_prime(episodes: pd.DataFrame, path=config.OCT_DR_XLSX) -> pd.DataFrame:
    """Held-out per-visit values for Prime visits, keyed by visit_id."""
    prime = episodes.loc[episodes.trial == "Prime_FULL", ["visit_id", "subject", "visit_label"]]
    parts = {
        "leakage_index": load_prime_field("DRSS", "Leakage Index", path),
        "drss_level": load_prime_field("DRSS", "DRSS Level", path),
        "injection": load_prime_injection(path),
    }
    out = prime.copy()
    for name, d in parts.items():
        out = out.merge(d[["subject", "visit_label", "value"]].rename(columns={"value": name}),
                        on=["subject", "visit_label"], how="left", validate="one_to_one")
    out["source_file"] = f"full_labels/{path.name}"
    return out.drop(columns=["subject", "visit_label"]).reset_index(drop=True)


def prime_alignment(episodes: pd.DataFrame, path=config.OCT_DR_XLSX) -> dict:
    """How many Prime image visits match the OCT-DR CST and BCVA sheets exactly."""
    prime = episodes[episodes.trial == "Prime_FULL"]
    res = {"visits": len(prime)}
    for sheet, col in (("CST", "cst"), ("BCVA", "bcva")):
        d = load_prime_field(sheet, sheet, path)
        m = prime.merge(d, on=["subject", "visit_label"], how="left")
        res[f"{col}_match"] = int((m.value == m[col]).sum())
    return res


def load_trex_visits(path=config.LABELS_DIR / "OCT-DME.xlsx") -> pd.DataFrame:
    """(subject, visit_number, etdrs, snellen, oct) from the Monthly/TREX/GILA sheets of OCT-DME.xlsx."""
    rows = []
    for sheet in ("Monthly", "TREX", "GILA"):
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        hdr = next(i for i in range(6) if str(raw.iat[i, 0]).strip() == "StudyID")
        visit = raw.iloc[hdr - 1].ffill()
        field = raw.iloc[hdr].astype(str).str.strip().str.upper()
        for i in range(hdr + 1, raw.shape[0]):
            subject = str(raw.iat[i, 0]).strip()
            if not re.fullmatch(r"\d{4}[GMT]O[DS]", subject):
                continue
            rec = {}
            for j in range(1, raw.shape[1]):
                vn = pd.to_numeric(visit[j], errors="coerce")
                if pd.isna(vn) or field[j] not in ("ETDRS", "SNELLEN", "OCT"):
                    continue
                rec.setdefault(int(vn), {})[field[j].lower()] = pd.to_numeric(raw.iat[i, j], errors="coerce")
            for vn, vals in rec.items():
                rows.append(dict(subject=subject, visit_number=vn, **vals))
    return pd.DataFrame(rows)


def trex_alignment(episodes: pd.DataFrame, path=config.LABELS_DIR / "OCT-DME.xlsx") -> dict:
    trex = episodes[episodes.trial == "TREX DME"]
    m = trex.merge(load_trex_visits(path), on=["subject", "visit_number"], how="left")
    return {"visits": len(trex), "cst_match": int((m.oct == m.cst).sum()),
            "bcva_match": int((m.etdrs == m.bcva).sum())}


def main():
    ep = pd.read_csv(config.PROCESSED_DIR / "episodes.csv")
    held = build_heldout_prime(ep)
    held.to_csv(config.PROCESSED_DIR / "heldout_prime.csv", index=False)
    print("held-out Prime visits:", len(held), "| non-missing:", held[list(HELDOUT_COLUMNS)].notna().sum().to_dict())
    print("Prime alignment:", prime_alignment(ep))
    print("TREX alignment:", trex_alignment(ep))


if __name__ == "__main__":
    main()
