# OLIVES Labels — Data Audit and Visit-Aggregation Strategy

All numbers below were computed directly from the files in `OLIVES_Dataset_Labels/`
(audit run 2026-10-22). They are reproduced by `python -m oct_retrieval.audit`.

## 1. Files

| File | Shape | Notes |
|---|---|---|
| `full_labels/Clinical_Data_Images.xlsx` | 78,185 × 5 | `File_Path, BCVA, CST, Eye_ID, Patient_ID`. One row per B-scan image. |
| `full_labels/Biomarker_Clinical_Data_Images.csv` | 9,408 × 22 | Path, `Scan (n/49)`, 16 binary biomarkers, `Eye_ID, BCVA, CST, Patient_ID`. |
| `full_labels/OCT-DME.xlsx` | 8 sheets | Wide, multi-row-header study spreadsheets (TREX-DME). Not used in Phase 1. |
| `full_labels/OCT-DR.xlsx` | 5 sheets | Wide study spreadsheets (Prime / DR). Only `Demographics` "Study Eye" used, for a cross-check. |
| `ml_centric_labels/*` | — | **Byte-identical** (same MD5) to the two `full_labels` image tables. Ignored. |

## 2. Path structure (the real identifiers)

The CSV header says `Trial/Arm/Folder/Visit/Eye/Image Name`, but this only holds for TREX-DME.
The two trials use different layouts:

| Trial | Layout | Example |
|---|---|---|
| `TREX DME` (DME, 56 eyes) | `/trial/arm/subject/Vn/eye/image` | `/TREX DME/GILA/0201GOD/V1/OD/TREXJ_000000.tif` |
| `Prime_FULL` (DR, 40 eyes) | `/trial/subject/Wn/eye/image` | `/Prime_FULL/01-001/W0/OS/0.tif` |

Verified mappings (both tables):

- `subject` folder ↔ `Eye_ID` is 1:1 (96 ↔ 96). Each `Eye_ID` has exactly one laterality, trial and arm.
- TREX subject code `0201GOD` = 4-digit patient (`0201` == `Patient_ID`, 100 % match) + arm letter
  (`G`=GILA, `M`=Monthly, `T`=TREX, 100 % consistent) + laterality (100 % match with the `eye` folder).
- Prime: `Patient_ID == Eye_ID` (57–96); laterality matches the OCT-DR Demographics "Study Eye" for all 40.
- 87 patients; **9 TREX patients contribute two eyes**, always in *different* arms (e.g. patient 201:
  `0201GOD` in GILA, `0201TOS` in TREX). Prime and TREX patient IDs do not collide.

## 3. Visits

- **Visit key = (`Eye_ID`, visit folder)**. There are **1,594 visits** in the clinical table
  (TREX: `V1`–`V27`; Prime: `W0`–`W104`). Visits per eye: 3–27 (mean 16.6).
- **CST and BCVA are constant within every visit** (1,594 / 1,594), so they are visit-level
  measurements repeated across B-scans. Rows ≠ independent observations.
- Images per visit: 49 for 1,589 visits; exceptions: Eye 8 V7 (36), Eye 29 V16 (48), Eye 81 W24 (46),
  Eye 32 V15 and V16 (97 — indices 0–96, a denser volume). All paths are kept.
- Chronology: TREX `Vn` ordinal number; Prime `Wn` week number. Order = numeric suffix.
  Prime has true study weeks; TREX `Vn` is ordinal only (intervals unknown, see §6).
- Image extensions are mixed (`.tif` mostly; some `.png`, `.jpg`); the paths are taken verbatim.

## 4. Missing values and conflicts

- Clinical table: CST and BCVA missing for exactly one visit (`0247GOD` V3, Eye 18; 49 rows).
  Kept as an episode with null measurements; never a ground-truth target for value questions.
- Biomarker table: 4 visual-field columns (`Partially/Fully attached vitreous face`, `Preretinal tissue/hemorrhage`,
  `Vitreous debris`) have 19 null cells in 4 visits (0206MOD V26, 0222MOS V26, 0222TOD V13, 0232TOS V21).
  Visit-level biomarker aggregates ignore nulls.
- **Laterality conflict:** Eye 79 (`02-016`) is `OS` in the clinical table and in OCT-DR Demographics, but its 98
  biomarker paths say `OD` (`/Prime_FULL/02-016/W0/OD/...`). CST/BCVA agree. We keep `OS` (2 sources agree) and
  record the conflict in the episode's `qc_flags`. This matters in Phase 2 when fetching image files.
- Where both tables cover the same path (9,310 of 9,408 biomarker rows), CST, BCVA and Eye_ID agree 100 %.

## 5. Biomarker coverage (important limitation)

Biomarker labels exist for **exactly 2 visits per eye** (192 visits): the first visit and the last visit of that eye
(100 % match with the clinical table's first/last visit). So biomarkers cannot drive retrieval across a full history
in Phase 1 — for any current visit, at most one labelled historical visit (baseline) exists. They are attached to
episodes as optional fields (`bm_frac_*`: fraction of B-scans positive; `bm_any_*`: any positive) and are **not**
used for Phase 1 question ground truth.

## 6. Ambiguities and assumptions

| Item | Decision |
|---|---|
| "Edema"/"swelling" in questions | Mapped to **CST as a proxy**. Higher CST ≈ more edema; this is a proxy, not a diagnosis. |
| "Worst/best vision" | Lowest/highest **BCVA** (ETDRS letters; higher = better). |
| TREX visit spacing | `Vn` treated as ordinal only. OCT-DME.xlsx has per-visit weeks, but we have not verified that its `VisitNum` aligns with the `Vn` image folders, so it is not used. |
| Baseline | The earliest visit with images for that eye (TREX `V1`, Prime `W0`). |
| Ties in extreme-value questions | All tied visits are relevant. |
| Biomarker CSV `Scan (n/49)` | Scan index within the 49-scan volume (1–49). Clinical table has no scan index; for TREX the image filename suffix is not guaranteed to equal it, so we do not derive one. |

## 7. Visit-aggregation strategy (implemented in `oct_retrieval/episodes.py`)

1. Parse each path with the trial-specific layout into `trial, arm, subject, visit_label, laterality, image_name`.
2. Group clinical rows by (`Eye_ID`, `visit_label`). Assert CST/BCVA are unique per group; take that value.
3. Order visits per eye by the numeric suffix → `visit_index` (0 = baseline), plus `week` (Prime only).
4. Keep the full list of image paths per visit in a separate `visit_images.csv` with source file + source row number.
5. Left-join visit-level biomarker aggregates from the biomarker CSV (192 visits).
6. Record `qc_flags` (`missing_cst_bcva`, `nonstandard_image_count`, `laterality_conflict_biomarker_table`).
7. Assign patient-level splits (grouped by `Patient_ID`, stratified by trial) so both eyes of a patient
   and all visits of an eye stay in the same partition.
