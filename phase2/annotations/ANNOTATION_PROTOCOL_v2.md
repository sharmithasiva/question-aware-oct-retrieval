# Tier B annotation protocol and relevance rubric: v2

Written 2026-10-26. This **supersedes the annotation procedure** in `phase2/QUESTION_EVAL_PROTOCOL.md` §4–§5 (v0.1) for
Tier B. Everything else in that protocol is unchanged: eligibility, systems, metrics, the dev/held-out split and freeze points.

The v0.1 packet (`tierB_dev_sheet.csv`, `packets/dev/`) is kept **unchanged** for the record. It must not be graded: v2 replaces it.

## 1. Why v2 (gaps found in the 2026-10-26 audit)

| v0.1 problem | v2 fix |
|---|---|
| Ungradable candidates were scored as grade 0 | Uncertain (`U`) and ungradable (`X`) are separate labels, **never** converted to 0 in the primary analysis (§5) |
| No "uncertain" option | `U` added |
| B1 and B2 in one sheet with the same candidate codes | A **separate packet per category**, each with its own query codes, candidate codes and shuffle |
| Real query visit IDs (e.g. `0210MOD_V7`) shown to annotators | Opaque query codes (`B1-Q07`); visit IDs only in the key |
| Packet images byte-identical to repository files (reversible by hashing) | Images re-encoded (lossless WebP) under random names |
| Rankings recomputed at each run | Rankings **frozen and hashed before any grade exists** (`phase2/freeze_rankings.py`) |
| CSV and contact sheets only | Self-contained HTML grading page, plus an equivalent CSV |

## 2. Who may grade what

| Packet | Question | Annotator requirement |
|---|---|---|
| **B1** `overall_appearance` | "Which earlier visit's OCT looks most like today's?" | Anyone trained on this rubric (record `annotator_role`). This is **visual similarity**, not a clinical judgement. |
| **B2** `fluid_similarity` | "At which earlier visit was the retinal fluid/thickening most similar to today?" | **A trained OCT grader**: ophthalmologist, optometrist or certified reading-centre grader. Non-expert B2 grades may be collected only as a **pilot of the workflow**. They are flagged and must not be reported as clinical relevance. |

- Grade B1 and B2 in separate sessions. If the same person grades both, grade B1 first and do not look back at B1 while grading B2.
- **Annotators never see:** system names, scores, ranks, CST/BCVA, weeks, visit labels or the key.

## 3. Relevance rubric (per candidate, relative to the query image)

| Label | B1: overall appearance | B2: fluid / thickening (trained grader) |
|---|---|---|
| **2: highly similar** | Same overall retinal profile: foveal contour, thickness profile, and the presence, size and position of dark (hypo-reflective) spaces all closely match | Same fluid status (IRF/SRF present or absent), same location (centre vs off-centre) and similar extent; similar diffuse thickening |
| **1: partially similar** | Same main pattern, but the extent clearly differs (e.g. smaller cysts, flatter contour) | Same fluid type and location, but clearly different extent; or the same thickening without matching fluid |
| **0: not similar** | Different overall profile (e.g. cystic vs dry fovea) | Different fluid status or a clearly different location |
| **U: uncertain** | The annotator cannot decide between grades | The grader cannot decide between grades (e.g. a borderline small cyst) |
| **X: ungradable** | The candidate image cannot be judged (artefact, signal loss, off-centre or cut-off scan) | As for B1, or the image quality is insufficient to assess fluid |

**Rules for every annotator:**
- **Ignore:** overall brightness, speckle noise, tilt or vertical offset, the "200 µm" scale bar and image borders.
- **Grade 2 can be given to several candidates, or to none.** "No candidate deserves 2" is a valid outcome.
- **Query-level:** if the query image itself cannot be judged, mark the query **not gradable**. The whole query is then excluded and counted, and its candidates need no grades.
- **Free text:** `rationale` is a short phrase, required for every 2, 1, U and X.
- **Calibration:** before the real packet, grade 2 practice queries from the dev packet with the protocol author (dev data only). Do not change the rubric after real grading has started; a change requires a new version (v3) and re-grading.

## 4. Packet mechanics (`phase2/make_annotation_packet_v2.py`)

- **Separate packets.** One packet per (set, category) at `phase2/annotations/v2/<set>/<B1|B2>/`, containing:
  - `index.html`: open it in a browser; it saves progress in the browser and has a **Download CSV** button;
  - `sheet.csv`: the same rows, for spreadsheet grading;
  - `images/`: query and candidate images under random names.
- **Seeded shuffle.** Query order and candidate order are shuffled with recorded seeds. Codes are drawn per category, so B1 codes say nothing about B2 codes.
- **Key.** `phase2/annotations/keys/v2/<set>_<category>_key.csv` maps codes → query visit, candidate visit and image. `keys/` is git-ignored. **Annotators must not open it.**
- **Manifest.** `phase2/annotations/v2/<set>/MANIFEST.json` records the seeds, code version, git commit, and SHA-256 of every sheet, key and image.
- **No overwriting.** The generator refuses to overwrite an existing packet.

## 5. How judgments are scored (`phase2/evaluate_annotations.py`)

**Join and pool:**
- **Join:** completed sheet → key (query code, candidate code) → candidate visit → frozen rankings (`phase2/freeze_rankings.py`).
- **Integrity checks:** every key row must match a ranking candidate of the same query, and every ranking must be a permutation of exactly that query's candidate pool. Otherwise the run stops.
- **Primary relevance:** grade 2. **Secondary:** grade ≥ 1.
- **nDCG@5 gains:** 2^g − 1, i.e. 0, 1 and 3.
- **`U` and `X` candidates (primary analysis):** removed from the candidate pool for **every** system alike; each ranking keeps its relative order. Counts are reported per query.
- **Sensitivity analysis:** `U`/`X` scored as 0 (pessimistic), reported separately. They are never merged into the primary result.

**Excluded queries** (each is excluded and counted):

| Case | Status |
|---|---|
| Any blank cell | **NOT EVALUATED** (missing labels); never scored 0 |
| Query marked not gradable | excluded |
| No candidate meets the relevance threshold | `no_relevant_candidate` |
| Fewer than 2 gradable candidates remain | excluded |
| Query image byte-identical to a candidate image | excluded |

**Systems and rankings:**
- **Metadata system abstaining:** the frozen Phase 1 system returns `unsupported_image` for every Tier B question. It is reported with coverage 0, not as a ranking.
- **Metadata baseline:** the pre-specified CST proxy (S2p).
- **Ties:** in system scores, ties were already broken by recency when the rankings were frozen. Ties in relevance are allowed, and all relevant candidates count.

**Reporting:**
- Hit@1, Hit@5, Recall@5, MRR and nDCG@5, per category (B1 and B2 are **never pooled**), per eye and over all evaluated queries.
- Paired differences vs `latest_prior` and vs the CST proxy, with eyes better / equal / worse.
- Exact random-ranking baselines.
- An eye-level bootstrap CI **only with ≥ 5 eyes**. Dev has 3, so no CI is reported.
- **No significance tests in the pilot.**

**Inter-annotator agreement** (when a second sheet is supplied): quadratic-weighted Cohen's κ on items both annotators graded 0/1/2, percent agreement on grade 2, and counts of `U`/`X` disagreements.

## 6. Dev vs held-out

- **Dev** (eyes 25, 11, 61; train split) is for calibrating the rubric, checking the workflow, and **choosing the S5 hybrid encoder**: the higher mean B1 nDCG@5 on dev; ResNet-50 on a tie. Record the choice in `ANNOTATION_LOG.md`.
- **Held-out** (val eyes 93, 74, 40, 14) is untouched until:
  1. the dev choice is recorded;
  2. the code and protocol hashes are logged;
  3. the 76 held-out images have been fetched (needs approval);
  4. held-out rankings are frozen;
  5. the held-out packets are graded blind and their hashes logged.

  It is then evaluated **once**. No tuning ever uses held-out labels.

## 7. Commands

Dev:

```bash
python phase2/freeze_rankings.py --set dev --tier B
```

```bash
python phase2/make_annotation_packet_v2.py --set dev
```

Grade `phase2/annotations/v2/dev/B1/index.html` and `.../B2/index.html` in a browser, then save each downloaded CSV as `phase2/annotations/v2/dev/<B1|B2>/completed_<annotatorID>.csv`. Then evaluate:

```bash
python phase2/evaluate_annotations.py --set dev --category B1 --sheet phase2/annotations/v2/dev/B1/completed_A1.csv
```

Add `--second-sheet <file>` for agreement. Outputs are written to a new versioned folder under `phase2/results/question_eval_v2/dev/`, and nothing is ever overwritten.
