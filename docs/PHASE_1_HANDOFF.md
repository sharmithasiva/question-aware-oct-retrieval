# Phase 1 Handoff — Question-Aware Episodic Retrieval for Longitudinal Retinal OCT

Handoff date: 2026-10-23. Status: **Phase 1 (metadata-only retrieval) is complete as a prototype baseline. It is NOT
clinically validated.** Phase 2 (OCT image retrieval) has not started.

> **Benchmark disclaimer.** The fresh_v1 evaluation is an **AI-generated preliminary benchmark with limited human
> review**.
> - Questions and labels were produced by language-model agents.
> - The researcher reviewed only 2 disagreements, 12 sampled labels and 8 Part B fixtures.
> - The annotator's independence from the generator **could not be verified**: the agent transcripts were unavailable, and both agents made unexpected extra tool calls. The original inter-agent agreement results are marked **potentially compromised**.
> - 14 of 120 questions were viewed by the developer during the guided review, after the code freeze.
> - **No clinician validated any question, label or result.** Nothing here establishes research novelty or clinical usefulness.

---

## 1. Research objective

Given a natural-language clinical question asked at a patient's current visit, retrieve the relevant **earlier
visits of the same eye**. This is instead of using the latest visit, a fixed window or the full history. Later phases
test whether question-aware retrieval improves retrieval quality, answer accuracy and cost when a vision-language
model (VLM) uses the retrieved OCT images. Dataset: **OLIVES** (TREX-DME and Prime trials).

## 2. What Phase 1 implemented

| Component | Implementation | Key files |
|---|---|---|
| Data audit and loading | Trial-specific path parsing. Labels and the study spreadsheets were cross-checked. | `src/oct_retrieval/load.py`, `docs/DATA_AUDIT.md`, `src/oct_retrieval/audit.py` |
| Episodic memory | One row per (eye, visit): CST, BCVA, order, week (Prime), image count, QC flags, provenance; every image path kept | `src/oct_retrieval/episodes.py` → `data/processed/episodes.csv`, `visit_images.csv` |
| Held-out study variables | Leakage index, DRSS and injection (Prime), **never** given to the retriever | `src/oct_retrieval/side_tables.py` → `data/processed/heldout_prime.csv` |
| v1 questions and retrievers | 8 types; keyword parser; latest / earliest / random / temporal / clinical / combined / rule-router | `src/oct_retrieval/questions.py`, `retrievers.py`, `evaluate.py` |
| v2 harder questions | 8 non-circular types; answer key independent of the retrievers | `src/oct_retrieval/ground_truth_v2.py`, `evaluate_v2.py`, `docs/PHASE1_V2_QUESTION_SPEC.md` |
| v3 intent parser and retrieval (**frozen system under test**) | Structured `QueryIntent`; 6 explicit outcomes (retrieved / no qualifying visit / insufficient data / unavailable variable / unsupported image / clarification); general operations | `src/oct_retrieval/intent.py`, `intent_retrieval.py`, `evaluate_v3_dev.py` |
| fresh_v1 evaluation | Pre-registered, sealed, one-shot; Parts A (parser routing / intent), B (retrieval correctness), C (visit retrieval vs baselines) | `docs/FRESH_EVALUATION_PROTOCOL.md`, `scripts/fresh_v1*.py`, `scripts/part_b_check.py` |

## 3. Dataset and metadata findings relevant to retrieval

- **Size:** 78,185 image rows correspond to **1,594 visits**, 96 eyes and 87 patients. CST and BCVA are constant within every visit, so the **visit** is the retrieval unit.
- **Path layouts:**
  - TREX DME: `/TREX DME/<arm>/<subject>/V<n>/<OD|OS>/<image>`;
  - Prime: `/Prime_FULL/<subject>/W<week>/<OD|OS>/<image>`.
  - Extensions: tif 75,735; jpg 2,107; png 343.
- **BCVA** is the ETDRS letter score; one line = 5 letters. Verified 660/660 (Prime) and 933/933 (TREX) against the study sheets.
- **Weeks** exist only for Prime: the `W` label is the study week (660/660 match). TREX visit numbers are ordinal, and dates are missing for 926 of 934 visits.
- **Biomarker labels** exist only for the first and last visit of each eye.
- **QC flags:**
  - missing CST/BCVA at `0247GOD_V3`;
  - 5 visits with a non-standard image count;
  - **laterality conflict for eye 79 (`02-016`)**: OS in the clinical table and demographics, OD in the biomarker CSV. This matters for Phase 2 image access.
- **Split:** patient-level, stratified by trial, seed 42 → train 52, val 17, test 18 patients. No patient or eye crosses splits.
- **Prime eyes** have no centre-involved DME, so CST there is thickness, not an edema measure.

## 4. Frozen evaluation setup

- **Code freeze:** `results/fresh_v1_prereg/CODE_FREEZE.sha256` (42 files: `src/oct_retrieval/*.py`, `tests/*.py`, dev paraphrases and manifest, `pyproject.toml`, protocol, analysis plan, `results/v3_dev/*`). Recorded **before** any fresh question existed.
- **Pre-registered analysis plan:** `results/fresh_v1_prereg/ANALYSIS_PLAN.md`. Approved clarification 01 (a): gold intent parameters come from the frozen type definitions (`PROPOSED_CLARIFICATION_01.md`).
- **Approved global fixed-rule baseline:** **`latest_prior`**, selected on dev data (mean dev Hit@1 0.274 over the 7 retrievable types; `global_fixed_rule.json`). The per-type rule (`results/v3_dev/selected_baselines.json`) is reported only as an **upper reference**, because it is given the true question type.
- **Restricted input:** the retriever sees visit order, CST, BCVA and week (Prime). `max_leakage` must route to "data unavailable". Image questions are judged on correct limitation handling, never on Hit@1.
- **Gold set:** `sealed/fresh_v1/gold.jsonl`: 120 items, 0 excluded, `GOLD_MANIFEST.sha256`.
- **Harness:** `scripts/fresh_v1_evaluate.py`, hashed in `HARNESS.sha256`. The one-shot run's outputs are hashed in `RESULTS_MANIFEST.sha256`.
- **Environment:** Python 3.12.10, pandas 2.2.2, numpy 2.2.4, openpyxl 3.1.5, pytest 9.1.0, Windows 11 (`environment.json`).

## 5. Main fresh_v1 results (one-shot, `reports/FRESH_V1_REPORT.md`)

**Part A** (unit = distinct question, n = 120; Wilson 95 % CI):

| Endpoint | Result |
|---|---|
| **Primary 1: routing accuracy** | 98/120 = **0.817** [0.738, 0.876] |
| **Primary 2: unsafe-retrieval rate** (lower is better) | 4/50 = **0.080** [0.032, 0.188] |
| Over-refusal rate | 14/70 = 0.200 [0.123, 0.308] |
| Correct limitation handling (exact / any refusal) | 20/25 = 0.80 / 23/25 = 0.92 |
| Exact intent on retrieve questions | 47/70 = 0.671 [0.555, 0.770] (dev phrasings: 100 %) |

**Part B** (retrieval correctness vs independent fixtures and reference, `results/fresh_v1_partB/PART_B_REPORT.md`):
- the answer key agrees with the reference everywhere (48/48 fixtures; 16,000 random cases; 9,712 real cases);
- the retriever agrees everywhere except KI-01;
- 6/8 planted bugs were detected;
- the researcher spot-checked 8/8 fixtures as OK.

**Part C** (test-split patients; paired Hit@1 difference, intent_v3 − `latest_prior`):
| Type | Difference [95 % CI] |
|---|---|
| largest_cst_rise | +0.349 [+0.235, +0.441] |
| largest_bcva_drop | +0.441 [+0.347, +0.521] |
| first_vision_gain_1line | +0.318 [+0.270, +0.360] |
| one_year_ago | +0.707 [+0.474, +0.849] (7 patients) |
| six_months_ago | +0.900 [+0.880, +0.919] (7 patients) |
| last_thicker_than_now | +0.156 [+0.105, +0.207] |
| last_vision_as_good | +0.023 [−0.078, +0.106] (no clear difference) |

**How to interpret these results:**
- **Part C CIs are too narrow.** They reflect patient variability only. Each type uses just 10 question texts, and intent_v3's Hit@1 mostly reflects how many of them parse correctly.
- **Small samples:** Prime-only types rest on 7 test patients, and per-type Part A figures (10 questions) are descriptive only.
- **The per-type upper reference** (given the true type) beats intent_v3 on `first_vision_gain_1line` (0.533 vs 0.421).
- **Partly circular, by design.** Part C's correctness *given a correct parse* follows from the operations implementing the frozen definitions. The informative quantity is parser generalisation (Part A).
- **Post-hoc error analysis** (`results/fresh_v1/POST_HOC_ERROR_ANALYSIS.md`, not pre-registered):
  - most failures are vocabulary gaps;
  - about 9 come from phrases deliberately removed before the freeze to avoid locked-set contamination;
  - 4 unsafe retrievals fetched an anchor visit for an unanswerable question (HbA1c, age, weeks elapsed, slope).
- **The fresh_v1 set is now exposed.** Any parser change needs a new freeze and a new sealed set (fresh_v2).

## 6. Known issues (`results/fresh_v1_prereg/KNOWN_ISSUES.md`)

- **KI-01:** the retriever does not enforce the "≥ 2 earlier visits" rule. It answers with one earlier visit where the spec says insufficient data. This cannot affect fresh_v1, because instances always have ≥ 2 earlier visits. Fix after Phase 1, under a new freeze.
- **KI-02:** a Part B harness limitation. The retriever returns a ranking, not a set, so an extra spurious tied candidate below the correct block is invisible. It cannot change Hit@1.
- **KI-03:** compound number words are not parsed ("twenty-six weeks" → 6, "fifty-two weeks" → 2). Found after the run; not fixed.
- **Other recorded limitations:**
  - the retired v2 locked set is developer-exposed;
  - the dev paraphrases were used for parser development, so all v3 dev scores are in-sample;
  - the v1/v3-dev 1.000 scores are pipeline checks / rule coverage, not evidence.

## 7. Paths

| What | Path |
|---|---|
| Reports | `reports/RETRIEVAL_REPORT.md` (v1), `reports/RETRIEVAL_REPORT_V2.md`, `reports/RETRIEVAL_REPORT_V3_DEV.md`, **`reports/FRESH_V1_REPORT.md`** |
| Specs and protocol | `docs/DATA_AUDIT.md`, `docs/PHASE1_V2_QUESTION_SPEC.md`, `docs/RETRIEVAL_IMPROVEMENT_PLAN.md`, `docs/DEV_RESULTS_AUDIT.md`, `docs/FRESH_EVALUATION_PROTOCOL.md` |
| Pre-registration and logs | `results/fresh_v1_prereg/` (`ANALYSIS_PLAN.md`, `STAGE_LOG.md`, `KNOWN_ISSUES.md`, `CODE_FREEZE.sha256`, `HARNESS.sha256`, `GOLD_MANIFEST.sha256`, `SEALED_ARTIFACTS.sha256`, `RESULTS_MANIFEST.sha256`, `global_fixed_rule.json`, `environment.json`) |
| Results | `results/v1_snapshot/`, `results/v2/`, `results/v3_dev/`, `results/fresh_v1_partB/`, **`results/fresh_v1/`**, `results/fresh_v1_harness_devtest/` (plumbing test, not results) |
| Sealed benchmark | `sealed/fresh_v1/` (now exposed; do not reuse as a test set) |
| Code | `src/oct_retrieval/` (frozen), `scripts/` (fresh_v1 utilities, Part B harness) |
| Tests | `tests/` (205 tests), `tests/fixtures/retrieval_cases_fresh_v1.jsonl`, `tests/reference/reference_ops.py` |

## 8. Reproduction commands (do not re-run the one-shot evaluation)

```bash
python -m pytest -q
```

```bash
python scripts/fresh_v1.py verify-freeze
```

```bash
sha256sum -c results/fresh_v1_prereg/HARNESS.sha256
```

```bash
sha256sum -c results/fresh_v1_prereg/RESULTS_MANIFEST.sha256
```

```bash
tr -d '\r' < results/fresh_v1_prereg/GOLD_MANIFEST.sha256 | sha256sum -c -
```

```bash
python scripts/part_b_check.py
```

```bash
python scripts/fresh_v1_evaluate.py --dev-test
```

- **Earlier-phase reruns:** `PYTHONPATH=src python -m oct_retrieval.evaluate` (v1), `oct_retrieval.evaluate_v2` and `oct_retrieval.evaluate_v3_dev`.
  - These **overwrite** their own output folders (`results/*.csv`, `results/v2/`, `results/v3_dev/`).
  - `results/v3_dev/*` is in the code freeze, so re-running v3_dev would break freeze verification.
  - Do not re-run these unless you intend to create new versions.
- **The sealed one-shot command must not be run again.** It refuses anyway, because `results/fresh_v1/` exists.

## 9. Phase 2 plan (recorded, NOT implemented)

1. **Inspect** the existing OLIVES image paths in `data/processed/visit_images.csv` (78,185 paths; tif/jpg/png). Decide how to access a **small subset** of images without downloading the full ~33.9 GB archive.
2. **Load a small image subset** (e.g. a few eyes from train/val with long histories, both trials). Do not download the full archive.
3. **Validate associations before any embedding:**
   - each image's patient, eye, laterality and visit match `episodes.csv`;
   - scan counts per visit match;
   - resolve the eye 79 OD/OS path conflict;
   - flag the 5 non-standard image-count visits.
4. **Build a visual retrieval baseline** (pretrained OCT encoder, e.g. RETFound or similar, on Colab GPU; cached embeddings).
5. **Compare** metadata-only, image-only and combined retrieval under the same patient-level splits. Use a new, pre-registered evaluation; fresh_v1 is exposed.
6. **Before Phase 2 changes any retrieval code**, create a new versioned freeze. Consider fixing KI-01 and KI-03 then, with regression tests.

**Constraints carried forward:**
- the laptop has 16 GB RAM, no GPU and 20–50 GB free;
- no new dependencies or image code without an explicit decision;
- image questions must stay separated from metadata answers in evaluation.

## 10. Next session: start here

- [ ] Read this file, then `results/fresh_v1_prereg/STAGE_LOG.md` and `KNOWN_ISSUES.md`.
- [ ] Verify nothing changed: `python scripts/fresh_v1.py verify-freeze` and `sha256sum -c results/fresh_v1_prereg/RESULTS_MANIFEST.sha256`. Both should report OK.
- [ ] Run `python -m pytest -q`. Expected: 205 passed.
- [ ] **First Phase 2 task (inspection only, no downloads, no new code files):** summarise the image paths that will be needed for a small subset.

  ```bash
  python -c "import pandas as pd; d=pd.read_csv('data/processed/visit_images.csv'); e=pd.read_csv('data/processed/episodes.csv'); m=d.merge(e[['visit_id','eye_id','patient_id','trial','laterality','split','visit_index','n_images']],on='visit_id'); print(m.groupby(['trial','split']).agg(eyes=('eye_id','nunique'),visits=('visit_id','nunique'),images=('file_path','size'))); print(m[m.eye_id==79].file_path.str.split('/').str[4].value_counts())"
  ```

- [ ] Decide with the researcher how images will be accessed (Colab, a partial archive or a remote source) **before** downloading anything.
- [ ] Only then start the association-validation step (§9.3).

## 11. Phase 1 status statement

Phase 1 is **complete as a prototype metadata-only baseline**:
- a validated visit-level memory;
- a transparent deterministic parser and retriever with explicit abstention;
- a pre-registered, sealed, one-shot evaluation on an AI-generated preliminary benchmark.

It is **not clinically validated**. Its results are preliminary and must be reported with the disclaimers above.
