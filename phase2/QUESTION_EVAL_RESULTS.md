# Question-conditioned retrieval evaluation: status and results

Written 2026-10-24. Protocol: `phase2/QUESTION_EVAL_PROTOCOL.md` (v0.1). Harness: `phase2/question_eval.py`.

> **Bottom line:**
> - **No result in this document measures whether images improve retrieval for image-dependent questions.** The human relevance labels that question needs (Tier B) **do not exist yet**, and every Tier B metric is **NOT EVALUATED**.
> - The only completed run is **Tier A on 3 development eyes**: questions whose relevance is defined by CST measurements through the frozen Phase 1 answer keys.

## 1. Protocol and data scope

| | Dev (train) | Held-out pilot (val) | Reserved |
|---|---|---|---|
| Eyes | 25, 11, 61 | 93, 74, 40, 14 (fixed selection rule) | test split |
| Images | 73 verified central scans, embedded (ResNet-50, DINOv2) | **not fetched** (76 central scans needed) | none |
| Tier A queries (all visits with ≥ 6 earlier visits) | 55 per category × 5 categories = 275 | not run (blocked by guard until the freeze) | — |
| Tier B queries (6 query visits per eye × 2 categories) | 36 queries, 18 query visits, 264 candidates | — | — |
| Tier B labels | **none** (blank sheet generated) | none | — |

**Systems:**
- S1 `latest_prior`;
- S2 metadata-only (frozen Phase 1 parser and retriever);
- S2p metadata CST proxy (Tier B only);
- S3/S4 image-only (ResNet-50 / DINOv2, saved embeddings reused; no encoder run);
- S5 hybrid RRF (k = 60, untuned) of metadata and image ranks.

All exact score ties are broken by recency.

## 2. Readiness

| Item | Status |
|---|---|
| Protocol, primary metrics, tie, missing-data and duplicate rules | **done**, written before any evaluation code ran |
| Harness (5 systems, per-query rankings and metrics, exact random baselines, eye-level reporting, NOT EVALUATED handling, held-out guard) | **done** |
| Tests (`phase2/tests/test_question_eval.py`, 14 tests): missing labels, partial labels, no relevant candidate, duplicate images, tie-breaking, candidate leakage, non-candidate rankings, metric maths, random baseline vs enumeration, Tier A labels vs Phase 1 instances, label parsing, held-out guard, blank packet | **pass** |
| Tier A dev run | **done** (§3) |
| Tier B dev annotation packet (blinded; seeded shuffle; key kept separately) | **done**; grades empty |
| Tier B dev grades | **missing**: needs an annotator |
| Clinical reference labels (fluid, progression, events) | **missing**: needs a trained OCT grader; out of the pilot's scope except B2 |
| Held-out images and embeddings | **missing**: needs approval for a 76-image fetch, then embedding |
| Freeze before held-out | **pending** (`phase2/annotations/ANNOTATION_LOG.md`) |

## 3. Completed experiment: Tier A, development eyes (metadata-referenced)

**What this run can and cannot show:**
- **Labels are real but not visual.** Relevance comes from the frozen Phase 1 answer keys (`questions.relevant_indices`, `ground_truth_v2.answer_key`) applied to recorded CST. Ties are all relevant. `label_source = metadata_rule`.
- **S2 is a ceiling, not a competitor.** It ranks on the same measurements, so its 1.000 everywhere is expected. It was also developed on these eyes (in-sample).
- **Scale.** 3 eyes from 2 trials, no CIs (n_eyes < 5), dev only.
- **The question it answers:** can image similarity **stand in for a CST measurement**? It cannot answer whether images help with image questions.

### 3.1 Primary metric: MRR, primary category A1 `similar_cst` (55 queries)

| System | Hit@1 | Hit@5 | Recall@5 | **MRR** | nDCG@5 | MRR per eye (11 / 25 / 61) | Eyes better / equal / worse vs `latest_prior` |
|---|---|---|---|---|---|---|---|
| random (exact expectation) | 0.095 | 0.420 | 0.398 | 0.260 | 0.244 | 0.24 / 0.24 / 0.31 | — |
| S1 `latest_prior` | 0.236 | 0.636 | 0.627 | **0.412** | 0.430 | 0.33 / 0.60 / 0.25 | — |
| S2 metadata (ceiling) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1 / 1 / 1 | 3/0/0 |
| S3 image ResNet-50 | 0.145 | 0.618 | 0.609 | **0.338** | 0.369 | 0.26 / 0.48 / 0.23 | **0/0/3** |
| S4 image DINOv2 | 0.073 | 0.545 | 0.530 | **0.265** | 0.295 | 0.18 / 0.38 / 0.22 | **0/0/3** |
| S5 hybrid RRF + ResNet-50 | 0.473 | 0.927 | 0.918 | 0.643 | 0.698 | 0.60 / 0.71 / 0.60 | 3/0/0 (−0.357 vs S2) |
| S5 hybrid RRF + DINOv2 | 0.400 | 0.855 | 0.855 | 0.579 | 0.631 | 0.41 / 0.72 / 0.59 | 3/0/0 (−0.421 vs S2) |

### 3.2 Secondary categories: MRR (n evaluated)

| Category | random | `latest_prior` | ResNet-50 | DINOv2 | S5 ResNet-50 (vs S2) | S5 DINOv2 (vs S2) |
|---|---|---|---|---|---|---|
| A2 `worst_edema` (55) | 0.244 | 0.174 | 0.199 | 0.174 | 0.385 (−0.615) | 0.547 (−0.453) |
| A3 `best_anatomy` (55) | 0.244 | 0.417 | 0.392 | 0.387 | 0.740 (−0.260) | 0.717 (−0.283) |
| A4 `last_thicker_than_now` (52; 3 with no relevant candidate) | 0.246 | 0.753 | 0.432 | 0.319 | 0.676 (−0.325) | 0.642 (−0.358) |
| A5 `largest_cst_rise` (54; 1 with no relevant candidate) | 0.241 | 0.297 | 0.261 | 0.266 | 0.587 (−0.413) | 0.493 (−0.507) |

S2 metadata = 1.000 in every category; the frozen parser read all five texts as intended (`parsed_intent` per query in `per_query_metrics.csv`).

### 3.3 Reading (dev, 3 eyes; not generalisable)

- **Images do not stand in for CST.**
  - Image-only ranking is **below `latest_prior`** on the primary category in all 3 eyes, for both encoders.
  - It is near random for DINOv2: MRR 0.265 vs random 0.260.
  - On A2–A5 it is at most slightly above random. On A4 it is well below `latest_prior`, because recency is close to the answer there.
- **The hybrid's gains over `latest_prior` come from its metadata half.** Every S5 variant is **below S2** in every category (−0.26 to −0.62 MRR). On these questions, adding image ranks to the metadata ranking **hurts**.
- These numbers say nothing about image-dependent questions. That is the Tier B experiment, and it is unevaluated.

**Sensitivity** (`--drop-visits 01-013_W12`, the byte-identical image): `similar_cst` MRR is ResNet-50 0.335, DINOv2 0.261 and `latest_prior` 0.411 (54 queries). The conclusions are unchanged. No query was excluded for a duplicate image in the main run: W12 is at t = 3, below the 6-candidate minimum.

## 4. Not yet evaluated

| Experiment | Status | Reason |
|---|---|---|
| Tier B dev: B1 `overall_appearance`, B2 `fluid_similarity` (all systems, all metrics) | **NOT EVALUATED** | relevance ground truth unavailable: 36 queries, 0 graded (harness output `phase2/results/question_eval/dev/tierB/summary.csv` shows NOT EVALUATED) |
| Tier A held-out | not run | images not fetched; protocol freeze pending |
| Tier B held-out | not run | images, packet and grades missing |
| Event and progression questions (e.g. last dry visit, onset of worsening) | not designed for this pilot | need per-visit clinical reference labels from a qualified grader; OLIVES expert biomarkers cover first and last visits only, and atlas labels for other visits are model-inferred |
| Weighted hybrid (α) | not run | only if RRF is insufficient; α is chosen on dev Tier B labels only |

## 5. Smallest next action that yields a valid result

**Grade the dev Tier B sheet for B1 `overall_appearance`:**
- 18 query visits and 264 candidate judgements, using the contact sheets in `phase2/annotations/packets/dev/`;
- follow the instructions in `phase2/annotations/README.md`, and leave B2 to a trained grader;
- then record the sheet's SHA-256 in `ANNOTATION_LOG.md` and run:

```bash
python phase2/question_eval.py --set dev --tier B --sheet phase2/annotations/tierB_dev_sheet.csv --key phase2/annotations/keys/tierB_dev_key.csv
```

**What it gives:** the first image-question results (dev; non-expert visual-similarity labels). It also decides the S5 encoder.

**After that:** approve the 76-image held-out fetch, freeze, grade the held-out packet blind, and run once with `--confirm-heldout-freeze`.

## 6. Reproduce

```bash
python phase2/question_eval.py --set dev --tier A
```

```bash
python phase2/question_eval.py --set dev --tier A --drop-visits 01-013_W12
```

```bash
python phase2/make_annotation_packet.py --set dev
```

```bash
python -m pytest -q phase2/tests
```

Outputs are in `phase2/results/question_eval/dev/<run>/`: `query_set.csv`, `labels_used.csv`, `rankings.csv`, `per_query_metrics.csv`, `summary.csv` and `run_meta.json`. Base Python with pandas and numpy is enough; no torch is needed, because saved embeddings are reused.
