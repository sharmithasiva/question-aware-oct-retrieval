# Question-Aware Episodic Retrieval for Longitudinal Retinal OCT Reasoning with Vision-Language Models

When a clinical question is asked at a patient's **current** OCT visit, the system should retrieve the relevant
**earlier visits of the same eye**, instead of always using the latest visit, a fixed window or the full history.
Later phases will test whether question-aware retrieval improves answers from a vision-language model (VLM) that
sees the retrieved OCT images.

The dataset is **OLIVES** (TREX-DME and Prime trials): 1,594 visits, 96 eyes, 87 patients.

> **Status:**
> - A research prototype. **Not clinically validated**: no clinician has reviewed any question, label or result.
> - The Phase 1 benchmark is AI-generated, with limited human review.
> - The Phase 2 image results are exploratory. No experiment so far shows that images improve retrieval for image-dependent questions.

| Phase | Goal | Status |
|---|---|---|
| 1 | Metadata-only retrieval: visit memory, question parser, retriever, baselines, sealed evaluation | **Complete and frozen** |
| 2 | Real OCT images: verified image subset, embeddings, image vs metadata retrieval, question-conditioned evaluation | **In progress**: pipeline, metrics and evaluation harness done; human relevance labels pending |
| 3 | VLM answer generation from retrieved images | Not started |

## Repository layout

```
OLIVES_Dataset_Labels/          raw OLIVES label files (unmodified)
data/processed/                 episodes.csv (1 row per eye-visit), visit_images.csv (1 row per B-scan path)
src/oct_retrieval/              Phase 1 package (FROZEN)
  load.py, episodes.py            read labels, build visit-level memory, patient-level splits
  questions.py, ground_truth_v2.py  question types and metadata answer keys (v1, v2)
  intent.py, intent_retrieval.py  structured intent parser + retrieval with explicit abstention (system under test)
  retrievers.py, metrics.py       baselines (latest_prior, ...), Hit@K / MRR / nDCG
  evaluate*.py, audit.py          v1 / v2 / v3-dev runners, audit numbers
scripts/                        fresh_v1 sealed-evaluation harness (FROZEN)
tests/                          Phase 1 tests (FROZEN; 205 tests)
sealed/fresh_v1/                fresh_v1 benchmark (now exposed; do not reuse as a test set)
results/, reports/              Phase 1 outputs, pre-registration and freeze manifests (results/fresh_v1_prereg/)
docs/                           data audit, specs, Phase 1/2 handoffs, Phase 2 image prototype notes
notebooks/01_image_processing.ipynb   Colab notebook: load, preprocess, embed, top-5 similarity, integration
phase2/                         Phase 2 (all new work; Phase 1 is imported read-only)
  fetch_atlas_subset.py           bounded, metadata-verified image fetcher (Retina OCT atlas)
  oct_image.py                    loading, preprocessing, ResNet-50 / DINOv2 encoders, cosine top-k
  embed_subset.py                 embed a verified subset from the command line
  image_vs_metadata.py            image vs frozen-metadata ranking comparison; prototype hybrid route
  plot_fullhist.py                per-eye similarity / CST-BCVA figure
  consolidate_metrics.py          rebuilds every reported metric from saved outputs
  question_eval.py                question-conditioned evaluation harness (5 systems)
  build_scan_manifest.py          offline plan of a 4,685-scan train-eye subset (scan_subset_v1); counts
  fetch_scan_subset.py            resumable, verified fetch of the planned scans (reuses local images)
  embed_cached.py                 batched, SHA-256-cached embedding; optional pooling variants
  hybrid_retriever.py             metadata + image hybrid: eligibility first, abstention preserved, inspectable
  large_eval.py                   E1/E2-definition, Tier A and post-hoc B1 metrics for any subset
  RETRIEVAL_V2.md                 stage-2 commands, hybrid policy, results and limitations
  make_annotation_packet.py       blinded annotation packets for human relevance grading
  make_colab_bundle.py            builds the zip the Colab notebook uploads
  data/atlas_subset/              96 verified B-scans (8 eyes x 4 visits x 3 scans) + manifest/provenance
  data/atlas_fullhist/            73 verified central B-scans (full history of 3 eyes)
  results/                        embeddings, figures, comparison outputs, performance/, question_eval/
  annotations/                    annotation template, dev sheet (no grades yet), packets, log
  tests/                          Phase 2 tests
  PERFORMANCE_REPORT.md           consolidated Phase 2 metrics with formulas and limitations
  QUESTION_EVAL_PROTOCOL.md       protocol for the question-conditioned evaluation
  QUESTION_EVAL_RESULTS.md        status and results of that evaluation
```

## Phase 1: metadata-only retrieval (frozen)

**What was built:**
- one memory record per eye-visit (CST, BCVA, visit order, Prime study week);
- a structured question parser;
- a retriever with six explicit outcomes: retrieved / no qualifying visit / insufficient data / unavailable variable / unsupported image / clarification.

**Baseline:** the approved fixed-rule baseline is **`latest_prior`**.

**fresh_v1 results** (one-shot, pre-registered; `reports/FRESH_V1_REPORT.md`):
- **Routing accuracy:** 98/120 = 0.817.
- **Unsafe retrieval:** 4/50 = 0.080.
- **Exact intent:** 47/70 = 0.671.
- **Image questions:** routed to `unsupported_image` in 6/7 cases (7/7 refused).

The benchmark is AI-generated, with limited human review and unverified annotator independence.

**Details:** `docs/PHASE_1_HANDOFF.md`. **Known issues:** `results/fresh_v1_prereg/KNOWN_ISSUES.md`.

> **Do not modify** `src/oct_retrieval/`, `tests/`, `scripts/`, `sealed/` or `results/`.
> **Do not re-run** the v1 / v2 / v3-dev evaluation commands: they overwrite frozen outputs.
> Never re-run the sealed one-shot command.

## Phase 2: OCT images (in progress)

**Image access:**
- **Source:** 169 real OLIVES B-scans were fetched from the Retina OCT atlas, a third-party index serving 504×496 WebP derivatives.
- **Verification:** each image is kept only if its archive path, eye, patient, laterality, visit, BCVA and CST all match our metadata. 0 were rejected.
- **Bounds:** test-split eyes and eyes with data issues are excluded; the full 33.9 GB archive was not downloaded.
- **Details:** `docs/PHASE_2_IMAGE_PROTOTYPE.md`.

**Pipeline:**
- **Preprocessing:** grayscale → resize to 224×224 → ImageNet normalisation.
- **Encoders:** two pretrained, non-gated models, ResNet-50 and DINOv2 ViT-S/14, run on GPU with a CPU fallback.
- **Retrieval:** cosine top-k.
- **Notebook:** `notebooks/01_image_processing.ipynb` runs in Colab. It was executed locally on CPU; it has not yet been run in Colab.

**Exploratory metrics** (full tables, formulas and baselines in `phase2/PERFORMANCE_REPORT.md`):

| Experiment | Metric | ResNet-50 | DINOv2 | Random |
|---|---|---|---|---|
| E1: 96 images, 8 eyes | top-5 neighbours from the same eye (identity, not relevance) | 0.623 | 0.640 | 0.116 |
| E2: 73 images, full history of 3 eyes | image top-1 is the previous visit | 0.299 | 0.224 | 0.123 |
| E2 | Spearman: cosine vs temporal closeness | 0.479 | 0.368 | 0 |
| E2 | partial Spearman: cosine vs CST closeness, given the time gap | 0.279 | 0.114 | 0 |

**Question-conditioned evaluation** (`phase2/QUESTION_EVAL_PROTOCOL.md`, `phase2/QUESTION_EVAL_RESULTS.md`): five systems are compared on a shared query set and candidate pool:
- `latest_prior`;
- metadata-only (the frozen Phase 1 system);
- image-only ResNet-50;
- image-only DINOv2;
- an untuned RRF hybrid.

**Results so far:**
- **Tier A** (CST questions; relevance from the frozen Phase 1 answer keys; 3 dev eyes):
  - On "When was the CST most similar to today?", MRR is `latest_prior` 0.412, ResNet-50 0.338, DINOv2 0.265 and random 0.260.
  - Images do **not** stand in for CST measurements, and the hybrid scores below metadata-only in every category.
- **Tier B B1** (visual similarity; 18 dev queries, 3 eyes, one non-expert annotator; frozen pre-label rankings):
  - nDCG@5 is `latest_prior` 0.786, ResNet-50 0.802, ResNet-50 RRF hybrid 0.823, DINOv2 0.682 and random 0.572.
  - The differences are small and there are too few eyes for a CI, so this is not evidence of improvement.
  - Details are in `phase2/RETRIEVAL_V2.md`.
- **Tier B B2** (fluid): **NOT EVALUATED**. No B2 grades exist.

**Next step:** grade the dev Tier B sheet, then fetch, freeze and blindly evaluate the held-out val eyes (protocol §7).

## Run

```bash
pip install pandas numpy openpyxl pytest
```

Phase 1 freeze and tests. Expected output: `freeze OK` and `205 passed`.

```bash
python scripts/fresh_v1.py verify-freeze
```

```bash
python -m pytest -q
```

Phase 2 tests and metrics. These need only pandas, numpy and Pillow, because saved embeddings are reused.

```bash
python -m pytest -q phase2/tests
```

```bash
python phase2/consolidate_metrics.py
```

```bash
python phase2/question_eval.py --set dev --tier A
```

Re-embedding images needs torch and torchvision, which are not project dependencies; Colab has them preinstalled.

```bash
python phase2/embed_subset.py --subset phase2/data/atlas_fullhist --encoder resnet50 --out phase2/results/fullhist/resnet50
```

To use **Colab**:
1. Build the bundle with `python phase2/make_colab_bundle.py`.
2. Upload `notebooks/01_image_processing.ipynb` to Colab.
3. Run all cells, and upload `phase2/dist/phase2_colab_bundle.zip` when prompted.

On PowerShell, set environment variables with `$env:NAME="value"`.

## Key data facts

These are verified; see `docs/DATA_AUDIT.md` and `docs/PHASE_2_HANDOFF.md`.

- A visit is (`eye_id`, visit folder). CST and BCVA are constant within a visit.
- Splits are made at patient level (seed 42): train 52, val 17, test 18 patients.
- Biomarker labels exist only for the first and last visit of each eye.
- Eye 79 has unresolved OD/OS laterality.
- The atlas serves byte-identical images for `01-013` W8 and W12.

## Data attribution

This repository includes OLIVES data and derived material (CC BY 4.0). See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) for the required citation, the image source and what was changed.
