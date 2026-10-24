# Phase 2 — Image processing prototype and preliminary integration

Written 2026-10-24. Builds on `docs/PHASE_2_HANDOFF.md`. All new work is under `phase2/`, `notebooks/` and this file;
**no Phase 1 file was modified**. After the work, `verify-freeze` reported OK, the RESULTS and HARNESS manifests were all OK,
and the frozen suite gave `205 passed`.

> **Disclaimers:**
> - A research prototype, not clinically validated, with no clinician review.
> - The images are third-party WebP derivatives (see §1).
> - §4 reports agreement with **no ground truth**. It does not show that images improve retrieval.

## 1. Image source (Task 1): done

**Source:** the Retina OCT atlas, <https://retinaoct.com>. It is a third-party index of OLIVES (CC BY 4.0; GitHub `NicoleMulla/retinaoct`), with a JSON API found in the page script:
- `/api/search` (filters `q`, `visit`, `eye`, …);
- `/api/image/<id>`, which returns the archive-relative `path`, e.g. `/TREX DME/TREX/0209TOD/V1/OD/TREXB_000019.tif`;
- `/img/<full_key>`, which serves the image.

**What was verified (from actual responses):**
- **Format:** WebP derivatives, **504 × 496**. They decode as RGB with channels differing by ≤ 1 grey level, so they are effectively grayscale.
- **Overlay:** each scan carries a burned-in 200 µm scale bar at the bottom left.
- **Not the originals:** the atlas states its images are downsampled from the source TIFF/PNG.
- **Identifiers:** trial, arm, subject, visit, eye, scan_index, patient_id, eye_id, BCVA, CST, plus the original archive path.
- **Extra images:** the atlas holds images **not in our labels**. For example, `01-013` W0 **OS** has atlas eye_id 100 and null BCVA/CST, while our metadata has only OD for that subject. Verification excludes such images by construction.

**Subset fetched** (`phase2/fetch_atlas_subset.py`, standard library only, capped at 100 images):
- **96 images, 0 rejected, 4.78 MB.**
- **Eyes:** 8 eyes, all **train/val**:
  - TREX: 25, 30, 11 (GILA arm), 24 (val);
  - Prime: 61, 67, 89, 72 (val).
- **Visits:** 4 per eye, evenly spaced from first to last visit (Prime W0/W28/W52/W104).
- **Scans:** indices 12, 24 (central) and 36 of each 49-scan volume.
- **Exclusions:** test-split eyes, eye 79 and the irregular-count eyes (8, 29, 32, 45, 46, 81) are refused in code.
- **Verification:** an image is kept only if the atlas `path` exactly equals a `file_path` of the expected visit in `visit_images.csv`, **and** eye_id, patient_id, laterality, visit label, subject, BCVA and CST all equal `episodes.csv`.
- **Outputs:** `manifest.csv`, `rejected.csv`, `provenance.json` and `SHA256SUMS` in `phase2/data/atlas_subset/`.
- **Not verified:** a byte-level match with the official Zenodo archive. Pixel provenance rests on the atlas.

**Full-history subset** (added the same day, `--all-visits --scans 24`, output in `phase2/data/atlas_fullhist/`):
- **73 images, 0 rejected, 3.4 MB.** Every visit of 3 train eyes, central B-scan only:
  - eye 25 (TREX, 27 visits);
  - eye 11 (TREX GILA, 25 visits);
  - eye 61 (Prime, 21 visits).
- Visit counts equal `episodes.csv`. The existing 100-image cap was not raised.
- **Data issue found:**
  - The atlas serves **byte-identical images for `01-013` W8 and W12**. This holds for scans 0, 12, 24, 36 and 48, each checked via SHA-256, although the atlas ids, paths and our BCVA (86 vs 89) differ.
  - Whether this comes from the source archive or from atlas ingest is **unknown** until it is checked against Zenodo.
  - It is reported automatically (`identical_image_content_across_visits`) and covered by a test. It is not "fixed".
  - It affects 1 of 67 full-history queries.

The `sequence_3d_*.npy` route was not used, and no array index was interpreted.

## 2. Image processing (Task 2): done, and executed locally

`notebooks/01_image_processing.ipynb` (23 cells) covers:
- loading with SHA-256 checks;
- a metadata table;
- a raw grid (8 eyes × 4 visits);
- preprocessing: bicubic resize to 224² (aspect distortion 1.6 %), fixed /255 scaling, 3-channel replication and ImageNet normalisation, with before/after images and histograms;
- embeddings: ResNet-50 (torchvision, default) or DINOv2 ViT-S/14 (torch.hub), on GPU if available, otherwise CPU;
- determinism and NaN checks;
- a cosine matrix and a top-5 view;
- saved `embeddings.npz` and `embedding_index.csv`;
- the integration (§4).

The logic lives in `phase2/oct_image.py`; tests are in `phase2/tests/` (10 pass).

**Execution status.**
- **Locally:** executed end to end with `nbclient` on **CPU**, in a separate venv (`C:\octvenv`; torch 2.14.1+cpu, torchvision 0.29.1). Executed copies: `phase2/results/executed_local_{resnet50,dinov2_vits14}.ipynb`. Figures and outputs: `phase2/results/<encoder>/`.
- **Colab:** **not run by me.** Colab and the GPU path are untested.

**Results** (96 scans; "random" = expected value under random ranking):

| | ResNet-50 (2048-d) | DINOv2 ViT-S/14 (384-d) | random |
|---|---|---|---|
| top-5 neighbours from the same eye | 0.623 | 0.640 | 0.116 |
| top-1 is another scan of the same visit | 0.042 | 0.104 | 0.021 |
| top-5 neighbours from the same trial | 0.823 | 0.825 | 0.495 |
| embedding time, 96 scans, CPU (last saved run, `run_info.json`; varies between runs) | 14.6 s | 19.1 s | |
| re-embedding max abs diff | 0 | 0 | |

**Reading:**
- Generic ImageNet features clearly carry eye-specific structure (about 5× chance).
- Same-visit top-1 is low because scans 12/24/36 come from different retinal locations.
- Cross-eye neighbours are common. For example, the query `0210MOD_V27` (TREX) returns Prime scans with a similar dry foveal profile.
- These are sanity checks, not clinical relevance measures.

## 3. Integration design (Task 3)

`phase2/image_vs_metadata.py` imports the **frozen** `oct_retrieval.intent.parse_intent` and `intent_retrieval.retrieve` read-only.

**Setup:**
- For each eye, it builds a `VisitMemory` from the **4 subset visits only**. This is not the full history, so it is not the Phase 1 setting.
- Each later visit (t = 1..3) is the "current" visit, and its earlier subset visits are the candidates.
- **Image ranking:** cosine similarity between visit embeddings (the mean of the 3 L2-normalised scan embeddings).
- **Metadata rankings:** obtained from the frozen parser for:
  - "…central subfield thickness most similar to today?" → `closest_to_current/cst`;
  - "…visual acuity closest to today?" → `closest_to_current/bcva`;
  - "Compare with the previous visit." → `anchor_previous`, which equals the approved `latest_prior` baseline;
  - "…changed since baseline?" → `anchor_baseline`.
- **Hybrid router (prototype):** "Which earlier scan looks most like today's OCT?" is routed by the frozen system to **`unsupported_image` (24/24 queries)**. The router answers those queries with the image ranking and leaves every other question to the metadata retriever.

**Mapping status:** the patient/eye/visit associations are verified for all 96 + 73 images. The full-history comparison is done for 3 train eyes (§4b); the other eyes are pending.

## 4. Preliminary integration results (agreement only, n is small)

Top-1 agreement is computed over the 16 queries with ≥ 2 candidates (chance 0.417). Spearman is over the 8 queries with 3 candidates.

| image ranking vs | ResNet-50 top-1 agree | DINOv2 top-1 agree | ResNet-50 ρ | DINOv2 ρ |
|---|---|---|---|---|
| closest CST | 0.500 | 0.438 | 0.19 | 0.38 |
| closest BCVA | 0.500 | 0.625 | 0.44 | 0.69 |
| previous visit (`latest_prior`) | 0.625 | 0.562 | 0.63 | 0.25 |
| baseline | 0.125 | 0.188 | −0.63 | −0.25 |

Within-eye visit pairs (n = 48): Spearman between cosine similarity and closeness in CST, BCVA and time.

| | ResNet-50 | DINOv2 |
|---|---|---|
| CST closeness | 0.67 (TREX 0.74, Prime 0.49) | 0.34 (TREX 0.33, Prime 0.26) |
| BCVA closeness | 0.44 | 0.50 |
| time closeness | 0.34 | 0.25 |

**Reading:**
- Image similarity is **positively associated** with clinical and temporal closeness. It tends to prefer recent visits over baseline.
- It is **not** a copy of any one metadata rule.
- The two encoders disagree on which variable they track best, so with 8 eyes these numbers are **unstable and exploratory**. They do **not** show that images help.

## 4b. Full-history comparison (3 train eyes, every visit)

**Setting:**
- Each eye's `VisitMemory` here is **identical to the Phase 1 memory**; the script asserts this.
- Every visit t ≥ 1 is a query, giving 70 queries. Top-1 agreement uses the 67 queries with ≥ 2 candidates (chance 0.123); Spearman uses the 64 queries with ≥ 3.

| image ranking vs | ResNet-50 top-1 agree | DINOv2 top-1 agree | ResNet-50 norm. rank* | DINOv2 norm. rank* |
|---|---|---|---|---|
| closest CST | 0.179 | 0.164 | 0.38 | 0.44 |
| closest BCVA | 0.269 | 0.134 | 0.34 | 0.46 |
| previous visit (`latest_prior`) | 0.299 | 0.224 | 0.30 | 0.32 |
| baseline | 0.104 | 0.060 | 0.72 | 0.57 |

\* The position of the metadata top-1 within the image ranking: 0 means it is the image top-1, 1 means it is ranked last, and random ranking gives 0.5.

Within-eye visit pairs (per eye, n = 300 / 351 / 210): rank partial correlation of cosine similarity with CST and BCVA closeness, **controlling for the time gap**.

| eye | ResNet-50 CST | ResNet-50 BCVA | DINOv2 CST | DINOv2 BCVA | raw time closeness (ResNet-50 / DINOv2) |
|---|---|---|---|---|---|
| 11 (TREX) | 0.45 | 0.07 | 0.14 | 0.09 | 0.32 / 0.36 |
| 25 (TREX) | 0.32 | 0.30 | 0.11 | 0.24 | 0.72 / 0.56 |
| 61 (Prime) | 0.32 | 0.15 | 0.38 | 0.23 | 0.67 / 0.56 |

**Reading:**
- **Recency.** Over full histories, image similarity is dominated by recency. The image top-1 is the previous visit in 22–30 % of queries, against 12 % by chance. The baseline visit is ranked low.
- **CST and BCVA.** Similarity is positively associated with CST closeness and BCVA closeness **beyond** time, in all 3 eyes with both encoders. It is weaker with DINOv2.
- **Metadata top-1.** The image ranking rarely picks the same top-1 as any metadata question (0.06–0.30), so images and metadata carry overlapping but different signals.
- **Caveats.** These are 3 eyes, the pairs are not independent and no significance test was run. This is exploratory; it does **not** show that images improve retrieval.
- **Figures:** `phase2/results/fullhist/<encoder>/fullhist_eyes.png`. Per-query and per-pair CSVs are alongside.

## 5. Exact Colab instructions (for the researcher)

1. **Locally**, rebuild the bundle (it is already built: `phase2/dist/phase2_colab_bundle.zip`, 5.4 MB):
   ```bash
   python phase2/make_colab_bundle.py
   ```
2. In Colab, open the notebook: File → Upload notebook → `notebooks/01_image_processing.ipynb`.
3. Optional: Runtime → Change runtime type → **T4 GPU**. The notebook falls back to CPU automatically; CPU takes about 20 s for 96 images.
4. Run **cell 1**. When prompted, upload `phase2_colab_bundle.zip`, or place it at `MyDrive/oct/phase2_colab_bundle.zip` beforehand. It is extracted to `/content/oct`.
5. Runtime → **Run all**. Expected printouts:
   - `checksum OK: 96/96`;
   - `embeddings (96, 2048)`;
   - the neighbour statistics;
   - the §9 JSON summary;
   - the §9b full-history summary and figure.
6. For DINOv2, set `ENCODER = "dinov2_vits14"` in cell 2 and Run all again. It downloads from GitHub and `dl.fbaipublicfiles.com`; no account is needed.
7. Download `/content/phase2_results_<encoder>.zip` from the Files pane.

**What should match:** on GPU, numbers should match the table in §2 up to small floating-point differences. If cosine values differ beyond about 1e-3, report it; do not assume.

## 6. What remains incomplete

- Full histories for the **other** eyes; only 3 train eyes are done. All 49 scans per visit, rather than the central scan only, would be about 1,000–1,300 images per eye and would need a raised cap.
- Resolve the `01-013` W8 = W12 duplication against the official archive.
- A byte-level spot check of atlas images against the official Zenodo archive.
- A domain-specific encoder (RETFound: access and licence unverified). Masking of the scale-bar overlay.
- A **pre-registered** evaluation with relevance labels for image questions, built on a new freeze. fresh_v1 is exposed.
- Colab/GPU execution: not yet run.
- Unchanged from before: eye 79 laterality, and KI-01 to KI-03.

## 7. Files added

| Path | What |
|---|---|
| `phase2/fetch_atlas_subset.py` | bounded, verified atlas fetcher (`--dry-run`, `--max-images`) |
| `phase2/oct_image.py` | loading, preprocessing, encoders, cosine/top-k, neighbour statistics |
| `phase2/image_vs_metadata.py` | read-only bridge to the frozen retriever; agreement statistics; hybrid route |
| `phase2/make_colab_bundle.py` | builds `phase2/dist/phase2_colab_bundle.zip` (8.8 MB) |
| `phase2/embed_subset.py` | embeds any verified subset with one encoder (no plots) |
| `phase2/plot_fullhist.py` | per-eye figure: visit × visit cosine matrix and CST/BCVA trajectories |
| `phase2/data/atlas_fullhist/` | 73 full-history central B-scans, manifest, provenance, SHA256SUMS |
| `phase2/results/fullhist/<encoder>/` | full-history embeddings, comparison CSVs/JSON, figure |
| `phase2/tests/test_phase2.py` | 10 tests, run with `python -m pytest -q phase2/tests` |
| `phase2/data/atlas_subset/` | 96 WebP images, manifest, rejected list, provenance, SHA256SUMS |
| `phase2/results/<encoder>/` | embeddings, index, figures, summaries, run_info |
| `notebooks/01_image_processing.ipynb` | Colab notebook (clean, unexecuted) |

**Citation for any use of these images:** Prabhushankar et al., "OLIVES Dataset: Ophthalmic Labels for Investigating Visual Eye Semantics", NeurIPS Datasets and Benchmarks 2022, doi:10.5281/zenodo.7105232. Please cite the dataset, not the atlas.
