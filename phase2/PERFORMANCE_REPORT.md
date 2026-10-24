# Phase 2 performance report: image embeddings and image-vs-metadata ranking

Written 2026-10-24. Every number below is recomputed from **saved outputs** by `phase2/consolidate_metrics.py`; no encoder is run and nothing is fetched. The machine-readable versions are:
- `phase2/results/performance/metrics.csv`: one row per metric, with formula, numerator, denominator, unit, baseline and source;
- `per_eye.csv`;
- `tie_audit.csv`;
- `consistency.json`: recomputed vs previously saved values. All match within the 3-dp rounding of the saved summaries.

> **Status:**
> - A research prototype. **Not clinically validated**; no clinician reviewed any image, label or result.
> - **No relevance ground truth exists for image questions**, so no retrieval-accuracy metric is reported.
> - No significance tests were run, and no claims of statistical significance are made.

---

## 1. Executive summary

- **What was measured.** Two small experiments, each with two generic pretrained encoders: ResNet-50 (ImageNet) and DINOv2 ViT-S/14. Both used real OLIVES B-scans whose patient, eye and visit identity was verified against our metadata.
  - **E1:** 96 images; 8 eyes × 4 visits × 3 scans.
  - **E2:** 73 images; the full visit history of 3 eyes, central scan only.
- **Identity.** Embeddings carry eye identity well above chance. In E1, 62–64 % of top-5 neighbours come from the query's eye, against 11.6 % by chance. This is **identity matching**, not clinical retrieval.
- **Recency.** Over full histories (E2), image similarity is dominated by recency:
  - the image top-1 is the previous visit in 30 % (ResNet-50) and 22 % (DINOv2) of queries, against 12 % by chance;
  - the Spearman correlation between cosine similarity and temporal closeness is 0.48 and 0.37.
- **CST.** After controlling for the visit gap, a weaker positive association with CST closeness remains: pooled partial ρ 0.28 (ResNet-50) and 0.11 (DINOv2), positive in all 3 E2 eyes. In E1 the pooled value is positive but **negative in 2–3 of 8 eyes**.
- **Agreement with metadata.** The image ranking rarely picks the same prior visit as the frozen metadata retriever. E2 top-1 agreement is 0.06–0.30, against chance 0.12. Images and metadata carry overlapping but distinct signals; this is **agreement, not accuracy**.
- **Routing.**
  - The frozen Phase 1 parser recognises image questions in its own benchmark: 6/7 exact routing, 7/7 refused, precision 6/7. These are Phase 1 numbers on an AI-generated benchmark.
  - In Phase 2, the hybrid route fired for 100 % of queries, but that is **one fixed sentence**, so it is coverage, not accuracy.
- **Not evaluated:** Hit@1, Hit@5/Recall@5, MRR, nDCG@5 and QA retrieval accuracy. **Relevance ground truth is unavailable** (§5).

### Review-slide summary (5 bullets)

1. We now process **real, identity-verified OLIVES OCT B-scans**: 96 + 73 images, and every label is checked against metadata.
2. Generic image embeddings **recognise the same eye** (top-5 same-eye 0.62–0.64 vs 0.12 chance) and **track time** (previous visit ranked first in 22–30 % of full-history queries vs 12 % chance).
3. Beyond time, image similarity shows a **weak, inconsistent link to CST** (partial ρ 0.11–0.28 pooled). It is not evidence of clinical relevance.
4. The frozen Phase 1 system **correctly refuses image questions**, and the hybrid route answers them with images. **Whether those answers are right is not yet measurable.**
5. **Next:** a small, pre-registered, annotated image-question set (relevance labels) on held-out eyes, to measure Hit@1, MRR and nDCG@5 against `latest_prior`.

---

## 2. Dataset and experimental setup

| | E1: 96-image experiment | E2: 73-image full-history experiment |
|---|---|---|
| Images | 96 WebP B-scans, 504×496 (`phase2/data/atlas_subset/`) | 73 WebP B-scans, 504×496 (`phase2/data/atlas_fullhist/`) |
| Eyes (split) | 8: TREX 25, 30, 11 (train), 24 (val); Prime 61, 67, 89 (train), 72 (val) | 3: TREX 25, 11; Prime 61 (all train) |
| Visits | 4 per eye (32), evenly spaced first → last | **every** visit: 27 + 25 + 21 = 73 |
| Scans per visit | 3 (indices 12, 24, 36) | 1 (index 24, central) |
| Visit memory for metadata ranking | the 4 subset visits only (**not** the Phase 1 history) | identical to the Phase 1 memory (asserted in code) |
| Queries for ranking metrics | t = 1..3 per eye → 24 (16 with ≥ 2 candidates) | t = 1..n−1 → 70 (67 with ≥ 2 candidates; 64 with ≥ 3) |
| Within-eye visit pairs | 48 (6 per eye) | 861 (351 + 300 + 210) |

**Common to both experiments:**
- **Source and verification:** Retina OCT atlas, a third-party index of OLIVES (CC BY 4.0). Each image is kept only if its archive path equals the expected `visit_images.csv` path **and** eye, patient, laterality, visit, subject, BCVA and CST match `episodes.csv`. Rejections: 0 of 96 and 0 of 73.
- **Preprocessing:** luminance → bicubic resize to 224×224 → /255 → 3-channel replication → ImageNet mean/std (`phase2/oct_image.py::preprocess`).
- **Encoders:**
  - ResNet-50, torchvision `IMAGENET1K_V2`, global-average-pooled 2048-d;
  - DINOv2 ViT-S/14 via `torch.hub`, 384-d CLS.
- **Similarity:** cosine on L2-normalised vectors. Visit embedding = normalised mean of its scans' normalised embeddings.
- **Image ranking of prior visits:** sort candidates by cosine to the current visit, descending. Ties keep the earlier visit (stable sort).
- **Metadata rankings:** the **frozen** Phase 1 `intent.parse_intent` + `intent_retrieval.retrieve`, imported read-only, for four questions:
  - closest CST → `closest_to_current/cst`;
  - closest BCVA → `closest_to_current/bcva`;
  - previous visit → `anchor_previous`, the same ranking as the approved `latest_prior` baseline;
  - baseline → `anchor_baseline`.

  Phase 1 tie-breaks prefer the more recent visit.
- **Image question:** "Which earlier scan looks most like today's OCT?". The frozen parser returns `unsupported_image`, and the prototype hybrid router answers with the image ranking.

---

## 3. E1 results: 96 images (8 eyes × 4 visits × 3 scans)

Unit, N and baseline are given per row. The baseline is the **expected value under uniformly random ranking**, computed analytically (formulas in §5.1). n/d = numerator/denominator.

### 3.1 Image-level neighbour metrics (unit: image query, N = 96, k = 5, the query itself excluded)

| Metric | Measures | ResNet-50 (n/d) | DINOv2 (n/d) | Random baseline |
|---|---|---|---|---|
| Top-5 same-eye rate | identity (eye) | **0.623** (299/480) | **0.640** (307/480) | 0.116 |
| Top-1 same-visit rate | identity (visit; other scan of the same visit) | 0.042 (4/96) | 0.104 (10/96) | 0.021 |
| Top-5 same-trial rate | acquisition/trial similarity | 0.823 (395/480) | 0.825 (396/480) | 0.495 |

Per-eye top-5 same-eye (8 eyes): ResNet-50 0.37–0.90, DINOv2 0.37–1.00 (`per_eye.csv`).

### 3.2 Visit ranking vs frozen metadata (unit: query visit with ≥ 2 subset candidates, N = 16)

| Metric | Measures | ResNet-50 | DINOv2 | Random baseline |
|---|---|---|---|---|
| Image top-1 = previous **subset** visit | recency | 0.625 (10/16) | 0.563 (9/16) | 0.417 |
| Top-1 agreement with closest-CST | agreement | 0.500 (8/16) | 0.438 (7/16) | 0.417 |
| Top-1 agreement with closest-BCVA | agreement | 0.500 (8/16) | 0.625 (10/16) | 0.417 |
| Top-1 agreement with previous visit (`latest_prior`) | agreement (**same quantity as the recency row**) | 0.625 (10/16) | 0.563 (9/16) | 0.417 |
| Top-1 agreement with baseline visit | agreement | 0.125 (2/16) | 0.188 (3/16) | 0.417 |
| Mean query-level Spearman, image vs closest-CST ranking (N = 8, 3 candidates) | agreement | 0.19 | 0.38 | 0 |
| Same, vs closest-BCVA | agreement | 0.44 | 0.69 | 0 |

"Previous" in E1 means the previous visit **within the 4-visit subset**, about 8 real visits earlier, not the true previous visit.

### 3.3 Within-eye visit pairs (unit: visit pair, N = 48 pooled, 6 per eye; pairs share visits, so they are not independent)

| Metric | Measures | ResNet-50 | DINOv2 | Baseline |
|---|---|---|---|---|
| Spearman(cos, −gap) | temporal similarity | 0.343 | 0.252 | 0 |
| Partial Spearman(cos, −\|ΔCST\| \| gap) | association with CST beyond time | 0.639 | 0.296 | 0 |
| Partial Spearman(cos, −\|ΔBCVA\| \| gap) | association with BCVA beyond time | 0.392 | 0.470 | 0 |

**Per-eye heterogeneity.** The partial CST correlation is **negative** in eyes 61, 72 and 89 for ResNet-50, and in 61 and 89 for DINOv2; per-eye values range from −0.93 to +0.93 on 6 pairs each. The gap is measured in subset positions (0–3), not weeks.

---

## 4. E2 results: 73 images (3 eyes, full history, central scan)

### 4.1 Image-level neighbour metrics (unit: image query, N = 73, k = 5)

| Metric | Measures | ResNet-50 | DINOv2 | Random baseline |
|---|---|---|---|---|
| Top-5 same-eye rate | identity (eye) | 0.981 (358/365) | 0.973 (355/365) | 0.328 |
| Top-1 same-visit rate | — | **not applicable** (one scan per visit) | n/a | — |
| Top-5 same-trial rate | trial similarity | 1.000 (365/365) | 0.989 (361/365) | 0.585 |

With only 3 eyes, each with 21–27 visits of very similar scans, these rates are close to trivial. They are reported for completeness and **must not be compared with E1**.

### 4.2 Visit ranking vs frozen metadata (unit: query visit with ≥ 2 candidates, N = 67; full Phase 1 memory)

| Metric | Measures | ResNet-50 | DINOv2 | Random baseline |
|---|---|---|---|---|
| Image top-1 = true previous visit | recency | **0.299** (20/67) | **0.224** (15/67) | 0.123 |
| Top-1 agreement with closest-CST | agreement | 0.179 (12/67) | 0.164 (11/67) | 0.123 |
| Top-1 agreement with closest-BCVA | agreement | 0.269 (18/67) | 0.134 (9/67) | 0.123 |
| Top-1 agreement with previous visit (`latest_prior`) | agreement (same quantity as the recency row) | 0.299 (20/67) | 0.224 (15/67) | 0.123 |
| Top-1 agreement with baseline visit | agreement | 0.104 (7/67) | 0.060 (4/67) | 0.123 |
| Mean normalised rank of metadata top-1 in image ranking (0 = image top-1, 1 = last) | agreement | CST 0.38, BCVA 0.34, previous 0.30, baseline 0.72 | CST 0.44, BCVA 0.46, previous 0.32, baseline 0.57 | 0.5 |
| Mean query-level Spearman, image vs metadata ranking (N = 64) | agreement | CST 0.39, BCVA 0.29, previous 0.48, baseline −0.48 | CST 0.24, BCVA 0.13, previous 0.41, baseline −0.41 | 0 |

### 4.3 Within-eye visit pairs (unit: visit pair, N = 861 pooled; per eye 300 / 351 / 210)

| Metric | Measures | ResNet-50 pooled (eyes 11 / 25 / 61) | DINOv2 pooled (eyes 11 / 25 / 61) | Baseline |
|---|---|---|---|---|
| Spearman(cos, −gap) | temporal similarity | **0.479** (0.32 / 0.72 / 0.67) | **0.368** (0.36 / 0.56 / 0.56) | 0 |
| Partial Spearman(cos, −\|ΔCST\| \| gap) | association with CST beyond time | **0.279** (0.45 / 0.32 / 0.32) | **0.114** (0.14 / 0.11 / 0.38) | 0 |
| Partial Spearman(cos, −\|ΔBCVA\| \| gap) | association with BCVA beyond time | 0.139 | 0.067 | 0 |

The gap is the difference in visit position (`visit_index`), not weeks. Prime weeks are irregular after W52.

### 4.4 Sensitivity: the duplicated visit removed (01-013 W12 dropped as a query and from all pairs)

| Metric | ResNet-50 | DINOv2 |
|---|---|---|
| Image top-1 = previous visit | 0.288 (19/66), baseline 0.120 | 0.212 (14/66), baseline 0.120 |
| Spearman(cos, −gap), 841 pairs | 0.471 | 0.356 |
| Partial CST \| gap, 841 pairs | 0.291 | 0.129 |

The duplicate does not drive the E2 conclusions.

---

## 5. Baselines and standard retrieval metrics

### 5.1 How each baseline was computed (all analytic; no simulation)

| Metric family | Baseline | Formula |
|---|---|---|
| Top-5 same-eye/trial rate | random ranking of the other N−1 images | mean over queries of (#other images in the query's group)/(N−1). E1 eye: 11/95 = 0.116; E2 eye: (25·24 + 27·26 + 21·20)/(73·72) = 0.328 |
| Top-1 same-visit rate | random ranking | mean over queries of (scans in the visit − 1)/(N−1) = 2/95 = 0.021 |
| Previous-visit rate; top-1 agreement | random ranking of candidates | mean over queries of 1/(#candidates). E1 (t = 2, 3): (1/2 + 1/3)/2 = 0.417; E2: 0.123 |
| Normalised rank | random ranking | 0.5 |
| Spearman / partial Spearman | no association | 0 |

**Comparability:**
- Random baselines are comparable **within** a row. They are **not** comparable across E1 and E2, which differ in candidate counts, group sizes and the meaning of "previous".
- The `latest_prior` rule is not a competing *score* here: it appears as one of the metadata rankings the image ranking is compared with.
- The image ranking was never scored against `latest_prior` on a relevance target, because no such target exists.

### 5.2 Standard retrieval metrics

| Metric | E1 | E2 | Reason |
|---|---|---|---|
| Hit@1 | **Not evaluated — relevance ground truth unavailable** | same | no relevance labels for the image question |
| Hit@5 / Recall@5 | **Not evaluated — relevance ground truth unavailable** | same | E1 has ≤ 3 candidates per query; E2 has no labels |
| MRR | **Not evaluated — relevance ground truth unavailable** | same | " |
| nDCG@5 | **Not evaluated — relevance ground truth unavailable** | same | no graded relevance |
| QA retrieval accuracy | **Not evaluated — relevance ground truth unavailable** | same | no VLM answers and no answer key (Phase 3) |

**What was checked before concluding this:**
- `sealed/fresh_v1/gold.jsonl` holds gold for **metadata** questions on **test** patients. Image questions there carry gold *routing* only, never relevant visits, and the benchmark is exposed.
- The `ground_truth_v2` answer keys define metadata answers.
- OLIVES expert biomarker labels exist only for the first and last visit of each eye, which is too few to rank histories.
- The atlas's per-scan biomarkers for other visits are **model-inferred** (`label_source = model`), not ground truth.

**Why the metadata answers are not used as Hit@1.** If one treated the metadata-question answers as "gold" for the image ranking, Hit@1 would simply equal the top-1 agreement already reported. It is deliberately **not** labelled Hit@1: the image ranking was not asked those questions, and the previous visit is not assumed relevant.

### 5.3 Routing (kept separate from retrieval)

| Metric | Value (n/d) | Unit / source | Measures |
|---|---|---|---|
| Image-question routing recall, exact (`requires_images` → `unsupported_image`) | 0.857 (6/7) | fresh_v1 items; `results/fresh_v1/part_a_items.csv` (frozen) | routing accuracy |
| Image-question refusal (not retrieved) | 1.000 (7/7) | same | routing safety |
| Precision of the `unsupported_image` route | 0.857 (6/7); 1 false route out of 113 non-image items (F112, gold `needs_clarification`) | same | routing accuracy |
| Hybrid image-route coverage, E1 | 1.000 (24/24) | `phase2/results/<enc>/image_vs_metadata_per_query.csv` | coverage of **one fixed sentence**; not accuracy |
| Hybrid image-route coverage, E2 | 1.000 (70/70) | `phase2/results/fullhist/<enc>/…` | same |

The fresh_v1 routing numbers come from an AI-generated benchmark with limited human review and only **7** image items (see `reports/FRESH_V1_REPORT.md`). They describe the **parser**, not image retrieval.

### 5.4 What each metric is: measured, proxy, or needs annotation

| Category | Metrics |
|---|---|
| **Genuinely measured** (exact counts on verified data, as defined) | top-5 same-eye/same-trial rates; top-1 same-visit rate (E1); previous-visit top-1 rate; top-1 agreement rates; normalised rank; query-level and pairwise Spearman; partial Spearman; hybrid route coverage; fresh_v1 image-routing recall, refusal and precision (on its AI-generated labels) |
| **Proxies** (measured, but standing in for something they do not establish) | same-eye/same-visit rates → "embeddings carry patient-specific structure" (identity, not relevance); previous-visit rate and temporal Spearman → recency sensitivity; partial CST/BCVA correlations → weak link to clinical variables (CST is one central scalar); agreement rates → consistency with metadata rules (no notion of correctness); route coverage → the route fires |
| **Need a new annotated evaluation set** | Hit@1, Hit@5/Recall@5, MRR and nDCG@5 for image questions; routing accuracy over **varied** image-question wordings (more than 7 items); VLM answer accuracy with retrieved images (Phase 3) |

---

## 6. Reproducibility

**Environment of the saved runs:**
- Windows 11, CPU only, Python 3.12.10;
- torch 2.14.1+cpu, torchvision 0.29.1+cpu, in a separate venv `C:\octvenv`;
- the notebook executed with nbclient; deterministic: re-embedding max |diff| = 0.

**Model weights:**
- ResNet-50: `resnet50-11ad3fa6.pth` (torchvision `IMAGENET1K_V2`).
- DINOv2: `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`. **The hub repository code is not pinned to a commit**, which is a reproducibility risk; the weights URL is fixed.

**Commands** (repository root; the venv's Python is needed only for the steps that embed):

```bash
python scripts/fresh_v1.py verify-freeze
```

```bash
python phase2/consolidate_metrics.py
```

The second command regenerates every table in this report from saved outputs. It needs only the base Python with pandas and numpy; no torch.

To regenerate the saved outputs themselves:

```bash
C:/octvenv/Scripts/python.exe phase2/embed_subset.py --subset phase2/data/atlas_fullhist --encoder resnet50 --out phase2/results/fullhist/resnet50
```

```bash
C:/octvenv/Scripts/python.exe phase2/image_vs_metadata.py --emb phase2/results/fullhist/resnet50/embeddings.npz --subset phase2/data/atlas_fullhist
```

**Notes:**
- Repeat with `dinov2_vits14`. For E1, use `--subset phase2/data/atlas_subset` and the `phase2/results/<encoder>/` paths.
- E1's neighbour statistics and `run_info.json` were produced by executing `notebooks/01_image_processing.ipynb`; executed copies are in `phase2/results/executed_local_*.ipynb`.
- The image files are pinned by `SHA256SUMS`. Re-fetching from the atlas (`phase2/fetch_atlas_subset.py`) may return different bytes if the atlas changes.
- **File provenance in E2:**
  - `embeddings.npz` was last written by the notebook (§9b);
  - `run_info.json` and `embedding_index.csv` were written earlier by `embed_subset.py`;
  - the embedding order matches the index (`consistency.json`), and all ranking metrics were identical across both runs.

---

## 7. Limitations and data-quality issues

- **Sample size:**
  - E1: 8 eyes and 16 informative ranking queries.
  - E2: **3 eyes**, all train-split, and one trial pair (2 TREX, 1 Prime).
  - Eye-level conclusions rest on 3–8 units. No confidence intervals are given because the units are dependent (next bullet); proportion CIs would be misleadingly narrow.
- **Dependence:** queries and pairs from the same eye share visits and embeddings. The 861 E2 pairs are not 861 independent observations, and pooling pairs across eyes mixes eye-specific scales.
- **Byte-identical images: eye 61 (`01-013`), W8 and W12.** The atlas serves identical images for the whole volume (scans 0/12/24/36/48 checked), while our metadata differs (BCVA 86 vs 89). The cause, archive or atlas, is unknown. It is flagged in every summary (`identical_image_content_across_visits`) and its effect is small (§4.4).
- **Metadata ties:**
  - E2: 32 of 67 closest-BCVA queries have more than one equally close prior visit, against 6 of 67 for CST. E1: 2 of 16 for BCVA and 0 for CST.
  - Phase 1 breaks ties by recency, so "agreement" partly reflects that rule.
  - Tie-aware agreement (image top-1 is any of the tied-best visits) is higher. E2 BCVA: ResNet-50 23/67 vs 18/67 strict; DINOv2 13/67 vs 9/67. Tie-aware chance is also higher, so these are not compared with 0.123 (`tie_audit.csv`).
- **E1 vs E2 are not comparable:** the history definition differs (subset vs full), "previous" means different things, scan counts per visit differ, and the baselines differ. They are never merged into one score.
- **Image provenance:** downsampled lossy WebP from a third party, with a burned-in scale bar. Not checked byte-for-byte against the Zenodo archive.
- **Encoders:** generic ImageNet/self-supervised models, not OCT-specific. The two encoders disagree on secondary associations (BCVA vs CST), which signals instability.
- **Mismatches found between docs and saved outputs:**
  1. `docs/PHASE_2_IMAGE_PROTOTYPE.md` quoted embedding times from an earlier run (15.0 / 19.4 s). It has been corrected to the saved `run_info.json` values (14.6 / 19.1 s).
  2. `docs/PHASE_2_HANDOFF.md` line 12 ("Nothing in Phase 2 so far has processed an image") is outdated. It is a dated historical handoff and was left unchanged; `PHASE_2_IMAGE_PROTOTYPE.md` and this report supersede it.
  3. All other quoted metrics match the saved outputs.
- **Clinical meaning: not supported by these results.**
  - Same-eye matching shows identity, not relevance.
  - Recency agreement could come from slow acquisition drift as much as from pathology.
  - The CST association is weak, inconsistent across eyes in E1, and based on one central scalar.
  - None of the metrics asks whether a retrieved visit helps answer a clinical question.

---

## 8. Recommended next experiment

**A pre-registered, annotated image-question evaluation on held-out eyes.**

1. **Data.** Fetch full histories (central scan, or 3 scans) for **val-split** eyes from both trials (e.g. 6–8 eyes). This uses the same verified fetcher, with an approved cap (about 160–220 images). The 3 train eyes stay development data.
2. **Questions.** Write a frozen set of image-dependent questions before any system output is seen, e.g.:
   - "When did the retina last look like this?";
   - "At which earlier visit was the fluid most similar?".

   Include varied wordings, so that routing accuracy can be measured on more than 7 items.
3. **Labels.** Per query visit, an annotator (ideally a clinician; otherwise documented as a non-expert) grades each candidate prior visit **0/1/2** for the stated visual criterion, blinded to system rankings. Record annotator identity and agreement on a double-graded subset.
4. **Metrics** (pre-registered):
   - Hit@1, MRR and nDCG@5 for: image ranking (ResNet-50, DINOv2), `latest_prior`, the metadata rules, and the hybrid router;
   - eye-level clustered bootstrap CIs;
   - routing accuracy on the new image questions, reported separately.
5. **Freeze** the Phase 2 code and the analysis plan first, as for fresh_v1. Resolve the W8/W12 duplicate against the official archive before the freeze.
