# Phase 3 consolidated results

E3: `phase2/results/e3/fullhist_dev_v1` (eyes [11, 25, 61]). Pathology: `phase2/results/pathology/atlas_subset_pilot_v1`. Every number is read from those saved outputs. Exploratory: the unit of independence is the eye/patient, and no CI is reported with fewer than 5.

## 1. Dataset and annotation coverage

| Source | Coverage | Notes |
|---|---|---|
| Visits / eyes / patients (all splits) | 1594 / 96 / 87 | OLIVES clinical table |
| CST and BCVA | 1593 / 1593 visits | every visit (structured) |
| Finding labels (16 biomarkers, per B-scan) | 192 visits = first + last visit of 96 eyes | OLIVES; none for intermediate visits |
| Clinical edema-severity grade | none | CST is a thickness measurement, not a grade |
| Labelled B-scans in atlas_subset (E1, local) | 48 scans, 8 eyes | exact path join |
| Labelled B-scans in atlas_fullhist (E2, local) | 6 scans, 3 eyes | exact path join |
| Labelled B-scans in scan_subset_v1 (planned) | 550 scans, 55 eyes | available once fetched + embedded |
| Visual-similarity judgments (B1) | 18 queries, 264 judgments, 3 dev eyes | one researcher_non_expert annotator |
| Fluid-similarity judgments (B2) | none | packet exists; not graded |

## 2. Question categories (primary metric: MRR, or nDCG@5 for B1)

| Category | Ground truth | Queries | Eyes | Patients | Metric | latest_prior | metadata-only | ResNet image-only | safe hybrid | question-aware | random |
|---|---|---|---|---|---|---|---|---|---|---|---|
| temporal_previous | structural (visit order) | 55 | 3 | 3 | MRR | 1.000 | 1.000 | 0.479 | 1.000 | 1.000 | 0.244 |
| temporal_baseline | structural (visit order) | 55 | 3 | 3 | MRR | 0.080 | 1.000 | 0.146 | 1.000 | 1.000 | 0.244 |
| cst_highest | metadata (frozen v1 key: max CST) | 55 | 3 | 3 | MRR | 0.174 | 1.000 | 0.199 | 1.000 | 1.000 | 0.244 |
| cst_lowest | metadata (frozen v1 key: min CST) | 55 | 3 | 3 | MRR | 0.417 | 1.000 | 0.392 | 1.000 | 1.000 | 0.244 |
| cst_most_similar | metadata (frozen v1 key) | 55 | 3 | 3 | MRR | 0.411 | 1.000 | 0.338 | 1.000 | 1.000 | 0.260 |
| cst_largest_rise | metadata (frozen v2 key) | 54 | 3 | 3 | MRR | 0.297 | 1.000 | 0.261 | 1.000 | 1.000 | 0.241 |
| cst_last_above_today | metadata (frozen v2 key) | 52 | 3 | 3 | MRR | 0.752 | 1.000 | 0.432 | 1.000 | 1.000 | 0.246 |
| edema_worst_proxy | metadata (frozen v1 'worst_edema' key = max CST; Phase 1 operationalisation, NOT a clinical edema grade) | 55 | 3 | 3 | MRR | 0.174 | 1.000 | 0.199 | 1.000 | 1.000 | 0.244 |
| image_similarity_B1 | human (B1, single researcher_non_expert annotator; visual similarity only) | 18 | 3 | 3 | NDCG5 | 0.786 | 0.000 | 0.802 | 0.823 | 0.823 | 0.572 |

Full Recall@1 / Recall@5 / Hit@1 / MRR / nDCG@5 per system and category: `summary.md` in the E3 folder.

## 3. Encoder comparison (E3 image-only, same queries)

| Category | image-only dinov2_vits14 | image-only resnet50 | image-only retfound_oct |
|---|---|---|---|
| temporal_previous | 0.382 | 0.479 | N/A |
| temporal_baseline | 0.145 | 0.146 | N/A |
| cst_highest | 0.174 | 0.199 | N/A |
| cst_lowest | 0.387 | 0.392 | N/A |
| cst_most_similar | 0.265 | 0.338 | N/A |
| cst_largest_rise | 0.266 | 0.261 | N/A |
| cst_last_above_today | 0.319 | 0.432 | N/A |
| edema_worst_proxy | 0.174 | 0.199 | N/A |
| image_similarity_B1 | 0.682 | 0.802 | N/A |

## 4. Pathology probe (per-B-scan OLIVES labels, folds grouped by patient)

| Task | Positives | Patients | Method | AUROC | AUROC 95% CI (patient) | AUPRC | Prevalence (AUPRC chance) |
|---|---|---|---|---|---|---|---|
| irf | 19 | 8 | position_only | 0.495 | [0.3697, 0.6727] | 0.386 | 0.396 |
| irf | 19 | 8 | metadata_cst | 0.623 | [0.3801, 0.8215] | 0.621 | 0.396 |
| irf | 19 | 8 | knn_dinov2_vits14 | 0.750 | [0.5703, 0.8911] | 0.652 | 0.396 |
| irf | 19 | 8 | probe_dinov2_vits14 | 0.813 | [0.6543, 0.9391] | 0.679 | 0.396 |
| irf | 19 | 8 | combined_dinov2_vits14 | 0.802 | [0.6835, 0.9231] | 0.667 | 0.396 |
| irf | 19 | 8 | knn_resnet50 | 0.656 | [0.525, 0.7739] | 0.538 | 0.396 |
| irf | 19 | 8 | probe_resnet50 | 0.779 | [0.598, 0.9286] | 0.741 | 0.396 |
| irf | 19 | 8 | combined_resnet50 | 0.750 | [0.5597, 0.9083] | 0.738 | 0.396 |
| srf | 1 | 8 | N/A (insufficient labels) |  |  |  |  |
| drt_me | 12 | 8 | position_only | 0.458 | [0.3688, 0.6528] | 0.233 | 0.250 |
| drt_me | 12 | 8 | metadata_cst | 0.743 | [0.4092, 0.9105] | 0.509 | 0.250 |
| drt_me | 12 | 8 | knn_dinov2_vits14 | 0.900 | [0.7725, 0.9875] | 0.794 | 0.250 |
| drt_me | 12 | 8 | probe_dinov2_vits14 | 0.794 | [0.6189, 0.9891] | 0.591 | 0.250 |
| drt_me | 12 | 8 | combined_dinov2_vits14 | 0.771 | [0.5859, 0.9716] | 0.542 | 0.250 |
| drt_me | 12 | 8 | knn_resnet50 | 0.811 | [0.6852, 0.9464] | 0.596 | 0.250 |
| drt_me | 12 | 8 | probe_resnet50 | 0.900 | [0.8182, 0.9773] | 0.741 | 0.250 |
| drt_me | 12 | 8 | combined_resnet50 | 0.903 | [0.8107, 0.993] | 0.735 | 0.250 |
| all |  |  | knn/probe_retfound_oct | N/A |  | N/A |  |

Subset `phase2/data/atlas_subset`; splits {'train': 6, 'val': 2}.

## 5. Ablation (primary metric per category)

| Category | A: Metadata-only | B: ResNet-only | C: OCT encoder-only | D: Hybrid ResNet (question-aware) | E: Hybrid OCT (question-aware) |
|---|---|---|---|---|---|
| temporal_previous | 1.000 | 0.479 | N/A | 1.000 | N/A |
| temporal_baseline | 1.000 | 0.146 | N/A | 1.000 | N/A |
| cst_highest | 1.000 | 0.199 | N/A | 1.000 | N/A |
| cst_lowest | 1.000 | 0.392 | N/A | 1.000 | N/A |
| cst_most_similar | 1.000 | 0.338 | N/A | 1.000 | N/A |
| cst_largest_rise | 1.000 | 0.261 | N/A | 1.000 | N/A |
| cst_last_above_today | 1.000 | 0.432 | N/A | 1.000 | N/A |
| edema_worst_proxy | 1.000 | 0.199 | N/A | 1.000 | N/A |
| image_similarity_B1 | 0.000 | 0.802 | N/A | 0.823 | N/A |

N/A = the system was not run (no embeddings for that encoder), not a zero.

