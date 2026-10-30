# Retrieval evaluation: `phase2\data\scan_subset_v1` (visit embedding: central)

Exploratory. Image relevance here is either identity/temporal structure (no labels), frozen metadata answer keys (Tier A) or one non-expert's visual-similarity judgments (B1, post-hoc). None is clinical ground truth.

## Inputs

```json
{
  "resnet50": "phase2\\results\\scan_subset_v1\\resnet50\\embeddings.npz"
}
```

## E1 definition: scan-level identity (not relevance)

```json
{
  "resnet50": {
    "n_scans": 2935,
    "n_eyes": 37,
    "dropped_duplicates": 5,
    "top5_same_eye_rate": 0.9027,
    "top5_same_eye_chance": 0.0298,
    "top5_same_eye_ci95_eye_bootstrap": [
      0.8292,
      0.9187
    ],
    "top1_same_visit_rate": 0.0491,
    "top1_same_visit_chance": 0.0013,
    "measures": "identity (same eye / same visit), not relevance"
  }
}
```

## E2 definition: within-eye structure (34 complete-history eyes)

```json
{
  "resnet50": {
    "n_eyes": 34,
    "n_queries": 497,
    "n_pairs": 4849,
    "image_top1_is_previous_rate": 0.2676,
    "image_top1_is_previous_chance": 0.1569,
    "image_top1_is_previous_ci95_eye_bootstrap": [
      0.2208,
      0.311
    ],
    "spearman_cos_vs_temporal_closeness_pooled": 0.289,
    "spearman_cos_vs_temporal_closeness_mean_over_eyes": 0.2827,
    "spearman_temporal_ci95_eye_bootstrap": [
      0.2131,
      0.3487
    ],
    "partial_spearman_cos_vs_cst_given_gap_pooled": 0.4794,
    "partial_spearman_cos_vs_cst_given_gap_mean_over_eyes": 0.2153,
    "partial_cst_ci95_eye_bootstrap": [
      0.1311,
      0.2972
    ],
    "measures": "temporal / CST association within an eye (no relevance labels)"
  }
}
```

## Tier A: CST questions, frozen metadata answer keys (primary metric MRR)

`S2_metadata` is the frozen Phase 1 system. `S5_hybrid_rrf_*` is the v0.1 prototype (fuses every metadata ranking and falls back to images when metadata abstains). `H_fuse_closest_*` is phase2/hybrid_retriever.py: fuses only `closest_to_current`, never overrides abstentions.

| category | system | n_evaluated | coverage | hit1 | hit5 | recall5 | mrr | ndcg5 | n_eyes | mrr_ci95_eye_bootstrap | mrr_minus_latest_prior | eyes_better_equal_worse_vs_latest_prior | mrr_minus_S2_metadata |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| similar_cst | S1_latest_prior | 362 | 1.0 | 0.2928 | 0.7044 | 0.651 | 0.4675 | 0.4701 | 33 | [0.443, 0.546] |  |  |  |
| similar_cst | S2_metadata | 362 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 33 | [1.000, 1.000] | 0.5325 | 32/1/0 |  |
| similar_cst | S3S4_image_resnet50 | 362 | 1.0 | 0.2099 | 0.674 | 0.602 | 0.4118 | 0.4109 | 33 | [0.387, 0.468] | -0.0557 | 10/1/22 |  |
| similar_cst | S5_hybrid_rrf_resnet50 | 362 | 1.0 | 0.5138 | 0.9613 | 0.9289 | 0.687 | 0.7226 | 33 | [0.669, 0.758] | 0.2195 | 31/1/1 | -0.313 |
| similar_cst | S3S4_image_dinov2_vits14 | 362 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 33 | [0.000, 0.000] | -0.4675 | 0/0/33 |  |
| similar_cst | S5_hybrid_rrf_dinov2_vits14 | 362 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 33 | [0.000, 0.000] | -0.4675 | 0/0/33 | -1.0 |
| similar_cst | H_fuse_closest_resnet50 | 362 | 1.0 | 0.5138 | 0.9613 | 0.9289 | 0.687 | 0.7226 | 33 | [0.669, 0.758] | 0.2195 | 31/1/1 |  |
| similar_cst | random_expected | 362 | 1.0 | 0.1288 | 0.5497 | 0.4773 | 0.3203 | 0.3022 | 33 | [0.321, 0.361] |  |  |  |
| worst_edema | S1_latest_prior | 362 | 1.0 | 0.047 | 0.1878 | 0.1865 | 0.1786 | 0.1153 | 33 | [0.141, 0.209] |  |  |  |
| worst_edema | S2_metadata | 362 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 33 | [1.000, 1.000] | 0.8214 | 33/0/0 |  |
| worst_edema | S3S4_image_resnet50 | 362 | 1.0 | 0.0691 | 0.3591 | 0.3591 | 0.2265 | 0.2046 | 33 | [0.194, 0.265] | 0.0479 | 24/4/5 |  |
| worst_edema | S5_hybrid_rrf_resnet50 | 362 | 1.0 | 0.2818 | 0.8895 | 0.8895 | 0.511 | 0.5924 | 33 | [0.445, 0.560] | 0.3324 | 33/0/0 | -0.489 |
| worst_edema | S3S4_image_dinov2_vits14 | 362 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 33 | [0.000, 0.000] | -0.1786 | 0/0/33 |  |
| worst_edema | S5_hybrid_rrf_dinov2_vits14 | 362 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 33 | [0.000, 0.000] | -0.1786 | 0/0/33 | -1.0 |
| worst_edema | H_fuse_closest_resnet50 | 362 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 33 | [1.000, 1.000] | 0.8214 | 33/0/0 |  |
| worst_edema | random_expected | 362 | 1.0 | 0.0956 | 0.4778 | 0.4773 | 0.2783 | 0.2815 | 33 | [0.282, 0.310] |  |  |  |
| best_anatomy | S1_latest_prior | 362 | 1.0 | 0.2707 | 0.7431 | 0.721 | 0.4643 | 0.5039 | 33 | [0.430, 0.582] |  |  |  |
| best_anatomy | S2_metadata | 362 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 33 | [1.000, 1.000] | 0.5357 | 31/2/0 |  |
| best_anatomy | S3S4_image_resnet50 | 362 | 1.0 | 0.2017 | 0.6215 | 0.5732 | 0.3945 | 0.3911 | 33 | [0.368, 0.507] | -0.0699 | 12/1/20 |  |
| best_anatomy | S5_hybrid_rrf_resnet50 | 362 | 1.0 | 0.4862 | 0.9448 | 0.9328 | 0.6689 | 0.7156 | 33 | [0.629, 0.761] | 0.2046 | 26/2/5 | -0.3311 |
| best_anatomy | S3S4_image_dinov2_vits14 | 362 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 33 | [0.000, 0.000] | -0.4643 | 0/0/33 |  |
| best_anatomy | S5_hybrid_rrf_dinov2_vits14 | 362 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 33 | [0.000, 0.000] | -0.4643 | 0/0/33 | -1.0 |
| best_anatomy | H_fuse_closest_resnet50 | 362 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 33 | [1.000, 1.000] | 0.5357 | 31/2/0 |  |
| best_anatomy | random_expected | 362 | 1.0 | 0.1126 | 0.5194 | 0.4773 | 0.3009 | 0.2927 | 33 | [0.301, 0.358] |  |  |  |
| last_thicker_than_now | S1_latest_prior | 341 | 1.0 | 0.5161 | 0.8622 | 0.8622 | 0.6695 | 0.7053 | 33 | [0.609, 0.709] |  |  |  |
| last_thicker_than_now | S2_metadata | 341 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 33 | [1.000, 1.000] | 0.3305 | 32/1/0 |  |
| last_thicker_than_now | S3S4_image_resnet50 | 341 | 1.0 | 0.1701 | 0.5953 | 0.5953 | 0.3643 | 0.3837 | 33 | [0.320, 0.398] | -0.3052 | 0/0/33 |  |
| last_thicker_than_now | S5_hybrid_rrf_resnet50 | 341 | 1.0 | 0.4839 | 0.9355 | 0.9355 | 0.6568 | 0.7187 | 33 | [0.603, 0.718] | -0.0127 | 15/1/17 | -0.3432 |
| last_thicker_than_now | S3S4_image_dinov2_vits14 | 341 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 33 | [0.000, 0.000] | -0.6695 | 0/0/33 |  |
| last_thicker_than_now | S5_hybrid_rrf_dinov2_vits14 | 341 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 33 | [0.000, 0.000] | -0.6695 | 0/0/33 | -1.0 |
| last_thicker_than_now | H_fuse_closest_resnet50 | 341 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 33 | [1.000, 1.000] | 0.3305 | 32/1/0 |  |
| last_thicker_than_now | random_expected | 341 | 1.0 | 0.0961 | 0.4803 | 0.4803 | 0.2792 | 0.2832 | 33 | [0.282, 0.310] |  |  |  |
| largest_cst_rise | S1_latest_prior | 353 | 1.0 | 0.1926 | 0.7422 | 0.7269 | 0.4159 | 0.4636 | 31 | [0.385, 0.501] |  |  |  |
| largest_cst_rise | S2_metadata | 353 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 31 | [1.000, 1.000] | 0.5841 | 31/0/0 |  |
| largest_cst_rise | S3S4_image_resnet50 | 353 | 1.0 | 0.1246 | 0.5241 | 0.5019 | 0.3024 | 0.305 | 31 | [0.266, 0.363] | -0.1135 | 6/0/25 |  |
| largest_cst_rise | S5_hybrid_rrf_resnet50 | 353 | 1.0 | 0.4023 | 0.8867 | 0.8796 | 0.5945 | 0.6478 | 31 | [0.552, 0.670] | 0.1786 | 25/0/6 | -0.4055 |
| largest_cst_rise | S3S4_image_dinov2_vits14 | 353 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 31 | [0.000, 0.000] | -0.4159 | 0/0/31 |  |
| largest_cst_rise | S5_hybrid_rrf_dinov2_vits14 | 353 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 31 | [0.000, 0.000] | -0.4159 | 0/0/31 | -1.0 |
| largest_cst_rise | H_fuse_closest_resnet50 | 353 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 31 | [1.000, 1.000] | 0.5841 | 31/0/0 |  |
| largest_cst_rise | random_expected | 353 | 1.0 | 0.1029 | 0.4925 | 0.4695 | 0.2873 | 0.2826 | 31 | [0.285, 0.315] |  |  |  |

## B1

NOT EVALUATED: B1 key/sheet missing or dev visits lack embeddings

