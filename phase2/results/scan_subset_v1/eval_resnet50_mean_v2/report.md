# Retrieval evaluation: `phase2\data\scan_subset_v1` (visit embedding: mean)

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

## E2 definition: within-eye structure (33 complete-history eyes)

```json
{
  "resnet50": {
    "n_eyes": 33,
    "n_queries": 478,
    "n_pairs": 4639,
    "image_top1_is_previous_rate": 0.3054,
    "image_top1_is_previous_chance": 0.1577,
    "image_top1_is_previous_ci95_eye_bootstrap": [
      0.2603,
      0.3595
    ],
    "spearman_cos_vs_temporal_closeness_pooled": 0.316,
    "spearman_cos_vs_temporal_closeness_mean_over_eyes": 0.3451,
    "spearman_temporal_ci95_eye_bootstrap": [
      0.2868,
      0.4022
    ],
    "partial_spearman_cos_vs_cst_given_gap_pooled": 0.4574,
    "partial_spearman_cos_vs_cst_given_gap_mean_over_eyes": 0.1857,
    "partial_cst_ci95_eye_bootstrap": [
      0.1122,
      0.2619
    ],
    "measures": "temporal / CST association within an eye (no relevance labels)"
  }
}
```

## Tier A: CST questions, frozen metadata answer keys (primary metric MRR)

`S2_metadata` is the frozen Phase 1 system. `S5_hybrid_rrf_*` is the v0.1 prototype (fuses every metadata ranking and falls back to images when metadata abstains). `H_fuse_closest_*` is phase2/hybrid_retriever.py: fuses only `closest_to_current`, never overrides abstentions.

| category | system | n_evaluated | coverage | hit1 | hit5 | recall5 | mrr | ndcg5 | n_eyes | mrr_ci95_eye_bootstrap | mrr_minus_latest_prior | eyes_better_equal_worse_vs_latest_prior | mrr_minus_S2_metadata |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| similar_cst | S1_latest_prior | 347 | 1.0 | 0.3026 | 0.7176 | 0.6633 | 0.4767 | 0.4806 | 32 | [0.452, 0.553] |  |  |  |
| similar_cst | S2_metadata | 347 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 32 | [1.000, 1.000] | 0.5233 | 31/1/0 |  |
| similar_cst | S3S4_image_resnet50 | 347 | 1.0 | 0.1931 | 0.6455 | 0.5764 | 0.3914 | 0.388 | 32 | [0.357, 0.441] | -0.0853 | 9/0/23 |  |
| similar_cst | S5_hybrid_rrf_resnet50 | 347 | 1.0 | 0.5101 | 0.9337 | 0.8977 | 0.6762 | 0.7051 | 32 | [0.653, 0.736] | 0.1995 | 30/1/1 | -0.3238 |
| similar_cst | S3S4_image_dinov2_vits14 | 347 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32 | [0.000, 0.000] | -0.4767 | 0/0/32 |  |
| similar_cst | S5_hybrid_rrf_dinov2_vits14 | 347 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32 | [0.000, 0.000] | -0.4767 | 0/0/32 | -1.0 |
| similar_cst | H_fuse_closest_resnet50 | 347 | 1.0 | 0.5101 | 0.9337 | 0.8977 | 0.6762 | 0.7051 | 32 | [0.653, 0.736] | 0.1995 | 30/1/1 |  |
| similar_cst | random_expected | 347 | 1.0 | 0.1286 | 0.5518 | 0.479 | 0.3207 | 0.3028 | 32 | [0.321, 0.363] |  |  |  |
| worst_edema | S1_latest_prior | 347 | 1.0 | 0.0432 | 0.1787 | 0.1772 | 0.1738 | 0.109 | 32 | [0.136, 0.206] |  |  |  |
| worst_edema | S2_metadata | 347 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 32 | [1.000, 1.000] | 0.8262 | 32/0/0 |  |
| worst_edema | S3S4_image_resnet50 | 347 | 1.0 | 0.0461 | 0.2738 | 0.2723 | 0.1946 | 0.1526 | 32 | [0.171, 0.236] | 0.0208 | 21/3/8 |  |
| worst_edema | S5_hybrid_rrf_resnet50 | 347 | 1.0 | 0.2882 | 0.8501 | 0.8501 | 0.497 | 0.5662 | 32 | [0.421, 0.550] | 0.3232 | 32/0/0 | -0.503 |
| worst_edema | S3S4_image_dinov2_vits14 | 347 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32 | [0.000, 0.000] | -0.1738 | 0/0/32 |  |
| worst_edema | S5_hybrid_rrf_dinov2_vits14 | 347 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32 | [0.000, 0.000] | -0.1738 | 0/0/32 | -1.0 |
| worst_edema | H_fuse_closest_resnet50 | 347 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 32 | [1.000, 1.000] | 0.8262 | 32/0/0 |  |
| worst_edema | random_expected | 347 | 1.0 | 0.0959 | 0.4796 | 0.479 | 0.279 | 0.2825 | 32 | [0.283, 0.312] |  |  |  |
| best_anatomy | S1_latest_prior | 347 | 1.0 | 0.2795 | 0.7608 | 0.7378 | 0.4748 | 0.5172 | 32 | [0.438, 0.593] |  |  |  |
| best_anatomy | S2_metadata | 347 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 32 | [1.000, 1.000] | 0.5252 | 30/2/0 |  |
| best_anatomy | S3S4_image_resnet50 | 347 | 1.0 | 0.2017 | 0.5764 | 0.5476 | 0.3875 | 0.3823 | 32 | [0.361, 0.469] | -0.0874 | 7/0/25 |  |
| best_anatomy | S5_hybrid_rrf_resnet50 | 347 | 1.0 | 0.4611 | 0.9424 | 0.926 | 0.6407 | 0.6959 | 32 | [0.597, 0.733] | 0.1659 | 24/1/7 | -0.3593 |
| best_anatomy | S3S4_image_dinov2_vits14 | 347 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32 | [0.000, 0.000] | -0.4748 | 0/0/32 |  |
| best_anatomy | S5_hybrid_rrf_dinov2_vits14 | 347 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32 | [0.000, 0.000] | -0.4748 | 0/0/32 | -1.0 |
| best_anatomy | H_fuse_closest_resnet50 | 347 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 32 | [1.000, 1.000] | 0.5252 | 30/2/0 |  |
| best_anatomy | random_expected | 347 | 1.0 | 0.1137 | 0.523 | 0.479 | 0.3026 | 0.2941 | 32 | [0.303, 0.362] |  |  |  |
| last_thicker_than_now | S1_latest_prior | 328 | 1.0 | 0.5183 | 0.8659 | 0.8659 | 0.6719 | 0.7084 | 32 | [0.610, 0.712] |  |  |  |
| last_thicker_than_now | S2_metadata | 328 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 32 | [1.000, 1.000] | 0.3281 | 31/1/0 |  |
| last_thicker_than_now | S3S4_image_resnet50 | 328 | 1.0 | 0.1799 | 0.622 | 0.622 | 0.377 | 0.4026 | 32 | [0.329, 0.409] | -0.2949 | 0/0/32 |  |
| last_thicker_than_now | S5_hybrid_rrf_resnet50 | 328 | 1.0 | 0.5091 | 0.939 | 0.939 | 0.6742 | 0.7329 | 32 | [0.613, 0.719] | 0.0023 | 18/1/13 | -0.3258 |
| last_thicker_than_now | S3S4_image_dinov2_vits14 | 328 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32 | [0.000, 0.000] | -0.6719 | 0/0/32 |  |
| last_thicker_than_now | S5_hybrid_rrf_dinov2_vits14 | 328 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32 | [0.000, 0.000] | -0.6719 | 0/0/32 | -1.0 |
| last_thicker_than_now | H_fuse_closest_resnet50 | 328 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 32 | [1.000, 1.000] | 0.3281 | 31/1/0 |  |
| last_thicker_than_now | random_expected | 328 | 1.0 | 0.0963 | 0.4813 | 0.4813 | 0.2796 | 0.2838 | 32 | [0.283, 0.312] |  |  |  |
| largest_cst_rise | S1_latest_prior | 339 | 1.0 | 0.1888 | 0.7345 | 0.7185 | 0.4113 | 0.4573 | 30 | [0.381, 0.501] |  |  |  |
| largest_cst_rise | S2_metadata | 339 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 30 | [1.000, 1.000] | 0.5887 | 30/0/0 |  |
| largest_cst_rise | S3S4_image_resnet50 | 339 | 1.0 | 0.0914 | 0.4779 | 0.4521 | 0.2773 | 0.2695 | 30 | [0.247, 0.344] | -0.134 | 5/0/25 |  |
| largest_cst_rise | S5_hybrid_rrf_resnet50 | 339 | 1.0 | 0.41 | 0.8938 | 0.8798 | 0.5963 | 0.6458 | 30 | [0.562, 0.674] | 0.185 | 23/0/7 | -0.4037 |
| largest_cst_rise | S3S4_image_dinov2_vits14 | 339 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 30 | [0.000, 0.000] | -0.4113 | 0/0/30 |  |
| largest_cst_rise | S5_hybrid_rrf_dinov2_vits14 | 339 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 30 | [0.000, 0.000] | -0.4113 | 0/0/30 | -1.0 |
| largest_cst_rise | H_fuse_closest_resnet50 | 339 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 30 | [1.000, 1.000] | 0.5887 | 30/0/0 |  |
| largest_cst_rise | random_expected | 339 | 1.0 | 0.1038 | 0.4959 | 0.4719 | 0.2888 | 0.2843 | 30 | [0.286, 0.317] |  |  |  |

