# E3 question-conditioned retrieval: `phase2/data/atlas_fullhist`

Created 2026-10-30T07:59:05+00:00. Eyes: [11, 25, 61]. Exploratory; see the caveats in the module docstring.

## temporal_previous  (temporal)

Ground truth: structural (visit order)

| system | n_queries | n_eyes | n_patients | coverage | recall1 | recall5 | hit1 | mrr | ndcg5 | primary_ci95_eye_bootstrap |
|---|---|---|---|---|---|---|---|---|---|---|
| latest_prior | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| metadata_only | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| image_resnet50 | 55 | 3 | 3 | 1.0 | 0.2909 | 0.6909 | 0.2909 | 0.4787 | 0.5046 | n_eyes < 5 |
| safe_hybrid_resnet50 | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| question_aware_resnet50 | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| image_dinov2_vits14 | 55 | 3 | 3 | 1.0 | 0.1818 | 0.6364 | 0.1818 | 0.3823 | 0.412 | n_eyes < 5 |
| safe_hybrid_dinov2_vits14 | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| question_aware_dinov2_vits14 | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| random_expected | 55 | 3 | 3 | 1.0 | 0.0796 | 0.398 | 0.0796 | 0.2439 | 0.2347 | n_eyes < 5 |

## temporal_baseline  (temporal)

Ground truth: structural (visit order)

| system | n_queries | n_eyes | n_patients | coverage | recall1 | recall5 | hit1 | mrr | ndcg5 | primary_ci95_eye_bootstrap |
|---|---|---|---|---|---|---|---|---|---|---|
| latest_prior | 55 | 3 | 3 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0796 | 0.0 | n_eyes < 5 |
| metadata_only | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| image_resnet50 | 55 | 3 | 3 | 1.0 | 0.0364 | 0.1455 | 0.0364 | 0.1465 | 0.0895 | n_eyes < 5 |
| safe_hybrid_resnet50 | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| question_aware_resnet50 | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| image_dinov2_vits14 | 55 | 3 | 3 | 1.0 | 0.0 | 0.2545 | 0.0 | 0.1454 | 0.1178 | n_eyes < 5 |
| safe_hybrid_dinov2_vits14 | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| question_aware_dinov2_vits14 | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| random_expected | 55 | 3 | 3 | 1.0 | 0.0796 | 0.398 | 0.0796 | 0.2439 | 0.2347 | n_eyes < 5 |

## cst_highest  (cst_metadata)

Ground truth: metadata (frozen v1 key: max CST)

| system | n_queries | n_eyes | n_patients | coverage | recall1 | recall5 | hit1 | mrr | ndcg5 | primary_ci95_eye_bootstrap |
|---|---|---|---|---|---|---|---|---|---|---|
| latest_prior | 55 | 3 | 3 | 1.0 | 0.0545 | 0.2 | 0.0545 | 0.1742 | 0.1254 | n_eyes < 5 |
| metadata_only | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| image_resnet50 | 55 | 3 | 3 | 1.0 | 0.0727 | 0.2545 | 0.0727 | 0.1993 | 0.1637 | n_eyes < 5 |
| safe_hybrid_resnet50 | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| question_aware_resnet50 | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| image_dinov2_vits14 | 55 | 3 | 3 | 1.0 | 0.0182 | 0.3091 | 0.0182 | 0.1739 | 0.1553 | n_eyes < 5 |
| safe_hybrid_dinov2_vits14 | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| question_aware_dinov2_vits14 | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| random_expected | 55 | 3 | 3 | 1.0 | 0.0796 | 0.398 | 0.0796 | 0.2439 | 0.2347 | n_eyes < 5 |

## cst_lowest  (cst_metadata)

Ground truth: metadata (frozen v1 key: min CST)

| system | n_queries | n_eyes | n_patients | coverage | recall1 | recall5 | hit1 | mrr | ndcg5 | primary_ci95_eye_bootstrap |
|---|---|---|---|---|---|---|---|---|---|---|
| latest_prior | 55 | 3 | 3 | 1.0 | 0.2364 | 0.6727 | 0.2364 | 0.4171 | 0.4502 | n_eyes < 5 |
| metadata_only | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| image_resnet50 | 55 | 3 | 3 | 1.0 | 0.2 | 0.6364 | 0.2 | 0.3918 | 0.4161 | n_eyes < 5 |
| safe_hybrid_resnet50 | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| question_aware_resnet50 | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| image_dinov2_vits14 | 55 | 3 | 3 | 1.0 | 0.1636 | 0.6364 | 0.1636 | 0.3867 | 0.4169 | n_eyes < 5 |
| safe_hybrid_dinov2_vits14 | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| question_aware_dinov2_vits14 | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| random_expected | 55 | 3 | 3 | 1.0 | 0.0796 | 0.398 | 0.0796 | 0.2439 | 0.2347 | n_eyes < 5 |

## cst_most_similar  (cst_metadata)

Ground truth: metadata (frozen v1 key)

| system | n_queries | n_eyes | n_patients | coverage | recall1 | recall5 | hit1 | mrr | ndcg5 | primary_ci95_eye_bootstrap |
|---|---|---|---|---|---|---|---|---|---|---|
| latest_prior | 55 | 3 | 3 | 1.0 | 0.2182 | 0.6273 | 0.2364 | 0.4115 | 0.4303 | n_eyes < 5 |
| metadata_only | 55 | 3 | 3 | 1.0 | 0.9515 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| image_resnet50 | 55 | 3 | 3 | 1.0 | 0.1364 | 0.6091 | 0.1455 | 0.3383 | 0.3693 | n_eyes < 5 |
| safe_hybrid_resnet50 | 55 | 3 | 3 | 1.0 | 0.9515 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| question_aware_resnet50 | 55 | 3 | 3 | 1.0 | 0.9515 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| image_dinov2_vits14 | 55 | 3 | 3 | 1.0 | 0.0727 | 0.5303 | 0.0727 | 0.2649 | 0.2951 | n_eyes < 5 |
| safe_hybrid_dinov2_vits14 | 55 | 3 | 3 | 1.0 | 0.9515 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| question_aware_dinov2_vits14 | 55 | 3 | 3 | 1.0 | 0.9515 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| random_expected | 55 | 3 | 3 | 1.0 | 0.0796 | 0.398 | 0.0948 | 0.2604 | 0.2444 | n_eyes < 5 |

## cst_largest_rise  (cst_metadata)

Ground truth: metadata (frozen v2 key)

| system | n_queries | n_eyes | n_patients | coverage | recall1 | recall5 | hit1 | mrr | ndcg5 | primary_ci95_eye_bootstrap |
|---|---|---|---|---|---|---|---|---|---|---|
| latest_prior | 54 | 3 | 3 | 1.0 | 0.1111 | 0.537 | 0.1111 | 0.2973 | 0.3204 | n_eyes < 5 |
| metadata_only | 54 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| image_resnet50 | 54 | 3 | 3 | 1.0 | 0.0741 | 0.4815 | 0.0741 | 0.2605 | 0.2801 | n_eyes < 5 |
| safe_hybrid_resnet50 | 54 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| question_aware_resnet50 | 54 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| image_dinov2_vits14 | 54 | 3 | 3 | 1.0 | 0.1111 | 0.3889 | 0.1111 | 0.2662 | 0.2494 | n_eyes < 5 |
| safe_hybrid_dinov2_vits14 | 54 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| question_aware_dinov2_vits14 | 54 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| random_expected | 54 | 3 | 3 | 1.0 | 0.078 | 0.3899 | 0.078 | 0.2409 | 0.2299 | n_eyes < 5 |

## cst_last_above_today  (cst_metadata)

Ground truth: metadata (frozen v2 key)

| system | n_queries | n_eyes | n_patients | coverage | recall1 | recall5 | hit1 | mrr | ndcg5 | primary_ci95_eye_bootstrap |
|---|---|---|---|---|---|---|---|---|---|---|
| latest_prior | 52 | 3 | 3 | 1.0 | 0.6346 | 0.9038 | 0.6346 | 0.7525 | 0.78 | n_eyes < 5 |
| metadata_only | 52 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| image_resnet50 | 52 | 3 | 3 | 1.0 | 0.25 | 0.6154 | 0.25 | 0.4321 | 0.4435 | n_eyes < 5 |
| safe_hybrid_resnet50 | 52 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| question_aware_resnet50 | 52 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| image_dinov2_vits14 | 52 | 3 | 3 | 1.0 | 0.0962 | 0.6154 | 0.0962 | 0.3185 | 0.3604 | n_eyes < 5 |
| safe_hybrid_dinov2_vits14 | 52 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| question_aware_dinov2_vits14 | 52 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| random_expected | 52 | 3 | 3 | 1.0 | 0.0806 | 0.4032 | 0.0806 | 0.246 | 0.2377 | n_eyes < 5 |

## edema_worst_proxy  (pathology_proxy)

Ground truth: metadata (frozen v1 'worst_edema' key = max CST; Phase 1 operationalisation, NOT a clinical edema grade)

| system | n_queries | n_eyes | n_patients | coverage | recall1 | recall5 | hit1 | mrr | ndcg5 | primary_ci95_eye_bootstrap |
|---|---|---|---|---|---|---|---|---|---|---|
| latest_prior | 55 | 3 | 3 | 1.0 | 0.0545 | 0.2 | 0.0545 | 0.1742 | 0.1254 | n_eyes < 5 |
| metadata_only | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| image_resnet50 | 55 | 3 | 3 | 1.0 | 0.0727 | 0.2545 | 0.0727 | 0.1993 | 0.1637 | n_eyes < 5 |
| safe_hybrid_resnet50 | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| question_aware_resnet50 | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| image_dinov2_vits14 | 55 | 3 | 3 | 1.0 | 0.0182 | 0.3091 | 0.0182 | 0.1739 | 0.1553 | n_eyes < 5 |
| safe_hybrid_dinov2_vits14 | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| question_aware_dinov2_vits14 | 55 | 3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n_eyes < 5 |
| random_expected | 55 | 3 | 3 | 1.0 | 0.0796 | 0.398 | 0.0796 | 0.2439 | 0.2347 | n_eyes < 5 |

## image_similarity_B1  (image_similarity)

Ground truth: human (B1, single researcher_non_expert annotator; visual similarity only)

| system | n_queries | n_eyes | n_patients | coverage | recall1 | recall5 | hit1 | mrr | ndcg5 | primary_ci95_eye_bootstrap |
|---|---|---|---|---|---|---|---|---|---|---|
| image_dinov2_vits14 | 18 | 3 | 3 | 1.0 | 0.1144 | 0.6103 | 0.5 | 0.6782 | 0.6824 | n_eyes < 5 |
| image_resnet50 | 18 | 3 | 3 | 1.0 | 0.2149 | 0.6558 | 0.7778 | 0.8565 | 0.8023 | n_eyes < 5 |
| latest_prior | 18 | 3 | 3 | 1.0 | 0.2208 | 0.6667 | 0.6111 | 0.7725 | 0.7857 | n_eyes < 5 |
| metadata_cst_proxy | 18 | 3 | 3 | 1.0 | 0.1829 | 0.6942 | 0.6667 | 0.7764 | 0.7815 | n_eyes < 5 |
| metadata_only | 18 | 3 | 3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | n_eyes < 5 |
| question_aware_dinov2_vits14 | 18 | 3 | 3 | 1.0 | 0.0894 | 0.6945 | 0.3889 | 0.6426 | 0.7252 | n_eyes < 5 |
| question_aware_resnet50 | 18 | 3 | 3 | 1.0 | 0.2704 | 0.7026 | 0.7778 | 0.8565 | 0.823 | n_eyes < 5 |
| safe_hybrid_dinov2_vits14 | 18 | 3 | 3 | 1.0 | 0.0894 | 0.6945 | 0.3889 | 0.6426 | 0.7252 | n_eyes < 5 |
| safe_hybrid_resnet50 | 18 | 3 | 3 | 1.0 | 0.2704 | 0.7026 | 0.7778 | 0.8565 | 0.823 | n_eyes < 5 |
| random_expected | 18 | 3 | 3 | 1.0 | 0.0859 | 0.4295 | 0.3355 | 0.5288 | 0.5718 | n_eyes < 5 |

## Not evaluable (N/A)

- **image_similarity (all eyes)**: no visual relevance labels beyond the 3 dev eyes of B1 (reported separately)
- **pathology: fluid presence / first appearance / improvement (visit ranking)**: OLIVES finding labels exist only for the first and last visit of each eye, so no visit ranking over a history can be labelled; scan-level finding detection is evaluated in pathology_probe.py
- **mixed (image + metadata)**: no ground truth exists for the combined answer
