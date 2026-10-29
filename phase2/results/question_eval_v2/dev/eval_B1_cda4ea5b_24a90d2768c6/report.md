# Tier B evaluation: dev / B1 (overall_appearance)

Run eval_B1_cda4ea5b_24a90d2768c6, 2026-10-29T12:14:10+00:00. Protocol: `phase2/annotations/ANNOTATION_PROTOCOL_v2.md`.

**18 evaluated queries.** Exploratory (development eyes); no significance testing.

## Inputs

```json
{
  "sheet": "phase2/annotations/v2/dev/B1/completed_1_dev-B1-v2.csv",
  "sheet_sha256": "cda4ea5b7c130b4b45de3857f76d177afd015dd56ea4063bbcebe9f7d9b21b29",
  "key": "/home/user/question-aware-oct-retrieval/phase2/annotations/keys/v2/dev_B1_key.csv",
  "key_sha256": "018e34dc2a31c1189c67434df7fd111156caa78ab01c43086fb7477265a5e498",
  "rankings": "/home/user/question-aware-oct-retrieval/phase2/results/question_eval_v2/dev/rankings_tierB_24a90d2768c6",
  "rankings_sha256": "24a90d2768c6076e45f68427ffde0f05960a17a6086ba5aeea5d3cae2f388837",
  "second_sheet": null,
  "git_head": "5abb064655e564b1889198a2de7c6225a22facb8"
}
```

## Counts

```json
{
  "n_queries": 18,
  "n_judgments": 264,
  "n_labelled": 264,
  "n_blank": 0,
  "labels": {
    "1": 129,
    "2": 73,
    "0": 61,
    "U": 1
  },
  "query_status": {
    "evaluated": 18
  },
  "annotators": [
    "1"
  ],
  "annotator_roles": [
    "researcher_non_expert"
  ],
  "n_eyes": 3
}
```

## Primary analysis (relevant = grade 2; U/X removed from pool; nDCG gains 0/1/3)

| system | n_evaluated | coverage | hit1 | hit5 | recall5 | mrr | ndcg5 | ndcg5_minus_S1_latest_prior | ndcg5_minus_S2p_metadata_proxy_cst | eyes_better_equal_worse_vs_S1_latest_prior | eyes_better_equal_worse_vs_S2p_metadata_proxy_cst | ndcg5_ci95_eye_bootstrap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_latest_prior | 18 | 1.0 | 0.6111 | 0.9444 | 0.6667 | 0.7725 | 0.7857 |  | 0.0041 |  | 2/0/1 | not reported (n_eyes < 5) |
| S2_metadata | 18 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -0.7857 | -0.7815 | 0/0/3 | 0/0/3 | not reported (n_eyes < 5) |
| S2p_metadata_proxy_cst | 18 | 1.0 | 0.6667 | 0.8889 | 0.6942 | 0.7764 | 0.7815 | -0.0041 |  | 1/0/2 |  | not reported (n_eyes < 5) |
| S3S4_image_dinov2_vits14 | 18 | 1.0 | 0.5 | 0.9444 | 0.6103 | 0.6782 | 0.6824 | -0.1032 | -0.0991 | 1/0/2 | 0/0/3 | not reported (n_eyes < 5) |
| S3S4_image_resnet50 | 18 | 1.0 | 0.7778 | 0.9444 | 0.6558 | 0.8565 | 0.8023 | 0.0167 | 0.0208 | 2/0/1 | 2/0/1 | not reported (n_eyes < 5) |
| S5_hybrid_rrf_dinov2_vits14 | 18 | 1.0 | 0.3889 | 0.9444 | 0.6945 | 0.6426 | 0.7252 | -0.0605 | -0.0564 | 1/0/2 | 0/0/3 | not reported (n_eyes < 5) |
| S5_hybrid_rrf_resnet50 | 18 | 1.0 | 0.7778 | 0.9444 | 0.7026 | 0.8565 | 0.823 | 0.0373 | 0.0414 | 2/0/1 | 2/0/1 | not reported (n_eyes < 5) |
| random_expected | 18 | 1.0 | 0.3355 | 0.7978 | 0.4295 | 0.5288 | 0.5718 |  |  |  |  | not reported (n_eyes < 5) |

S2_metadata (frozen Phase 1) abstains on every Tier B question (`unsupported_image`): coverage 0, scored 0 when abstaining. `random_expected` = exact expectation under random ranking of the same pool.

## Per eye (primary)

| system | eye_id | n_queries | hit1 | hit5 | recall5 | mrr | ndcg5 |
|---|---|---|---|---|---|---|---|
| S1_latest_prior | 11 | 6 | 0.3333 | 0.8333 | 0.4905 | 0.5675 | 0.586 |
| S1_latest_prior | 25 | 6 | 1.0 | 1.0 | 0.8764 | 1.0 | 0.9807 |
| S1_latest_prior | 61 | 6 | 0.5 | 1.0 | 0.6333 | 0.75 | 0.7902 |
| S2_metadata | 11 | 6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S2_metadata | 25 | 6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S2_metadata | 61 | 6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S2p_metadata_proxy_cst | 11 | 6 | 0.5 | 0.8333 | 0.6714 | 0.6833 | 0.73 |
| S2p_metadata_proxy_cst | 25 | 6 | 0.6667 | 1.0 | 0.8333 | 0.7917 | 0.8513 |
| S2p_metadata_proxy_cst | 61 | 6 | 0.8333 | 0.8333 | 0.5778 | 0.8542 | 0.7633 |
| S3S4_image_dinov2_vits14 | 11 | 6 | 0.3333 | 0.8333 | 0.5976 | 0.5068 | 0.6134 |
| S3S4_image_dinov2_vits14 | 25 | 6 | 0.5 | 1.0 | 0.6417 | 0.6944 | 0.69 |
| S3S4_image_dinov2_vits14 | 61 | 6 | 0.6667 | 1.0 | 0.5917 | 0.8333 | 0.7439 |
| S3S4_image_resnet50 | 11 | 6 | 0.5 | 0.8333 | 0.5131 | 0.6528 | 0.6621 |
| S3S4_image_resnet50 | 25 | 6 | 1.0 | 1.0 | 0.8486 | 1.0 | 0.9263 |
| S3S4_image_resnet50 | 61 | 6 | 0.8333 | 1.0 | 0.6056 | 0.9167 | 0.8187 |
| S5_hybrid_rrf_dinov2_vits14 | 11 | 6 | 0.1667 | 1.0 | 0.6571 | 0.4833 | 0.6159 |
| S5_hybrid_rrf_dinov2_vits14 | 25 | 6 | 0.6667 | 1.0 | 0.7792 | 0.8333 | 0.8087 |
| S5_hybrid_rrf_dinov2_vits14 | 61 | 6 | 0.3333 | 0.8333 | 0.6472 | 0.6111 | 0.7508 |
| S5_hybrid_rrf_resnet50 | 11 | 6 | 0.6667 | 0.8333 | 0.5369 | 0.7778 | 0.6952 |
| S5_hybrid_rrf_resnet50 | 25 | 6 | 1.0 | 1.0 | 0.8819 | 1.0 | 0.9627 |
| S5_hybrid_rrf_resnet50 | 61 | 6 | 0.6667 | 1.0 | 0.6889 | 0.7917 | 0.811 |
| random_expected | 11 | 6 | 0.2385 | 0.7556 | 0.4395 | 0.4465 | 0.5388 |
| random_expected | 25 | 6 | 0.3668 | 0.7507 | 0.398 | 0.5406 | 0.5355 |
| random_expected | 61 | 6 | 0.4012 | 0.887 | 0.4511 | 0.5992 | 0.6412 |

## Sensitivity: U/X scored 0 (pessimistic)

| system | n_evaluated | hit1 | hit5 | recall5 | mrr | ndcg5 |
|---|---|---|---|---|---|---|
| S1_latest_prior | 18 | 0.6111 | 0.9444 | 0.6667 | 0.7725 | 0.781 |
| S2_metadata | 18 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S2p_metadata_proxy_cst | 18 | 0.6667 | 0.8889 | 0.6942 | 0.7671 | 0.767 |
| S3S4_image_dinov2_vits14 | 18 | 0.5 | 0.9444 | 0.6103 | 0.6782 | 0.6824 |
| S3S4_image_resnet50 | 18 | 0.7778 | 0.9444 | 0.6558 | 0.8565 | 0.7971 |
| S5_hybrid_rrf_dinov2_vits14 | 18 | 0.3889 | 0.9444 | 0.6945 | 0.6426 | 0.7196 |
| S5_hybrid_rrf_resnet50 | 18 | 0.7222 | 0.9444 | 0.7026 | 0.8287 | 0.8023 |
| random_expected | 18 | 0.3336 | 0.7885 | 0.4203 | 0.5261 | 0.5646 |

## Secondary: relevant = grade >= 1

| system | n_evaluated | hit1 | hit5 | recall5 | mrr | ndcg5 |
|---|---|---|---|---|---|---|
| S1_latest_prior | 18 | 0.8889 | 1.0 | 0.5069 | 0.9444 | 0.7857 |
| S2_metadata | 18 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S2p_metadata_proxy_cst | 18 | 0.8889 | 1.0 | 0.5328 | 0.9352 | 0.7815 |
| S3S4_image_dinov2_vits14 | 18 | 0.9444 | 1.0 | 0.4545 | 0.9722 | 0.6824 |
| S3S4_image_resnet50 | 18 | 1.0 | 1.0 | 0.503 | 1.0 | 0.8023 |
| S5_hybrid_rrf_dinov2_vits14 | 18 | 0.9444 | 1.0 | 0.4838 | 0.9722 | 0.7252 |
| S5_hybrid_rrf_resnet50 | 18 | 0.9444 | 1.0 | 0.5042 | 0.9722 | 0.823 |
| random_expected | 18 | 0.824 | 0.9701 | 0.4295 | 0.889 | 0.5718 |

## Inter-annotator agreement

No second sheet supplied.

## Caveats

- Development eyes only (3 eyes): used for rubric calibration and the S5 encoder choice. No CI with < 5 eyes.
- Queries of one eye share candidates; the eye, not the query, is the unit of independence.
- B1 measures visual similarity, not clinical relevance. B2 is clinical only if graded by a trained grader.
