# Tier B evaluation: dev / B1 (overall_appearance)

Run eval_B1_5f0f4b4d_24a90d2768c6, 2026-10-26T09:12:54+00:00. Protocol: `phase2/annotations/ANNOTATION_PROTOCOL_v2.md`.

**NOT EVALUATED: no complete, gradable query with a relevant candidate.**

## Inputs

```json
{
  "sheet": "phase2\\annotations\\v2\\dev\\B1\\sheet.csv",
  "sheet_sha256": "5f0f4b4dd390e963a769b88b30bcf099e1513a3f2f53dc7832800e912eb559a2",
  "key": "C:\\question aware retinal oct reasoning\\phase2\\annotations\\keys\\v2\\dev_B1_key.csv",
  "key_sha256": "018e34dc2a31c1189c67434df7fd111156caa78ab01c43086fb7477265a5e498",
  "rankings": "C:\\question aware retinal oct reasoning\\phase2\\results\\question_eval_v2\\dev\\rankings_tierB_24a90d2768c6",
  "rankings_sha256": "24a90d2768c6076e45f68427ffde0f05960a17a6086ba5aeea5d3cae2f388837",
  "second_sheet": null,
  "git_head": "7fd4f3ed986abd0a3cedcae9573f87fbda00dd1c"
}
```

## Counts

```json
{
  "n_queries": 18,
  "n_judgments": 264,
  "n_labelled": 0,
  "n_blank": 264,
  "labels": {
    "blank": 264
  },
  "query_status": {
    "not_evaluated_missing_labels": 18
  },
  "annotators": [],
  "annotator_roles": [],
  "n_eyes": 3
}
```

## Primary analysis (relevant = grade 2; U/X removed from pool; nDCG gains 0/1/3)

| system | n_evaluated | hit1 | hit5 | recall5 | mrr | ndcg5 |
|---|---|---|---|---|---|---|
| S1_latest_prior | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| S2_metadata | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| S2p_metadata_proxy_cst | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| S3S4_image_dinov2_vits14 | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| S3S4_image_resnet50 | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| S5_hybrid_rrf_dinov2_vits14 | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| S5_hybrid_rrf_resnet50 | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |

S2_metadata (frozen Phase 1) abstains on every Tier B question (`unsupported_image`): coverage 0, scored 0 when abstaining. `random_expected` = exact expectation under random ranking of the same pool.

## Per eye (primary)

NOT EVALUATED

## Sensitivity: U/X scored 0 (pessimistic)

| system | n_evaluated | hit1 | hit5 | recall5 | mrr | ndcg5 |
|---|---|---|---|---|---|---|
| S1_latest_prior | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| S2_metadata | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| S2p_metadata_proxy_cst | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| S3S4_image_dinov2_vits14 | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| S3S4_image_resnet50 | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| S5_hybrid_rrf_dinov2_vits14 | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| S5_hybrid_rrf_resnet50 | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |

## Secondary: relevant = grade >= 1

| system | n_evaluated | hit1 | hit5 | recall5 | mrr | ndcg5 |
|---|---|---|---|---|---|---|
| S1_latest_prior | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| S2_metadata | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| S2p_metadata_proxy_cst | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| S3S4_image_dinov2_vits14 | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| S3S4_image_resnet50 | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| S5_hybrid_rrf_dinov2_vits14 | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| S5_hybrid_rrf_resnet50 | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |

## Inter-annotator agreement

No second sheet supplied.

## Caveats

- Development eyes only (3 eyes): used for rubric calibration and the S5 encoder choice. No CI with < 5 eyes.
- Queries of one eye share candidates; the eye, not the query, is the unit of independence.
- B1 measures visual similarity, not clinical relevance. B2 is clinical only if graded by a trained grader.
