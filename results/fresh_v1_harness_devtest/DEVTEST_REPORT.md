# fresh_v1 evaluation report (DEV-TEST)

> **AI-generated preliminary benchmark with limited human review. NOT clinician-validated.** Questions and labels were produced by language-model agents; a human researcher reviewed only 2 disagreements and a 12-item sample. The annotator's independence could not be verified, and the original inter-agent agreement results are marked **potentially compromised** (`results/fresh_v1_prereg/STAGE_LOG.md`). 14 of 120 items were viewed by the developer during the guided review (after the code freeze). Results do not establish research novelty or clinical usefulness.

> **HARNESS PLUMBING TEST ONLY.** Questions are the developer-seen DEV paraphrases and Part C uses train+val patients. These numbers are NOT results and must not be reported as such.

## 1. Items (unit = distinct formulation)

- Distinct formulations: **55** (55 items). By category: supported 40, images_or_unavailable 15.
- By gold routing: retrieve 35, data_unavailable 11, requires_images 9.
- Developer-viewed items: 55.

## 2. Part A1 — routing and availability (Wilson 95 % CI over formulations)

| Endpoint | Result |
|---|---|
| **Primary 1: routing accuracy** | 55/55 = 1.000 [0.935, 1.000] |
| **Primary 2: unsafe-retrieval rate** (gold ≠ retrieve, system retrieves; lower is better) | 0/20 = 0.000 [0.000, 0.161] |
| Secondary: over-refusal rate (gold retrieve, system does not; lower is better) | 0/35 = 0.000 [0.000, 0.099] |
| Secondary: correct limitation handling, exact outcome (images / data unavailable) | 20/20 = 1.000 [0.839, 1.000] |
| Secondary: limitation items not retrieved (any refusal) | 20/20 = 1.000 [0.839, 1.000] |

**Confusion matrix** (rows: gold routing; columns: system outcome)

| gold \ system | retrieve | unavailable_variable | unsupported_image |
|---|---|---|---|
| data_unavailable | 0 | 11 | 0 |
| requires_images | 0 | 0 | 9 |
| retrieve | 35 | 0 | 0 |

**Routing accuracy by category and by developer-viewed stratum (descriptive)**

| Stratum | Routing accuracy |
|---|---|
| category = images_or_unavailable | 15/15 = 1.000 [0.796, 1.000] |
| category = supported | 40/40 = 1.000 [0.912, 1.000] |
| developer_viewed = True | 55/55 = 1.000 [0.935, 1.000] |

## 3. Part A2 — intent accuracy on gold-`retrieve` items (gold parameters from type definitions, clarification 01 (a))

- Exact intent: 35/35 = 1.000 [0.901, 1.000]
- operation: 35/35 = 1.000 [0.901, 1.000]
- target: 25/25 = 1.000 [0.867, 1.000]
- direction: 15/15 = 1.000 [0.796, 1.000]
- threshold: 5/5 = 1.000 [0.566, 1.000]
- offset_weeks: 10/10 = 1.000 [0.722, 1.000]
- tolerance_weeks: 10/10 = 1.000 [0.722, 1.000]
- comparator: 10/10 = 1.000 [0.722, 1.000]
- recency: 10/10 = 1.000 [0.722, 1.000]

| Type (descriptive) | Routing correct | Exact intent |
|---|---|---|
| first_vision_gain_1line | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] |
| largest_bcva_drop | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] |
| largest_cst_rise | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] |
| last_thicker_than_now | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] |
| last_vision_as_good | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] |
| max_leakage | 5/5 = 1.000 [0.566, 1.000] | n/a (routing only) |
| one_year_ago | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] |
| six_months_ago | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] |

## 4. Part C — visit retrieval (gold-`retrieve` items only; patients: train, val)

- Global fixed rule (fair baseline): **latest_prior**. Per-type rule = UPPER REFERENCE (it is given the true question type). Items with gold routing ≠ retrieve (incl. max_leakage, image questions) are never scored here.
- Distinct question texts per type used: {'largest_cst_rise': 5, 'largest_bcva_drop': 5, 'first_vision_gain_1line': 5, 'one_year_ago': 5, 'six_months_ago': 5, 'last_thicker_than_now': 5, 'last_vision_as_good': 5}. Hit@1 = top-ranked visit acceptable; an abstention on an answerable instance scores 0. 95 % patient-cluster bootstrap CIs.

| Type | System | Hit@1 [95 % CI] | MRR | Instances | Patients |
|---|---|---|---|---|---|
| largest_cst_rise | intent_v3 | 1.000 [1.000, 1.000] | 1.000 | 907 | 64 |
| largest_cst_rise | global_fixed_rule | 0.259 [0.224, 0.296] | 0.470 | 907 | 64 |
| largest_cst_rise | per_type_rule_UPPER_REF | 0.259 [0.224, 0.296] | 0.470 | 907 | 64 |
| largest_cst_rise | combined | 0.136 [0.107, 0.168] | 0.374 | 907 | 64 |
| largest_cst_rise | rule_router | 0.204 [0.173, 0.239] | 0.444 | 907 | 64 |
| largest_bcva_drop | intent_v3 | 1.000 [1.000, 1.000] | 1.000 | 995 | 68 |
| largest_bcva_drop | global_fixed_rule | 0.224 [0.190, 0.257] | 0.436 | 995 | 68 |
| largest_bcva_drop | per_type_rule_UPPER_REF | 0.276 [0.201, 0.366] | 0.524 | 995 | 68 |
| largest_bcva_drop | combined | 0.137 [0.107, 0.168] | 0.341 | 995 | 68 |
| largest_bcva_drop | rule_router | 0.185 [0.152, 0.218] | 0.390 | 995 | 68 |
| first_vision_gain_1line | intent_v3 | 1.000 [1.000, 1.000] | 1.000 | 806 | 57 |
| first_vision_gain_1line | global_fixed_rule | 0.120 [0.093, 0.160] | 0.300 | 806 | 57 |
| first_vision_gain_1line | per_type_rule_UPPER_REF | 0.415 [0.295, 0.547] | 0.538 | 806 | 57 |
| first_vision_gain_1line | combined | 0.023 [0.013, 0.034] | 0.264 | 806 | 57 |
| first_vision_gain_1line | rule_router | 0.023 [0.013, 0.033] | 0.263 | 806 | 57 |
| one_year_ago | intent_v3 | 1.000 [1.000, 1.000] | 1.000 | 145 | 26 |
| one_year_ago | global_fixed_rule | 0.000 [0.000, 0.000] | 0.141 | 145 | 26 |
| one_year_ago | per_type_rule_UPPER_REF | 0.266 [0.170, 0.383] | 0.369 | 145 | 26 |
| one_year_ago | combined | 0.013 [0.000, 0.031] | 0.152 | 145 | 26 |
| one_year_ago | rule_router | 0.000 [0.000, 0.000] | 0.141 | 145 | 26 |
| six_months_ago | intent_v3 | 1.000 [1.000, 1.000] | 1.000 | 289 | 31 |
| six_months_ago | global_fixed_rule | 0.035 [0.000, 0.106] | 0.248 | 289 | 31 |
| six_months_ago | per_type_rule_UPPER_REF | 0.258 [0.195, 0.337] | 0.406 | 289 | 31 |
| six_months_ago | combined | 0.041 [0.011, 0.083] | 0.263 | 289 | 31 |
| six_months_ago | rule_router | 0.035 [0.000, 0.106] | 0.248 | 289 | 31 |
| last_thicker_than_now | intent_v3 | 1.000 [1.000, 1.000] | 1.000 | 1091 | 69 |
| last_thicker_than_now | global_fixed_rule | 0.582 [0.551, 0.615] | 0.726 | 1091 | 69 |
| last_thicker_than_now | per_type_rule_UPPER_REF | 0.582 [0.551, 0.615] | 0.726 | 1091 | 69 |
| last_thicker_than_now | combined | 0.448 [0.417, 0.477] | 0.606 | 1091 | 69 |
| last_thicker_than_now | rule_router | 0.498 [0.472, 0.526] | 0.644 | 1091 | 69 |
| last_vision_as_good | intent_v3 | 1.000 [1.000, 1.000] | 1.000 | 922 | 68 |
| last_vision_as_good | global_fixed_rule | 0.695 [0.667, 0.725] | 0.810 | 922 | 68 |
| last_vision_as_good | per_type_rule_UPPER_REF | 0.695 [0.667, 0.725] | 0.810 | 922 | 68 |
| last_vision_as_good | combined | 0.495 [0.454, 0.537] | 0.645 | 922 | 68 |
| last_vision_as_good | rule_router | 0.563 [0.531, 0.596] | 0.697 | 922 | 68 |

**Paired difference: intent_v3 − global fixed rule (Hit@1)**

| Type | Difference [95 % CI] | Instances | Patients |
|---|---|---|---|
| largest_cst_rise | +0.741 [+0.704, +0.776] | 907 | 64 |
| largest_bcva_drop | +0.776 [+0.743, +0.810] | 995 | 68 |
| first_vision_gain_1line | +0.880 [+0.840, +0.907] | 806 | 57 |
| one_year_ago | +1.000 [+1.000, +1.000] | 145 | 26 |
| six_months_ago | +0.965 [+0.894, +1.000] | 289 | 31 |
| last_thicker_than_now | +0.418 [+0.385, +0.449] | 1091 | 69 |
| last_vision_as_good | +0.305 [+0.275, +0.333] | 922 | 68 |

**Abstention (intent_v3; fixed rules cannot abstain)**

| Type | Answerable | False abstention | IE instances | Correct abstention |
|---|---|---|---|---|
| largest_cst_rise | 907 | 0.000 | 230 | 1.000 |
| largest_bcva_drop | 995 | 0.000 | 142 | 1.000 |
| first_vision_gain_1line | 806 | 0.000 | 331 | 1.000 |
| one_year_ago | 145 | 0.000 | 1 | 1.000 |
| six_months_ago | 289 | 0.000 | 18 | 1.000 |
| last_thicker_than_now | 1091 | 0.000 | 45 | 1.000 |
| last_vision_as_good | 922 | 0.000 | 214 | 1.000 |

## 5. Known issues and limitations

- KI-01 (`results/fresh_v1_prereg/KNOWN_ISSUES.md`): the retriever does not enforce the ≥ 2-earlier-visits rule; benchmark instances always have ≥ 2 earlier visits, so it cannot affect Part C.
- With ~10 formulations per type, per-type figures are descriptive; inference uses pooled A1 endpoints.
- Part C rests on the test split (18 patients; 7–8 for Prime-only types); intervals are wide.
- Questions were written by an AI agent from the developer's definitions; real clinician wording may differ.
