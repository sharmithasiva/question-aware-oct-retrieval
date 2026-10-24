# fresh_v1 evaluation report (SEALED, ONE-SHOT)

> **AI-generated preliminary benchmark with limited human review. NOT clinician-validated.** Questions and labels were produced by language-model agents; a human researcher reviewed only 2 disagreements and a 12-item sample. The annotator's independence could not be verified, and the original inter-agent agreement results are marked **potentially compromised** (`results/fresh_v1_prereg/STAGE_LOG.md`). 14 of 120 items were viewed by the developer during the guided review (after the code freeze). Results do not establish research novelty or clinical usefulness.

## 1. Items (unit = distinct formulation)

- Distinct formulations: **120** (120 items). By category: plausible_unsupported 15, supported 80, ambiguous_underspecified_offtopic 10, images_or_unavailable 15.
- By gold routing: unsupported_operation 15, retrieve 70, needs_clarification 10, requires_images 7, data_unavailable 18.
- Developer-viewed items: 14.

## 2. Part A1 — routing and availability (Wilson 95 % CI over formulations)

| Endpoint | Result |
|---|---|
| **Primary 1: routing accuracy** | 98/120 = 0.817 [0.738, 0.876] |
| **Primary 2: unsafe-retrieval rate** (gold ≠ retrieve, system retrieves; lower is better) | 4/50 = 0.080 [0.032, 0.188] |
| Secondary: over-refusal rate (gold retrieve, system does not; lower is better) | 14/70 = 0.200 [0.123, 0.308] |
| Secondary: correct limitation handling, exact outcome (images / data unavailable) | 20/25 = 0.800 [0.609, 0.911] |
| Secondary: limitation items not retrieved (any refusal) | 23/25 = 0.920 [0.750, 0.978] |

**Confusion matrix** (rows: gold routing; columns: system outcome)

| gold \ system | clarify | retrieve | unavailable_variable | unsupported_image |
|---|---|---|---|---|
| data_unavailable | 2 | 2 | 14 | 0 |
| needs_clarification | 9 | 0 | 0 | 1 |
| requires_images | 1 | 0 | 0 | 6 |
| retrieve | 14 | 56 | 0 | 0 |
| unsupported_operation | 13 | 2 | 0 | 0 |

**Routing accuracy by category and by developer-viewed stratum (descriptive)**

| Stratum | Routing accuracy |
|---|---|
| category = ambiguous_underspecified_offtopic | 9/10 = 0.900 [0.596, 0.982] |
| category = images_or_unavailable | 10/15 = 0.667 [0.417, 0.848] |
| category = plausible_unsupported | 13/15 = 0.867 [0.621, 0.963] |
| category = supported | 66/80 = 0.825 [0.727, 0.893] |
| developer_viewed = False | 86/106 = 0.811 [0.726, 0.874] |
| developer_viewed = True | 12/14 = 0.857 [0.601, 0.960] |

## 3. Part A2 — intent accuracy on gold-`retrieve` items (gold parameters from type definitions, clarification 01 (a))

- Exact intent: 47/70 = 0.671 [0.555, 0.770]
- operation: 50/70 = 0.714 [0.599, 0.807]
- target: 47/50 = 0.940 [0.838, 0.979]
- direction: 17/30 = 0.567 [0.392, 0.726]
- threshold: 4/10 = 0.400 [0.168, 0.687]
- offset_weeks: 17/20 = 0.850 [0.640, 0.948]
- tolerance_weeks: 19/20 = 0.950 [0.764, 0.991]
- comparator: 13/20 = 0.650 [0.433, 0.819]
- recency: 14/20 = 0.700 [0.481, 0.855]

| Type (descriptive) | Routing correct | Exact intent |
|---|---|---|
| first_vision_gain_1line | 6/10 = 0.600 [0.313, 0.832] | 4/10 = 0.400 [0.168, 0.687] |
| largest_bcva_drop | 7/10 = 0.700 [0.397, 0.892] | 7/10 = 0.700 [0.397, 0.892] |
| largest_cst_rise | 7/10 = 0.700 [0.397, 0.892] | 6/10 = 0.600 [0.313, 0.832] |
| last_thicker_than_now | 9/10 = 0.900 [0.596, 0.982] | 7/10 = 0.700 [0.397, 0.892] |
| last_vision_as_good | 8/10 = 0.800 [0.490, 0.943] | 6/10 = 0.600 [0.313, 0.832] |
| max_leakage | 10/10 = 1.000 [0.722, 1.000] | n/a (routing only) |
| one_year_ago | 9/10 = 0.900 [0.596, 0.982] | 8/10 = 0.800 [0.490, 0.943] |
| six_months_ago | 10/10 = 1.000 [0.722, 1.000] | 9/10 = 0.900 [0.596, 0.982] |

## 4. Part C — visit retrieval (gold-`retrieve` items only; patients: test)

- Global fixed rule (fair baseline): **latest_prior**. Per-type rule = UPPER REFERENCE (it is given the true question type). Items with gold routing ≠ retrieve (incl. max_leakage, image questions) are never scored here.
- Distinct question texts per type used: {'largest_cst_rise': 10, 'largest_bcva_drop': 10, 'first_vision_gain_1line': 10, 'one_year_ago': 10, 'six_months_ago': 10, 'last_thicker_than_now': 10, 'last_vision_as_good': 10}. Hit@1 = top-ranked visit acceptable; an abstention on an answerable instance scores 0. 95 % patient-cluster bootstrap CIs.

| Type | System | Hit@1 [95 % CI] | MRR | Instances | Patients |
|---|---|---|---|---|---|
| largest_cst_rise | intent_v3 | 0.606 [0.581, 0.630] | 0.624 | 206 | 16 |
| largest_cst_rise | global_fixed_rule | 0.257 [0.176, 0.360] | 0.459 | 206 | 16 |
| largest_cst_rise | per_type_rule_UPPER_REF | 0.257 [0.176, 0.360] | 0.459 | 206 | 16 |
| largest_cst_rise | combined | 0.144 [0.100, 0.194] | 0.347 | 206 | 16 |
| largest_cst_rise | rule_router | 0.216 [0.138, 0.310] | 0.433 | 206 | 16 |
| largest_bcva_drop | intent_v3 | 0.744 [0.705, 0.792] | 0.744 | 233 | 18 |
| largest_bcva_drop | global_fixed_rule | 0.303 [0.195, 0.437] | 0.494 | 233 | 18 |
| largest_bcva_drop | per_type_rule_UPPER_REF | 0.306 [0.135, 0.515] | 0.519 | 233 | 18 |
| largest_bcva_drop | combined | 0.294 [0.185, 0.429] | 0.464 | 233 | 18 |
| largest_bcva_drop | rule_router | 0.294 [0.183, 0.431] | 0.480 | 233 | 18 |
| first_vision_gain_1line | intent_v3 | 0.421 [0.375, 0.466] | 0.496 | 182 | 14 |
| first_vision_gain_1line | global_fixed_rule | 0.103 [0.078, 0.136] | 0.287 | 182 | 14 |
| first_vision_gain_1line | per_type_rule_UPPER_REF | 0.533 [0.286, 0.786] | 0.624 | 182 | 14 |
| first_vision_gain_1line | combined | 0.043 [0.015, 0.076] | 0.326 | 182 | 14 |
| first_vision_gain_1line | rule_router | 0.039 [0.009, 0.074] | 0.321 | 182 | 14 |
| one_year_ago | intent_v3 | 0.707 [0.474, 0.849] | 0.720 | 46 | 7 |
| one_year_ago | global_fixed_rule | 0.000 [0.000, 0.000] | 0.112 | 46 | 7 |
| one_year_ago | per_type_rule_UPPER_REF | 0.258 [0.130, 0.508] | 0.365 | 46 | 7 |
| one_year_ago | combined | 0.000 [0.000, 0.000] | 0.137 | 46 | 7 |
| one_year_ago | rule_router | 0.000 [0.000, 0.000] | 0.112 | 46 | 7 |
| six_months_ago | intent_v3 | 0.900 [0.880, 0.919] | 0.920 | 82 | 7 |
| six_months_ago | global_fixed_rule | 0.000 [0.000, 0.000] | 0.233 | 82 | 7 |
| six_months_ago | per_type_rule_UPPER_REF | 0.155 [0.092, 0.212] | 0.318 | 82 | 7 |
| six_months_ago | combined | 0.021 [0.000, 0.043] | 0.250 | 82 | 7 |
| six_months_ago | rule_router | 0.000 [0.000, 0.000] | 0.233 | 82 | 7 |
| last_thicker_than_now | intent_v3 | 0.781 [0.745, 0.822] | 0.819 | 257 | 18 |
| last_thicker_than_now | global_fixed_rule | 0.625 [0.551, 0.706] | 0.751 | 257 | 18 |
| last_thicker_than_now | per_type_rule_UPPER_REF | 0.625 [0.551, 0.706] | 0.751 | 257 | 18 |
| last_thicker_than_now | combined | 0.457 [0.379, 0.554] | 0.622 | 257 | 18 |
| last_thicker_than_now | rule_router | 0.502 [0.428, 0.594] | 0.653 | 257 | 18 |
| last_vision_as_good | intent_v3 | 0.756 [0.687, 0.819] | 0.769 | 219 | 18 |
| last_vision_as_good | global_fixed_rule | 0.732 [0.674, 0.795] | 0.833 | 219 | 18 |
| last_vision_as_good | per_type_rule_UPPER_REF | 0.732 [0.674, 0.795] | 0.833 | 219 | 18 |
| last_vision_as_good | combined | 0.553 [0.472, 0.639] | 0.683 | 219 | 18 |
| last_vision_as_good | rule_router | 0.643 [0.577, 0.716] | 0.751 | 219 | 18 |

**Paired difference: intent_v3 − global fixed rule (Hit@1)**

| Type | Difference [95 % CI] | Instances | Patients |
|---|---|---|---|
| largest_cst_rise | +0.349 [+0.235, +0.441] | 206 | 16 |
| largest_bcva_drop | +0.441 [+0.347, +0.521] | 233 | 18 |
| first_vision_gain_1line | +0.318 [+0.270, +0.360] | 182 | 14 |
| one_year_ago | +0.707 [+0.474, +0.849] | 46 | 7 |
| six_months_ago | +0.900 [+0.880, +0.919] | 82 | 7 |
| last_thicker_than_now | +0.156 [+0.105, +0.207] | 257 | 18 |
| last_vision_as_good | +0.023 [-0.078, +0.106] | 219 | 18 |

**Abstention (intent_v3; fixed rules cannot abstain)**

| Type | Answerable | False abstention | IE instances | Correct abstention |
|---|---|---|---|---|
| largest_cst_rise | 206 | 0.301 | 59 | 0.627 |
| largest_bcva_drop | 233 | 0.292 | 32 | 0.656 |
| first_vision_gain_1line | 182 | 0.390 | 83 | 0.361 |
| one_year_ago | 46 | 0.152 | 0 | – |
| six_months_ago | 82 | 0.000 | 3 | 1.000 |
| last_thicker_than_now | 257 | 0.105 | 8 | 0.875 |
| last_vision_as_good | 219 | 0.205 | 46 | 0.630 |

## 5. Known issues and limitations

- KI-01 (`results/fresh_v1_prereg/KNOWN_ISSUES.md`): the retriever does not enforce the ≥ 2-earlier-visits rule; benchmark instances always have ≥ 2 earlier visits, so it cannot affect Part C.
- With ~10 formulations per type, per-type figures are descriptive; inference uses pooled A1 endpoints.
- Part C rests on the test split (18 patients; 7–8 for Prime-only types); intervals are wide.
- Questions were written by an AI agent from the developer's definitions; real clinician wording may differ.
