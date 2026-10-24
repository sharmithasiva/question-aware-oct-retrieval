# fresh_v1 — Pre-registered analysis plan

> **This is an AI-generated preliminary benchmark with limited human review. The questions and labels were produced
> by language-model agents; a human researcher reviewed only the disagreements and a small sample. Results are not
> clinician-validated and do not establish research novelty or clinical usefulness.**

Written and hashed **before** any fresh question exists. Protocol: `docs/FRESH_EVALUATION_PROTOCOL.md` (revision 2),
approved with decisions D1–D4 at their defaults and two added safeguards (below).

## Frozen system under test

The files and versions listed in `results/fresh_v1_prereg/CODE_FREEZE.sha256` and `environment.json`.
- Any change to a listed file voids the fresh_v1 run.
- The step-9 evaluation harness will be written after this freeze and hashed separately **before** the sealed set is opened. It may only import frozen modules, and it may not modify them.

## Restricted-input setup

The retriever sees visit order, CST, BCVA and study week (one cohort only). Leakage index, injections and DRSS are
not available to it.

## Routing mapping (fixed)

| Gold routing | Accepted system outcome (`intent.parse_intent(...).task`) |
|---|---|
| `retrieve` | `retrieve` |
| `data_unavailable` | `unavailable_variable` |
| `requires_images` | `unsupported_image` |
| `needs_clarification` | `clarify` |
| `unsupported_operation` | `clarify` |

## Safeguard 2: routing and visit retrieval are evaluated separately

**Part A1 — routing and availability (all gold items, unit = distinct formulation):**
- **routing accuracy**: the system outcome equals the mapped gold outcome; with a confusion matrix;
- **correct limitation handling** for gold `requires_images` and `data_unavailable` items (this includes all `max_leakage` items, whose gold routing is `data_unavailable` under the restricted input): the share routed to exactly the right limitation outcome, and the share routed to *any* non-retrieval outcome;
- **unsafe-retrieval rate**: gold routing ≠ `retrieve`, but the system retrieves;
- **over-refusal rate**: gold routing = `retrieve`, but the system does not retrieve.

**Part A2 — intent accuracy (gold-`retrieve` items only):**
- exact match on the parsed operation, plus the operation's parameters: target, direction, comparator, recency, time offset, threshold;
- per-field accuracy.

**Part C — visit retrieval (gold-`retrieve` items only, test-split patients only):**
- Hit@1, plus MRR as secondary, against the unchanged `ground_truth_v2.answer_key`;
- items with gold routing ≠ `retrieve`, including `max_leakage` and image-dependent questions, are **never** scored with Hit@1;
- **comparators:**
  - the global fixed rule in `global_fixed_rule.json` (the fair baseline);
  - the per-type rule in `results/v3_dev/selected_baselines.json`, an upper reference because it is given the true type;
  - frozen v1 `combined` and `rule_router`;
- per-type paired Hit@1 difference vs the global rule, with a 95 % patient-cluster bootstrap CI (1,000 reps, seed 42);
- insufficient-evidence instances are reported separately: correct abstention and false abstention.

## Endpoints

- **Primary:**
  - (1) pooled routing accuracy (A1);
  - (2) unsafe-retrieval rate (A1).
- **Secondary:**
  - exact-intent accuracy (A2);
  - over-refusal rate (A1);
  - correct limitation handling (A1);
  - Part C Hit@1 difference vs the global fixed rule.
- **Descriptive only:** everything else, including all per-type figures. With about 10 formulations per type, single-type Wilson intervals are wide.
- **Uncertainty:**
  - Wilson 95 % intervals over formulations for A1 and A2;
  - a patient-cluster bootstrap for Part C.

## Exclusions (fixed)

An item is excluded if it is:
- your `drop` decision;
- a near-duplicate within the set (normalised similarity ≥ 0.9);
- overlapping with dev paraphrases, v1 templates or v1 held-out paraphrases.

Items you mark `ambiguous` move to gold routing `needs_clarification`. The retired locked set is not checked (D3).
