# Proposed clarification 01 — source of gold intent parameters (NOT APPROVED, NOT IMPLEMENTED)

> AI-generated preliminary benchmark with limited human review. Not clinician-validated.

**Status: APPROVED, option (a), by the researcher on 2026-10-23, before the sealed set was opened.** Gold intent
parameters come from the frozen type definitions in the table below. The frozen `ANALYSIS_PLAN.md` is unchanged:
this file clarifies how its endpoint A2 obtains gold parameters and does not alter any endpoint. This file is outside
the code freeze.

*Original status, kept for the record:* proposal only; nothing applied until approved before opening.

## Issue

The analysis plan's secondary endpoint A2 ("exact-intent accuracy on gold-`retrieve` items") compares the parser's
operation *and parameters* with gold. The plan does not say where the gold parameters come from.

The free-form `parameters` fields written by the two agents agree on only **40 %** of items
(`agreement_summary.json`, `parameters_exact_agreement` = 0.4). That is low enough to suggest inconsistent formatting
or interpretation, for example `"CST"` vs `"cst"`, `52` vs `"52 weeks"`, or omitted keys. The developer has not
inspected the values. The agreement figures are also marked potentially compromised (`STAGE_LOG.md`).

## Proposed clarification

Take gold parameters for A2 from each item's adjudicated **`intended_type`**, using the frozen definitions. Do not use the agents' free-form `parameters`.

| Intended type | Gold operation and parameters (from `docs/PHASE1_V2_QUESTION_SPEC.md`) |
|---|---|
| largest_cst_rise | consecutive change, target CST, direction increase |
| largest_bcva_drop | consecutive change, target BCVA, direction decrease |
| first_vision_gain_1line | first threshold crossing vs baseline, target BCVA, increase, threshold 5 letters |
| one_year_ago | time offset 52 weeks (tolerance 4) |
| six_months_ago | time offset 26 weeks (tolerance 4) |
| max_leakage | no retrieval parameters; gold routing `data_unavailable` (A1 only) |
| last_thicker_than_now | compare to current, target CST, comparator `>`, most recent |
| last_vision_as_good | compare to current, target BCVA, comparator `≥`, most recent |

The agents' `parameters` would be kept in the files unchanged and reported descriptively only.

## Trade-off

- **For:** a single, frozen, consistent source of gold. It avoids scoring the parser against noisy agent formatting.
- **Against:** an item that clearly asks for a different parameter would be scored against its type's default. Examples: "about 50 weeks ago", or a 2-line threshold labelled as `first_vision_gain_1line`.
  - Proposed handling: during your review, any such item can be marked `ambiguous` or `drop`.
  - Alternatively, it can be scored on routing (A1) only.

## Options

- **(a)** Approve as written.
- **(b)** Approve, but score A2 on the operation and target only (ignore numeric parameters).
- **(c)** Reject. A2 then uses the agents' free-form parameters, which would first need a normalisation rule to be specified.
