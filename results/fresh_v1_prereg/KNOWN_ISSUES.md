# fresh_v1 — known issues (recorded before opening the sealed set)

> AI-generated preliminary benchmark with limited human review. Not clinician-validated.

## KI-01 — Retriever does not enforce the "at least 2 earlier visits" rule

- **Found by:** Part B (`results/fresh_v1_partB/PART_B_REPORT.md`).
- **Behaviour:** when the current visit has exactly one earlier visit, `intent_retrieval.retrieve` still returns `retrieved` or `no_qualifying_visit`. The spec (`docs/PHASE1_V2_QUESTION_SPEC.md` §2), the answer key and the independent reference all return insufficient data.
- **Evidence:**
  - 4/48 fixtures;
  - 664/664 random disagreements and 263/263 real disagreements, all at exactly one earlier visit;
  - agreement on every case with ≥ 2 earlier visits.
- **Affected types:** first_vision_gain_1line, max_leakage, last_thicker_than_now, last_vision_as_good. The other four reach insufficient data for other reasons.
- **Effect on fresh_v1 scoring: none expected.**
  - `ground_truth_v2.generate_instances_v2` creates questions only at visits with ≥ 2 earlier visits (`MIN_HISTORY = 2`; the loop starts at `t = 2`), so Part C never presents the triggering condition.
  - Part A scores the parser only.
  - The fresh_v1 report must state this issue.
- **Decision (researcher, option A):**
  - the code freeze is kept;
  - no change is made to `intent_retrieval.py` before or during the fresh_v1 run;
  - it will be fixed after the fresh_v1 run, in a new versioned freeze with its own regression test.
- **Proposed fix (not implemented):** at the start of `retrieve`, return `insufficient_data` when the current position is below 2, which mirrors the spec's common rule. Add a regression test using the four failing Part B fixtures.

## KI-03 — Compound number words are not parsed (found AFTER the one-shot run)

`intent.parse_intent` does not handle hyphenated number words. "twenty-six weeks" was parsed as 6 weeks (F075), and
"fifty-two weeks" as 2 weeks (F091). It was found in the fresh_v1 results (`results/fresh_v1/POST_HOC_ERROR_ANALYSIS.md`).
**Not fixed:** the code freeze stands, and any fix needs a new versioned freeze and a new fresh set.

## KI-02 — Part B harness cannot see extra tied candidates (harness limitation)

The retriever returns a ranking, not a set of acceptable visits. A spurious extra candidate ranked just below the correct block is therefore invisible to the harness (mutant M4). This cannot change Hit@1. It would only matter if acceptable *sets* were reported. Not a retriever defect; recorded for transparency.
