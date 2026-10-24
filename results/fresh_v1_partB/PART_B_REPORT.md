# fresh_v1 — Part B: retrieval correctness on controlled histories

> **AI-generated preliminary benchmark with limited human review. Not clinician-validated.** Part B measures only
> whether the retrieval code implements the *written definitions*. It is not retrieval performance, and it says
> nothing about clinical validity or research novelty. Three implementations agreeing on a wrong definition would
> still be wrong.

Run: `python scripts/part_b_check.py` (after the code freeze; frozen modules imported read-only; mutants exist in
memory only; `sealed/` never touched). Raw outputs: `partB_summary.json`, `partB_disagreements.jsonl`.

## What was compared

| Implementation | Author | Input |
|---|---|---|
| **Retriever**: `intent_retrieval.retrieve` (frozen) | developer | structured intent (parser bypassed); `max_leakage` tested as a *declared-variable* lookup |
| **Answer key**: `ground_truth_v2.answer_key` (frozen) | developer | DataFrame adapter |
| **Reference**: `tests/reference/reference_ops.py` | independent agent; definitions only; plain Python, no imports | fixture-format dict |
| **Fixtures**: `tests/fixtures/retrieval_cases_fresh_v1.jsonl` | a different independent agent; hand-derived answers with arithmetic rationale | 48 cases, 6 per type |

**Retriever scoring:** the retriever returns a ranking, so it counts as correct when the outcome matches and its top-k block (k = number of acceptable visits) equals the acceptable set.

## Results

**Fixtures** (cases passed / 48 against the hand-specified answers):

| Type | Retriever | Answer key | Reference |
|---|---|---|---|
| largest_cst_rise | 6/6 | 6/6 | 6/6 |
| largest_bcva_drop | 6/6 | 6/6 | 6/6 |
| first_vision_gain_1line | **5/6** | 6/6 | 6/6 |
| one_year_ago | 6/6 | 6/6 | 6/6 |
| six_months_ago | 6/6 | 6/6 | 6/6 |
| max_leakage | **5/6** | 6/6 | 6/6 |
| last_thicker_than_now | **5/6** | 6/6 | 6/6 |
| last_vision_as_good | **5/6** | 6/6 | 6/6 |
| **Total** | **44/48** | **48/48** | **48/48** |

**Three-way agreement** (no hand answers; the reference is the comparator):

| Source | Cases | Answer key vs reference | Retriever vs reference |
|---|---|---|---|
| Seeded random histories (ties, NaNs, irregular weeks) | 16,000 (2,000/type) | **16,000 / 16,000** | 15,336 / 16,000 |
| Real train+val histories (every current visit t ≥ 1) | 9,712 (1,214/type) | **9,712 / 9,712** | 9,449 / 9,712 |

## Finding: the retriever does not enforce the "at least 2 earlier visits" rule

- **Every** retriever disagreement occurs when the current visit has exactly **one** earlier visit: 664 of 664 random, 263 of 263 real, and the 4 failing fixtures.
- In that situation:
  - the spec, the answer key and the reference all return `insufficient_data`;
  - the retriever answers anyway: `visits` or `no_qualifying_visit`.
- Affected types: first one-line gain, max leakage, last thicker than now, last vision as good.
- The other four types agree only incidentally:
  - a consecutive change needs two earlier visits anyway;
  - the time types fail the week rule at such early visits.
- **Impact on the planned evaluation: none for fresh_v1 scoring, as far as the code shows.**
  - `generate_instances_v2` only creates benchmark questions at visits with ≥ 2 earlier visits (`MIN_HISTORY = 2`, loop starting at `t = 2`), so Part C never presents a 1-earlier-visit case.
  - Part A scores the parser only.
  - It *is* a spec-conformance defect for any use outside the benchmark.
- **Not fixed.** `intent_retrieval.py` is in the code freeze, and any change voids the fresh_v1 run. See the decision below.

Outside that one condition, the retriever agreed with the reference on **every** case: all 15,336 random and all 9,449 real cases with ≥ 2 earlier visits. That includes ties, missing values and tolerance boundaries.

## Mutation tests (planted bugs; scored only on cases the unmutated retriever gets right)

| Mutant | New fixture failures | New random failures | Detected? |
|---|---|---|---|
| M1 strict `>` becomes `≥` | 1 | 398 | yes |
| M2 tolerance +1 week | 2 | 59 | yes |
| M3 current visit allowed as a candidate | 17 | 4,850 (1,640 crashes) | yes |
| M4 baseline given a "change" | 0 | 0 | **no** (see below) |
| M5 threshold `≥` becomes `>` | 1 | 303 | yes |
| M6 "first" becomes "latest" | 3 | 610 | yes |
| M7 target week off by one | 2 | 295 | yes |
| M8 tie-break reversed | 0 | 0 | **no**, by design |

**Why two mutants were not detected:**
- **M8** reverses the order of *tied* visits. The spec makes all tied visits acceptable, so tie order is not part of the definition. This mutant is equivalent with respect to the spec.
- **M4** copies visit 1's change onto the baseline. The baseline can then tie with visit 1, but the recency tie-break always ranks visit 1 first, so the top-k block is unchanged. This exposes a **harness limitation**: because the retriever returns a ranking rather than an acceptable set, an extra spurious candidate ranked just below the correct block cannot be seen. The same limitation means it cannot change Hit@1.

**Note on scoring:** an earlier run scored mutants against all cases, including the ones the unmutated retriever already fails. That inflated the kill counts (e.g. M8 appeared "killed"). It was corrected before this report, and the earlier numbers were discarded.

## Metamorphic checks (16,000 random cases each)

| Check | Retriever | Answer key | Reference |
|---|---|---|---|
| Add 37 µm to every CST value (incl. current) → same answer | 16,000/16,000 | 16,000/16,000 | 16,000/16,000 |
| Append 1–3 extreme visits *after* the current one → same answer (no future leakage) | 16,000/16,000 | n/a | n/a |

Renaming visit IDs is a no-op here, because all three implementations work on positions; it was not tested separately.

## Reference author's interpretation choices (reported by the agent, verbatim in substance)

1. The candidates are positions 0 … current−1. Position 0 can be an answer except where the definition requires i ≥ 1 (types 1–3).
2. The "≥ 2 earlier visits" check runs first.
3. A change counts only if both i and i−1 are earlier visits. The change into the current visit never counts.
4. Type 3 skips missing BCVA values.
5. For types 4–6, the cohort check (TREX → insufficient) runs before the other checks.
6. For types 4–5, a current week of exactly 52/26 is allowed (target 0), and a distance ≤ 4 qualifies.
7. Ties and the 4-week threshold use a 1e-9 floating-point tolerance.
8. Types 7–8 skip missing values, and "latest" means the highest position.
9. Type 6 skips missing leakage.
10. Position equals list index.
11. An unknown type raises an error.

None of these conflicts with `docs/PHASE1_V2_QUESTION_SPEC.md`.

## Independence caveats

- The fixture agent and the reference agent each made **2** tool calls where 1 was expected (a single file write). Their transcripts are unavailable, so the extra calls cannot be identified.
- Both were instructed not to read any file, but compliance cannot be verified. Their 100 % agreement with the answer key is consistent with either:
  - correct independent work from the same spec;
  - exposure to project code.
- The fixture *coverage* tags are self-reported by the fixture agent. Ties are "not applicable" for the "first" and "latest" types (by definition), and "no answer" is not applicable to max_leakage when leakage values exist.

## Spot-check for the researcher (one fixture per type; check the arithmetic in each rationale)

| Case | Data (positions 0 → current) | Expected | Rationale |
|---|---|---|---|
| B_largest_cst_rise_1 (TREX) | CST 400, 450, 380, 430, 410, **350 (current)** | visits {1, 3} | changes +50, −70, +50, −20 → tie at +50 |
| B_largest_bcva_drop_1 (TREX) | BCVA 70, 62, 68, 60, 65, **55** | visits {1, 3} | drops 8, −6, 8, −5 → tie at 8 |
| B_first_vision_gain_1line_1 (TREX) | BCVA 60, 64, 65, 70, **72** | visits {2} | threshold 65; +4 at 1 not enough; +5 at 2 qualifies |
| B_one_year_ago_1 (Prime) | weeks 0, 12, 24, 36, 44, 56, **100** | visits {4} | target 48; distances 48, 36, 24, 12, **4**, 8 → exactly 4 allowed |
| B_six_months_ago_1 (Prime) | weeks 0, 8, 16, 24, **46** | visits {2, 3} | target 20; distances 20, 12, 4, 4 → tie |
| B_max_leakage_1 (Prime) | leakage 0.42, 0.65, 0.30, 0.65, 0.20, **0.90** | visits {1, 3} | max 0.65 tie; current 0.90 not used |
| B_last_thicker_than_now_1 (TREX) | CST 450, 380, 360, 340, **360** | visits {1} | strictly > 360: positions 0 and 1; 360 (equal) excluded; latest = 1 |
| B_last_vision_as_good_1 (TREX) | BCVA 70, 65, 60, 55, **60** | visits {2} | ≥ 60: positions 0, 1, 2; latest = 2 (equal counts) |

The developer also checked these 8 by hand and found no arithmetic errors. The protocol still asks for the researcher's own check.

## Decision needed (min-history defect)

- **(A) Recommended:** keep the code freeze. Record the defect as a known spec-conformance issue that the code shows cannot affect fresh_v1 scoring. Fix it only after the fresh_v1 run, in a new versioned freeze.
- **(B)** Fix now. This changes a frozen file, which voids fresh_v1 under the protocol and requires a new freeze and a new sealed question set (fresh_v2).
