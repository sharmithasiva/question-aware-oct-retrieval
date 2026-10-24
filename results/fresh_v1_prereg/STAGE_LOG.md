# fresh_v1 — stage log (no question text)

> AI-generated preliminary benchmark with limited human review. Not clinician-validated.

| Step | Time (IST, 2026-10-23) | Result |
|---|---|---|
| Pre-check | before freeze | v1 snapshot MD5 OK; v1/v2 frozen artifacts OK; 205/205 tests passed |
| 1. Code freeze | before generation | 42 files hashed in `CODE_FREEZE.sha256` (sha256 `fd06be3aec6a61f6e68353340947cd5efc7441b42807b6fb02802e8e96ad20f6`); global fixed rule `latest_prior`; environment recorded |
| 2. Generator | `questions.jsonl` modified 10:14:39, `generator_labels.jsonl` 10:16:20 | 120 questions, 120 distinct texts, 120 labels, schema OK, quotas exact |
| 3. Annotator | `annotator_labels.jsonl` modified 10:18:36 | 120 labels, schema OK |
| 4. Compare | 10:19 | 2 disagreements (routing only); review sample of 12 drawn (seed 42) |

| 5a. Pre-review check | after step 4 | Code freeze re-verified OK; v1 snapshot and v1/v2 MD5 OK; sealed-file SHA-256 unchanged since step 4; hashes recorded in `SEALED_ARTIFACTS.sha256` |
| 5b. Researcher decision | — | **Annotation will not be re-run.** Proceed with the existing artifacts and document the leakage concern (below). |
| 5c. Human review | guided, item by item | Disagreements: F065 → `generator`, F080 → `generator` (both routing `needs_clarification`). Review sample (12): all `ok`; **wrong = 0** (limit: > 2 stops), ambiguous = 0, drop = 0. `adjudication.csv` filled in: new SHA-256 `470b6f56b6aa3c862fbbb94ee4ad49c0be3fd3a61df5b226ddc8d3bb720e47b7`. The template hash in `SEALED_ARTIFACTS.sha256` is kept as the record of the blank template. All other sealed files are unchanged, and the code freeze re-verified OK. |
| 5d. Parameter policy | before opening | `PROPOSED_CLARIFICATION_01.md` **approved, option (a)**: gold intent parameters come from the frozen type definitions. The frozen `ANALYSIS_PLAN.md` is unchanged. |
| 6. Gold set | after review | Built with `scripts/fresh_v1_build_gold.py`, a new file written after the freeze that only imports frozen code read-only. 120 gold items, 0 excluded (no drops, near-duplicates or dev/v1 overlaps). 118 agreed + 2 adjudicated (`generator`). Expected system outcome: retrieve 70, unavailable variable 18 (incl. 10 `max_leakage`), unsupported image 7, clarify 25. 70 retrieve items carry gold intent parameters; 0 inconsistencies with type definitions. 14 items flagged `developer_viewed`. `gold.jsonl` SHA-256 `b9f9b06ba337629b25f8e7dd860f3f9018bd3c088c971bbbca412740611c846c`; `GOLD_MANIFEST.sha256` (11 files) SHA-256 `c0d984f904fdc35d4e204c9f2253a963c9021e8f25a23765762ee4f5c89a24c7`, verified. Code freeze re-verified OK. |
| — | — | Stopped before Part B and before opening, as instructed. |
| 7. Part B | after step 6, researcher-approved | Independent fixture agent: 48 cases. Independent reference agent: `reference_ops.py`. Harness `scripts/part_b_check.py` (new, outside the freeze). **Answer key = reference on 48/48 fixtures, 16,000/16,000 random and 9,712/9,712 real cases. Retriever:** 44/48 fixtures; it fails only when the current visit has exactly 1 earlier visit (all 664 random and 263 real disagreements). The retriever does not enforce the "≥ 2 earlier visits" rule. **Not fixed** (frozen); per `generate_instances_v2` (`MIN_HISTORY = 2`) this cannot occur in fresh_v1 instances. Mutation: 6/8 detected; M8 (tie order) is equivalent by spec, and M4 is undetectable through the ranking interface. Metamorphic checks: all pass. Both agents made 2 tool calls where 1 was expected; their transcripts are unavailable. Harness bugs fixed during the run: (1) mutant modules were not registered (crash); (2) mutant crashes were not caught; (3) mutation kills were initially counted against cases the unmutated retriever already fails, which inflated the kills. Corrected before reporting. Code freeze re-verified OK. Report: `results/fresh_v1_partB/PART_B_REPORT.md`. |
| — | — | Stopped before opening the sealed set, as instructed. |
| 7b. Defect decision | before opening | Researcher chose **option A**: keep the code freeze. The min-history defect is recorded in `KNOWN_ISSUES.md` (KI-01) as a known issue that cannot affect fresh_v1 scoring. It will be fixed only after the fresh_v1 run, in a new versioned freeze. No code changed; freeze re-verified OK. |
| 7c. Researcher spot-check | before opening | The researcher checked all 8 Part B spot-check fixtures (one per type: B_largest_cst_rise_1, B_largest_bcva_drop_1, B_first_vision_gain_1line_1, B_one_year_ago_1, B_six_months_ago_1, B_max_leakage_1, B_last_thicker_than_now_1, B_last_vision_as_good_1): **all 8 OK**, with no arithmetic or expected-answer errors. |
| — | — | At the final approval gate (protocol §3.3). The sealed set is unopened and no evaluation has been run. |
| 8. Evaluation harness | after Part B, researcher-authorised | Wrote `scripts/fresh_v1_evaluate.py`, a new file outside the code freeze that imports frozen modules read-only. It implements the pre-registered A1 / A2 / C analysis, routing kept separate from retrieval (safeguard 2), and puts the disclaimers and caveats at the top of the report. **Tested in `--dev-test` mode only** (dev paraphrases; train+val patients; no sealed file read; output in `results/fresh_v1_harness_devtest/`, labelled "plumbing test, not results"). Frozen-system Part C numbers reproduce the earlier v3 dev results. **Guards tested:** `--open-sealed` without `--i-have-explicit-approval` refuses; with it but before hashing, it refuses ("HARNESS.sha256 missing"). The sealed run also refuses if the code freeze, gold manifest or harness hash fail, or if `results/fresh_v1/` already exists (one-shot rule). Hashed in `HARNESS.sha256` (SHA-256 `7a898fc069235318e462412aff7b913693f4613708a99eb4d4aac22b83910d3a`; covers `fresh_v1_evaluate.py`, `fresh_v1_build_gold.py`, `fresh_v1.py`; LF line endings, verifies with plain `sha256sum -c`). Re-verified: code freeze OK, gold manifest OK, v1/v2 checksums OK, 205/205 tests pass, `results/fresh_v1/` absent. |
| — | — | At the final approval gate. |
| 9. **One-shot sealed run** | researcher explicitly authorised | Ran `python scripts/fresh_v1_evaluate.py --open-sealed --i-have-explicit-approval` once. Preflight passed: freeze, gold manifest, harness hash, results directory absent. **Primary:** routing accuracy 98/120 = 0.817 [0.738, 0.876]; unsafe-retrieval 4/50 = 0.080 [0.032, 0.188]. **Secondary:** over-refusal 14/70 = 0.200; exact limitation handling 20/25; exact intent 47/70 = 0.671. Part C: see `reports/FRESH_V1_REPORT.md`. Outputs hashed in `RESULTS_MANIFEST.sha256`. **No code was changed afterwards**; freeze and harness re-verified OK. The set is now **exposed**: any parser change requires fresh_v2. |
| 10. Post-hoc documentation | after the run | `results/fresh_v1/POST_HOC_ERROR_ANALYSIS.md` (descriptive, not pre-registered; changes no score). New known issue KI-03 (compound number words) recorded. No code changes. |

**Note on manifest line endings:** Python on Windows writes the manifests with CRLF line endings, so plain `sha256sum -c` reports "No such file" for every line. Verify with Python (`python scripts/fresh_v1.py verify-freeze` for the code freeze), or with `tr -d '\r' < MANIFEST | sha256sum -c -`. Both report OK.

## Integrity notes

### Annotator independence: NOT VERIFIED

The annotator's independence from the generator **could not be verified**, for two reasons:
- **The agent transcripts were unavailable.** The transcript files on disk are empty (0 bytes), so the agents' actions cannot be inspected.
- **Both agents made unexpected extra tool calls.** Each reported 3 tool calls where 2 were expected (generator: 2 writes; annotator: 1 read + 1 write). What the extra calls were cannot be determined.

The annotator was instructed not to open `generator_labels.jsonl`. Whether it complied cannot be established. File last-access times are no help, because they were overwritten by the developer's own counts-only `seal-info` run at 10:18:42.

**This log does not claim that leakage occurred, and it does not claim that independent annotation was verified.**

### Status of the original agreement results: POTENTIALLY COMPROMISED

- The agreement figures in `sealed/fresh_v1/agreement_summary.json` are marked **potentially compromised**: category κ = 1.00; routing κ = 0.97; question type 100 %; answerability 100 %; 2 disagreements.
- They are unusually high, and the annotator's category counts match quotas it was never given. Two explanations cannot be told apart:
  - the questions are prototypical, written to fit the definitions;
  - the annotator saw the generator's labels.
- **Consequences:**
  - agreement must not be cited as evidence of label quality or as inter-annotator reliability;
  - the effective human check is the researcher's review of the 2 disagreements and the 12-item sample;
  - any report produced from fresh_v1 must repeat this caveat next to the AI-generated, not-clinician-validated disclaimer.
- The original files are preserved unchanged (hashes in `SEALED_ARTIFACTS.sha256`).

### Developer access

The developer has not opened:
- `questions.jsonl`;
- either label file;
- `disagreement_report.md`;
- `review_sample.md`.

The only operations on them were counts-, IDs- and hashes-only scripts (`scripts/fresh_v1.py seal-info`, `compare`), plus `grep` for item ID header lines.

### Developer exposure during the guided review (deviation from §3.2, researcher-requested)

At the researcher's request, the developer is displaying the review items, one at a time, to guide the human review. This means the developer now sees the text and labels of up to 14 of the 120 items: F065, F080 and the 12 sample items.
- **Mitigation:** the code freeze (`CODE_FREEZE.sha256`) was recorded before any question existed, and it is re-verified before opening. Any change to the parser, retrieval code, tests or analysis plan voids the run. This exposure therefore cannot feed into the system under test.
- **Reporting:** results for these 14 items are flagged "developer-viewed" in any report.

Items displayed so far: F065, F080 and all 12 sample items (F002, F089, F082, F059, F056, F097, F011, F075, F017,
F014, F060, F118). No other items have been viewed.

### Pending, not implemented

`PROPOSED_CLARIFICATION_01.md` sets out the ground-truth parameter policy. It is **proposed only**; the frozen `ANALYSIS_PLAN.md` is unchanged.
