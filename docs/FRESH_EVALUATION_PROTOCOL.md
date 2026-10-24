# Fresh Evaluation Protocol — AI-assisted preliminary benchmark (revision 2, for approval)

Status: **proposal only.** No code has been changed, no fresh questions have been generated, and no evaluation has
been run. This revision replaces the earlier human-authored design (about 384 questions) with a practical AI-assisted
design of **120 questions** and limited human review.

> **Mandatory disclaimer (to be repeated at the top of every results report produced under this protocol).**
> This is an **AI-generated preliminary benchmark with limited human review**. The questions and their labels were
> produced by language-model agents. A human researcher reviewed only the disagreements and a small sample.
> The results are **not clinician-validated** and must not be described as such. They do not establish research
> novelty or clinical usefulness.

---

## 0. What is preserved and what is already "spent"

| Artifact | Status under this protocol |
|---|---|
| `results/v1_snapshot/` (MD5 manifest), `results/v2/`, `results/v3_dev/`, all reports, `data/processed/` | **Preserved; never overwritten.** Checked against `results/v1_snapshot/MANIFEST.md5` and `results/v3_dev/frozen_artifacts.md5` before and after every step. |
| Dev paraphrases (55), v1 templates (24), v1 held-out paraphrases (9) | **Used for parser development.** Every existing dev score is in-sample (`docs/DEV_RESULTS_AUDIT.md` §5). |
| Retired locked paraphrases (55, `tests/data/paraphrases_v2_locked.jsonl`) | **Developer-exposed and retired.** Not opened, not run, and not used in any check (see decision D3). |
| `results/v3_dev/selected_baselines.json` | Per-type fixed rules chosen on dev; frozen. Used unchanged in Part C. |

All new outputs go to new locations only: `sealed/fresh_v1/`, `results/fresh_v1_prereg/`, `results/fresh_v1/`.

---

## 1. Fresh question set (Part A: parser generalisation)

### 1.1 Composition: 120 distinct formulations

| Category | n | Gold routing (what the system *should* do) |
|---|---|---|
| **Supported**: 10 for each of the 8 benchmark types (§1.2) | 80 | Retrieve, except `max_leakage`, whose correct routing is *data unavailable to the retriever* (decision D1) |
| **Plausible but unsupported**: realistic questions about CST/BCVA/timing that need an operation the system does not define (averages, trends, counts, intervals, predictions, sets of visits, comparisons across eyes) | 15 | Do not retrieve: *unsupported operation* |
| **Images or unavailable information**: need OCT image findings, or data not in the retriever's memory (injections, severity grades, demographics, lab values) | 15 | Do not retrieve: *requires images* or *data unavailable* |
| **Ambiguous, underspecified or off-topic** | 10 | Do not retrieve: *needs clarification* |

The report always states the number of **distinct formulations** per category, after exclusions, separately from
any patient-instance counts.

### 1.2 The 8 supported types (plain-English definitions given to both agents)

The same text goes into both agent briefs (Appendix A and B):

1. **Largest thickness rise**: the earlier visit at which CST increased the most compared with the visit immediately before it.
2. **Largest vision drop**: the earlier visit at which BCVA decreased the most compared with the visit immediately before it.
3. **First one-line vision gain**: the first visit after the baseline visit at which BCVA was at least 5 ETDRS letters (one line) above baseline.
4. **About one year ago**: the earlier visit closest to 52 weeks before the current visit, if one lies within ±4 weeks. Only the cohort with recorded study weeks.
5. **About six months ago**: as above, with 26 weeks.
6. **Greatest angiographic leakage**: the earlier visit with the highest fluorescein-angiography leakage index. This value exists in the dataset for one cohort but is **not available to the retrieval system**.
7. **Last time thicker than now**: the most recent earlier visit whose CST was strictly greater than the current CST.
8. **Last time vision as good as now**: the most recent earlier visit whose BCVA was greater than or equal to the current BCVA.

### 1.3 Dataset capability description (given to both agents)

- **Per visit, the retriever can see:** visit order, CST (µm, higher = thicker), BCVA (ETDRS letters, higher = better), and the study week for one cohort only.
- **Exists in the dataset but is NOT available to the retriever:** leakage index, injection records, retinopathy severity grade (DRSS), demographics and lab values.
- **Images:** OCT images exist but are not used.
- **Biomarker labels** (fluid, layer disruption and similar) exist only for the first and last visit of each eye.
- **Other operations the system also supports** (needed so that "unsupported" questions are genuinely unsupported):
  - the baseline visit;
  - the previous visit;
  - the visit with the highest or lowest CST or BCVA;
  - the visit with CST closest to the current value;
  - simple variants of the 8 types above that change only the direction, time span or threshold.

### 1.4 Label schema (identical for both agents)

`id`, `category` (4 values, §1.1), `intended_type` (one of the 8, or `none`),
`routing` ∈ {`retrieve`, `data_unavailable`, `requires_images`, `unsupported_operation`, `needs_clarification`},
`required_fields` ⊆ {cst, bcva, visit_order, week, leakage_index, injection, severity_grade, oct_images,
biomarker_labels, other},
`answerability` ∈ {`metadata_answerable`, `needs_unavailable_data`, `needs_images`, `not_answerable_as_posed`},
optional `parameters` {target, direction, time_offset_weeks, threshold, comparator},
and a one-sentence `rationale`.

**Mapping to system outputs.** The system has four outcomes for a question (retrieve, unavailable variable, unsupported image, clarify), so:

| Gold routing | Accepted system outcome |
|---|---|
| `retrieve` | retrieve |
| `data_unavailable` | unavailable variable |
| `requires_images` | unsupported image |
| `needs_clarification` | clarify |
| `unsupported_operation` | clarify (the system has no separate label for this) |

This mapping is fixed now, before any question exists.

---

## 2. Exact workflow

| Step | Who | Action | Output (location) | Developer may read? |
|---|---|---|---|---|
| 0 | You | Approve this protocol, the agent briefs (Appendices A–D) and decisions D1–D4 | — | — |
| 1 | Developer | **Code freeze and pre-registration** (§3) | `results/fresh_v1_prereg/` (hashes, analysis plan copy) | yes |
| 2 | **Generator agent** (fresh agent, Appendix A) | Writes 120 questions **directly to disk**. Returns only "N written + SHA-256" to the developer. | `sealed/fresh_v1/questions.jsonl` (`id`, `text`); `sealed/fresh_v1/generator_labels.jsonl` | **no** (hash and counts only) |
| 3 | **Annotator agent** (a different fresh agent, Appendix B) | Reads only `questions.jsonl`, never the generator labels. Labels each item. Returns only "N labelled + SHA-256". | `sealed/fresh_v1/annotator_labels.jsonl` | **no** |
| 4 | Developer (script) | Compares the two label sets, with output split by audience (§2.1) | `sealed/fresh_v1/agreement_summary.json` (counts and IDs only); `sealed/fresh_v1/disagreement_report.md` and `review_sample.md` (contain text) | summary **yes**; reports **no** |
| 5 | **You** (about 30–45 min) | Review every disagreement plus a 12-item sample (§2.2). Record decisions **by ID only**. | `sealed/fresh_v1/adjudication.csv` (`id`, `decision`) | yes (no text) |
| 6 | Developer (script) | Build the final gold set: agreed items + your decisions; drop excluded items; hash everything | `sealed/fresh_v1/gold.jsonl` + `sealed/fresh_v1/MANIFEST.sha256`; counts summary | counts and hashes only |
| 7 | Independent agents + developer | **Part B** retrieval-correctness tests (§4). This can run in parallel with steps 2–6, since it uses no natural language. | `tests/fixtures/`, `tests/reference/` | yes |
| 8 | **You** | Explicit written approval to open (§3.3) | — | — |
| 9 | Developer | **One-shot** evaluation of Parts A and C (§5) | `results/fresh_v1/`, `reports/FRESH_V1_REPORT.md` | yes, after opening |

### 2.1 Comparison and disagreement report (step 4)

- **Agreement statistics:** per field, % agreement; Cohen's κ for `category`, `routing` and `intended_type`; exact-match rate on `parameters`.
- **Disagreement report:** every item where `category`, `routing`, `intended_type` or `answerability` differ, with the text and both label sets side by side.
- **Automatic checks** (counts and IDs only):
  - near-duplicates within the set (normalised similarity ≥ 0.9);
  - overlap with the dev paraphrases and v1 templates, which are excluded if found;
  - category quotas met.
- **Agreement is not proof of correctness.** Both agents come from the same model family as the developer, so they share linguistic and conceptual priors. Their agreement is inflated relative to independent humans, and it says nothing about clinical correctness. The report states this.

### 2.2 Your review (step 5)

- **Every disagreement.** For each one, choose: `generator`, `annotator`, `ambiguous` (moves the item to the clarification category), or `drop`.
- **Random sample of 12 agreed items** (seeded, stratified: one per supported type + one per other category + one random). Mark `ok`, `ambiguous`, `wrong` or `drop`.
- **Escalation rule:** if more than 2 of the 12 sample items are judged `wrong`, the agreed labels are not trusted. The labelling step is redone with a revised brief, and the set is not opened.
- Decisions are recorded by ID only, so they can be shared with the developer without revealing any text.

### 2.3 Exclusions

Items are dropped if they are `drop` decisions, near-duplicates, or overlap with development text. The final count per
category is reported. If any supported type falls below 8 items, a top-up batch is generated by the same generator
brief and goes through steps 3–5 again.

---

## 3. Keeping the set separate, and when it may be opened

### 3.1 Code freeze and pre-registration (step 1, before any question exists)

- **SHA-256 manifest** of `src/oct_retrieval/*.py`, `tests/*.py`, `results/v3_dev/selected_baselines.json` and this protocol file.
- **The global fixed rule for Part C** (§5.2) is selected on dev data and frozen at this step.
- **Analysis plan:** primary and secondary endpoints (§5.3), the routing mapping (§1.4) and the exclusion rules, copied into `results/fresh_v1_prereg/ANALYSIS_PLAN.md` and hashed.
- **Full test suite** passes, and the v1/v2 checksums verify.

Freezing *before* generation means the developer cannot adjust the parser in response to the fresh set, even accidentally.

### 3.2 Sealing rules

- `sealed/fresh_v1/` is never printed, read or summarised by the developer except through scripts that output **only counts, IDs and hashes**.
- Agents are instructed to return only counts and hashes, never question text.
- The developer does not open agent transcripts.
- **Limit:** this is an honour-system seal, because the developer has file-system access. To strengthen it, you may move `sealed/fresh_v1/` outside the project folder after step 6 and return it only at step 9.

### 3.3 Opening conditions (all required, in order)

1. The code-freeze manifest is unchanged (any change voids the run).
2. All tests pass, and the v1/v2/v3 checksums verify.
3. The gold manifest is hashed after your adjudication.
4. The Part B report is complete.
5. You give explicit approval in chat.

The evaluation runs **once**. Afterwards the set is "exposed": any parser change requires a new set (`fresh_v2`)
generated under the same workflow.

---

## 4. Part B: retrieval correctness on controlled histories (no natural language)

Structured question specifications are passed straight to the retrieval function, so the parser is not involved.

1. **Fixture agent** (Appendix C). Writes 48 controlled cases (6 per supported type) from the definitions and a fixture schema only, without seeing any code. Each case contains:
   - the visit history;
   - the current visit;
   - the expected outcome (visit set, *no qualifying visit*, or *insufficient data*);
   - a step-by-step arithmetic rationale.

   **Required coverage per type:**
   - a tie;
   - a missing value;
   - a boundary (±4 vs 5 weeks; +4 vs +5 letters; equal-to-current for > vs ≥);
   - a no-answer case;
   - a case where the latest visit is wrong;
   - a case where the earliest visit is wrong.
2. **Reference-implementation agent** (Appendix D, a different agent). Writes a plain-Python reference (lists and loops, no numpy) from the definitions only. It must list every ambiguity it had to decide, and each one is brought to you.
3. **Three-way check:** the existing retriever, the reference and the existing answer key are compared on:
   - the 48 fixtures;
   - about 2,000 seeded random histories per type, rich in ties and missing values;
   - all real train/val histories.

   Every disagreement is reported as a minimal case.
4. **Mutation and metamorphic tests**, written by the developer; these encode no expected answers.
   - Mutations: `>`↔`≥`, tolerance off by one, the current visit allowed as a candidate, baseline allowed as a change visit, tie-break reversed. The report shows which mutants the tests catch.
   - Metamorphic checks: adding a constant to CST, appending visits after the current one, renaming IDs.
5. **Your spot-check:** 8 fixtures (one per type), checking the arithmetic in the rationale. About 10 minutes.

**Reporting:** cases passed / cases, per type. This measures **implementation correctness against the written definitions only**. It is not retrieval performance, and three agreeing implementations of a wrong definition are still wrong.

**How shared mistakes are prevented:**
- the expected answers are written by an agent that has not seen the retriever or the answer key;
- the reference implementation uses a different paradigm and has a different author;
- the written definitions are the only shared input, and every ambiguity is logged and resolved by you, not silently;
- mutation testing measures whether the tests can detect typical errors.

---

## 5. Evaluation after opening (step 9)

### 5.1 Part A metrics (unit = distinct formulation)

- **Routing accuracy** over all 120 items (after exclusions), using the §1.4 mapping, with a confusion matrix.
- **Unsafe-retrieval rate:** the share of items whose gold routing is *not* `retrieve` (about 50 items, including the 10 leakage items) for which the system retrieves.
- **Over-refusal rate:** the share of gold-`retrieve` items (about 70) that the system does not retrieve.
- **Exact-intent and per-field accuracy** on gold-`retrieve` items, compared with the parser's structured output: operation, target, direction, offset, threshold, comparator.
- **Uncertainty:** Wilson 95 % intervals. With 10 items per type, a single type's interval is wide (e.g. 9/10 → 0.60–0.98), so **per-type results are descriptive only**. Inference uses the pooled categories.

### 5.2 Part C: end-to-end comparison on test patients

- **Instances:** test-split patients only (18 patients; 7–8 for the Prime-only types). Gold-`retrieve` fresh questions are rotated per type, and the answer key is unchanged.
- **Systems:**
  - frozen `intent_v3`;
  - frozen v1 `combined` and `rule_router`;
  - **(a) per-type fixed rule** from `selected_baselines.json`. This baseline is given the true question type (an oracle advantage), so it is an *upper reference*, not a fair competitor.
  - **(b) single global fixed rule**, chosen on dev at step 1 (the fair, type-agnostic baseline).
- **Paired comparison:** per type, the Hit@1 difference vs (b) on the same instances, with a 95 % patient-cluster bootstrap CI. MRR is secondary.
- **Abstention:** insufficient-evidence questions are reported separately (correct abstention vs false abstention). Fixed rules cannot abstain.
- **Decomposition:** end-to-end Hit@1 is shown next to the parse accuracy of the same texts, to show where errors come from.

### 5.3 Pre-registered endpoints

- **Primary:**
  - (1) pooled routing accuracy;
  - (2) unsafe-retrieval rate.
- **Secondary:**
  - exact-intent accuracy on gold-`retrieve` items;
  - over-refusal rate;
  - Part C Hit@1 difference vs the global fixed rule.
- Everything else is descriptive.

### 5.4 Claims allowed and not allowed

- **Allowed:** "On an AI-generated preliminary set of N distinct questions with limited human review, the frozen parser routed X % correctly and retrieved on Y % of questions it should have refused."
- **Not allowed:**
  - clinician validation;
  - clinical usefulness;
  - research novelty;
  - visual reasoning;
  - any reading of perfect Part B results as evidence of retrieval performance.
- **Small samples:** 18 test patients, and 10 questions per type, give wide intervals. This is stated in the report.

---

## 6. Known limitations of this design

1. Generator, annotator and developer are all the same model family. Their agreement and the parser's success may share blind spots that human writing would expose.
2. The definitions given to the agents were written by the developer who wrote the parser, so a definition-level bias is unavoidable.
3. Human review covers only the disagreements and 12 sampled items. Systematic labelling errors that both agents share may go undetected. The escalation rule (§2.2) limits but does not remove this risk.
4. With 120 questions, per-type estimates rest on 10 formulations each.
5. Overlap with the retired locked set is not checked, because checking would require opening it (see D3).
6. The seal is procedural, not technical.

---

## 7. Decisions needed before step 1

- **D1 — `max_leakage` among the 80 supported questions.**
  - (a) *Default:* keep it as one of the 8 types. Gold routing = `data_unavailable`, so the 10 items count toward the should-not-retrieve set.
  - (b) Replace it with a retrievable v1 type, e.g. "visit with the highest CST".
- **D2 — Review sample size:** 12 items (default) or larger.
- **D3 — Retired locked set:** leave it unchecked (default; you asked that it not be reopened). Alternatively, allow a script that reads it only to report the *count* of near-duplicates with the fresh set, never the text.
- **D4 — Sealed folder:** keep `sealed/fresh_v1/` in the project (default), or move it outside the project between steps 6 and 9 for a stronger seal.

---

## Appendix A — Generator agent brief (exact text to be used)

> You are writing evaluation questions for a research benchmark about retrieving earlier clinic visits of one eye from
> its visit history. Do not read, open or search any files or directories, and use no tools except writing the two
> output files named below. You have not seen, and must not try to find, any software, keyword lists or earlier
> example questions.
>
> [§1.3 dataset capability description inserted verbatim]
> [§1.2 definitions of the 8 supported types inserted verbatim]
>
> Write exactly 120 distinct questions, as a clinician, researcher or patient might ask them while looking at the
> current visit:
> - 10 per supported type (80);
> - 15 plausible questions about CST, BCVA or visit timing that need a *different kind* of computation from all
>   supported operations (for example an average, a trend, a count, a time interval, a prediction, a set of several
>   visits, or a comparison with the other eye). Do not merely change the direction, time span or threshold of a
>   supported type;
> - 15 that need OCT image findings or information not available to the retriever;
> - 10 that are ambiguous (two reasonable readings), missing a needed detail, or off-topic.
>
> Vary register (clinician shorthand, full sentences, lay language, requests embedded in longer instructions),
> syntax, length, and number format (digits, words, units). Do not reuse sentence templates across items.
>
> Write `sealed/fresh_v1/questions.jsonl` with one JSON object per line: {"id": "F001", "text": "..."}, ids F001–F120
> in random order of category. Write `sealed/fresh_v1/generator_labels.jsonl` with the label schema
> [§1.4 inserted verbatim] for each id.
>
> Your final reply must contain ONLY the number of items written and the SHA-256 of each file. Do not include any
> question text in your reply.

## Appendix B — Annotator agent brief (exact text to be used)

> You are labelling evaluation questions. Read ONLY `sealed/fresh_v1/questions.jsonl`. Do not open any other file,
> including any other file in `sealed/`, and do not search the file system. [§1.3 and §1.2 inserted verbatim]
> For each question, assign the label schema [§1.4 inserted verbatim] based only on the question text and the
> definitions above. If a question has two reasonable readings, label it `needs_clarification` and say why.
> Write `sealed/fresh_v1/annotator_labels.jsonl`. Your final reply must contain ONLY the number of items labelled
> and the file's SHA-256, with no question text.

## Appendix C — Fixture agent brief (summary)

> From the §1.2 definitions and a fixture schema only (no code, no files), write 48 controlled visit histories, 6 per
> type, with the required coverage in §4.1, the expected outcome and a step-by-step arithmetic rationale. Write
> them to `tests/fixtures/retrieval_cases_fresh_v1.jsonl`.

## Appendix D — Reference-implementation agent brief (summary)

> From the §1.2 definitions only (no access to project code), write `tests/reference/reference_ops.py`: one plain
> Python function per type, using lists and loops, no numpy, returning (outcome, set of acceptable positions). List
> every interpretation choice you had to make in a docstring at the top of the file.
