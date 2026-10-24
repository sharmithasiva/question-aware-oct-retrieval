# Audit of the Phase 1.3 development results (`reports/RETRIEVAL_REPORT_V3_DEV.md`)

Scope: development data only (`train` + `val` patients, dev paraphrases). No code was changed and no locked file
was read for this audit. The verification script is a scratch file that reads only `results/v3_dev/*`,
`data/processed/episodes.csv` and `data/processed/heldout_prime.csv` (the latter only to display the leakage
values in the `max_leakage` example below).

**Bottom line.**
- The 1.000 scores are computed correctly. They come from real visit selection scored against the answer key's visit IDs, and no question type label or held-out value leaks into the retriever.
- They are **circular by design**. The retriever's operations implement the same frozen definitions as the answer key, and the scoring text is the text the parser was built on.
- So they establish **implementation correctness and coverage of seen phrasing**, not retrieval performance.

---

## 1. Which function computes `intent_v3 Hit@1`

`run_new_system(instances, memories)` in `src/oct_retrieval/evaluate_v3_dev.py` (lines 81–103):

```python
q, res = answer(inst.question, mem, t)                      # parse + retrieve, from the question TEXT only
ranked = [mem.visit_ids[i] for i in res.ranked]             # positions -> visit IDs
assert inst.current_visit_id not in ranked
rel = set(inst.relevant_visit_ids.split(";")) if inst.relevant_visit_ids else set()
...
if inst.answerable:
    ok = res.status == RETRIEVED
    row.update({"hit@1": hit_at_k(ranked, rel, 1) if ok else 0.0, ...})
```

- **Inputs:**
  - `instances`: rows from `ground_truth_v2.generate_instances_v2`, with `question`, `current_visit_index`, `relevant_visit_ids` and `answerable`;
  - `memories`: one `VisitMemory` per eye, from `intent_retrieval.build_memories`.
- **Output:** one row per instance with `status`, `ranked_visit_ids` and the metrics.
- `hit_at_k` (`metrics.py`) returns 1.0 if any of the top-k visit IDs is in `rel`.
- The per-type figure is `eye_macro(d, METRICS)` (mean within eye, then across eyes), inside `retrieval_table`.

## 2. Is the question type passed to the retriever?

**No.** `answer(question, mem, t)` (`intent_retrieval.py`, line 182) receives only the question text, the eye's memory and the current position. It calls `intent.parse_intent(question)` and then `retrieve(intent, mem, t)`.
- `inst.qtype` is recorded in the output row but never passed in.
- Each dev question text maps to exactly one operation across all instances (verified: max 1 distinct operation per text). The intent therefore depends only on the text, not on the instance or its label.

## 3. Is Hit@1 visit-ID agreement with the independent key? How are ties handled?

**Yes.** `rel` holds the visit IDs produced by `ground_truth_v2.answer_key`, which `intent.py` and `intent_retrieval.py` do not import (enforced by `test_new_retriever_does_not_import_answer_key`). The retriever's top visit ID is compared against that set.

**Ties:** all tied visits are acceptable. Hit@1 = 1 if the top visit is any of them. Recall@1 = 1/|acceptable| in that case. The retriever ranks tied candidates together, with the most recent first (`retrievers._order`, reused).

**Verification:**
- recomputing Hit@1 from `intent_v3_outcomes_dev.csv` and `instances_dev.csv` gives **0 disagreements** with the stored values;
- every multi-answer instance has its top visit in the acceptable set: `six_months_ago` 220, `largest_bcva_drop` 117, `largest_cst_rise` 44, `one_year_ago` 1.

## 4. Why exactly 1.000 on seven types

Two things combine:
1. **Parsing.** Every dev paraphrase (5 per type) parses to its gold intent. The parser was developed against exactly these sentences, plus v1 templates and generic tests.
2. **Definitional coupling.** Given the correct intent, `retrieve` computes the same mathematical quantity the answer key computes, from the same visible data:

| Type | Retriever operation (`intent_retrieval.retrieve`) | Answer key (`ground_truth_v2.answer_key`) |
|---|---|---|
| `largest_cst_rise` / `largest_bcva_drop` | rank by `np.diff` of CST (or −BCVA); `no_qualifying_visit` if max ≤ 0 | argmax of the same consecutive change; IE if ≤ 0 |
| `first_vision_gain_1line` | first i ≥ 1 with BCVA_i − BCVA_0 ≥ parsed threshold (5) | first i ≥ 1 with BCVA_i ≥ BCVA_0 + 5 |
| `one_year_ago` / `six_months_ago` | nearest week to current − parsed offset, within `TEMPORAL_TOLERANCE_WEEKS` = 4 | same, `WEEK_TOLERANCE` = 4 |
| `last_thicker_than_now` / `last_vision_as_good` | most recent i with CST > current / BCVA ≥ current | same |

The answer key does **not** influence selection at run time: no import, the qtype is not passed, and memory fields are whitelisted. But the retriever was **designed from the same frozen definitions**, as the Phase 1.3 brief requested ("based on their existing frozen definitions"). The same parameters appear in both (4-week tolerance, 5-letter line).

So once parsing succeeds, a correct implementation **must** score 1.000. The result is a consistency check between two implementations, both written by the same developer from the same spec. They agree on 5,155 answerable (excluding `max_leakage`) and 981 insufficient-evidence dev questions. That catches implementation bugs. It cannot catch a shared misreading of the spec.

## 5. Overlap between development and scoring data

| Component | Developed on | Scored on (dev report) | Overlap |
|---|---|---|---|
| Parser lexicon / rules | 45 dev paraphrases + 10 dev unsupported, v1 templates and held-out, developer-written generic sentences | the same 45 + 10 dev paraphrases | **100 %** |
| Retrieval operations | written from `docs/PHASE1_V2_QUESTION_SPEC.md`; not fitted to any patient data | train + val patients | no fitted parameters, but the definitions are shared with the key |
| Fixed-rule baseline selection | dev Hit@1 (`selected_baselines.json`) | the same dev data | 100 % (selection bias favours the baseline slightly) |
| Combined α = 0.05 | v1 train split, v1 question types | train + val, v2 types | partial (train patients) |
| Locked paraphrases (55) | not used | not used | disjoint from dev (`test_paraphrases_frozen_and_disjoint`) |

**Effective sample size.** Each type's hundreds of instances use only **5 distinct question texts**, so parser evidence per type is n = 5. The patient bootstrap CI [1.000, 1.000] is degenerate: it varies patients, not wording.

## 6. Why `max_leakage` = 0.000

The retriever **abstains**; it neither returns a wrong visit nor sees the value.
- `parse_intent` routes leakage wording to `task = unavailable_variable` (stage 2, `VARIABLE_TERMS`), because `leakage_index` is not in `MEMORY_VARIABLES`.
- `retrieve` then returns `RetrievalResult(UNAVAILABLE)` with an empty ranking.
- `run_new_system` scores abstention on an answerable question as 0 (`ok = False`).
- All 459 dev instances have status `unavailable_variable`.
- The key's answer exists, e.g. `02-010_W0` (leakage 1.97) in the example below, but the system correctly reports that it cannot see leakage.

## 7. What the frozen `combined` column measures, and is the comparison fair?

`combined` is `parse_question` (v1 keyword parser) plus `retrievers.Combined(α = 0.05)`, run through `evaluate.run_retriever`:
- same instances;
- same question texts;
- same answer key;
- identical denominators per type (verified, e.g. 907 vs 907 for `largest_cst_rise`).

It measures **what the previous system does** on these questions.
- **Fair** as a before/after reference.
- **Not a fair method comparison.** Combined has no operation for any v2 type and no access to visit weeks, so the gap measures missing capability rather than ranking quality.
- The more meaningful reference is the **dev-selected fixed rule**. Even that is optimistic for the baseline, because it was selected on the same dev data.

## 8. One example per type (actual dev instances)

| Type (eye, split) | Question (dev) | Relevant history | Selected | Key | Why correct |
|---|---|---|---|---|---|
| `largest_cst_rise` (eye 1, train) | "Max sequential rise in OCT central subfield thickness: which earlier visit was that?" | CST V1–V6: 575, 367, 294, 313, 420, 331; current V7 283 | V5 | V5 | Consecutive changes −208, −73, +19, **+107**, −89; the largest rise is at V5. Latest prior (V6) would be wrong. |
| `largest_bcva_drop` (eye 1, train) | "Identify the prior exam showing the steepest single-step fall in ETDRS letter score." | BCVA V1–V8: 61, 63, 69, 70, 73, 78, 75, 79; current V9 (76) | V7 | V7 | Consecutive changes +2, +6, +1, +3, +5, **−3**, +4. The only drop in history is at V7. |
| `first_vision_gain_1line` (eye 1, train) | "Find the earliest post-baseline exam reaching a one-line improvement from the initial BCVA." | BCVA V1 61, V2 63, V3 69, V4 70; current V5 | V3 | V3 | +2 at V2 is below 5 letters; +8 at V3 is the first ≥ 5. |
| `one_year_ago` (eye 60, val) | "Show me the visit from about a year ago." | weeks W0…W48 every 4; current W52 | W0 | W0 | Target week 52 − 52 = 0; W0 is at distance 0. Latest prior (W48) would be wrong. |
| `six_months_ago` (eye 57, val) | "Compare with the 26-weeks-prior visit." | weeks W0, W4, …, W24; current W28 | W4 | W0; W4 | Target 2: W0 and W4 are both 2 weeks away (tie, both acceptable); top = W4 (recency tie-break). |
| `max_leakage` (eye 57, val) | "Find the prior exam where angiographic leakage was greatest." | leakage (hidden): W0 1.97, W4 n/a, W8 0.84, W12 0.83; current W16 | **none** (abstained) | W0 | Counted **incorrect** (Hit@1 = 0): status `unavailable_variable`. |
| `last_thicker_than_now` (eye 1, train) | "Going backwards from today, which is the first visit where CST was higher than now?" | CST V1–V4: 575, 367, 294, 313; current V5 420 | V1 | V1 | Only V1 (575) exceeds 420. Latest prior (V4, 313) would be wrong. |
| `last_vision_as_good` (eye 1, train) | "Going back from today, which is the first visit with VA equal to or better than now?" | BCVA V5–V10: 73, 78, 75, 79, 76, 73; current V11 78 | V8 | V8 | Most recent with BCVA ≥ 78 is V8 (79). V10 (73) and V9 (76) do not qualify. |
| IE example, `largest_cst_rise` | "…biggest jump in central subfield thickness compared with the visit right before it?" | CST 575, 367 | no visit | IE | CST never rose → `no_qualifying_visit` (correct abstention). |

## 9. Split integrity and hidden fields (verified)

- **Patients per split:** train 52, val 17, test 18. Patients in more than one split: **0**. Eyes in more than one split: **0**. Dev instances come only from `train`/`val`, asserted in `dev_episodes`.
- **`episodes.csv`** has no leakage, DRSS or injection columns.
- **`build_memories`:**
  - raises an error if any `HELDOUT_COLUMNS` are present;
  - is called without `declared_extra` in the runner;
  - every `VisitMemory.extra` is empty (`test_real_memories_carry_only_declared_fields`).
- **`heldout_prime.csv`** is read only by `generate_instances_v2` (the answer key).
- **The dev runner** never references the locked set (`test_dev_runner_never_touches_locked_or_test`).

## 10. Are these results sufficient to proceed to locked evaluation?

**What they establish:**
- the six-outcome pipeline runs end to end;
- the operations agree exactly with the independently written answer key, including ties, missing values, the ±4-week boundary, the 5-letter threshold and abstention cases;
- the parser covers the dev phrasings;
- the system abstains correctly on insufficient-evidence questions and never returns a visit for unrecognised or unsupported questions on dev.

**What they do not establish:**
- parser generalisation to new wording (n = 5 seen texts per type);
- any retrieval advantage over fixed rules on unseen data;
- clinical validity;
- novelty.

The dev set is now **saturated** (at ceiling), so it cannot discriminate further changes.

**Conclusion.** The pipeline is ready for one locked evaluation, which is the only remaining informative test. The headline quantity should be **parser/routing accuracy on unseen text**, with retrieval Hit@1 reported as conditional on correct parsing. Because of the developer's prior exposure, that evaluation should use a **fresh, never-read locked paraphrase set**, with the existing locked set reported separately as "developer-exposed". Freeze the code with hashes before running.
