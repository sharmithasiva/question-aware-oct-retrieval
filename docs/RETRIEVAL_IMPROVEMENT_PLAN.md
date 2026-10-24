# Phase 1.3 — Retrieval Improvement Plan (metadata only)

Status: diagnosis written **before** any code change. All diagnosis uses the **dev** paraphrases
(`tests/data/paraphrases_v2_dev.jsonl`) and v1 templates only. The locked file was not opened for this work.

> **Contamination disclosure.** In an earlier session step the developer (Claude) transcribed the independent
> agent's output, including the locked paraphrases, into `paraphrases_v2_raw.jsonl`. That text is in the
> developer's context. It was not reopened or consulted for this plan, but prior exposure cannot be ruled out.
> The locked-test protocol (§6) therefore proposes a **fresh, never-seen locked set**.

## 1. Real entry points (current code)

| Step | Function | File |
|---|---|---|
| Parse question | `parse_question(text) -> QuestionSpec(target, intent, direction, evidence)` | `src/oct_retrieval/questions.py` |
| Retriever input | `History(visit_ids, cst, bcva, n_images)`, built by `eye_histories` and sliced in `run_retriever` | `retrievers.py`, `evaluate.py` |
| Route / rank | `RuleRouter.rank`, `Combined.rank`, `TemporalOnly.rank`, `ClinicalOnly.rank` → always a list of positions | `retrievers.py` |
| Scoring helpers | `temporal_score`, `clinical_score`, `_order` | `retrievers.py` |
| v2 answer key | `answer_key`, `generate_instances_v2` (returns `INELIGIBLE` / `IE` / positions) | `ground_truth_v2.py` |
| Runner | `run_retriever` (calls `parse_question` then `rank`), `eye_macro`, `bootstrap_ci` | `evaluate.py`; v2 driver `evaluate_v2.py` |

## 2. Current behaviour per question type (frozen v1 parser on the 5 dev paraphrases per type)

| Type | v1 parse on dev (count) | Strategy invoked (`rule_router` / `combined`) | Fallback | Failure source |
|---|---|---|---|---|
| `largest_cst_rise` | extreme-max cst ×2, previous ×1, unknown ×2 | argmax CST / latest / latest (rule_router); closeness-to-current CST (combined, for unknown) | latest | **Parsing** (no "change between consecutive visits" concept) **and ranking** (no consecutive-change scorer exists) |
| `largest_bcva_drop` | unknown ×3 (1 with no target), extreme-**max** ×1, previous ×1 | latest; argmax BCVA (wrong direction) | latest | **Parsing defect:** "best-corrected" triggers `_BEST`. "VA" is not a BCVA word. No change operation. |
| `first_vision_gain_1line` | extreme-**min** ×2, baseline ×1, unknown ×2 | argmin BCVA; earliest; latest | latest | **Parsing defect:** "at least" matches `_MIN` (`least`). No threshold-crossing operation. |
| `one_year_ago` | unknown ×5 (1 with spurious target bcva) | latest (rule_router); closeness (combined, when target found) | latest | **Parsing:** no time-expression parsing. **Defect:** `\bsee\b` in `_BCVA_WORDS` ("so I can see…"). **Interface defect:** `History` carries no `week`, so no strategy could rank by time. |
| `six_months_ago` | unknown ×5 | latest | latest | Same as above |
| `max_leakage` | unknown ×5 (no target) | latest — **silently retrieves** | latest | **Routing:** the variable is deliberately absent from memory (Tier A). The correct behaviour is an explicit "variable unavailable" outcome, not a latest-visit guess. |
| `last_thicker_than_now` | previous ×3, baseline ×1, unknown ×1 | latest; earliest | latest | **Parsing:** no "compared with now + most recent qualifying" concept. "First" in "going back… first visit" read as baseline. |
| `last_vision_as_good` | previous ×1, extreme-min ×1, unknown ×3 (2 no target) | latest; argmin BCVA | latest | **Parsing defects:** "at least" → min; "eyesight", "VA" missing from lexicon. No comparison-to-current concept. |
| `last_injection` (not benchmarked) | unknown ×5 | latest — silently retrieves | latest | Should be "variable unavailable" (injection is not in memory) |
| unsupported (image) | image ×4, metadata-unknown ×6 | latest — **silently retrieves for 6/10** | latest | **Routing defect:** the image vocabulary is incomplete, and unknown silently falls back to retrieval |
| v1 types (8) | 24/24 templates correct; 9/9 v1 held-out paraphrases | rule rules | — | Circular by construction (see v1 report) |

## 3. Confirmed code defects (vs. algorithmic gaps)

**Confirmed defects (behaviour contradicts the stated design or the Phase 1.3 requirements):**
1. `RuleRouter.rank` returns `LatestPrior` order for any unrecognised intent: a silent fallback (`retrievers.py`, last line).
2. There is no abstention channel. `Retriever.rank` must return a list, so "unrecognised", "unsupported" and "recognised but nothing qualifies" are indistinguishable.
3. `clinical_score` treats *every* non-`extreme` intent (including `baseline`, `previous`, `unknown`) as "closeness to the current value". This is why `combined` ranks by CST similarity on unrecognised questions.
4. Lexicon collisions in `parse_question`:
   - `least` (inside "at least") → `_MIN`;
   - `best` (inside "best-corrected") → `_BEST`;
   - `see` → BCVA target;
   - "first visit" → baseline even when "first" is the operator.
5. The image check runs only when no CST/BCVA word is found, so image questions that mention "thick"/"retina" are routed to CST.
6. The retriever interface (`History`) omits `week`, which is retriever-visible metadata. Temporal questions are unanswerable by design of the interface, not because of data.

**Algorithmic gaps (new capability, not bugs):** the parser has no concept of consecutive change, threshold crossing relative to baseline, comparison with the current value plus recency, time offsets, or variables outside memory.

**Not a defect, but must be stated:** for Tier B types, any retriever that implements the frozen definitions will score ≈1.0 when parsing succeeds. This is **rule coverage**, not evidence of retrieval superiority. For those types the informative quantities are **parser/routing accuracy on independent paraphrases** and **abstention behaviour**.

## 4. Proposed corrections (implemented in new modules; v1/v2 code untouched)

| Correction | Where | Notes |
|---|---|---|
| Structured intent `QueryIntent(task, target, operation, direction, comparator, reference, offset_weeks, tolerance_weeks, threshold, missing_variable, reason)` | new `intent.py` | Explicit, testable categories; `task` ∈ {`retrieve`, `unavailable_variable`, `unsupported_image`, `clarify`} |
| Deterministic parser `parse_intent` with ordered rule stages: time offset → out-of-memory variables → image-only findings → consecutive change → threshold vs baseline → comparison with current → similarity → extremum → baseline anchor → previous anchor → **clarify** | `intent.py` | No silent fallback. Phrase-level patterns (not single words) fix defect 4. Image check has priority over weak thickness words (defect 5). |
| `VisitMemory(visit_ids, visit_index, cst, bcva, week, n_images, trial)` with a declared `MEMORY_FIELDS` whitelist; refuses held-out columns | new `intent_retrieval.py` | Fixes defect 6 without changing the frozen `History` |
| `retrieve(intent, memory, t)` → `RetrievalResult(status, ranked_positions, reason)`, status ∈ {`retrieved`, `no_qualifying_visit`, `insufficient_data`, `unavailable_variable`, `unsupported_image`, `clarification_needed`} | `intent_retrieval.py` | Fixes defects 1–2. Six distinct outcomes. |
| General operations: anchor (baseline/previous), extremum, closest-to-current, max consecutive change (±), first threshold crossing vs baseline (±, any threshold), last/first visit satisfying a comparison with the current value, nearest visit to a time offset with tolerance | `intent_retrieval.py` | Does **not** import `ground_truth_v2`. Operations are parameterised (any target, direction, threshold, offset), not per-benchmark-type. |
| Leakage / injection / DRSS | `intent.py` → `unavailable_variable` | A lookup runs only if the variable is in a declared `available_fields`; the default memory excludes them. No proxy is used (decision D1 below). |

## 5. How each correction is evaluated without locked data

- **Development data only:** dev paraphrases (5 per type + 5 `last_injection` + 10 unsupported) × patients in `train` and `val`. `test`-split patients are excluded in code (assertion), and the locked file is never opened by the dev runner.
- **Separate measurements:**
  - (a) parser exact-intent accuracy;
  - (b) routing accuracy (task category);
  - (c) retrieval Hit@1 / MRR on answerable instances (patient-cluster bootstrap);
  - (d) abstention: correct abstention on IE instances and false abstention on answerable ones.
- **Baselines:** fixed rules (latest, earliest, second visit, max/min CST, max/min BCVA, random) and the frozen v1 `rule_router`/`combined`. **The best fixed rule per type is selected on dev** and written to `results/v3_dev/selected_baselines.json` before any locked evaluation.
- **Statistical caveat:** each type has only 5 dev paraphrases. Parser accuracy on them is a count out of 5, not a generalisable estimate; patient bootstrap CIs do not capture paraphrase variability.

## 6. Decisions to raise (not implemented unilaterally)

- **D1 `max_leakage`:** the frozen definition hides leakage from the retriever, so the correct system output is `unavailable_variable`, which scores 0 on Hit@1 by design. A *declared proxy* mode (e.g. "rank by CST" or "earliest", chosen on dev) is possible, but it changes what the system claims to do. **Not implemented; awaiting your decision.**
- **D2 fresh locked set:** see the contamination disclosure above.

## 7. Implementation status (added after implementation, before any locked run)

- Implemented in new files only:
  - `src/oct_retrieval/intent.py`, `src/oct_retrieval/intent_retrieval.py`, `src/oct_retrieval/evaluate_v3_dev.py`;
  - `dev_by_type()` added to `paraphrases.py`;
  - tests `tests/test_v3_intent.py`, `tests/test_v3_retrieval.py`.
- The frozen `questions.py`, `retrievers.py`, `ground_truth_v2.py`, v1/v2 results and reports are unchanged. This is verified by `results/v3_dev/frozen_artifacts.md5` and `test_v1_v2_artifacts_unchanged`.
- **Lexicon hygiene.** After the first draft, terms the developer recognised as appearing only in the locked set were removed. Examples: "back-to-back", "from one visit to the next", "no worse than", "at or above", "present", "IVT", "choroid", "when I started treatment", "starting visit", "closest previous", "see this well". The image vocabulary was rebuilt from the dataset's own biomarker column names, the dev "unsupported" questions, the project brief and the v1 held-out set.
- **Fixes driven by developer-written generic tests (not benchmark text):**
  - "N letters above/below baseline" is now a baseline reference, and the preposition sets the direction;
  - "most" counts as a maximum for declared lookup variables.
- **Dev outcome:** every dev paraphrase and v1 template parses to its gold intent. Every recognised type retrieves with Hit@1 = 1.000 and abstains correctly on 100 % of insufficient-evidence questions. This is **expected by construction** (rule coverage on development text) and is not evidence of improvement.
