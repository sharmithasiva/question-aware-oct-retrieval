# Proposal: Harder Phase 1 Question Types (for review — nothing implemented yet)

> **Review outcome (2026-10-22):** approved with conditions. Decisions:
> - #3/#4 kept, but as secondary benchmarks.
> - `last_injection` deferred and documented as a structured lookup.
> - Independent paraphrases approved.
> - `max_leakage` prioritised.
>
> The final definitions, counts and tie rules are in [PHASE1_V2_QUESTION_SPEC.md](PHASE1_V2_QUESTION_SPEC.md); that document supersedes §3 below.

Every number below was measured on the actual label files (all splits, current visits with ≥2 prior visits).
Analysis script: exploratory, not yet part of the pipeline. No question type below is implemented until approved.

## 1. Problems with the current benchmark that this proposal addresses

1. **Circularity.** For the 8 existing types, `rule_router`/`combined` implement the same rule that defines the
   ground truth, so their 1.00 scores are a pipeline check, not evidence. The report will relabel them that way.
2. **Fixed heuristics already solve some types.** `worst_edema` is the baseline visit in 78.7 % of cases.
3. **Prime eyes have no centre-involved DME.** Per `Summaries _DR_DME_Studies.docx`, PRIME enrolled
   "NPDR/PDR without center-involved DME". "Edema" questions phrased via CST are clinically weak for the 40 Prime
   eyes; CST is still a valid *thickness* measurement there. New CST questions are phrased as thickness, not edema,
   when they apply to Prime.

## 2. Evaluation protocol for the new types (non-circular by design)

| Tier | Ground-truth source | What the evaluated retrievers can see | Why it is not circular |
|---|---|---|---|
| **A — held-out variable** | A dataset variable that is **never loaded into the episodic memory** (OCT-DR leakage index) | CST, BCVA, visit order, week | The retriever can only reach the answer through proxies. |
| **B — compositional** | An independent ground-truth module computing the definition from CST/BCVA/order/week | Same metadata | Evaluated only with the **frozen** Phase 1 retrievers (code unchanged, tagged `v1`). None of them implements these rules. |

Rules:
- The existing retrievers are frozen as `v1` before these types exist. Any future retriever extension that handles
  these compositions will be reported in a separate "rule coverage" table, never as the headline result.
- **Insufficient-evidence instances** are kept as their own set. Examples: CST never rose, or vision is now the best ever recorded. The correct output is "no such visit". The frozen retrievers cannot abstain, so they score 0 on these. That is the honest baseline for Component 5's "explicit indication when evidence is insufficient".
- Each type is reported on its own, with Hit@1/3, MRR and nDCG@3 for every existing baseline. Each type also gets a **heuristic ceiling**: the best Hit@1 achieved by any fixed rule (latest, earliest, max/min CST, max/min BCVA, second visit).

## 3. Proposed question types

`Best fixed` is the highest share of instances already solved by one fixed visit choice. Lower is better for
requirement 4. `Undef.` is the share of eligible instances where no visit satisfies the definition; these go to
the insufficient-evidence set.

| # | Type | Example question | Ground-truth definition (within history only, current visit excluded) | Tier | Cohort | Instances (eyes) | Undef. | Best fixed |
|---|---|---|---|---|---|---|---|---|
| 1 | `largest_cst_rise` | "At which visit did the retina thicken the most since the visit before it?" | visit *i* ≥ 1 maximising CST*i* − CST*i−1*; undefined if no rise > 0 | B | all | 1,113 (89) | 21 % | 22 % (latest) |
| 2 | `largest_bcva_drop` | "When did vision drop the most between consecutive visits?" | visit *i* ≥ 1 maximising BCVA*i−1* − BCVA*i*; undefined if no drop > 0 | B | all | 1,228 (95) | 12 % | 34 % (min BCVA) |
| 3 | `last_thicker_than_now` | "When was the retina last thicker than it is today?" | most recent visit with CST > current CST; undefined if none | B | all | 1,348 (96) | 4 % | 57 % (latest) |
| 4 | `last_vision_as_good` | "When was vision last at least as good as today?" | most recent visit with BCVA ≥ current BCVA; undefined if none | B | all | 1,141 (95) | 19 % | 68 % (latest) |
| 5 | `first_vision_gain_1line` | "When did vision first improve by at least one line from baseline?" | first visit *i* ≥ 1 with BCVA*i* ≥ BCVA*0* + 5 letters; undefined if none | B | all | 988 (eyes not yet counted) | 30 % | 49 % (second visit) / 32 % (max BCVA) |
| 6 | `one_year_ago` | "How does today compare with about a year ago?" | visit with week closest to current week − 52, only if within ±4 weeks; undefined otherwise | B | Prime only | 191 (33) | 67 % | 16 % (earliest) |
| 7 | `six_months_ago` | "Compared with roughly six months ago, has thickness changed?" | visit with week closest to current week − 26, within ±4 weeks (ties all relevant) | B | Prime only | 371 (38) | 36 % | 25 % (max BCVA) |
| 8 | `max_leakage` | "Compare with the visit when angiographic leakage was greatest." | visit(s) with the highest panretinal leakage index (OCT-DR `DRSS` sheet) | **A** | Prime only | 580 (40) | 0 % | 68 % (earliest) |

Justifications and caveats:
- **#5 "one line = 5 letters"** is not my threshold. The dataset uses it itself: OCT-DME.xlsx defines
  "1 line gainers/losers". Without a threshold, 1-letter changes would be measurement noise.
- **#6/#7 use real weeks, so they are Prime only.** Prime `W` folder labels are study weeks, and the CST
  sheet in OCT-DR.xlsx matches **660/660** Prime visits with Screen = W0. TREX `VisitDate`/`Week` are blank in 926 of 934 visits. The ±4-week tolerance is half of Prime's largest scheduled gap (8 weeks after W52). This is a design choice and is stated as one.
- **#8 is the only fully held-out signal.** Leakage index comes from ultra-widefield fluorescein angiography (UWFA). This is confirmed by the study summary's acronym list. It is *not* an OCT measurement. Spearman correlation with CST is only 0.13, so it tests whether the CST/time proxies can find a non-OCT clinical extreme. OCT images may not resolve it either. That is a Phase 2 question, and the results will say so.
- #3 and #4 are partly solved by recency (57 % / 68 %). I propose keeping them because they are the most
  clinically natural "relative to now" questions, and the gap between recency and the correct visit is still 30–40 %.

## 4. Evaluated and rejected (data-supported, but fail a requirement)

| Candidate | Why rejected (measured) |
|---|---|
| `worst_drss` (most severe DR, Prime) | Baseline is relevant in 97 % of instances, with 2.9 tied visits on average. A fixed heuristic solves it. DRSS is graded on fundus photography, not OCT. |
| `post_peak_nadir` (thinnest since the worst edema) | Equal to the global-minimum-CST visit in 93 % of cases, because the peak is usually the baseline. |
| `before_worst_edema` (visit before the peak) | Defined in only 21 % of instances (298 instances, 28 eyes), because the peak is usually the baseline. |
| `last_injection` (Prime, 100 % coverage in `Data` sheet) | **Deferred — your call.** If injection status is loaded into memory, retrieval is a trivial lookup (circular). If it is held out, nothing in CST/BCVA predicts it, so it tests nothing beyond recency (latest = 51 %). |

## 5. Not supported by Phase 1 metadata (reported as unsupported, not implemented)

| Question family | Why | Where it could be evaluated |
|---|---|---|
| Fluid/biomarker trajectory ("when did IRF resolve?", "when did DRIL appear?") | Biomarker labels exist only at the first and last visit of each eye | Phase 3 answer accuracy on the baseline-vs-final pair only |
| Structural similarity ("when did the scan look most like today?") | No label defines structural similarity | Phase 2 needs images; there is no ground truth except proxies |
| Any time-interval question for TREX DME | Dates and weeks are missing for 926 of 934 TREX visits | Unsupported |
| TREX treatment events | Per-visit injection data is only in the Year-3 sheets, which lie beyond the imaged visits | Unsupported |
| DME severity grade, treatment-arm effects | No per-visit severity grade; arm is constant within an eye | Unsupported |
| Fellow-eye comparisons | Retrieval is same-eye by design; only 9 patients have both eyes, in different trial arms | Out of scope |

## 6. Reducing parser-test leakage

1. **Independent paraphrases.** A separate agent is given only the plain-English definitions in §3 and §5, without the parser code or the existing templates. It writes 10 paraphrases per type, plus about 20 unsupported or ambiguous questions. They are saved to `tests/data/paraphrases_v2.jsonl` with a hash before any parser change.
2. **Dev/locked split.** Half of the paraphrases may be used for parser development. The locked half is evaluated once, at the end, and reported as-is, including failures.
3. The benchmark instances use the locked paraphrases. Parser errors therefore show up in the retrieval numbers instead of being hidden.
4. The frozen `v1` parser will mis-parse most of the new types. This is reported. It is part of what the frozen baselines measure.

## 7. Implementation plan once approved

- `side_tables.py`: parse the OCT-DR `DRSS` sheet (leakage) with an alignment check (660/660). The values are kept out of `episodes.csv` and stored in `data/processed/heldout_prime.csv`.
- `ground_truth_v2.py`: an independent ground-truth module for types 1–8, kept separate from `retrievers.py`.
- Regression tests:
  - the existing 52 tests must still pass;
  - the existing 8-type results must reproduce exactly;
  - hand-built histories for each new definition, including ties and undefined cases;
  - the alignment check;
  - a test that held-out variables never appear in retriever inputs.
- Report:
  - a new per-type section with the heuristic ceiling;
  - an insufficient-evidence table;
  - the unsupported list;
  - existing results relabelled as a pipeline check.

## 8. Decisions needed from you

1. Approve, drop or modify types 1–8. In particular: keep #3/#4 despite 57–68 % recency, and keep #5 despite 49 % "second visit".
2. `last_injection`: defer (recommended), hold out, or add to memory as a structured lookup.
3. OK to use a separate agent to write the independent paraphrases (§6.1)?
