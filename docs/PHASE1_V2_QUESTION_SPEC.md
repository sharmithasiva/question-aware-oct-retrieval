# Phase 1 v2 — Final Question Definitions (for sign-off; not implemented yet)

Status: specification only. The v1 pipeline is unchanged. v1 outputs are preserved with checksums in
`results/v1_snapshot/` (`MANIFEST.md5`). All counts below were measured on the label files on 2026-10-22.

---

## 1. Verifications requested before implementation

### 1.1 BCVA scoring convention (needed for `largest_bcva_drop`, `first_vision_gain_1line`)

| Check | Result |
|---|---|
| Values in the image table | Integers 27–100 for every visit. No decimals, so not logMAR or a decimal Snellen fraction. |
| Prime vs OCT-DR `BCVA` sheet; Demographics/Data label it **"ETDRS BCVA"** | 660 / 660 visits identical |
| TREX vs OCT-DME `ETDRS` column (same `VisitNum` = `Vn`) | 933 / 933 visits identical; the 1 remaining visit is missing in both |
| TREX ETDRS vs the recorded Snellen equivalent, using letters ≈ 85 + 50·log10(20/D) | Within ±5 letters in 98.1 % of 2,984 sheet rows; Spearman −0.98 against the Snellen denominator |

**Conclusion:** BCVA is the **ETDRS letter score**. Higher is better, and **one line = 5 letters**. OCT-DME.xlsx uses the same convention in its "1/2/3-line gainers/losers" columns.

### 1.2 Leakage index (`max_leakage`)

| Check | Result |
|---|---|
| Source | OCT-DR.xlsx, sheet `DRSS`, column "Leakage Index" per visit. 40 subject rows, all unique. Legend rows below row 41 are ignored. |
| What it measures | Panretinal leakage index from ultra-widefield fluorescein angiography (study summary acronym list). **Not an OCT measurement.** |
| Visit alignment | The same `Screen`/`Week n` → `W0`/`Wn` mapping used for the CST sheet, which matches 660/660 Prime visits |
| Coverage | 655 / 660 Prime visits. **Missing (blank cells):** 02-010 W4, 02-008 W4, 02-008 W8, 02-015 W16, 02-042 W8 |
| Values | 0.00–10.72, two decimals, median 0.78 |
| Ties at the maximum | 0.5 % of answerable instances |
| Isolation | `episodes.csv` contains no leakage, DRSS or injection column (checked). In v2 it is stored only in `data/processed/heldout_prime.csv`, which only the ground-truth module reads. A test will assert that no retriever input contains it. |

### 1.3 Week / timing (temporal questions)

- Prime `Wn` folder = study week *n*. The OCT-DR CST sheet confirms this (660/660, Screen = W0).
- TREX `Vn` is ordinal only. `VisitDate`/`Week` are blank for 926 / 934 TREX visits. **Temporal questions are therefore Prime only.**

---

## 2. Common definitions (apply to every type)

| Term | Definition |
|---|---|
| Current visit | Any visit with `visit_index` ≥ 2, i.e. at least two earlier visits of the same eye |
| History (candidates) | All visits of the same eye with `visit_index` < current. The current visit and later visits are never candidates. |
| Ineligible | The type cannot be posed at this visit (wrong cohort, or a required input such as the current measurement is missing). **No instance is generated.** |
| Insufficient evidence (IE) | The question can be posed, but no history visit satisfies the definition. An instance is generated with expected output "no such visit". |
| Missing measurements | A visit with a missing value for the type's variable is never relevant for that type. A change involving a missing value is undefined. Only 1 visit lacks CST/BCVA (`0247GOD_V3`), and 5 Prime visits lack leakage. |
| Ties | All tied visits are relevant. Hit@K and MRR count any relevant visit. Recall@K uses \|relevant\| as the denominator, so it cannot reach 1 when there are more relevant visits than K. nDCG uses the ideal ranking over min(K, \|relevant\|) relevant visits. |
| "Most recent" / "first" | Defined on `visit_index`, so they never produce ties |
| Consecutive change at visit *i* | value*i* − value*i−1*, where *i−1* is the immediately preceding visit of the same eye, both in history (*i* ≥ 1) |

---

## 3. Final question types

**Primary** = headline v2 retrieval benchmark. **Secondary** = reported separately, because recency already does well.

| # | Type | Tier | Cohort | Eligible when | Relevant visit(s) | Insufficient evidence when | Ties |
|---|---|---|---|---|---|---|---|
| P1 | `largest_cst_rise` | B | all | ≥1 defined consecutive CST change in history | argmax over *i* of (CST*i* − CST*i−1*) | max change ≤ 0 (thickness never rose) | all tied *i* relevant |
| P2 | `largest_bcva_drop` | B | all | ≥1 defined consecutive BCVA change | argmax over *i* of (BCVA*i−1* − BCVA*i*), in letters | max drop ≤ 0 (vision never fell) | all tied *i* relevant |
| P3 | `first_vision_gain_1line` | B | all | baseline BCVA present | smallest *i* ≥ 1 with BCVA*i* ≥ BCVA*0* + 5 | no such visit | none possible |
| P4 | `one_year_ago` | B | Prime | current week ≥ 52 | visit(s) minimising \|week − (current week − 52)\|, provided that distance ≤ 4 | no prior visit within ±4 weeks of the target | all equidistant visits relevant |
| P5 | `six_months_ago` | B | Prime | current week ≥ 26 | as P4, with 26 weeks | as P4 | as P4 (frequent, see §5) |
| P6 | `max_leakage` | **A** | Prime | ≥1 history visit with non-missing leakage | argmax leakage index over history visits with a value | never occurs (every eligible history has a value) | all tied visits relevant |
| S1 | `last_thicker_than_now` | B | all | current CST present | largest *i* with CST*i* > current CST (strict) | no prior visit thicker than now | none possible |
| S2 | `last_vision_as_good` | B | all | current BCVA present | largest *i* with BCVA*i* ≥ current BCVA | current vision is the best ever recorded | none possible |

Notes on the definitions:
- **P1/P2: baseline can never be relevant**, because a change needs a preceding visit. No minimum magnitude is applied, since the dataset gives no convention for CST. For P2 the largest drop is under 5 letters in only 0.1 % of answerable instances, so a 1-line threshold would change almost nothing.
- **P4/P5:** the ±4-week tolerance is half of Prime's largest scheduled gap (8 weeks after W52). The retrieved visit's CST/BCVA may be missing; retrieval does not require a measurement.
- **P6:** the current visit's own leakage is irrelevant and never read.

---

## 4. Eligible sample counts (measured)

Instances = (current visit × type). Answerable = eligible − insufficient-evidence. Eyes and patients are counted
over answerable instances.

| Type | Train elig / ans (eyes, pts) | Val elig / ans (eyes, pts) | **Test** elig / ans (eyes, pts) | All elig / ans | IE share |
|---|---|---|---|---|---|
| P1 `largest_cst_rise` | 890 / 715 (54, 47) | 247 / 192 (18, 17) | **265 / 206 (17, 16)** | 1,402 / 1,113 | 20.6 % |
| P2 `largest_bcva_drop` | 890 / 788 (58, 51) | 247 / 207 (18, 17) | **265 / 233 (19, 18)** | 1,402 / 1,228 | 12.4 % |
| P3 `first_vision_gain_1line` | 890 / 624 (49, 42) | 247 / 182 (16, 15) | **265 / 182 (15, 14)** | 1,402 / 988 | 29.5 % |
| P4 `one_year_ago` | 113 / 112 (20, 20) | 33 / 33 (6, 6) | **46 / 46 (7, 7)** | 192 / 191 | 0.5 % |
| P5 `six_months_ago` | 229 / 215 (23, 23) | 78 / 74 (8, 8) | **85 / 82 (7, 7)** | 392 / 371 | 5.4 % |
| P6 `max_leakage` | 341 / 341 (24, 24) | 118 / 118 (8, 8) | **121 / 121 (8, 8)** | 580 / 580 | 0 % |
| S1 `last_thicker_than_now` | 889 / 854 (59, 52) | 247 / 237 (18, 17) | **265 / 257 (19, 18)** | 1,401 / 1,348 | 3.8 % |
| S2 `last_vision_as_good` | 889 / 728 (58, 51) | 247 / 194 (18, 17) | **265 / 219 (19, 18)** | 1,401 / 1,141 | 18.6 % |

## 5. Ties and fixed-heuristic hit rates (answerable instances, all splits)

A heuristic "hits" when its single chosen visit is among the relevant visits. This is the "heuristic ceiling" the report will show for each type.

| Type | Instances with ties | Mean relevant visits | Latest | Earliest | 2nd visit | Max CST | Min CST | Max BCVA | Min BCVA |
|---|---|---|---|---|---|---|---|---|---|
| P1 `largest_cst_rise` | 4.0 % | 1.05 | **21.8 %** | 0 | 6.2 % | 17.1 % | 0 | 17.2 % | 12.0 % |
| P2 `largest_bcva_drop` | 11.1 % | 1.16 | 19.2 % | 0 | 12.7 % | 4.5 % | 17.9 % | 0 | **34.4 %** |
| P3 `first_vision_gain_1line` | 0 | 1.00 | 8.1 % | 0 | **48.9 %** | 1.3 % | 16.2 % | 32.2 % | 0 |
| P4 `one_year_ago` | 0.5 % | 1.01 | 0 | **16.2 %** | 0 | 15.2 % | 7.9 % | 11.5 % | 15.2 % |
| P5 `six_months_ago` | **75.2 %** | 1.75 | 0.8 % | 9.7 % | 18.6 % | 14.3 % | 14.8 % | **25.3 %** | 20.5 % |
| P6 `max_leakage` | 0.5 % | 1.01 | 7.6 % | **68.1 %** | 2.8 % | 64.0 % | 2.4 % | 9.5 % | 34.1 % |
| S1 `last_thicker_than_now` | 0 | 1.00 | **57.0 %** | 7.3 % | 10.0 % | 13.2 % | 23.8 % | 26.7 % | 14.5 % |
| S2 `last_vision_as_good` | 0 | 1.00 | **67.7 %** | 0.5 % | 5.2 % | 3.9 % | 32.0 % | 42.5 % | 3.4 % |

**P5 tie issue:** Prime visits are 4-weekly until W52. A 26-week offset therefore lands exactly between two visits for most current visits (e.g. W60 → target W34 → W32 and W36 are both 2 weeks away), so 75 % of instances have two relevant visits. The rule handles this correctly (both are relevant), but it makes Hit@1 easier. See decision D1.

---

## 6. Deferred: `last_injection` as a structured lookup (not a retrieval benchmark)

- Source: OCT-DR `Data` sheet, "Injection" Yes/No per visit. Covers 660/660 Prime visits; Yes on 51.8 %.
- Definition, if built: most recent history visit with Injection = Yes. Insufficient evidence if none.
- Why it is not a retrieval benchmark: with injection status in memory, the answer is a deterministic lookup. Without it, CST/BCVA carry no signal (latest prior = 51 %). It will be documented in the report as a structured-lookup task. It will not be loaded into memory and will not appear in any retrieval table.

---

## 7. How v2 will be evaluated

- **Frozen v1 components.** The parser, all retrievers and Combined α = 0.05 (tuned in v1 on train, old types only) are tagged `v1` and not modified. v2 ground truth lives in a separate module (`ground_truth_v2.py`) and does not import `retrievers.py`.
- **Per-type reporting.** Every type gets its own table with Hit@1, Hit@3, MRR, nDCG@3 and Recall@1 for: latest, earliest, random, temporal-only, clinical-only, combined (v1), rule router (v1), plus the heuristic ceiling. Uncertainty comes from a 95 % **patient-cluster bootstrap** (1,000 reps, resampling patients) per type.
- **Insufficient-evidence set.** Reported separately: count per type and abstention rate. The v1 retrievers never abstain, so they score 0 %. IE instances are excluded from ranking metrics.
- **Paraphrases.** An independent agent writes the paraphrases from §3 and §6 text only (no parser, keywords or existing templates). The dev and locked sets are fixed with a SHA-256 hash before any parser work. Benchmark instances use **locked** paraphrases, so frozen-parser errors propagate into the scores. The parser is not tuned on the locked set.
- **Regression.** The 52 existing tests must still pass, and a new test asserts that v1 summary metrics reproduce `results/v1_snapshot/summary.csv` exactly. v2 outputs go to `results/v2/`; v1 files are not overwritten.
- **Relabelling.** In the v1 report, the 1.00 scores of `combined`/`rule_router` become **"pipeline check (ground truth and retriever share the rule)"**, not evidence of superiority. Retrieval metrics stay separate from any future VLM answer metrics.
- **Planned tests, written before the evaluation changes:**
  - hand-built histories per type covering ties, missing values, IE and ineligible cases;
  - Prime-only gating;
  - the ±4-week boundary (exactly 4 → eligible, 5 → IE);
  - the week ≥ 52 / ≥ 26 eligibility rule;
  - leakage isolation;
  - the BCVA convention, with sheet alignment (660/660 and 933/933) as integration tests.

---

## 8. Decisions needed before implementation

- **D1 — P5 offset.** (a) Keep 26 weeks, with the frequent two-visit ties reported. (b) Use 24 weeks: it lands on a scheduled visit before W52, but it is a less literal "six months". *Recommendation: (a), because it is the literal definition and ties are handled explicitly.*
- **D2 — small Prime test sets.** P4–P6 have only 7–8 test patients. Nothing in the frozen v1 system was fitted on these types. I propose a **secondary all-patient table** (40 Prime patients, clearly labelled) next to the primary test-split table. *Recommendation: yes; the test split remains the headline.*
- **D3 — S1/S2.** Confirmed secondary; reported in their own table below the primary types.
