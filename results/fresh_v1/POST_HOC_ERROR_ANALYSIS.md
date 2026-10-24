# fresh_v1 — post-hoc error analysis (descriptive; NOT pre-registered)

> AI-generated preliminary benchmark with limited human review. NOT clinician-validated. The annotator's independence
> is unverified, and the original agreement results are potentially compromised. Written **after** the one-shot run;
> nothing here changes any score in `reports/FRESH_V1_REPORT.md`. No code was changed.

## Routing errors (22 / 120)

| Gold routing → system | n | Items | Pattern |
|---|---|---|---|
| retrieve → clarify (**over-refusal**) | 14 | F032, F047, F052, F083 (first gain); F019, F089, F119 (BCVA drop); F043, F066, F094 (CST rise); F059 (last thicker); F087, F106 (last vision); F058 (one year) | Phrasings outside the parser's vocabulary (see below) |
| data_unavailable → retrieve (**unsafe**) | 2 | F064 (HbA1c at the highest-CST visit), F103 (age at baseline) | Parser recognises the visit anchor but not the unavailable variable |
| unsupported_operation → retrieve (**unsafe**) | 2 | F027 (weeks since baseline), F040 (CST slope since baseline) | Retrieves the baseline visit instead of declining the computation |
| data_unavailable → clarify | 2 | F081 (drug switch), F120 (blood pressure) | Safe, but the reason given is wrong |
| requires_images → clarify | 1 | F036 ("intraretinal cysts") | Safe; "cyst" is not in the image vocabulary |
| needs_clarification → unsupported_image | 1 | F112 ("when did the numbers last look like this?") | Safe; "look like" is read as an image question |

**Note on the 4 unsafe retrievals:** retrieving the anchor visit is a plausible *first step* for each of them (e.g. finding the highest-CST visit before reading its HbA1c). They are still counted as unsafe under the pre-registered rule, because the system cannot answer the question as posed.

## Routed correctly but wrong intent (9 / 56 retrieve-routed)

| Item | Gold type | Parsed as | Cause |
|---|---|---|---|
| F075 "…approximately **twenty-six** weeks…" | six_months_ago | time offset **6** weeks | **Defect:** compound number words are not parsed; only "six" is matched (KI-03) |
| F091 "…approximately **fifty-two** weeks…" | one_year_ago | time offset **2** weeks | Same defect (KI-03) |
| F022, F068 | first_vision_gain_1line | baseline anchor | "first … from baseline" / "beat baseline by five" not recognised as a threshold crossing |
| F056 "this good **or better**" | last_vision_as_good | comparator `>` instead of `≥` | "or better" was removed from the lexicon (see below) |
| F079 "last time CST **>** current" | last_thicker_than_now | previous-visit anchor | The symbol ">" is not handled |
| F098 "**closest previous** visit where thickness was above…" | last_thicker_than_now | closest to current | "closest previous" was removed from the lexicon; "closest" then triggers similarity |
| F100 "biggest positive CST **delta**" | largest_cst_rise | previous-visit anchor | "delta" is not a change word |
| F117 "**at or above** today's level" | last_vision_as_good | previous-visit anchor | "at or above" was removed from the lexicon |

## A measurable cost of the contamination safeguard

Before the freeze, phrases the developer recognised from the **retired, developer-exposed** locked set were deliberately
removed from the parser (`docs/RETRIEVAL_IMPROVEMENT_PLAN.md` §7). They were: "back-to-back", "from one visit to the
next", "see this well", "no worse than", "at or above", "or better", "closest previous", "spiked", "starting".

The independent generator used several of these **same** phrasings. They account for these failures:
- F019 (back-to-back);
- F119 (from one visit to the next);
- F087 (see this well);
- F106 ("no lower than", a close variant);
- F117 (at or above);
- F056 (or better);
- F098 (closest previous);
- F066 (spiked);
- F052 (starting vision).

**Interpretation:** these are common, natural phrasings, not quirks of the locked set. Removing them kept the fresh test
uncontaminated, but cost accuracy. A future parser version could add them back **only with a new fresh set (fresh_v2)**,
because these fresh_v1 items are now exposed.

## Other vocabulary gaps (not related to the safeguard)

- "BL" (baseline abbreviation);
- "a full line";
- "≥5 ETDRS letters" (unit word between the number and "letters");
- "went up by the largest amount";
- "swelling measurement went up";
- "last year's visit";
- "a higher CST than this one" (no "today/now/current" word);
- "HbA1c", "age", "blood pressure", "drug switch" (unavailable variables not in the vocabulary);
- "slope" and "how many weeks" (computations).

## Reading Part C correctly

Each type's Part C instances rotate through its 10 gold texts. intent_v3's Hit@1 therefore mostly reflects **which of
those 10 texts parse correctly**. For example, largest_cst_rise reaches 0.606 with 6 of 10 texts exact.

The patient-bootstrap CIs in Part C (e.g. [0.581, 0.630]) capture **patient** variability only. They do not capture
**wording** variability, which is the dominant source of uncertainty here (n = 10 texts per type). Part C intervals
for intent_v3 should therefore be read as too narrow.

Also: "correct abstention" on insufficient-evidence instances requires the specific outcome `no_qualifying_visit`.
When a text is misparsed, the system answers `clarify` instead. That is still an abstention, but it is not counted as
correct, which is why those rates fall below 1.
