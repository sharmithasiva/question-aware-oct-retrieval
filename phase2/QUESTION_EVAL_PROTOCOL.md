# Question-conditioned retrieval evaluation: protocol (pilot, v0.1)

Written 2026-10-24, **before** any evaluation code was run. Inputs read:
- `phase2/PERFORMANCE_REPORT.md`;
- `docs/PHASE_2_IMAGE_PROTOTYPE.md` (it lives in `docs/`, not `phase2/`);
- `docs/PHASE_1_HANDOFF.md` and `docs/PHASE_2_HANDOFF.md`;
- the frozen question schemas `src/oct_retrieval/questions.py` (v1) and `ground_truth_v2.py` (v2), plus `intent.py` and `intent_retrieval.py`;
- `data/processed/episodes.csv` and `visit_images.csv`;
- the Phase 2 manifests and saved embeddings.

**Question.** For questions asked at a patient's current OCT visit, do image representations retrieve the **relevant earlier visits of the same eye** better than `latest_prior`, metadata-only retrieval, or each other?

**What this protocol does not test:**
- whether embeddings recognise the same eye;
- clinical validity, unless clinical reference labels (§4.3) are obtained.

---

## 1. Data scope and eligibility

| Set | Eyes (project `eye_id`) | Split | Images | Status |
|---|---|---|---|---|
| **Development (dev)** | 25 (TREX, 27 visits), 11 (TREX, 25), 61 (Prime, 21) | train | E2 full-history central B-scans, `phase2/data/atlas_fullhist/`, already embedded | available |
| **Held-out pilot** | 93 (Prime, 21), 74 (Prime, 20), 40 (TREX, 19), 14 (TREX, 16) | val | **not yet fetched**: 76 central scans, within the fetcher's 100-image cap | needs fetch + embedding |
| Reserved | all test-split eyes | test | none | **untouched**; reserved for a later confirmatory study |

**Held-out selection rule** (fixed before any of these eyes' images were seen):
- val split;
- exclude eye 79 (unresolved laterality), eye 45 (irregular image names), and eyes 24 and 72 (4 of their visits were viewed in E1);
- then the 2 eyes with the longest history per trial, ties broken by lower `eye_id`.

**Further rules:**
- **Patient-level separation.** Splits are patient-level (Phase 1, seed 42), so no patient is in both dev and held-out.
- **Prior exposure, declared:**
  - The metadata-only system (Phase 1 v3) was developed on train and val data, so its metadata scores are **in-sample** on every non-test eye.
  - The image encoders are generic pretrained models; none was tuned on OLIVES.
- **Query visits.** A visit t of an eligible eye is a query if it has **≥ 6 earlier visits** (t ≥ 6, 0-based). This guarantees that @5 metrics are a real selection rather than "all candidates".
  - Tier A uses every eligible visit.
  - Tier B uses **6 query visits per eye**, at positions `round(linspace(6, n−1, 6))`, which keeps annotation bounded.
- **Candidate pool.** Exactly the visits 0..t−1 of the **same eye**: the query visit itself, later visits and other eyes are never candidates. This is enforced and tested (`test_candidate_pool_has_no_leakage`).
- **Image used.** The central B-scan (index 24) of each visit, the only scan available in dev. A visit with no verified image is **excluded as a candidate for image systems** and kept, flagged, for metadata systems. In dev and the planned held-out set every visit has an image.

## 2. Systems compared (all on the same query set and candidate pool)

| ID | System | Ranking rule |
|---|---|---|
| S1 | `latest_prior` (approved Phase 1 baseline) | candidates by recency: t−1, t−2, … |
| S2 | metadata-only (frozen Phase 1 v3) | `intent.parse_intent(question_text)` then `intent_retrieval.retrieve` on the eye's **full** `VisitMemory`, imported read-only. An abstention yields **no ranking**. |
| S2p | metadata proxy (Tier B only, pre-specified) | the frozen `retrieve` with the fixed intent `closest_to_current / cst`: the most CST-similar visit first. This is the stated metadata stand-in for "looks similar" and "similar fluid/thickening". |
| S3 | image-only, ResNet-50 | cosine between the query's and each candidate's L2-normalised embedding, descending |
| S4 | image-only, DINOv2 ViT-S/14 | as S3 |
| S5 | hybrid, reciprocal-rank fusion (RRF) | score = 1/(60 + rank_meta) + 1/(60 + rank_img), with 1-based ranks; rank_meta is from S2 (Tier A) or S2p (Tier B), and rank_img from S3 or S4. If the metadata ranking is absent, the image ranking alone is used. |

**About the hybrid (S5):**
- RRF with the conventional k = 60 (Cormack, Clarke & Büttcher, SIGIR 2009) is **parameter-free and untuned**. It is a pre-specified candidate, **not a validated method**.
- **Encoder choice for the held-out run:** whichever of S5-ResNet-50 / S5-DINOv2 has the higher mean nDCG@5 on **dev Tier B labels**; ResNet-50 if they are equal or dev labels are absent. This is decided and recorded before held-out labels exist.
- **Optional weighted fusion** α·norm(meta) + (1−α)·cos, if ever wanted: α is chosen on dev only, from the grid {0, 0.25, 0.5, 0.75, 1}, by mean nDCG@5, then frozen. The held-out set is never used for tuning.

**Tie-breaking (all systems, deterministic):**
- exact score ties are broken by **recency** (more recent first), the Phase 1 convention;
- byte-identical images therefore tie exactly and are resolved the same way;
- no randomness is used, apart from the seeded bootstrap (§6).

## 3. Question categories

### Tier A: metadata-referenced (labels exist now, from frozen Phase 1 answer keys)

Relevance is computed by the **frozen** Phase 1 answer-key functions, imported read-only. These functions are independent of every retriever and of any embedding.

| ID | Question text (first frozen template) | Frozen answer key | Relevant set |
|---|---|---|---|
| A1 `similar_cst` | "When was the CST most similar to today's measurement?" | `questions.relevant_indices` (intent `similar`) | all prior visits with minimal \|CST − current CST\| (ties all relevant) |
| A2 `worst_edema` | "Compared with when the edema was worst, has it improved?" | `relevant_indices` (extreme, max) | all prior visits with the maximal CST |
| A3 `best_anatomy` | "Compared with the visit with the lowest CST, has thickness increased?" | `relevant_indices` (extreme, min) | all prior visits with the minimal CST |
| A4 `last_thicker_than_now` | "Most recent prior visit with CST above today's value?" (first entry of `tests/data/paraphrases_v2_dev.jsonl`) | `ground_truth_v2.answer_key` | the most recent prior visit with CST > current |
| A5 `largest_cst_rise` | "Which prior visit showed the biggest jump in central subfield thickness compared with the visit right before it?" (first dev paraphrase) | `ground_truth_v2.answer_key` | the visit(s) with the largest consecutive CST increase |

- **Primary Tier A category: A1.** It is the only similarity question, and therefore the closest metadata analogue of image similarity. A2–A5 are secondary.
- **Excluded:**
  - anchor questions (baseline, previous visit), whose relevance is definitional, so `latest_prior` or the baseline rule is trivially correct;
  - BCVA, time-offset and leakage questions, where images are not a plausible source.
- **Caveat that must accompany every Tier A result:**
  - Relevance is defined from **CST measurements**, and S2 retrieves on the same measurements. S2 is therefore a near-**ceiling reference**, not a competitor.
  - Tier A tests whether images can **stand in** for a CST measurement (for example when it is missing). It does **not** test visual relevance.
  - Question texts are fixed above: v1 first templates for A1–A3, and the first **dev** v2 paraphrase for A4–A5; the locked paraphrases are not used. The frozen parser's reading of each text is recorded per query.

### Tier B: image-dependent (human labels required; none exist yet)

| ID | Question text | Relevance criterion (annotator) | Annotator requirement |
|---|---|---|---|
| B1 `overall_appearance` | "Which earlier visit's OCT looks most like today's?" | Visual similarity of the **retinal profile** in the central B-scan: foveal contour, overall thickness profile, presence and extent of dark (hypo-reflective) spaces, layer visibility. **Ignore** brightness, noise, tilt and the scale-bar overlay. | trained non-expert permitted; record `annotator_role` |
| B2 `fluid_similarity` | "At which earlier visit was the retinal fluid/thickening most similar to today?" | Similarity of intraretinal and subretinal **fluid** and diffuse thickening: presence, location (centre vs off-centre) and approximate extent | **trained OCT grader required** for any clinical interpretation; non-expert labels are allowed only as a harness pilot and are flagged |

**Grades (per candidate):**
- **2 = highly similar** on the criterion; a clinician would accept it as "the visit that looks like today".
- **1 = partially similar**: same main feature, clearly different extent.
- **0 = not similar.**

The annotator may also mark the candidate `ungradable` (image quality) and the query `unanswerable` (e.g. the query image is ungradable).

**Binary relevance:**
- primary: **grade = 2**;
- secondary (sensitivity): **grade ≥ 1**;
- nDCG uses the graded gains 2^g − 1 (0, 1, 3).

**Deliberately not in the pilot:**
- event questions ("When was the retina last dry?") and progression questions ("When did the edema start worsening?");
- per-scan biomarker questions.

These need **per-visit clinical reference labels** from a qualified grader: IRF/SRF/DRT presence per visit, or reading-centre grades. OLIVES expert biomarker labels cover only the first and last visit of each eye, and the atlas labels for other visits are model-inferred, so **neither is a valid reference for ranking a history**.

## 4. Annotation procedure (Tier B)

1. **Packet generation** (`phase2/make_annotation_packet.py`):
   - For each Tier B query, the candidates are **shuffled with a fixed seed** and given opaque codes (C01, C02, …).
   - One contact sheet is rendered per query visit: the query image plus the coded candidate images. **No visit labels, weeks, CST, BCVA or system scores appear.**
   - The code → visit key is written to `phase2/annotations/keys/`, which the annotator **must not open** until grading is submitted.
2. **Grading:** the annotator fills `grade`, `ungradable`, `rationale` and `evidence_source`, and the query-level `answerable`, in the sheet `phase2/annotations/tierB_<set>_sheet.csv`. **No system output may be generated or viewed for the held-out set before the held-out sheet is complete and its SHA-256 recorded.**
3. **Double grading:** if a second annotator is available, 25 % of queries (seeded selection) are graded independently. Report weighted Cohen's κ (quadratic) on grades and percent agreement on grade = 2. Disagreements of ≥ 2 grades are adjudicated; record `adjudication_status`.
4. **Freeze:** after grading, the sheet hash, annotator IDs and roles are recorded in `phase2/annotations/ANNOTATION_LOG.md`, and only then is the harness run.
5. **Programmatic vs human:** the packet generator fills only identifiers, candidate codes and question text. All `grade` cells start empty, and the harness treats an empty cell as **missing**, never as 0. Rows carry `label_source = human`. Tier A labels carry `label_source = metadata_rule:<frozen function>`.

## 5. Handling special cases

| Case | Rule |
|---|---|
| Missing labels (blank grade, or no sheet) | the query is **NOT EVALUATED**: excluded from every metric and counted; never scored 0 |
| Partially graded query (some candidates blank) | NOT EVALUATED (a partially labelled ranking cannot be scored) |
| No relevant candidate (Tier A IE/empty set; Tier B no grade ≥ threshold) | excluded from Hit/MRR/Recall/nDCG (undefined) and counted as `no_relevant_candidate` |
| Ineligible question (frozen key returns INELIGIBLE, e.g. CST missing) | not a query; counted |
| Unanswerable (annotator) | excluded and counted |
| Ties in relevance | all tied visits are relevant (frozen-key convention) |
| Ties in system scores | broken by recency (§2) |
| Missing metadata for a candidate | the candidate stays in the pool; metadata systems rank it as the frozen code does (NaN last) |
| Metadata system abstains | no ranking: Hit/RR/Recall/nDCG = 0 for that system on that query in **all-query** metrics; **coverage** and **answered-only** metrics are reported alongside |
| Byte-identical images (known: `01-013` W8 = W12) | a query whose own image is byte-identical to a candidate image is **excluded** (image leakage) and counted. Identical candidates stay and tie exactly. Sensitivity analysis: rerun with visit W12 removed from eye 61's pool. |
| Image missing for a visit | excluded from image-system pools only; flagged |

## 6. Metrics, baselines and uncertainty

For a ranked list with relevance r_i ∈ {0, 1}, or graded g_i, and R relevant candidates:
- **Hit@1** = r_1.
- **Hit@5** = 1[Σ_{i≤5} r_i ≥ 1].
- **Recall@5** = Σ_{i≤5} r_i / R. It is reported separately from Hit@5.
- **MRR** = mean over queries of 1/(rank of the first relevant).
- **nDCG@5** = DCG@5 / IDCG@5, where DCG@5 = Σ_{i≤5} (2^{g_i} − 1)/log2(i + 1). With binary labels, g = r.

**Primary metrics** (declared before any results):
- **Tier A:** MRR, because relevant sets are small and binary.
- **Tier B:** nDCG@5 on graded labels.
- All other metrics are secondary.
- The **primary comparison** is each system vs S1 `latest_prior` on the primary metric, per category.

**Random baseline** (exact, analytic, per query, then averaged), for n candidates and R relevant:
- Hit@1 = R/n;
- Hit@5 = 1 − C(n−R, 5)/C(n, 5);
- Recall@5 = min(5, n)/n;
- MRR = Σ_k (1/k)·P(first relevant at rank k);
- nDCG@5 = (mean gain · Σ_{i≤min(5,n)} 1/log2(i + 1)) / IDCG@5.

**Reporting:**
1. Per category, per system: the metric mean over evaluated queries; the counts of evaluated, not-evaluated, no-relevant and excluded queries; and coverage for S2.
2. **Per-eye means**, because queries within an eye are dependent and the eye is the unit for uncertainty.
3. **Paired comparison vs S1 (`latest_prior`)**: per-query difference, then the per-eye mean difference; report the number of eyes where the system is better, equal or worse.
4. **95 % CIs**: eye-level cluster bootstrap (10,000 resamples of eyes, seed 20261024, percentile intervals), **only when ≥ 5 eyes** are evaluated. Otherwise "CI not reported (n_eyes < 5)".
5. **No hypothesis tests in the pilot.** Results are labelled *development* or *held-out pilot*. No claims of significance or clinical validity are made.

## 7. Order of work and freeze points

1. **Done with this protocol:** the harness and tests, the Tier B dev packet, and the Tier A dev run (labels exist).
2. **Dev Tier B grading** by the annotator; run the harness on dev, then select the S5 encoder (§2) and record it.
3. **Freeze:** hash the harness, protocol, query sets and selection in `phase2/annotations/ANNOTATION_LOG.md`.
4. **Held-out:**
   - fetch and embed the 76 held-out central scans (explicit approval needed);
   - generate the held-out packet and grade it blind;
   - record the sheet hash;
   - run the harness **once**.
5. **Report** dev and held-out separately. Never pool them.
