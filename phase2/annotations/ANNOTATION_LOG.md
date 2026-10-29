# Annotation and freeze log

Append-only. Record hashes with `sha256sum <file>`.

## 2026-10-24: packet generation (dev)

- `tierB_dev_sheet.csv` SHA-256 `991ee7fd40de997e6798a3b8f2fa1cdab783791d08e0dbd527f9b90fb9bce0b8` (blank grades).
- `keys/tierB_dev_key.csv` SHA-256 `99b2b6502408ac20a35ad5bbbed0755a04462da90da54b1a09126dec57d8f3b4`.
- Dev eyes 25, 11, 61 (train). 18 query visits, 36 queries, 264 coded candidates, 528 sheet rows.
- **Exposure:** the developer (Claude) viewed one dev contact sheet (`0232GOD_V14`) to check the layout. The developer assigns no grades.

## 2026-10-24: incident (held-out)

- While smoke-testing the CLI, `question_eval.py --set heldout --tier A` was run once. Held-out eyes 93, 74, 40 and 14 have no images yet, so only S1/S2 rankings existed.
- Only the status counts were displayed (257 evaluated, 3 with no relevant candidate). **No metric value was viewed.**
- The output folder was deleted unread.
- Tier A held-out labels are computed by frozen code from metadata, so no human label was exposed.
- A guard now requires `--confirm-heldout-freeze` for any held-out run.

## Pending

- [ ] Dev Tier B grading. Record annotator ID(s) and role(s), the completed sheet's SHA-256, and the double-graded subset if any.
- [ ] Choose the S5 encoder on dev (protocol §2) and record it here.
- [ ] Freeze: SHA-256 of `phase2/question_eval.py`, `QUESTION_EVAL_PROTOCOL.md`, `make_annotation_packet.py`, and the held-out query set.
- [ ] Held-out: approval to fetch 76 images; fetch, embed, generate the packet, grade blind, hash the sheet; then one run with `--confirm-heldout-freeze`.

## 2026-10-26: v2 workflow (supersedes the v0.1 packet for grading)

- **Audit:** the v0.1 packet has 0 grades. It is kept unchanged and must **not** be graded. Gaps fixed in v2 are listed in `ANNOTATION_PROTOCOL_v2.md` §1. Note that 264 judgments is **per category**: the total is 528.
- **Protocol** `ANNOTATION_PROTOCOL_v2.md`: SHA-256 `14d735d56e2364470ffed4e2a8892a594d51a1dc8c369fef9a8eb318b7577e21`.
- **Rankings frozen before any label existed:** `phase2/results/question_eval_v2/dev/rankings_tierB_24a90d2768c6/`.
  - rankings SHA-256 `24a90d2768c6076e45f68427ffde0f05960a17a6086ba5aeea5d3cae2f388837`;
  - 36 queries, 3,168 rows, 6 ranked systems;
  - `S2_metadata` abstains on 36/36;
  - git HEAD `7fd4f3e`.
- **Packets** (dev, eyes 25 / 11 / 61): `v2/dev/B1/` and `v2/dev/B2/`, 18 queries × 264 judgments each, 73 re-encoded images each.
  - `MANIFEST.json` SHA-256 `55342da230fd4c9ca6bcc92612b38838a08c7090ad28bd8740404347d1056593`.
  - `B1/sheet.csv` `5f0f4b4dd390e963a769b88b30bcf099e1513a3f2f53dc7832800e912eb559a2`; `B2/sheet.csv` `3c3309acdd26cc5774c75211d7c65defd25166b81bd34c5783340b598d4d930e` (all labels blank).
  - Keys (git-ignored; **back them up outside the repository**): `keys/v2/dev_B1_key.csv` `018e34dc2a31c1189c67434df7fd111156caa78ab01c43086fb7477265a5e498`, `keys/v2/dev_B2_key.csv` `e34ba936b5d7b79101ff06884e3e9e37dad979c0c369cfbdcccf04f47f4266d8`.
  - Seeds: B1 queries 3174602428, B1 image names 1667648305, B2 queries 2913323342, B2 image names 757443869. Candidate order is seeded per query (see the manifest).
- **Code:**
  - `freeze_rankings.py` `b21eb15a649a60366d011ed1215b48ac8f4531e71b12f535c0ed4ce8c0c24edf`;
  - `make_annotation_packet_v2.py` `f190449decab80c0b29d9029f3640c3fbf5ad6c948eb615dddb194ccf953440a`;
  - `evaluate_annotations.py` `2d5914ff673923a1412df26624d1c331f184030af6090f02eadb1edeb969b324`;
  - `question_eval.py` (unchanged since v0.1) `1e9e103b43487022de9cd307f8c8a7fca27fb7361a7192529cc9809d03c4f382`.
- **Sample evaluation** on the blank sheets: `eval_B1_5f0f4b4d_24a90d2768c6/` and `eval_B2_3c3309ac_24a90d2768c6/`. All 18/18 queries are NOT EVALUATED; no labels exist.
- **Exposure:** the developer (Claude) did UI tests of the HTML page with throwaway clicks. They were held in browser memory, then cleared; nothing was saved. The developer assigns no grades.

## Pending (v2)

- [ ] B1 dev grading (`v2/dev/B1/index.html`). Record annotator ID, role and the completed CSV's SHA-256 here.
- [ ] B2 dev grading by a **trained OCT grader**. Record role and SHA-256.
- [ ] Optional: a second B1 annotator on the full packet, for κ.
- [ ] Choose the S5 encoder on dev B1 nDCG@5 (protocol v2 §6) and record it here.
- [ ] Freeze before held-out. After approval to fetch: fetch and embed the 76 held-out images, then `freeze_rankings.py --set heldout --confirm-heldout-freeze`, generate the held-out packets, grade them blind, and run the evaluation once.

## 2026-10-26: B1 dev annotations completed

- **File:** `v2/dev/B1/completed_1_dev-B1-v2.csv`; SHA-256 `cda4ea5b7c130b4b45de3857f76d177afd015dd56ea4063bbcebe9f7d9b21b29`.
- **Annotator ID:** `1`. **Role:** `researcher_non_expert`. **Annotation date:** 2026-10-26. All three are as recorded in the file.
- **Packet:** `dev-B1-v2`, 18 queries, 264 judgments.
  - Labels: **2** 73, **1** 129, **0** 61, **U** 1, **X** 0; blank 0.
  - Queries marked not gradable: 0.
  - Rationales are present for all 203 labels of 2, 1, U or X.
- **Validation** (read-only, 2026-10-26):
  - columns are identical to `sheet.csv`;
  - row IDs and the packet-defined columns are unchanged;
  - the file joins one-to-one to `keys/v2/dev_B1_key.csv`;
  - every candidate pool matches the frozen rankings `rankings_tierB_24a90d2768c6`, whose hash was verified.
- **Single annotator;** no second sheet yet. B1 measures **visual similarity, not clinical relevance**.
- **Not yet done:** the evaluation (`evaluate_annotations.py`) has not been run on this file, and the S5 encoder choice is not yet recorded. B2 is **not** annotated.

## 2026-10-29: B1 dev evaluation run (frozen rankings)

- **Key:** `keys/v2/dev_B1_key.csv` was absent from this clone (git-ignored). It was **regenerated** with the frozen
  generator's own `build_category` and the recorded seeds, in a scratch directory. It matches the logged SHA-256
  `018e34dc2a31c1189c67434df7fd111156caa78ab01c43086fb7477265a5e498` byte for byte (the original was written with
  CRLF line endings), and the regenerated blank sheet matches `5f0f4b4d...` too. Only the B1 key was regenerated;
  the B2 key was not. The key is stored in `keys/` (git-ignored) and has not been committed.
- **Run:** `python phase2/evaluate_annotations.py --set dev --category B1 --sheet phase2/annotations/v2/dev/B1/completed_1_dev-B1-v2.csv`
  → `phase2/results/question_eval_v2/dev/eval_B1_cda4ea5b_24a90d2768c6/`. Evaluator SHA-256 unchanged
  (`2d5914ff...`); frozen rankings hash verified (`24a90d27...`). 18/18 queries evaluated; 1 `U` removed from its pool.
- **Primary nDCG@5 (grade 2):** S1 latest_prior 0.786, S2p CST proxy 0.782, S3S4 ResNet-50 0.802, S3S4 DINOv2
  0.682, S5 RRF ResNet-50 0.823, S5 RRF DINOv2 0.725, random 0.572. S2 (frozen Phase 1) abstains on 18/18.
  3 eyes, so no CI; single non-expert annotator; B1 is visual similarity, not clinical relevance.
- **S5 encoder rule (protocol v2 §6, higher dev B1 nDCG@5, ResNet-50 on a tie):** the rule's outcome is
  **ResNet-50** (S3S4 0.802 vs 0.682; S5 0.823 vs 0.725). To be confirmed by the study lead before the held-out freeze.
- **Exposure notice:** the committed blank-sheet sample evaluations `eval_B1_5f0f4b4d_*` and `eval_B2_3c3309ac_*`
  contain `judgments_joined.csv`, which includes the code → visit mapping (`candidate_visit_id`) for both packets.
  The **B2 key is therefore readable in the repository**. B2 has not been graded. Before B2 grading, either keep
  the grader away from `phase2/results/` or regenerate the B2 packet under a new version.
