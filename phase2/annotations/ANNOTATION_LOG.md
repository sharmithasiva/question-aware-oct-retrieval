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
