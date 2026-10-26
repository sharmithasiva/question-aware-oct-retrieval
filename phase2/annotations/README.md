# Annotations for the question-conditioned evaluation

Protocol: `phase2/QUESTION_EVAL_PROTOCOL.md`, sections 3–5. **No human labels exist yet.** Every `grade` cell is empty.

## Files

| File | Who uses it | Content |
|---|---|---|
| `relevance_template.csv` | reference | header-only schema (the columns below) |
| `tierB_dev_sheet.csv` | **annotator** | 528 rows: 18 dev query visits × 2 categories × their coded candidates; grade cells empty |
| `packets/dev/<query_visit>/` | **annotator** | `QUERY.webp`, `C01.webp` … (coded candidates), `contact_sheet.png` |
| `keys/tierB_dev_key.csv` | harness only | code → candidate visit. **Do not open before grading is submitted.** |
| `ANNOTATION_LOG.md` | everyone | hashes, annotator IDs and roles, freeze points, incidents |

Regenerate the dev packet (deterministic seeds):

```bash
python phase2/make_annotation_packet.py --set dev
```

## Columns (`relevance_template.csv`)

| Column | Filled by | Meaning |
|---|---|---|
| `annotation_row_id` | generator | `<query_id>::<candidate_code>` |
| `query_id` | generator | `<set>-<tier>-<category>-<query_visit_id>` |
| `set`, `tier`, `category`, `question` | generator | dev/heldout; B; `overall_appearance` or `fluid_similarity`; the question text |
| `eye_id`, `patient_id` | generator | project-local IDs from `episodes.csv` (not trial identifiers) |
| `query_visit_id` | generator | today's visit (it is the query, so it reveals nothing about the candidates) |
| `query_image`, `candidate_image`, `contact_sheet` | generator | paths relative to this folder |
| `candidate_code` | generator | opaque code; its visit is only in the key |
| `presentation_order` | generator | the seeded shuffle order |
| `grade` | **annotator** | **2** highly similar on the criterion; **1** partially similar; **0** not similar. Leave blank only if not yet graded (blank = missing). |
| `ungradable` | **annotator** | `yes` if the candidate image cannot be judged (scored as grade 0 and counted) |
| `query_answerable` | **annotator** | `no` if the query image itself cannot be judged (the query is excluded and counted) |
| `missing_data_flags` | generator / annotator | e.g. `candidate_image_missing`; the annotator may add notes |
| `evidence_source` | **annotator** | what was looked at, e.g. `central B-scan, contact sheet + full-size image` |
| `rationale` | **annotator** | one short phrase per grade-2 or grade-1 decision |
| `annotator_id`, `annotator_role` | **annotator** | e.g. `A1`; role is one of `ophthalmologist`, `optometrist`, `certified_grader`, `trained_non_expert`, `researcher_non_expert` |
| `annotation_date` | **annotator** | ISO date |
| `adjudication_status` | generator → adjudicator | `single`, `double_agree`, `double_adjudicated` |
| `label_source` | generator | always `human` here. Tier A labels from frozen answer keys are `metadata_rule:<function>` and are never written to this sheet. |
| `candidate_generated_by` | generator | how the candidate list was produced (all earlier visits of the eye), with the shuffle seed |

## Grading instructions (summary of protocol §3–§4)

**B1 `overall_appearance`: "Which earlier visit's OCT looks most like today's?"**
- Compare the **retinal profile** of the central B-scan: foveal contour, thickness profile, extent of dark (hypo-reflective) spaces, and layer visibility.
- Ignore brightness, speckle noise, tilt and the "200 µm" scale bar.

**B2 `fluid_similarity`: "At which earlier visit was the retinal fluid/thickening most similar to today?"**
- Compare the presence, location (central vs off-centre) and approximate extent of intraretinal/subretinal fluid and diffuse thickening.
- **Clinical use requires a trained OCT grader.** Non-expert grades are accepted only for the pilot and must declare `annotator_role`.

**Rules for every annotator:**
- Grade every candidate of a query before moving on.
- **Never view system rankings, CST/BCVA values or the key while grading.**
- Several candidates may receive grade 2.
- If no candidate deserves 2, that is a valid outcome: the harness counts it as "no relevant candidate".

## v2 (current, 2026-10-26)

**Grade the v2 packets, not the v0.1 sheet above.** The protocol and rubric are in `ANNOTATION_PROTOCOL_v2.md`.

- **B1 (visual similarity):** open `v2/dev/B1/index.html` in a browser (Chrome, Edge or Firefox).
- **B2 (fluid / thickening):** a trained OCT grader only; open `v2/dev/B2/index.html`.
- **Labels:** 2 / 1 / 0, `U` (uncertain) or `X` (ungradable); a query whose image cannot be judged is marked "not gradable".
- **Progress:** autosaved in the browser. When finished, click **Download CSV**, save the file as `v2/dev/<B1|B2>/completed_<annotatorID>.csv`, and record its SHA-256 in `ANNOTATION_LOG.md`.
- **Spreadsheet alternative:** fill `label`, `rationale` and `query_gradable` in a copy of `v2/dev/<B1|B2>/sheet.csv`.
- **Do not open `keys/`.** It is git-ignored; back it up outside the repository.
