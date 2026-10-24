# Phase 2 Handoff — Question-Aware Episodic Retrieval for Longitudinal Retinal OCT Reasoning with VLMs

Written 2026-10-24. Repository root: `C:\question aware retinal oct reasoning` (Windows; **not a git repository**,
so the SHA-256/MD5 manifests are the only change record). Companion document: `docs/PHASE_1_HANDOFF.md`, which
remains authoritative for Phase 1 detail.

> **Disclaimers that must travel with every result:**
> - Phase 1 is a **prototype metadata-only baseline, NOT clinically validated**.
> - fresh_v1 is an **AI-generated preliminary benchmark with limited human review**.
> - The annotator's independence was **not verified**, and the original agreement results are marked potentially compromised.
> - No clinician validated any question, label or result.
> - Nothing in Phase 2 so far has processed an image.

---

## 1. Objective and architecture

**Research question.** Given a natural-language clinical question asked at a patient's *current* OCT visit, can a
system retrieve the relevant **earlier visits of the same eye**, using clinical, temporal and (later) visual
signals? And does that beat fixed history-selection rules (latest visit, fixed window, full history) on retrieval
quality, downstream VLM answer accuracy and cost?

**Architecture:**
- question understanding → structured intent;
- patient-specific **episodic memory** (one record per eye-visit);
- question-aware ranking of prior visits;
- top-K selection;
- VLM answer generation from the current and retrieved OCT images;
- separate evaluation of retrieval, answering and efficiency.

| Phase | Goal | Status |
|---|---|---|
| 1 | Metadata-only retrieval: visit memory, parser, retriever, baselines, sealed evaluation | **Complete and frozen** |
| 2 | Image access and visual retrieval: verify the image ↔ visit mapping, then visual embeddings and a visual-retrieval baseline; later compare metadata / image / combined | **Investigation only; nothing implemented** |
| 3 | VLM answer generation with retrieved images | Not started |

## 2. Repository map (all paths verified to exist on 2026-10-24)

| Area | Path | Notes |
|---|---|---|
| Raw labels (input; unchanged) | `OLIVES_Dataset_Labels/full_labels/` | `Clinical_Data_Images.xlsx`, `Biomarker_Clinical_Data_Images.csv`, `OCT-DME.xlsx`, `OCT-DR.xlsx`, `Summaries _DR_DME_Studies.docx`. `ml_centric_labels/` holds byte-identical copies of the first two. |
| Processed metadata | `data/processed/episodes.csv` (1,594 × 56), `visit_images.csv` (78,185 × 5), `heldout_prime.csv` (660 × 5; leakage/DRSS/injection; never for the retriever) | |
| Phase 1 source (**frozen**) | `src/oct_retrieval/` | `load.py`, `episodes.py`, `side_tables.py`, `questions.py`, `retrievers.py`, `metrics.py`, `evaluate.py`, `evaluate_v2.py`, `ground_truth_v2.py`, `paraphrases.py`, `intent.py`, `intent_retrieval.py`, `evaluate_v3_dev.py`, `audit.py`, `config.py` |
| Evaluation scripts | `scripts/fresh_v1.py`, `scripts/fresh_v1_build_gold.py`, `scripts/fresh_v1_evaluate.py`, `scripts/part_b_check.py` | |
| Tests (**frozen**) | `tests/*.py` (205 tests), `tests/data/` (paraphrases + manifest), `tests/fixtures/retrieval_cases_fresh_v1.jsonl`, `tests/reference/reference_ops.py` | |
| Reports | `reports/RETRIEVAL_REPORT.md`, `RETRIEVAL_REPORT_V2.md`, `RETRIEVAL_REPORT_V3_DEV.md`, **`FRESH_V1_REPORT.md`** | |
| Results | `results/v1_snapshot/`, `results/v2/`, `results/v3_dev/`, `results/fresh_v1/`, `results/fresh_v1_partB/`, `results/fresh_v1_harness_devtest/` (plumbing test only), plus top-level v1 CSVs | |
| Pre-registration, logs, manifests | `results/fresh_v1_prereg/` | `CODE_FREEZE.sha256`, `HARNESS.sha256`, `GOLD_MANIFEST.sha256`, `SEALED_ARTIFACTS.sha256`, `RESULTS_MANIFEST.sha256`, `ANALYSIS_PLAN.md`, `STAGE_LOG.md`, `KNOWN_ISSUES.md`, `PROPOSED_CLARIFICATION_01.md`, `global_fixed_rule.json`, `environment.json` |
| Sealed benchmark (**now exposed**) | `sealed/fresh_v1/` | Do not reuse as a test set |
| Documentation | `docs/DATA_AUDIT.md`, `PHASE1_V2_QUESTION_SPEC.md`, `RETRIEVAL_IMPROVEMENT_PLAN.md`, `DEV_RESULTS_AUDIT.md`, `FRESH_EVALUATION_PROTOCOL.md`, `PHASE1_HARDER_QUESTIONS_PROPOSAL.md`, `PHASE_1_HANDOFF.md`, this file | |
| Notebooks | **none exist** (no `notebooks/` folder, no `.ipynb` anywhere) | |
| Config | `pyproject.toml` | Dependencies: pandas, numpy, openpyxl; dev: pytest. No image or ML dependencies. |

## 3. Phase 1 status (frozen; do not modify)

- **System under test:** the `intent.parse_intent` structured parser plus `intent_retrieval.retrieve`. It has 6 explicit outcomes: retrieved / no qualifying visit / insufficient data / unavailable variable / unsupported image / clarification.
- **Baseline:** the approved global fixed rule is **`latest_prior`** (`results/fresh_v1_prereg/global_fixed_rule.json`). The per-type rules in `results/v3_dev/selected_baselines.json` are an upper reference only.
- **fresh_v1 one-shot results** (`reports/FRESH_V1_REPORT.md`; 120 distinct AI-generated questions):
  - routing accuracy **98/120 = 0.817** [0.738, 0.876];
  - unsafe-retrieval rate **4/50 = 0.080** [0.032, 0.188];
  - over-refusal 14/70;
  - exact intent 47/70 = 0.671.
  - Part C, intent_v3 − `latest_prior` Hit@1 on test patients: positive on 6 of 7 types; `last_vision_as_good` +0.023 [−0.078, +0.106]. Prime types rest on only 7 patients, and the Part C CIs ignore wording variability, so they are too narrow.
- **Limitations:**
  - an AI-generated benchmark with limited human review, and unverified annotator independence;
  - 14 of 120 items were viewed by the developer after the freeze;
  - dev scores are in-sample;
  - v1/v3-dev 1.000 scores are pipeline checks, not evidence.
- **Known issues** (`results/fresh_v1_prereg/KNOWN_ISSUES.md`):
  - **KI-01:** the retriever does not enforce "≥ 2 earlier visits". This cannot affect fresh_v1; fix only under a new freeze.
  - **KI-02:** a Part B harness limitation. The retriever returns a ranking, so a spurious tied candidate below the correct block is invisible. It cannot change Hit@1.
  - **KI-03:** "twenty-six weeks" was parsed as 6 weeks, and "fifty-two weeks" as 2 weeks.
- **Rule:** the Phase 1 source, tests, benchmark, results and manifests **must not be modified**. Do not re-run the v1/v2/v3-dev evaluations (they overwrite outputs, and v3-dev outputs are in the freeze). Never run the sealed one-shot command again.
- **Verify the freeze** (all must report OK; last confirmed OK 2026-10-24):

```bash
python scripts/fresh_v1.py verify-freeze
```

```bash
sha256sum -c results/fresh_v1_prereg/RESULTS_MANIFEST.sha256
```

```bash
sha256sum -c results/fresh_v1_prereg/HARNESS.sha256
```

```bash
python -m pytest -q
```

Expected: `freeze OK`; every manifest line OK; `205 passed`.

## 4. Dataset status

### 4.1 Verified from actual responses or local files
| Fact | Evidence |
|---|---|
| Official source: Zenodo record **7105232** (DOI 10.5281/zenodo.7105232), CC-BY-4.0, open access, no authentication | Zenodo API, HTTP 200 |
| `OLIVES.zip` = **33,913,178,281 bytes**, MD5 `e08f4198aa919d6900bd30525c137933`, URL `https://zenodo.org/api/records/7105232/files/OLIVES.zip/content`; `last-modified: Wed, 14 Jan 2026` | Zenodo API + HEAD |
| `OLIVES_Dataset_Labels.zip` = 6,324,881 bytes, MD5 `e0b14b14f457e4abcafafdd60b8d3952`. The **local copy** (`C:\Users\sharmitha\Downloads\OLIVES_Dataset_Labels.zip`) has the identical MD5 and size | local `md5sum` |
| GitHub README (`olivesgatech/OLIVES_Dataset`, branch `main`): images and labels on that Zenodo DOI; a Hugging Face copy "as separate disease classification and biomarker detection splits" (2025-02-10 update) | raw README |
| Hugging Face `gOLIVES/OLIVES_Dataset`: not gated, MIT, parquet shards, configs `biomarker_detection` / `disease_classification`. Features include `Image`, `BCVA`, `CST`, `Eye_ID`, `Patient_ID`, **no file-path / visit / week column**. 78,822 train rows (vs 78,185 metadata rows) | HF API card metadata only (no shard downloaded) |
| Zenodo serves **HTTP range requests** (HTTP 206 `content-range: bytes 0-0/33913178281`), although HEAD omits `Accept-Ranges` | 1-byte range probe |
| `OLIVES.zip` has **3 entries** (ZIP64): `OLIVES/` (folder); `OLIVES/TREX_DME.zip` (deflate, 19,523,012,677 compressed / 19,549,520,242 uncompressed, local header at byte 69); `OLIVES/Prime_FULL.zip` (deflate, 14,390,164,856 / 14,408,556,388, local header at byte **19,523,012,871**) | last 64 KB, central directory parsed in memory |
| Outer Prime entry: deflate, sizes deferred to a data descriptor (flag bit 3), ZIP64. Inner entries seen carry sizes in their local headers (flags 0x0000) | 3 MB probe |
| First inner entries: `Prime_FULL/`, `Prime_FULL/01-001/`, `Prime_FULL/01-001/sequence_3d_58.npy` (deflate; 29,306,265 compressed / 41,545,856 bytes) | 3 MB probe |
| Local image files: **0 of 78,185** referenced paths exist on this machine; no image archive anywhere checked | local audit |

### 4.2 Image metadata (local, verified)
- **`visit_images.csv` columns:** `visit_id, file_path, image_name, source_file, source_row`.
- **Paths are archive-relative**, starting with "/":
  - TREX: `/TREX DME/<arm>/<subject>/V<n>/<OD|OS>/<image>` (6 segments, every path contains a space);
  - Prime: `/Prime_FULL/<subject>/W<week>/<OD|OS>/<image>` (5 segments).
  - Subject, visit and laterality agree 100 % with `episodes.csv`.
  - No duplicate paths, and one extension per visit.
- **Extensions:** Prime tif 31,994 / png 343; TREX tif 43,741 / jpg 2,107.
- **Irregularities:**
  - 98 double-index names (`TREXL_000004_000022.tif`) in `0228TOS_V11` (eye 45, val) and `0230TOS_V1` (eye 46, train);
  - index gaps in `02-018_W24` (eye 81, train), `0219MOS_V16` (eye 29, train) and `0225GOS_V7` (eye 8, test);
  - 97-image visits `0248MOD_V15`/`V16` (eye 32, train).
- **Image counts per split:**

| Trial | Split | Eyes | Visits | Images |
|---|---|---|---|---|
| Prime_FULL | train / val / test | 24 / 8 / 8 | 389 / 134 / 137 | 19,058 / 6,566 / 6,713 |
| TREX DME | train / val / test | 35 / 10 / 11 | 619 / 149 / 166 | 30,426 / 7,301 / 8,121 |

- **Eye 79 (`02-016`, val): unresolved.**
  - OS in the clinical table (686 paths, 14 visits) and in the OCT-DR demographics.
  - OD in the biomarker CSV (98 paths, W0 and W52), and those paths match the clinical paths exactly once OD is replaced with OS.
  - No laterality has been chosen.

### 4.3 Uncertain or unverified
- Whether `OLIVES.zip` contains **any** individual `.tif`/`.png`/`.jpg` B-scans matching our paths. Only the first inner entry has been seen.
- The internal layout of `TREX_DME.zip` (never probed).
- Whether the archive was restructured on 2026-01-14 relative to the labels (only an inference from `last-modified`).
- What the Hugging Face rows correspond to, and whether any reliable mapping to visits exists.
- Colab ↔ Zenodo throughput.
- Whether Colab's `bsdtar` handles this nested ZIP. The Windows bsdtar 3.8.9 test **failed** with "(null)" and then "OLIVES/Prime_FULL.zip: Not found in archive".

## 5. Important Phase 2 discovery: `sequence_3d_58.npy`
- **Observed from the `.npy` header only** (dtype and shape, read from ≤ 4 KB decompressed in memory): `Prime_FULL/01-001/sequence_3d_58.npy`, **shape `(18, 46, 224, 224)`, dtype `uint8`** (`|u1`), 41,545,856 bytes.
- **Our metadata** says `01-001` is **eye 58, OS, test split**, with **20 visits × 49 B-scans** (W0–W52 every 4 weeks, then W68, W76, W84, W92, W100, W104).
- **Consequences:**
  - The array's first two dimensions (18, 46) do **not** match 20 × 49.
  - **Do not assume that index 0 is a visit, that index 1 is a B-scan, or that indices follow chronological order.** No slice-to-visit or slice-to-scan mapping is verified. The images also appear resized to 224 × 224.
  - "58" matching our Eye_ID 58 is a plausible naming link, **not a verified one**.
  - If the archive is organised per eye as arrays, eye 79's OD/OS question may not be answerable from folder names.
- **Exposure note:** the few in-memory bytes beyond the header belonged to a **test-split** eye. They were never decoded, displayed or saved.

## 6. Implementation status

| Item | Status |
|---|---|
| Phase 1 system, evaluation, manifests | Implemented, tested, frozen |
| Phase 2 local path/schema audit | Done as **read-only checks**; no files created |
| Remote checks (HEAD, range probes, central directory, npy header) | Done. About **25.3 MB** transferred over 5 bounded range requests (largest 8 MB); nothing persisted |
| Streaming Prime lister `olives_prime_lister.py` (standard library only) | **Written and tested, but not in the repository.** It lives only in the previous session's scratchpad: `C:\Users\SHARMI~1\AppData\Local\Temp\claude\C--question-aware-retinal-oct-reasoning\8cc7b8fd-e5c8-499e-a6aa-be493773ecee\scratchpad\olives_prime_lister.py`, SHA-256 `8629cace40f64ba66dee09f7b5db498e0811d657858ed39552ef637b33cdfe02`. The full text is in **Appendix A**. Tested: an 8 MB real bounded run correctly gives INCOMPLETE (exit 2), and an offline synthetic nested ZIP gives COMPLETE / INCOMPLETE as expected. **Not tested:** the full 14.39 GB pass, so the real completion path has never run. |
| Full Prime listing pass (~14,390,165,410 bytes) | **Not run.** It needs your approval. A documentation check of what `sequence_3d_*.npy` contains was recommended first and has not been done. |
| Any image extraction, visualisation, embedding, notebook | **Not implemented** |
| Pretrained encoder choice (e.g. RETFound or a generic model) | **Proposed only**; access requirements unverified |

## 7. Next task: `notebooks/01_image_data_audit.ipynb` (Google Colab)

**Goal:** a self-contained visual-pipeline prototype, kept **separate from longitudinal retrieval evaluation** until mappings are verified.

1. **Input: one small NumPy array** (no full archive). Two options, and **the researcher must choose**:
   - **(a)** A researcher-provided sample `.npy` uploaded to Colab.
   - **(b)** A bounded fetch of only the first inner member, `sequence_3d_58.npy`. This would mean reading the start of the Prime entry until that member is complete. The member is 29,306,265 bytes compressed inside the inner ZIP, so the outer transfer is **estimated (not verified) at ~30 MB**. This is a **test-split eye**: use it only as a pipeline smoke test, never for tuning or evaluation, and record that in the notebook.
2. **Inspect:** shape, dtype, min/max/mean per axis, constant or empty slices, and duplicates across indices. Report findings neutrally, with **no** visit or scan interpretation.
3. **Visualise** a few slices, e.g. a grid over `[i, j]` for a handful of i and j. Note orientation and intensity range; check whether the resizing to 224 × 224 looks like an aspect-ratio distortion.
4. **Preprocessing check:** convert uint8 to float, normalise and replicate to 3 channels for a standard encoder, and document each choice.
5. **Encoder smoke test:** load one pretrained image encoder available in Colab without gated access (e.g. a torchvision ImageNet ResNet-50). Embed a handful of slices, then check embedding shape, determinism and cosine-similarity sanity. Domain-specific encoders (RETFound) only after their weight access and licence are verified.
6. **Outputs:** keep them in Colab (`/content/`). Nothing goes into the frozen Phase 1 folders.

**Explicitly out of scope until mappings are verified:** linking arrays to `episodes.csv` visits, building visit-level image memory, and any retrieval evaluation or comparison with Phase 1.

## 8. Execution instructions

| Where | What |
|---|---|
| **Local** (Windows; Python 3.12.10, pandas 2.2.2, numpy 2.2.4, openpyxl 3.1.5, pytest 9.1.0) | Freeze verification and tests (§3); creating the notebook file in a new `notebooks/` folder; reading metadata CSVs. **No** image or ML packages installed locally. |
| **Colab** (researcher runs) | Upload or fetch the sample array; run the notebook cells; any `pip install` (Colab only); GPU optional for the smoke test. The default Colab image's preinstalled packages (numpy, matplotlib, torch, torchvision) are **assumed, not verified**. |

**Option (b) fetch:** it needs a bounded variant of the Appendix A logic that stops after the first `.npy` member completes and saves only that member. **This does not exist yet**; it must be written, reviewed and approved before use.

**Full Prime listing** (only after approval and the documentation check): copy Appendix A to `/content/olives_prime_lister.py` in Colab, then run:

```bash
OLIVES_OUT=/content/prime_listing.tsv python /content/olives_prime_lister.py
```

- **Success:** the last line is `# SUMMARY {"status": "COMPLETE", ...}` and the exit code is 0.
- **Failure:** `INCOMPLETE` with a reason, exit code 2, or a missing SUMMARY line. Any of these means re-run from the start (no resume).
- **Transfer:** about 14.39 GB. Storage: listing only.

## 9. Constraints (carry forward)
- No changes to frozen Phase 1 source, tests, benchmark, results or manifests. New work goes in new paths (e.g. `notebooks/`, `docs/`).
- No full archive download (33.9 GB) to the laptop or to Google Drive. No image extraction beyond an explicitly approved small sample.
- No fabricated or assumed visit/scan mappings for `.npy` arrays. No resolution of eye 79 by assumption.
- Prefer train/val eyes. Any test-split image use must be flagged and limited to non-evaluative smoke tests.
- No claims of clinical validation, novelty or retrieval benefit from images until measured under a new, pre-registered evaluation.
- Bounded network tests must stay within an agreed size and be reported with actual responses.

## 10. Next-session prompt (copy into a fresh Claude chat)

```text
I'm continuing my research project "Question-Aware Episodic Retrieval for Longitudinal Retinal OCT Reasoning
with Vision-Language Models". The repository is at C:\question aware retinal oct reasoning.

Read docs/PHASE_2_HANDOFF.md completely (and docs/PHASE_1_HANDOFF.md only if you need Phase 1 detail). Do not
repeat broad dataset research. The source, archive structure and access facts are already verified there.

Then:
1. Verify the Phase 1 freeze with: python scripts/fresh_v1.py verify-freeze, sha256sum -c
   results/fresh_v1_prereg/RESULTS_MANIFEST.sha256 and python -m pytest -q. Stop if anything fails.
2. Ask me which input option to use for the notebook: (a) I upload a small sample .npy to Colab, or (b) a bounded
   fetch of only sequence_3d_58.npy (~30 MB estimated, a test-split eye, smoke test only). Do not fetch anything
   before I answer.
3. Implement notebooks/01_image_data_audit.ipynb for Google Colab as specified in section 7 of the handoff:
   load one small NumPy array, report shape/dtype/statistics neutrally, visualise a few slices, check
   preprocessing, and run a pretrained image-encoder smoke test (non-gated model first).

Constraints: do not modify any frozen Phase 1 file; do not download the full archive; do not assume array indices
are visits or B-scans; keep this prototype separate from retrieval evaluation; do not claim clinical validation.
Report exactly what you ran and what I must run in Colab.
```

---

## Appendix A — `olives_prime_lister.py` (verbatim copy of the tested scratchpad file)

The SHA-256 of the original file is `8629cace40f64ba66dee09f7b5db498e0811d657858ed39552ef637b33cdfe02`. A copy pasted from this page may differ only in line endings; compare contents if the hash differs.

```python
"""Stream-list the inner Prime_FULL.zip inside Zenodo OLIVES.zip without storing it. Standard library only.

Reads the outer entry by HTTP range, inflates it on the fly, walks the inner ZIP's local headers, records each
entry (name, method, flags, sizes, CRC) and, for .npy members, only the npy header (dtype, shape; first <=4 KB).
Nothing but the listing is written. Completion requires: outer deflate stream finished, inner central directory
reached, outer data-descriptor CRC/size match the inflated bytes, and inflated size == expected.

Env: OLIVES_OUT (path or "-" for stdout; default /content/prime_listing.tsv), OLIVES_MAX_BYTES (cap; 0 = none),
     OLIVES_START / OLIVES_EXPECT_USIZE / OLIVES_URL (defaults = values verified from the Zenodo central directory).
"""
import ast, json, os, struct, sys, urllib.request, zlib

URL = os.environ.get("OLIVES_URL", "https://zenodo.org/api/records/7105232/files/OLIVES.zip/content")
START = int(os.environ.get("OLIVES_START", "19523012871"))           # local header of OLIVES/Prime_FULL.zip
EXPECT_USIZE = int(os.environ.get("OLIVES_EXPECT_USIZE", "14408556388"))
ENTRY = b"OLIVES/Prime_FULL.zip"
MAX = int(os.environ.get("OLIVES_MAX_BYTES", "0")) or None
OUT = os.environ.get("OLIVES_OUT", "/content/prime_listing.tsv")


def npy_header(m, method):
    try:
        b = zlib.decompressobj(-15).decompress(m) if method == 8 else m
        if b[:6] != b"\x93NUMPY":
            return "", ""
        hl = struct.unpack("<H", b[8:10])[0] if b[6] == 1 else struct.unpack("<I", b[8:12])[0]
        o = 10 if b[6] == 1 else 12
        h = ast.literal_eval(b[o:o + hl].decode("latin1").strip())
        return h["descr"], str(h["shape"])
    except Exception:
        return "?", "?"


def main(urlopen=urllib.request.urlopen):
    out = sys.stdout if OUT == "-" else open(OUT, "w")
    out.write("name\tmethod\tflags\tcsize\tusize\tcrc32\tnpy_dtype\tnpy_shape\n")
    rng = f"bytes={START}-" + (str(START + MAX - 1) if MAX else "")
    resp = urlopen(urllib.request.Request(URL, headers={"Range": rng}), timeout=120)
    print(f"# HTTP {resp.status} {resp.headers.get('Content-Range')}", file=sys.stderr)
    raw = resp.read(30)
    nl, xl = struct.unpack("<HH", raw[26:30])
    raw += resp.read(nl + xl)
    assert raw[:4] == b"PK\x03\x04" and raw[30:30 + nl] == ENTRY, "outer entry header not found at START"
    dec = zlib.decompressobj(-15)
    crc, total_in, total_out = 0, len(raw), 0
    buf, skip, head, head_need, cur, n_entries, inner_cd, err = b"", 0, None, 0, None, 0, False, None

    def emit(c, h):
        dt, sh = npy_header(h, c[1]) if c[0].endswith(".npy") and h else ("", "")
        out.write(f"{c[0]}\t{c[1]}\t{c[2]}\t{c[3]}\t{c[4]}\t{c[5]:08x}\t{dt}\t{sh}\n")

    try:
        while not dec.eof:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            total_in += len(chunk)
            data = dec.decompress(chunk)
            crc = zlib.crc32(data, crc)
            total_out += len(data)
            buf += data
            while True:
                if skip:
                    take = min(skip, len(buf))
                    if head is not None and head_need:
                        h = buf[:min(take, head_need)]
                        head += h
                        head_need -= len(h)
                    buf = buf[take:]
                    skip -= take
                    if skip:
                        break
                    emit(cur, head)
                    head = None
                if inner_cd:
                    buf = b""
                    break
                if len(buf) < 30:
                    break
                sig = buf[:4]
                if sig == b"PK\x01\x02":
                    inner_cd = True
                    continue
                if sig != b"PK\x03\x04":
                    err = f"unexpected signature {sig!r} after {n_entries} entries"
                    break
                _, _, fl, m, _, _, c32, cs, us, n2, x2 = struct.unpack("<IHHHHHIIIHH", buf[:30])
                if len(buf) < 30 + n2 + x2:
                    break
                name = buf[30:30 + n2].decode("utf8", "replace")
                ex = buf[30 + n2:30 + n2 + x2]
                q = 0
                while q + 4 <= len(ex):
                    hid, ln = struct.unpack("<HH", ex[q:q + 4])
                    if hid == 1:
                        v = list(struct.unpack("<" + "Q" * (ln // 8), ex[q + 4:q + 4 + 8 * (ln // 8)]))
                        if us == 0xFFFFFFFF:
                            us = v.pop(0)
                        if cs == 0xFFFFFFFF:
                            cs = v.pop(0)
                    q += 4 + ln
                if fl & 8:
                    err = f"inner entry {name!r} uses a data descriptor (unsupported by this lister)"
                    break
                buf = buf[30 + n2 + x2:]
                n_entries += 1
                cur = (name, m, fl, cs, us, c32)
                head = b"" if name.endswith(".npy") else None
                head_need = 4096 if head is not None else 0
                if cs == 0:
                    emit(cur, head)
                    head = None
                else:
                    skip = cs
            if err:
                break
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    tail = dec.unused_data if dec.eof else b""
    if dec.eof and len(tail) < 24:
        tail += resp.read(24 - len(tail))
    desc_ok = None
    if dec.eof and tail[:4] == b"PK\x07\x08":
        d_crc, _d_cs, d_us = struct.unpack("<IQQ", tail[4:24])
        desc_ok = d_crc == crc and d_us == total_out
    ok = dec.eof and inner_cd and not err and desc_ok and total_out == EXPECT_USIZE
    summary = dict(status="COMPLETE" if ok else "INCOMPLETE", error=err, compressed_bytes_read=total_in,
                   inflated_bytes=total_out, expected_inflated=EXPECT_USIZE, outer_stream_finished=dec.eof,
                   inner_central_directory_reached=inner_cd, outer_crc_and_size_match=desc_ok,
                   entries_listed=n_entries, byte_cap=MAX)
    out.write("# SUMMARY " + json.dumps(summary) + "\n")
    out.flush()
    print("# SUMMARY " + json.dumps(summary), file=sys.stderr)
    return summary


if __name__ == "__main__":
    s = main()
    sys.exit(0 if s["status"] == "COMPLETE" else 2)
```
