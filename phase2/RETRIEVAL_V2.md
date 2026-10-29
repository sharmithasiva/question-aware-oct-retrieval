# Phase 2, stage 2: larger image subset, cached embeddings, hybrid retriever

Status 2026-10-29. This is a new experiment (`scan_subset_v1`). The 169-image artifacts, E1/E2, Tier A and the frozen
Tier B rankings are unchanged. No VLM is involved.

## 1. What is new

| File | Role |
|---|---|
| `build_scan_manifest.py` | Offline plan from `visit_images.csv` + `episodes.csv`; inclusion/exclusion counts |
| `fetch_scan_subset.py` | Fetches only the planned scans: resumable, verified with the existing atlas checks, reuses the 169 local images, marks byte-identical images |
| `embed_cached.py` | Streaming batched embedding (GPU if available), SHA-256-keyed cache, decoding audit, optional pooling variants |
| `hybrid_retriever.py` | Metadata + image hybrid with hard eligibility and preserved abstention; inspectable rankings |
| `large_eval.py` | E1/E2-definition sanity metrics, Tier A and post-hoc B1 for any subset, from saved embeddings |
| `tests/test_{scan_subset,embed_cached,hybrid_retriever,large_eval}.py` | 24 tests; no network, and pretrained weights are not needed |

## 2. The subset (`phase2/data/scan_subset_v1/plan_counts.json`)

| Step | Count |
|---|---|
| Visits in the dataset | 1,594 |
| Train-split visits (59 eyes) | 1,008 |
| Eyes excluded (existing `EXCLUDED_EYES`: 29, 32, 46, 81) | 4 |
| Visits excluded by a QC flag (`0247GOD_V3`, missing CST/BCVA) | 1 |
| **Planned** | **4,685 scans = 937 visits × 5 positions (12, 18, 24, 30, 36); 55 eyes, 48 patients** |
| By trial | Prime 23 eyes / 375 visits / 1,875 scans; TREX DME 32 / 562 / 2,810 |
| Already local (copied, not downloaded) | 133 |
| To download | ~4,552 (~230 MB of WebP) |

**Why train only:**
- `val` holds the protocol's held-out eyes (93, 74, 40, 14) and `test` is never used.
- The dev annotation eyes (25, 11, 61) are train eyes, so they are covered.

**What the plan preserves:**
- Every visit of every kept eye, so histories are complete. Eye 18 is the one exception: it loses its flagged visit.
- Scan identity is the archive path, read from `visit_images.csv` and matched exactly by the fetcher.

**Correlated scans:**
- Scans of one eye are never treated as independent.
- Per-eye means and eye-level bootstrap CIs are reported (only with ≥ 5 eyes).
- Patients with two eyes: 7.

## 3. Commands

The images and weights are not reachable from the Claude cloud container: retinaoct.com, zenodo.org,
download.pytorch.org, github.com and huggingface.co are blocked by its network policy. Run steps 2–3 where those
hosts are reachable, for example Colab with a GPU or your own machine. You need torch + torchvision for step 3 only.

1. Build the plan. It is deterministic and already committed.

   ```bash
   python phase2/build_scan_manifest.py
   ```

2. Fetch the images. The download is resumable (re-run after an interruption) and needs network access to
   retinaoct.com. The dry run makes no requests.

   ```bash
   python phase2/fetch_scan_subset.py --dry-run
   python phase2/fetch_scan_subset.py
   ```

3. Embed. The cache in `phase2/cache/` is git-ignored. The first command is the reproducibility check.
   - **What it compares:** the same 73 SHA-256-verified images, through identical decoding, preprocessing and
     pretrained modules. On CPU with the same weights, `embed_cached.py` and `embed_subset.py` agree exactly (tested).
   - **What may differ:** only the hardware and library builds. The reference was made on CPU with torch 2.14.1.
   - **Expected tolerance:** `min_cosine` ≥ 0.999 per image. On a GPU, TF32 convolutions (the PyTorch default on
     Ampere and newer) and different kernels typically give 0.9995–0.99999. Setting `NVIDIA_TF32_OVERRIDE=0`
     brings it to ≥ 0.99999.
   - **Investigate before continuing:** a `min_cosine` below 0.999, or `n_common` other than 73. Causes include a
     different weights version, a changed preprocessing hash, or a mismatched image.

   ```bash
   python phase2/embed_cached.py --subset phase2/data/atlas_fullhist --encoder resnet50 --out phase2/results/repro_check/fullhist_resnet50 --check-against phase2/results/fullhist/resnet50/embeddings.npz
   python phase2/embed_cached.py --subset phase2/data/scan_subset_v1 --encoder resnet50
   python phase2/embed_cached.py --subset phase2/data/scan_subset_v1 --encoder dinov2_vits14
   ```

   Optional pooling variants (only after the baseline is in):

   ```bash
   python phase2/embed_cached.py --subset phase2/data/scan_subset_v1 --encoder resnet50 --pooling gem
   python phase2/embed_cached.py --subset phase2/data/scan_subset_v1 --encoder dinov2_vits14 --pooling patchmean
   ```

4. Evaluate. This runs on CPU in seconds, from saved embeddings. Use the encoder names `resnet50` / `dinov2_vits14`
   as keys so the question_eval systems S3S4/S5 are included. Repeat with `--visit-mode mean` for the multi-scan
   visit embedding.

   ```bash
   python phase2/large_eval.py --subset phase2/data/scan_subset_v1 --emb resnet50=phase2/results/scan_subset_v1/resnet50/embeddings.npz --emb dinov2_vits14=phase2/results/scan_subset_v1/dinov2_vits14/embeddings.npz --visit-mode central --b1 --out phase2/results/scan_subset_v1/eval_central
   ```

5. Tests. The phase2 test `test_consolidated_metrics_match_saved_outputs` rewrites
   `phase2/results/performance/*` with the platform's line endings. On Linux this shows as a whitespace-only diff
   against the committed CRLF files; do not commit it.

   ```bash
   python -m pytest -q
   python -m pytest -q phase2/tests
   ```

## 4. Hybrid retriever policy (`hybrid_retriever.py`)

1. **Eligibility first.** The candidates are the strictly earlier visits of the same eye, taken from the frozen
   `VisitMemory`. Images never add a visit or re-admit one.
2. **The frozen metadata system decides the route.** It uses `parse_intent` + `retrieve`, imported read-only.
   - `retrieved`: the result is the metadata ranking. Image ranks are RRF-fused only for operations listed in
     `fuse_ops`, which is empty by default. Filters and anchors (previous, baseline, "last visit thicker than now",
     thresholds) are therefore never reordered by appearance.
   - `unsupported_image`: the image ranking is RRF-fused with the CST-closeness proxy. This is the pre-specified
     S5 design, and the result is identical to the frozen S5 rankings (tested).
     - It abstains if the current visit has no image, or no candidate has one.
     - Candidates without images keep only their metadata term.
   - Any other abstention is returned unchanged. The image ranking is still computed, for inspection only.
3. **Inspection.** `HybridResult.table()` lists the metadata, proxy, image and final rank, plus the cosine, for every
   candidate.

**Difference from the v0.1 prototype S5 in `question_eval.py`, which stays unchanged.** The prototype falls back to
the image ranking when metadata abstains. On dev Tier A this happens on 4 `no_qualifying_visit` queries, which the
new module abstains on.

## 5. Results so far (exploratory; nothing here is clinical ground truth)

### B1 visual similarity: first evaluation of the completed sheet

Source: frozen pre-label rankings, `question_eval_v2/dev/eval_B1_cda4ea5b_24a90d2768c6/`.

- **Set:** 18 queries, 3 dev eyes.
- **Annotator:** 1 non-expert (`researcher_non_expert`).
- **Relevance:** grade 2. Labels: 73 × 2, 129 × 1, 61 × 0, 1 × U.

| System | Hit@1 | Hit@5 | MRR | nDCG@5 | Δ nDCG@5 vs latest_prior (eyes better/equal/worse) |
|---|---|---|---|---|---|
| S1 latest_prior | 0.611 | 0.944 | 0.773 | 0.786 | — |
| S2p metadata proxy (CST closeness) | 0.667 | 0.889 | 0.776 | 0.782 | −0.004 (1/0/2) |
| S3S4 image ResNet-50 | 0.778 | 0.944 | 0.857 | 0.802 | +0.017 (2/0/1) |
| S3S4 image DINOv2 | 0.500 | 0.944 | 0.678 | 0.682 | −0.103 (1/0/2) |
| S5 hybrid RRF ResNet-50 | 0.778 | 0.944 | 0.857 | **0.823** | +0.037 (2/0/1) |
| S5 hybrid RRF DINOv2 | 0.389 | 0.944 | 0.643 | 0.725 | −0.061 (1/0/2) |
| random (exact expectation) | 0.336 | 0.798 | 0.529 | 0.572 | |

**How to read this:**
- ResNet-50 image retrieval and the ResNet-50 hybrid score above `latest_prior` on this small set.
- DINOv2 scores below it.
- The gains are small (+0.02 to +0.04 nDCG@5), with 2 of 3 eyes better and no CI possible. This is **not evidence
  of an improvement**. It does fix the protocol's dev choice of S5 encoder: ResNet-50.
- The metadata system (S2) abstains on every B1 question, as designed.
- B2 (fluid similarity) is **not evaluable**: no completed B2 sheet exists.

### Reproduction checks (same saved embeddings → same numbers)

- `large_eval.py` on `atlas_fullhist` reproduces:
  - E2: ResNet-50 top-1 = previous 0.299, Spearman (temporal) 0.479, partial Spearman (CST | gap) 0.279;
    DINOv2 0.224 / 0.368 / 0.114.
  - The saved dev Tier A summary for every system.
- `hybrid_retriever.py` reproduces the frozen S2p, S3S4 and S5 rankings of all 36 dev Tier B queries.
- Tier A (dev, metadata answer keys): fusing images into `closest_to_current` lowers MRR from 1.0 (metadata-only,
  which is correct by construction) to 0.64 (ResNet-50). Images do not stand in for CST, so fusion is off by
  default for metadata questions.

### scan_subset_v1

**NOT RUN YET.** It needs the fetch and the embedding (section 3). A synthetic run with random embeddings checked
only the plumbing: 4,685 scans, 54 complete-history eyes, eye-level CIs, 22 s on CPU. Image systems scored near the
random baseline there, which suggests no label leakage through the pipeline. No synthetic number is reported as a
result.

## 6. What the larger subset can and cannot show

**What it can show:**
- **Sanity checks over 54 eyes instead of 3–8, with eye-level CIs.** These cover identity and temporal structure
  (E1/E2 definitions) and Tier A on metadata labels.
- **ResNet-50 vs DINOv2, pooling variants, and central-scan vs 5-scan visit embeddings.** All of these can be
  compared on the B1 judgments, because the dev eyes are in the subset. This comparison is post-hoc and covers
  only 18 queries.

**What it cannot show:** human relevance at scale. B1 covers 3 eyes, and B2 has no grades. Any claim that images
improve image-dependent questions still rests on 18 non-expert-graded queries.

**Held-out:** the held-out val eyes are untouched. Protocol v2 §6 remains the route to a confirmatory result.

## 7. Limitations and open issues

- **Image source.** The images are the atlas's WebP derivatives, not the Zenodo originals, and the atlas is a
  third-party service. If it is unavailable, the fallback is the official Zenodo archive (33.9 GB). The plan's
  `archive_path` column is already the archive-relative path.
- **B2 key exposure.** The committed sample evaluations (`eval_B2_3c3309ac_*/judgments_joined.csv`) contain the B2
  code→visit mapping. See `annotations/ANNOTATION_LOG.md` (2026-10-29) and section 8.
- **E1 definition on the new subset.** With 5 scans per visit, other scans of the same visit fill much of each scan's
  top 5. So top-5 same-eye is even more an identity measure than in E1 (3 scans per visit). Compare it only with
  its own chance value, never with the reported E1 number.
- **No held-out guard in `large_eval.py`.** It evaluates whatever subset it is given. `scan_subset_v1` is train-only
  by construction. Do not point it at a held-out subset.

## 8. Blinding and privacy (development limitations)

**Rules:**
- **Never commit** keys (`phase2/annotations/keys/`, git-ignored), `judgments_joined.csv`, or any per-query file
  that pairs annotation codes with visits.
- `.gitignore` now excludes such files under `phase2/results/`, including the `b1_posthoc_per_query.csv` written by
  `large_eval.py --b1`. Files that were already committed before this rule stay tracked (below).

**B1 (graded):**
- **Key:** regenerated by the developer (Claude) from the frozen generator and seeds, after grading was complete.
  The developer has therefore seen the code→visit mapping.
- **Grades:** not changed. The completed sheet's SHA-256 `cda4ea5b...` is unchanged.
- **Metrics:** the developer has seen the B1 metrics.
- **What counts as pre-registered:** only the frozen-rankings evaluation (`eval_B1_cda4ea5b_24a90d2768c6`). Any
  system or variant scored on B1 afterwards (`large_eval.py --b1`) is post-hoc dev analysis, not confirmatory.
- **Independence:** the same developer wrote the protocol, the systems and the evaluation; there was a single
  non-expert annotator.

**B2 (ungraded):**
- **Exposure:** the full code→visit mapping has been in the pushed history since commit `7b8771e`, in
  `eval_B2_3c3309ac_24a90d2768c6/judgments_joined.csv`, and the query mapping is in `per_query_primary.csv`. Anyone
  with repository access, including a prospective grader, can de-blind the packet.
- **What the developer did:** did not regenerate the B2 key and did not open the B2 mapping file.
- **Before grading:**
  - Either regenerate B2 as a new packet version with new seeds and keep it out of git,
  - or ensure the grader never has repository access, and record that in `ANNOTATION_LOG.md`.
- Removing the file from the published history would require rewriting pushed history. That is a decision for the
  repository owner.
- **v0.1 test needs the key.** `test_dev_packet_sheet_has_no_grades_and_matches_key` fails on a clean clone,
  because the v0.1 key is git-ignored and was not regenerated.
- **No fine-tuning.** There is no supervision signal beyond 18 graded queries, so none was attempted.
