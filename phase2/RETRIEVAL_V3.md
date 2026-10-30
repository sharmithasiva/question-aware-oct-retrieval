# Phase 3: unified question-aware hybrid retrieval, pathology probe and paper-style evaluation

Status 2026-10-30. This builds on `RETRIEVAL_V2.md`.
- **Unchanged:** frozen Phase 1 code (`src/`, `tests/`, `scripts/`, `sealed/`, `results/`) and every earlier output.
- **New results:** in new folders under `phase2/results/{e3,pathology,report_v3}/`.
- **Status of the work:**
  - **Implemented and tested offline:** the unified router, evidence fusion and demo; E3; the pathology probe; the RETFound encoder option; the report generator.
  - **Run here:** only on the local 169-image subsets, so every result is a pilot (3 eyes for E3, 8 patients for the probe).
  - **Pending:** the 4,685-scan subset, RETFound weights and expert labels.

## 1. Audit (what existed and what was missing)

| Item | Finding |
|---|---|
| Frozen parser | Maps "edema worst", "swelling", "retina thickest" to `extremum/cst/max`; fluid / cyst / finding questions to `unsupported_image`; any question mixing image and metadata to `clarify` |
| Existing hybrid | Answered every `unsupported_image` question by visual similarity to today, including "when did fluid first appear?", which similarity cannot answer |
| Image encoders | ResNet-50 (ImageNet-supervised, global average pooling) and DINOv2 ViT-S/14 (natural-image self-supervision). Neither was trained on OCT or with any fluid or edema objective, so neither should be described as understanding retinal pathology. |
| Finding labels | OLIVES, 16 binary biomarkers per B-scan (including IRF, SRF, DRT/ME), only for the **first and last visit** of each eye (192 visits). Joined to images by exact archive path. No clinical edema-severity grade exists. |
| Reproducible metrics | E1/E2, Tier A and B1 (frozen rankings) reproduce exactly from saved embeddings; E3 and `large_eval.py` tests check this |
| OCT encoder | `RETFound_mae_natureOCT`: ViT-L/16 (about 300M parameters) pretrained on OCT B-scans; CC BY-NC 4.0; gated on Hugging Face (account plus form); 224-pixel input. Feasible on an 8 GB RTX 5050. Not downloadable in the Claude cloud container (Hugging Face is blocked there). |

## 2. Architecture (one user-facing demo: `phase2/demo.py`)

```
question + eye + current visit
  -> question_router.QuestionAwareRetriever          (outside frozen Phase 1)
       frozen intent.parse_intent  +  a small routing lexicon
       eligibility: strictly earlier visits of the same eye (frozen VisitMemory)
       components, by category:
         metadata          frozen intent_retrieval.retrieve (via hybrid_retriever)
         image_similarity  hybrid_retriever image route (embedding cosine + CST-proxy RRF, frozen S5 design)
         biomarker_labels  OLIVES per-visit finding labels (eligible visits + today only)
         metadata_change   CST/BCVA change from the image-selected visit to today
  -> answer.build_unified_evidence                   one evidence package: rows merged by visit with their
                                                      sources; B-scans derived from the rows; disclosures;
                                                      computed facts; finding labels
  -> answer.answer (evidence-only or Claude vision)  abstentions never reach the model; grounding checks on
                                                      visit IDs, CST/BCVA values and the "Evidence used:" line
  -> demo page: routing and sources table, disclosures, evidence table, B-scans, labels, answer, all ranks
```

## 3. Question-to-retrieval routing

| Question | Category | Evidence used | Behaviour |
|---|---|---|---|
| When was CST highest? | metadata | frozen CST rule | metadata ranking; no images |
| Which earlier OCT looks most similar? | image_similarity | embeddings + CST closeness | B-scans attached; disclosed as not a finding detector |
| When was edema worst? | edema_proxy | frozen CST rule, **disclosed as a proxy** | B-scans attached; DRT/ME and IRF labels shown where they exist |
| When did intraretinal fluid first appear? | pathology | OLIVES IRF labels | **abstains** (`insufficient_labels`) unless every eligible visit is labelled |
| What was the largest consecutive CST increase? | metadata | frozen rule | metadata ranking |
| Compare anatomical appearance across visits | image_similarity | embeddings + CST closeness | as above |
| Which earlier visit looks similar, and how did CST change? | mixed | image ranking + recorded CST of the selected visit and today | both components shown; change computed, not generated |
| Which visit looks similar and why is the sky blue? | partially_supported | none | abstains: never answers only part of a question |
| What changed in the retinal layers? | unsupported_image_finding | none | abstains |
| Injections, leakage, DR severity; unparseable questions | abstain | none | frozen abstention unchanged |

**Hard rules, each covered by tests in `tests/test_question_router.py`:**
- Images never reorder metadata rankings and never replace an abstention.
- Finding labels of future visits are never read.
- The B-scans sent to the model are exactly the evidence visits (plus today).

## 4. Results (pilot, local data only). Full tables: `results/report_v3/local_pilot_v1/report.md`

**E3** (`results/e3/fullhist_dev_v1`): 3 dev eyes (3 patients), 52–55 queries per metadata category, 18 for B1. No confidence intervals (fewer than 5 eyes).
- **Metadata-key categories** (CST highest/lowest/most similar, largest rise, last above today, edema proxy): metadata-only and both hybrids score MRR 1.000 by construction, because the keys come from the same CST column. ResNet image-only scores 0.20–0.43 against a random baseline of 0.24–0.26. **This shows the routing is correct, not clinical understanding.**
- **Structural categories:** previous visit and baseline are 1.000 for metadata-only (rule lookups). Image-only gets 0.48 and 0.15.
- **B1 visual similarity** (human, non-expert): nDCG@5 is 0.823 for the ResNet hybrid, 0.802 for ResNet image-only, 0.786 for `latest_prior` and 0.572 for random. Metadata-only abstains (coverage 0).
- **Question-aware vs safe hybrid:** identical on every evaluable category. The new behaviour (edema disclosure, pathology abstention, mixed decomposition) applies only to categories that have no ground truth, so **no retrieval improvement from question-aware routing is measured.**

**Pathology probe** (`results/pathology/atlas_subset_pilot_v1`): 48 labelled B-scans from 8 patients (6 train, 2 val; not the protocol held-out eyes).
- **IRF:** ResNet probe AUROC 0.78 (95% CI 0.60–0.93) and DINOv2 probe 0.81 (0.65–0.94). CST alone gives 0.62, and the scan-position-only control 0.49.
- **DRT/ME:** ResNet probe 0.90 (0.82–0.98), CST alone 0.74, position-only 0.46.
- **SRF:** N/A (1 positive).
- **How to read it:** generic embeddings carry *some* label-correlated signal beyond scan position, at pilot scale only. The intervals are wide, the scans of one visit are correlated, and this is not evidence of pathology understanding.

**Encoder comparison:** ResNet-50 ≥ DINOv2 on most E3 categories and on B1. RETFound is **N/A** (not run).

**Ablation A–E:** A, B and D were measured (tables above). C and E are N/A.

## 5. Not completed, and why

| Experiment | Status | Why / what is needed |
|---|---|---|
| E3, probe and ablation on `scan_subset_v1` (54 eyes, 550 labelled scans) | pending | images and embeddings live on the GPU machine; commands below |
| RETFound-OCT (systems C and E) | pending | gated weights must be downloaded on the GPU machine |
| Visit-level pathology retrieval ("when did fluid first appear?") | not evaluable | labels exist only for the first and last visit |
| Edema severity | not evaluable | no severity grade exists; CST is a proxy, and DRT/ME is only a binary per-scan label |
| Mixed questions | not evaluable | no ground truth |
| B2 fluid similarity | pending | no grader; the B2 key is exposed in git history (see `RETRIEVAL_V2.md` §8) |
| Fine-tuning | not attempted | no defensible supervision beyond 2 labelled visits per eye; the probe is the label-efficient test |

**Proposed expert annotation (pending).** Two graders would label IRF/SRF presence and central-subfield extent (none, small, large) on the central B-scan of 10 intermediate visits per eye, for about 20 train eyes.
- **Blinding:** blinded to system output, CST and visit order.
- **Workflow:** reuse the v2 packet tooling, with keys kept out of git.
- **What it enables:** visit-level "first appearance", "worst" and "improved" ground truth.

## 6. Commands

Local (CPU, saved embeddings):

```bash
python phase2/e3_question_eval.py --subset phase2/data/atlas_fullhist --emb resnet50=phase2/results/fullhist/resnet50/embeddings.npz --emb dinov2_vits14=phase2/results/fullhist/dinov2_vits14/embeddings.npz --out phase2/results/e3/fullhist_dev_v1
python phase2/pathology_probe.py --subset phase2/data/atlas_subset --emb resnet50=phase2/results/resnet50/embeddings.npz --emb dinov2_vits14=phase2/results/dinov2_vits14/embeddings.npz --out phase2/results/pathology/atlas_subset_pilot_v1
python phase2/report_v3.py --e3 phase2/results/e3/fullhist_dev_v1 --pathology phase2/results/pathology/atlas_subset_pilot_v1 --out phase2/results/report_v3/local_pilot_v1
python phase2/demo.py --eye 25 --question "Which earlier visit looks similar, and how did CST change?" --html demo.html
```

GPU machine, after `bash phase2/run_gpu_pipeline.sh`.
- **torch:** use a CUDA build that supports the RTX 5050. For RTX 50-series GPUs this likely means torch 2.7+ with CUDA 12.8; don't use RETFound's `torch==2.5.1` pin.
- **RETFound weights:** download `RETFound_mae_natureOCT.pth` from the gated Hugging Face repo `YukunZhou/RETFound_mae_natureOCT` after requesting access. The licence is CC BY-NC 4.0, so non-commercial use only.
- **Commands:** install timm, embed with RETFound, then run E3, the probe and the report.

```bash
pip install timm
python phase2/embed_cached.py --subset phase2/data/scan_subset_v1 --encoder retfound_oct --weights /path/to/RETFound_mae_natureOCT.pth
python phase2/e3_question_eval.py --subset phase2/data/scan_subset_v1 --emb resnet50=phase2/results/scan_subset_v1/resnet50/embeddings.npz --emb retfound_oct=phase2/results/scan_subset_v1/retfound_oct/embeddings.npz --out phase2/results/e3/scan_subset_v1_v1
python phase2/pathology_probe.py --subset phase2/data/scan_subset_v1 --emb resnet50=phase2/results/scan_subset_v1/resnet50/embeddings.npz --emb retfound_oct=phase2/results/scan_subset_v1/retfound_oct/embeddings.npz --out phase2/results/pathology/scan_subset_v1_v1
python phase2/report_v3.py --e3 phase2/results/e3/scan_subset_v1_v1 --pathology phase2/results/pathology/scan_subset_v1_v1 --out phase2/results/report_v3/scan_subset_v1_v1
```

- **If the checkpoint won't load:** it may pickle training arguments, in which case loading with `weights_only=True` fails. Add `--trust-checkpoint` only for the official file.
- **Tests (offline):**

```bash
python -m pytest -q
```

```bash
python -m pytest -q phase2/tests/test_question_router.py phase2/tests/test_answer.py phase2/tests/test_answer_offline.py phase2/tests/test_hybrid_retriever.py phase2/tests/test_e3_and_pathology.py phase2/tests/test_retfound_encoder.py phase2/tests/test_embed_cached.py phase2/tests/test_large_eval.py phase2/tests/test_scan_subset.py
```

## 7. Research claim, stated honestly

**Hypothesis (untested at scale):** question-aware fusion of longitudinal clinical metadata and retinal-structure-aware visual representations improves evidence retrieval and grounded question answering compared with metadata-only and generic image-retrieval baselines.

**What the current evidence supports:**
1. Routing questions to the correct evidence source matters a great deal. For metadata-defined questions, images alone are near random.
2. For visual similarity, a ResNet hybrid scores slightly above `latest_prior`, on 3 eyes with one non-expert annotator.
3. Generic embeddings carry some finding-label signal beyond scan position, in a pilot of 8 patients.

**What it does not support:**
- that question-aware routing improves any measured retrieval metric;
- that any encoder understands edema or fluid;
- that RETFound helps.

**Main risks of bias:**
- metadata-derived keys favour metadata systems;
- a single non-expert annotator;
- correlated scans within a visit;
- dev eyes reused across analyses;
- WebP derivatives rather than the original images.
