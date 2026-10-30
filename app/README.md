# Question-Aware Multimodal OCT Assistant (demo)

A chat app over the OLIVES OCT dataset. Every question goes through the project's question-aware hybrid retrieval
(frozen Phase 1 parser and eligibility rules, Phase 2 hybrid retriever, Phase 3 question router) **before** any
model sees it. A vision-language model (VLM) can only phrase an answer from the evidence package it receives:
retrieved visits, recorded CST/BCVA, dataset finding labels and the B-scans. Abstentions never reach a model.

```
question ──► Assistant.route ─┬─ history ──────────► question_router (frozen parse_intent + eligibility) ─┐
 (+ eye, visit,               ├─ scan_visual ──────► selected visit's B-scan or the uploaded scan ─────────┤
  optional upload)            ├─ upload_similarity ► ResNet-50 cosine of the upload vs this eye's visits ──┤
                              └─ followup ─────────► re-shows the previous turn's evidence                 │
                                                                                                           ▼
                        evidence package (answer.build_unified_evidence: rows, images, labels, disclosures)
                                       │ abstained? ──► abstention text, no model call
                                       ▼
                 backend: evidence-only │ OpenAI-compatible VLM (OpenRouter / Ollama / vLLM) │ Claude
                                       ▼
                 grounding check (visit IDs, CST/BCVA values, "Evidence used:" line) ──► answer + evidence panel
```

| Question type | Category | What controls the answer |
|---|---|---|
| "When was CST highest?" / "largest CST increase" | `metadata` | frozen rule over recorded CST of eligible earlier visits |
| "When was the edema worst?" | `edema_proxy` | CST used as a proxy; disclosed every time (CST does not measure edema) |
| "Which earlier visit's OCT looks most like today's?" | `image_similarity` | ResNet-50 cosine fused with CST closeness (RRF); similarity ≠ clinical similarity |
| "When was the retinal anatomy best?" | `anatomy_proxy` | lowest CST, disclosed as a proxy (no anatomy grade exists) |
| "How has the retina changed since baseline?" | `trajectory` | baseline + previous visit, recorded CST deltas, images |
| "When did IRF first appear?" | `pathology` | OLIVES labels exist only for first/last visit → abstains (`insufficient_labels`) |
| "What abnormalities are visible in this scan?" | `scan_visual` | the VLM's visual observation, labelled as such, next to dataset labels |
| Upload + "which visit looks most like this scan?" | `upload_similarity` | cosine vs the selected eye's embedded visits |

Uploaded images are **external**: they are never linked to a patient, eye, visit or date unless they are byte- or
pixel-identical to a dataset B-scan (then the answer says which visit it matched). The dataset has no calendar
dates; visits are shown by their visit labels.

## Setup (Windows PowerShell)

From the repository root, in the environment that already has PyTorch with CUDA:

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
pip install -r app\requirements-app.txt
```

```powershell
Copy-Item .env.example .env
```

Edit `.env` and choose a provider (see below). Real environment variables override `.env`, for example
`$env:OCT_VLM_PROVIDER = "evidence"`. `.env` and `app\.runtime\` (uploads, hash cache) are git-ignored.

### Provider

* **Evidence only** (default, `OCT_VLM_PROVIDER=evidence`): no model; answers are written from the evidence. It
  cannot describe image content and says so.
* **Ollama, local and free (recommended for the 8 GB RTX 5050)**: install Ollama ≥ 0.7, then

  ```powershell
  ollama pull qwen2.5vl:7b
  ```

  (about 6 GB; `qwen2.5vl:3b`, about 3.2 GB, if memory is tight). Set `OCT_VLM_PROVIDER=openai_compatible`,
  `OCT_VLM_BASE_URL=http://localhost:11434/v1`, `OCT_VLM_MODEL=qwen2.5vl:7b`. No key, works offline.
* **OpenRouter, hosted**: `OCT_VLM_BASE_URL=https://openrouter.ai/api/v1` and `OPENROUTER_API_KEY`. Free `:free`
  models are limited to about 20 requests/min and 50 requests/day (1,000/day after buying ≥ $10 credits) and the
  list of free vision models changes, so check before a demo with `python -m app.check_vlm --list-only`.
* **Claude** (`OCT_VLM_PROVIDER=anthropic`, `pip install anthropic`, `ANTHROPIC_API_KEY`): paid.

Check the configured model with one real B-scan (the key is never printed):

```powershell
python -m app.check_vlm
```

### Run

```powershell
streamlit run app\streamlit_app.py
```

Optional JSON API (`/health`, `/eyes`, `/eyes/{eye}/visits`, `/scans/{visit_id}`, `POST /ask` with form fields
and an optional `file`):

```powershell
uvicorn app.api:app --port 8000
```

The default data is the committed full-history subset (`phase2/data/atlas_fullhist`); the eyes offered are the
protocol's train eyes (`OCT_SPLITS=train`), so held-out eyes stay out of the demo. After the GPU pipeline has
produced the larger subset, point `OCT_SUBSET` and `OCT_EMBEDDINGS` at it (see `.env.example`).

### Tests

```powershell
python -m pytest -q app\tests
```

## Demo workflow

1. Pick an eye and a current visit in the sidebar (eyes with the most OCT images in the subset are listed first).
2. Ask "When was CST highest?" — the panel shows the selected visit, its CST, the timeline and the rule used.
3. Ask "When was the edema worst?" — same visit, with the CST-proxy disclosure.
4. Ask "Which earlier visit's OCT looks most like today's?" — similar scans with cosines and a similarity
   disclaimer.
5. Ask "When did IRF first appear?" — abstention: labels exist only for the first and last visit.
6. Upload a PNG/JPEG B-scan and ask "What abnormalities are visible in this scan?" — the VLM's observation is
   labelled as a model observation of an external image; no history is attributed to it.
7. Ask "Which earlier visit looks most like this scan?" — cosine ranking of the selected eye's visits.
8. Ask "What evidence supports your answer?" — the previous turn's evidence, unchanged.

The evidence panel colours sources separately: recorded metadata, image retrieval, dataset labels and model
observations.

## Limitations

* Not a clinical tool. Image similarity is ResNet-50 (ImageNet) feature cosine, not clinical similarity.
* CST is the only quantitative anatomy signal; edema and "anatomy" questions are CST proxies.
* Finding labels exist only for the first and last visit of each eye; onset questions abstain.
* The demo subset is small (73 images); the larger subset requires the GPU pipeline.
* VLM observations of OCT images are unvalidated; free hosted models can be rate-limited or withdrawn.
* No calendar dates exist in the dataset.
