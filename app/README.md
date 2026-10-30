# Question-Aware Multimodal OCT Assistant (demo)

A chat app over the OLIVES OCT dataset. **Retrieval organises the evidence; the vision-language model (Qwen2.5-VL
by default) writes the answer.** Every question first goes through the project's question-aware hybrid retrieval
(frozen Phase 1 parser and eligibility rules, Phase 2 hybrid retriever, Phase 3 question router). `app/reasoning.py`
then works out what the original question needs, computes the exact numbers, and gives the model the question, the
structured evidence and the B-scans it must look at. Abstentions never reach a model.

```
question ──► Assistant.route ─┬─ history ──────────► question_router (frozen parse_intent + eligibility) ─┐
 (+ eye, visit,               ├─ scan_visual ──────► selected visit's B-scan or the uploaded scan ─────────┤
  optional upload)            ├─ upload_similarity ► ResNet-50 cosine vs this eye's individual B-scans ───┤
                              └─ followup ─────────► re-shows the previous turn's evidence                 │
                                                                                                           ▼
                 evidence package: same-eye earlier visits, CST/BCVA, OLIVES labels, scans, disclosures
                                       │ abstained? ──► abstention text, no model call
                                       ▼
                 reasoning.plan_question: numeric | labels | visual | visual+metadata (from the ORIGINAL question)
                 reasoning.computed_facts: exact rule results, CST/BCVA differences, visit order (deterministic)
                                       ▼
                 VLM prompt: original question + evidence (visits by label, no IDs) + the scans the plan needs,
                 downscaled and fitted to the context budget ──► the model writes the answer
                                       ▼
                 validation (numbers, visit labels, dates, CST≠edema, label coverage, uncertainty) ──►
                 answer prose | exact values | provenance table (visit IDs, roles, scans sent) — kept separate
```

| Question | Plan | What the model receives |
|---|---|---|
| "When was CST highest?" / "largest CST increase" | numeric | the frozen rule's result and exact differences as text; no scans |
| "When was the edema worst?" | visual + metadata | today's scan + the highest-CST earlier visits as *candidates*; CST is a shortlist cue, not an edema grade; labels |
| "Which visit had the best anatomy?" | visual + metadata | today's scan + the lowest-CST candidates; the model judges anatomy on the scans |
| "How has the retina changed since baseline / over time?" | visual + metadata | today's, baseline (and previous) scans + exact CST/BCVA changes |
| "Which earlier visit's OCT looks most like today's?" | visual + metadata | today's scan + the top embedding candidates (cosine ≠ clinical similarity) |
| "Which earlier visit looks similar, and how did CST change?" | visual + metadata | scans + the exact CST change |
| "What abnormalities are visible in this scan?" | visual | that one scan (+ dataset labels when the visit is labelled) |
| Upload + "what is visible?" | visual | the uploaded scan (external) |
| Upload + "which previous visit looks most like this scan?" | visual | the upload + the best-matching stored B-scan of each top visit; ranking fixed by embeddings |
| "When did IRF first appear?" | — | abstains: labels exist only for the first and last visit |

Without a model (evidence-only mode) questions about recorded values list the computed evidence, and questions that
need visual interpretation say that a vision-language model is needed: they are never answered from metadata alone.

**Uploaded-scan similarity.** The upload is embedded with the same ResNet-50 function (`embed_cached.make_encoder
("resnet50", "gap")`), decoding and preprocessing (`oct_image.load_gray` + `preprocess`: 224 × 224 bicubic, /255,
ImageNet mean/std) and L2 normalisation as the stored embeddings, and compared with every **individual** stored
B-scan (`OCT_EMBEDDINGS` holds one 2048-d vector per B-scan) of the selected eye's visits before the selected visit.
Each visit is scored by its best-matching B-scan (that scan, its file and cosine are kept); visits are ranked by that
score, ties to the more recent visit. Visit-level vectors (central or mean, `OCT_VISIT_MODE`) are not used for this.
The model sees the upload and the top visits' best-matching scans and explains similarities and differences; it does
not rank. If embedding fails, the chat shows the actual error (e.g. a failed weights download). To check the
encoder on your machine: `$env:OCT_TEST_RESNET="1"; python -m pytest -q app\tests -k real_resnet50`.

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

* **Evidence only** (default, `OCT_VLM_PROVIDER=evidence`): no model. Questions about recorded values list the
  computed evidence; questions that need the scans to be interpreted say that a model is needed.
* **Ollama, local and free (recommended for the 8 GB RTX 5050)**: install Ollama ≥ 0.7, then

  ```powershell
  ollama pull qwen2.5vl:3b
  ```

  (about 3.2 GB; `qwen2.5vl:7b`, about 6 GB, for better descriptions). Set `OCT_VLM_PROVIDER=ollama`,
  `OCT_VLM_BASE_URL=http://localhost:11434`, `OCT_VLM_MODEL=qwen2.5vl:3b`. No key, works offline. The native
  provider sends `OCT_VLM_NUM_CTX` as Ollama's `num_ctx`; the OpenAI-compatible `/v1` endpoint cannot set it.
* **OpenRouter, hosted**: `OCT_VLM_BASE_URL=https://openrouter.ai/api/v1` and `OPENROUTER_API_KEY`. Free `:free`
  models are limited to about 20 requests/min and 50 requests/day (1,000/day after buying ≥ $10 credits) and the
  list of free vision models changes, so check before a demo with `python -m app.check_vlm --list-only`.
* **Claude** (`OCT_VLM_PROVIDER=anthropic`, `pip install anthropic`, `ANTHROPIC_API_KEY`): paid.

### What the model receives, and how it is checked

* **The original question**, never a rewritten one (the router's interpretation is used only for retrieval).
* **Structured evidence**: today's visit, the retrieved visits with their role (rule result, CST-shortlisted
  candidate, image-similarity candidate, reference visit), recorded CST/BCVA, the CST/BCVA timeline of the eligible
  visits, OLIVES labels with their coverage ("labels exist for V1, V27 only"), and the disclosures. Visits are named
  by their visit label (V3, W12); visit IDs never appear in the prompt, so provenance stays out of the prose.
* **Exact computed facts**: rule results, differences, ranges and visit order are computed by the app; the model is
  told to use them and not to calculate.
* **Only the scans the question needs**: 1 for a single-scan question; today's + 2–3 candidates for edema, anatomy
  and similarity questions; today's, baseline and previous for change questions; none for numeric questions.
  Identical files are sent once; `OCT_VLM_MAX_IMAGES` caps optional scans.
* **Checks on the answer** (shown as warnings, the text is never rewritten): numbers that are not recorded values or
  differences of them, visit labels not in this eye's evidence, dates (the dataset has none), visit IDs in the
  prose, CST presented as edema severity, no mention of the incomplete finding labels when findings are involved,
  and visual descriptions without any acknowledgement of uncertainty.
* **Provenance** is shown next to the answer as a table (visit ID, label, role, how it was retrieved, CST, BCVA,
  whether its scan was sent, dataset labels), with the exact computed values listed separately.
* **Context budget.** Scans are converted to greyscale and downscaled to `OCT_VLM_IMAGE_MAX_SIDE` px (default
  448; the atlas scans are 504 × 496; never upscaled). The prompt size is estimated conservatively (about one token
  per 14 × 14 pixels, three characters per text token). If the request plus 256 answer tokens exceeds
  `OCT_VLM_CONTEXT` (default: `OCT_VLM_NUM_CTX`, else 4096 for Ollama), the scans are first shrunk step by step down
  to `OCT_VLM_IMAGE_MIN_SIDE` (280), then optional scans are dropped; if it still does not fit, the model is not
  called and the chat says why. At 4096, three scans go at about 336 px.
* **Failures** (budget, rate limit, connection, context errors from the server) show a clear message with the
  exact values and provenance kept; no answer is substituted.

For 3–4 scans at full size on Ollama, raise the window: `OCT_VLM_NUM_CTX=8192` (uses more GPU memory).

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

1. Pick an eye and a current visit in the sidebar (eyes with the most OCT images in the subset are listed first),
   and choose the Ollama provider under **Answer model**.
2. Ask "When was CST highest?" — the model states the frozen rule's result; the exact values and the provenance
   table sit under the answer.
3. Ask "When was the edema worst?" — the model compares today's scan with the highest-CST candidates and says what
   it sees; CST is presented as a shortlist cue, not as edema severity.
4. Ask "How has the retina changed since baseline?" — today's and baseline scans plus the exact CST/BCVA changes.
5. Ask "Which earlier visit's OCT looks most like today's?" — the model compares the embedding candidates' scans.
6. Ask "When did IRF first appear?" — abstention: labels exist only for the first and last visit.
7. Upload a PNG/JPEG B-scan and ask "What abnormalities are visible in this scan?" — a model observation of an
   external image; no history is attributed to it.
8. Ask "What evidence supports your answer?" — the previous turn's evidence, unchanged.

The evidence panel colours sources separately: recorded metadata, image retrieval, dataset labels and model
observations.

## Limitations

* Not a clinical tool. Image similarity is ResNet-50 (ImageNet) feature cosine, not clinical similarity.
* CST is the only quantitative anatomy signal. Edema and anatomy candidates are shortlisted by CST (the frozen
  proxy rule), so a visit outside the shortlist is not shown to the model; the judgement on the scans is the
  model's own and unvalidated.
* A small local model (qwen2.5vl:3b) at a 4096-token window sees downscaled single central B-scans; its visual
  reading can be wrong. The answer checks catch invented numbers, visits and dates, not wrong visual descriptions.
* Finding labels exist only for the first and last visit of each eye; onset questions abstain.
* The demo subset is small (73 images); the larger subset requires the GPU pipeline.
* VLM observations of OCT images are unvalidated; free hosted models can be rate-limited or withdrawn.
* No calendar dates exist in the dataset.
