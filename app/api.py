"""Optional FastAPI backend over the same Assistant (the Streamlit app calls the Assistant directly).

  GET  /health                     provider, model, data source (never the API key)
  GET  /eyes                       selectable eyes with visit / OCT / label counts
  GET  /eyes/{eye}/visits          visit table (visit label, study week, CST, BCVA, OCT and label availability)
  GET  /scans/{visit_id}           the visit's central B-scan (image file)
  POST /ask                        form: question, eye?, visit?, backend? (evidence|vlm|claude), file? (PNG/JPEG)

Run (from the repository root):  uvicorn app.api:app --port 8000
"""
from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

from app.assistant import Assistant  # noqa: E402
from app.config import load_config  # noqa: E402
from app.uploads import UploadError, validate_upload  # noqa: E402
from app.vlm import make_vlm  # noqa: E402

app = FastAPI(title="Question-aware OCT assistant (research prototype)")


@lru_cache(maxsize=1)
def get_assistant() -> Assistant:
    cfg = load_config()
    return Assistant(cfg, vlm=make_vlm(cfg.provider, cfg.model, cfg.base_url, cfg.api_key, cfg.timeout, cfg.max_tokens))


def _clean(x):
    """JSON-safe copy: NaN -> None, paths -> file names (no local paths leave the server)."""
    if isinstance(x, dict):
        return {k: (Path(v).name if k == "path" and v else _clean(v)) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, float) and math.isnan(x):
        return None
    if hasattr(x, "item"):
        return x.item()
    return x


@app.get("/health")
def health():
    a = get_assistant()
    return dict(status="ok", provider=a.cfg.provider, model=a.cfg.model or None, subset=a.cfg.subset.name,
                embeddings=a.cfg.embeddings.name, api_key="set" if a.cfg.api_key else "not set")


@app.get("/eyes")
def eyes():
    return _clean(get_assistant().eyes().to_dict("records"))


@app.get("/eyes/{eye}/visits")
def visits(eye: int):
    v = get_assistant().visits(eye)
    if v.empty:
        raise HTTPException(404, f"eye {eye} not found")
    return _clean(v.to_dict("records"))


@app.get("/scans/{visit_id}")
def scan(visit_id: str):
    p = get_assistant().scan_path(visit_id)
    if p is None:
        raise HTTPException(404, f"no B-scan for visit {visit_id}")
    return FileResponse(p)


@app.post("/ask")
async def ask(question: str = Form(...), eye: int | None = Form(None), visit: str | None = Form(None),
              backend: str = Form("evidence"), file: UploadFile | None = File(None)):
    a = get_assistant()
    upload = None
    if file is not None:
        try:
            upload = a.match_upload(validate_upload(await file.read(), file.filename or "upload",
                                                    a.cfg.workdir / "uploads", a.cfg.upload_max_mb))
        except UploadError as ex:
            raise HTTPException(400, f"upload rejected: {ex}")
    if backend == "vlm" and a.vlm is None:
        raise HTTPException(400, "no VLM provider is configured (OCT_VLM_PROVIDER)")
    try:
        t = a.ask(question, eye, visit, upload, backend=backend)
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    return _clean(dict(question=t.question, route=t.route, category=t.category, answer=t.answer, notes=t.notes,
                       evidence={k: t.evidence.get(k) for k in ("status", "reason", "interpreted_as", "eye_id",
                                                                 "current", "retrieved", "images", "components",
                                                                 "disclosures", "facts", "biomarkers", "ranking_basis")},
                       upload=None if upload is None else dict(filename=upload.filename, external=upload.external,
                                                               matched_visit=upload.matched_visit,
                                                               warnings=upload.warnings)))
