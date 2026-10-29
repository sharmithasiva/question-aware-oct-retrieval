"""Answer generation over hybrid-retriever evidence (Phase 3 demo). Retrieval decides; the model only explains.

  build_evidence(result, episodes, images)  -> dict of exactly what the answer may use
  answer(evidence, backend)                  -> dict(answer, backend, evidence_used, ...)

Rules enforced in code, not left to the model:
  * an abstaining retrieval (any non-`retrieved` status) is answered with the abstention reason; no model call;
  * metadata routes send structured evidence only (no images);
  * image routes send the current visit's central B-scan plus the top retrieved visits' scans, each labelled;
  * the model sees only the listed visits; nothing else about the patient exists for it.
Backends: "evidence" (deterministic text from the evidence; no model) and "claude" (Anthropic API, vision).
"""
from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd

MODEL = "claude-opus-5-5"
TOP_N = 3            # retrieved visits passed as evidence
TOP_IMAGES = 2       # retrieved visits whose image is attached (image routes only)
IMAGE_ROUTES = ("image", "image+proxy_rrf")

SYSTEM = """You answer a clinician's question about the OCT visit history of one eye, for a research demo.

Use only the evidence in the message: the current visit, the retrieved earlier visits with their measurements \
(CST in micrometres, BCVA in ETDRS letters, study week when recorded) and any attached OCT B-scans, each labelled \
with its visit. The retrieval system has already chosen and ranked the visits; explain what that evidence shows \
for the question. Do not invent visits, dates, measurements, treatments, diagnoses or imaging findings. When you \
describe an image, describe only what is visible in that single B-scan and say that one B-scan does not represent \
the whole volume. If the evidence does not answer the question, say so plainly.

This is a research prototype, not clinical advice. Answer in 3 to 6 sentences. End with one line starting \
"Evidence used:" that lists the visit IDs you relied on."""


def _visit_row(ep: pd.DataFrame, vid: str) -> dict:
    r = ep.loc[vid]
    week = None if pd.isna(r.week) else int(r.week)
    return dict(visit_id=vid, visit_label=r.visit_label, visit_index=int(r.visit_index), week=week,
                cst=None if pd.isna(r.cst) else float(r.cst), bcva=None if pd.isna(r.bcva) else float(r.bcva))


def build_evidence(result, episodes: pd.DataFrame, images: dict[str, Path], question: str) -> dict:
    """Evidence for one HybridResult. `images`: visit_id -> central B-scan path (visits without one are absent)."""
    ep = episodes.set_index("visit_id")
    ev = dict(question=question, status=result.status, route=result.route, reason=result.reason,
              intent=result.intent, current=_visit_row(ep, result.query_visit_id), retrieved=[], images=[])
    ev["eye_id"] = int(ep.loc[result.query_visit_id, "eye_id"])
    if result.final_ranking is None:
        return ev
    for rank, vid in enumerate(result.final_ranking[:TOP_N], 1):
        row = _visit_row(ep, vid)
        row.update(rank=rank, image_cosine=None if vid not in result.image_scores
                   else round(result.image_scores[vid], 4))
        ev["retrieved"].append(row)
    if result.route in IMAGE_ROUTES:
        want = [result.query_visit_id] + result.final_ranking[:TOP_IMAGES]
        ev["images"] = [dict(visit_id=v, role="current" if v == result.query_visit_id else "retrieved",
                             path=str(images[v])) for v in want if v in images]
    return ev


def _fmt(v: dict) -> str:
    wk = f", week {v['week']}" if v["week"] is not None else ""
    cst = "not recorded" if v["cst"] is None else f"{v['cst']:.0f} µm"
    bcva = "not recorded" if v["bcva"] is None else f"{v['bcva']:.0f} letters"
    return f"{v['visit_id']} (visit {v['visit_label']}{wk}): CST {cst}, BCVA {bcva}"


def evidence_text(ev: dict) -> str:
    lines = [f"Question: {ev['question']}", f"Eye: {ev['eye_id']}", f"Current visit: {_fmt(ev['current'])}",
             f"Retrieval route: {ev['route']} ({ev['reason'] or ev['status']})", "Retrieved earlier visits, best first:"]
    for v in ev["retrieved"]:
        sim = "" if v["image_cosine"] is None else f"; image similarity {v['image_cosine']:.3f}"
        lines.append(f"  {v['rank']}. {_fmt(v)}{sim}")
    if ev["images"]:
        lines.append("Attached B-scans (central scan of each visit, in this order): "
                     + ", ".join(f"{i['visit_id']} ({i['role']})" for i in ev["images"]))
    return "\n".join(lines)


def _abstention(ev: dict) -> str:
    reason = ev["reason"].strip().rstrip(".")
    return (f"No answer: the retriever abstained ({ev['status']}): {reason}. "
            "The system does not guess when the visit history cannot support an answer.")


def answer_evidence_only(ev: dict) -> str:
    if ev["status"] != "retrieved":
        return _abstention(ev)
    top = ev["retrieved"][0]
    kind = "most similar-looking earlier scan" if ev["route"] in IMAGE_ROUTES else "best-matching earlier visit"
    return (f"The {kind} for this question is {_fmt(top)}. Current visit: {_fmt(ev['current'])}. "
            "(Evidence-only mode: no language model was used.)\nEvidence used: "
            + ", ".join(v["visit_id"] for v in ev["retrieved"]))


def _image_block(path: str) -> dict:
    data = base64.standard_b64encode(Path(path).read_bytes()).decode()
    media = {".webp": "image/webp", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}[
        Path(path).suffix.lower()]
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}


def claude_request(ev: dict) -> dict:
    """Request body for the Anthropic Messages API (built separately so it can be tested without a network)."""
    content = []
    for im in ev["images"]:
        content += [{"type": "text", "text": f"B-scan: {im['visit_id']} ({im['role']} visit)"}, _image_block(im["path"])]
    content.append({"type": "text", "text": evidence_text(ev)})
    return dict(model=MODEL, max_tokens=16000, system=SYSTEM, output_config={"effort": "medium"},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                messages=[{"role": "user", "content": content}])


def answer_claude(ev: dict, client=None) -> str:
    if ev["status"] != "retrieved":
        return _abstention(ev)
    import anthropic
    client = client or anthropic.Anthropic()
    resp = client.beta.messages.create(**claude_request(ev))
    if resp.stop_reason == "refusal":
        return "The language model declined to answer; the retrieved evidence is shown above."
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return text or "The language model returned no text; the retrieved evidence is shown above."


def answer(ev: dict, backend: str = "evidence", client=None) -> dict:
    if backend == "claude":
        text = answer_claude(ev, client)
    elif backend == "evidence":
        text = answer_evidence_only(ev)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    used = "abstained: no model call" if ev["status"] != "retrieved" else (
        f"{len(ev['retrieved'])} retrieved visit(s)" + (f", {len(ev['images'])} B-scan(s)" if ev["images"] else ""))
    return dict(answer=text, backend=backend if ev["status"] == "retrieved" else "none", evidence_used=used)
