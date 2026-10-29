"""Answer generation over hybrid-retriever evidence (Phase 3 demo). Retrieval decides; the model only explains.

  build_evidence(result, episodes, images, question) -> dict of exactly what the answer may use
  answer(evidence, backend, client=None)             -> dict(status, answer, backend, evidence_used, warnings, ...)

Rules enforced in code, not left to the model:
  * an abstaining retrieval (any non-`retrieved` status) is answered with the abstention reason and guidance;
    no model call is made;
  * metadata routes send structured evidence only (no images). For hard-rule operations (comparison with today,
    threshold crossing, last event) only the selected visit is passed: the frozen retriever ranks qualifying visits
    first and then the rest by recency, so lower ranks may NOT satisfy the condition;
  * image routes send the current visit's central B-scan plus the top retrieved visits' scans, each labelled, and
    state the ranking basis (image-embedding similarity, optionally fused with CST closeness; no fluid detection);
  * a model failure (missing package or key, network, API error, refusal, empty output) is reported as such with
    the evidence unchanged; no substitute answer is produced, and error text is redacted of credentials;
  * model output is checked against the evidence (visit IDs, CST/BCVA values, the "Evidence used:" line);
    mismatches are returned as warnings, never silently corrected or hidden.
Backends: "evidence" (deterministic text from the evidence; no model) and "claude" (Anthropic API, vision).
Research prototype: nothing here is a diagnosis or a clinically validated output.
"""
from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oct_retrieval import intent as I  # noqa: E402  (frozen, read-only: operation names only)

MODEL = "claude-opus-5-5"
TOP_N = 3            # retrieved visits passed as evidence (similarity-type rankings)
TOP_IMAGES = 2       # retrieved visits whose image is attached (image routes only)
IMAGE_ROUTES = ("image", "image+proxy_rrf")
# Operations whose ranking is "visits meeting a hard condition first, then the rest": only rank 1 is evidence.
FILTER_OPS = (I.COMPARE_TO_CURRENT, I.THRESHOLD_FROM_BASELINE, I.LAST_EVENT)
MEDIA_TYPES = {".webp": "image/webp", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

DISCLAIMER = ("Research prototype on OLIVES data. Not clinically validated and not a diagnosis or treatment "
              "recommendation. Retrieval ranks and similarity scores describe how visits were selected, not "
              "clinical findings.")

ROUTE_BASIS = {
    "metadata": "Selected by the structured metadata rule for this question (CST, BCVA, visit order); "
                "OCT images were not used.",
    "metadata+image_rrf": "Metadata rule ranking fused with OCT image similarity (reciprocal-rank fusion).",
    "image+proxy_rrf": "Image question: ranked by OCT image-embedding similarity to today's central B-scan, fused "
                       "with closeness of CST to today's value. The ranking does not detect fluid or any specific "
                       "finding.",
    "image": "Image question: ranked by OCT image-embedding similarity to today's central B-scan only. The ranking "
             "does not detect fluid or any specific finding.",
    "abstain": "No visit was selected.",
}

ABSTENTION_GUIDANCE = {
    "no_qualifying_visit": "The question was understood, but no earlier visit of this eye meets its condition.",
    "insufficient_data": "The question was understood, but the data it needs are missing for this eye or visit.",
    "unavailable_variable": "The question needs a variable this system does not hold (for example injections, "
                            "leakage or DR severity).",
    "unsupported_image": "This image question cannot be answered here, usually because the needed OCT images "
                         "or embeddings are not available.",
    "clarification_needed": "The question was not understood. Try rephrasing, for example: \"When was the CST "
                            "most similar to today's?\", \"Compare with the previous visit.\" or \"Which earlier "
                            "visit's OCT looks most like today's?\"",
}

SYSTEM = """You answer a clinician's question about the OCT visit history of one eye, for a research demo.

Use only the evidence in the message: the current visit, the retrieved earlier visits with their measurements \
(CST in micrometres, BCVA in ETDRS letters, study week when recorded) and any attached OCT B-scans, each labelled \
with its visit. The retrieval system has already chosen and ranked the visits and the message states how; the \
visit marked "selected" is the retrieval result, and any "runner-up" visits are context only. Explain what that \
evidence shows for the question. Do not invent visits, dates, measurements, treatments, diagnoses or imaging \
findings, and do not treat a similarity score as a clinical measurement. When you describe an image, describe only \
what is visible in that single B-scan and say that one B-scan does not represent the whole volume. If the evidence \
does not answer the question, say so plainly.

This is a research prototype, not clinical advice: do not give a diagnosis or a treatment recommendation. Answer \
in 3 to 6 sentences. End with one line starting "Evidence used:" that lists the visit IDs you relied on."""

VISIT_ID_RE = re.compile(r"\b(?:\d{4}[A-Z]{3}_V\d+|\d{2}-\d{3}_W\d+)\b")
MEASURE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:µm|um|μm|microns?|micrometres?|micrometers?|letters?)\b", re.I)
SECRET_RE = re.compile(r"(sk-ant-[A-Za-z0-9_\-]+|(?i:x-api-key|api[_-]?key|authorization)\s*[:=]\s*(?i:bearer\s+)?\S+)")


def _visit_row(ep: pd.DataFrame, vid: str) -> dict:
    r = ep.loc[vid]
    week = None if pd.isna(r.week) else int(r.week)
    return dict(visit_id=vid, visit_label=str(r.visit_label), visit_index=int(r.visit_index), week=week,
                cst=None if pd.isna(r.cst) else float(r.cst), bcva=None if pd.isna(r.bcva) else float(r.bcva))


def _operation(intent: str) -> str | None:
    parts = intent.split("/")
    return parts[1] if len(parts) > 1 and parts[1] != "None" else None


def build_evidence(result, episodes: pd.DataFrame, images: dict[str, Path], question: str) -> dict:
    """Evidence for one HybridResult. `images`: visit_id -> central B-scan path (visits without one are absent)."""
    ep = episodes.set_index("visit_id")
    op = _operation(result.intent)
    ev = dict(question=question, status=result.status, route=result.route, reason=result.reason or "",
              intent=result.intent, operation=op, current=_visit_row(ep, result.query_visit_id), retrieved=[],
              images=[], ranking_basis=ROUTE_BASIS.get(result.route, ""), n_candidates=len(result.candidates),
              guidance=ABSTENTION_GUIDANCE.get(result.status, "") if result.status != "retrieved" else "")
    ev["eye_id"] = int(ep.loc[result.query_visit_id, "eye_id"])
    if result.final_ranking is None:
        return ev
    n = 1 if (result.route == "metadata" and op in FILTER_OPS) else TOP_N
    if n == 1:
        ev["ranking_basis"] += (" The selected visit is the one meeting the question's condition; other visits are "
                                "not passed on because lower ranks may not meet it.")
    for rank, vid in enumerate(result.final_ranking[:n], 1):
        row = _visit_row(ep, vid)
        row.update(rank=rank, role="selected" if rank == 1 else "runner-up",
                   image_cosine=None if vid not in result.image_scores else round(result.image_scores[vid], 4))
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
             f"Retrieval route: {ev['route']} (status {ev['status']}{'; ' + ev['reason'] if ev['reason'] else ''})",
             f"How visits were selected: {ev['ranking_basis']}"]
    if ev["retrieved"]:
        lines.append(f"Retrieved earlier visits ({ev['n_candidates']} eligible earlier visits of this eye), best first:")
        for v in ev["retrieved"]:
            sim = "" if v["image_cosine"] is None else f"; image-embedding cosine {v['image_cosine']:.3f}"
            lines.append(f"  {v['rank']}. [{v['role']}] {_fmt(v)}{sim}")
    else:
        lines.append("No earlier visit was retrieved.")
    if ev["images"]:
        lines.append("Attached B-scans (central scan of each visit, in this order): "
                     + ", ".join(f"{i['visit_id']} ({i['role']})" for i in ev["images"]))
    return "\n".join(lines)


def _abstention(ev: dict) -> str:
    reason = (ev["reason"] or "no reason given").strip().rstrip(".")
    text = f"No answer: the retriever abstained ({ev['status']}): {reason}."
    if ev.get("guidance"):
        text += " " + ev["guidance"]
    return text + " The system does not guess when the visit history cannot support an answer."


def answer_evidence_only(ev: dict) -> str:
    if ev["status"] != "retrieved":
        return _abstention(ev)
    top = ev["retrieved"][0]
    kind = {"image+proxy_rrf": "top-ranked earlier visit (image similarity fused with CST closeness)",
            "image": "top-ranked earlier visit by image similarity",
            "metadata+image_rrf": "top-ranked earlier visit (metadata rule fused with image similarity)"}.get(
        ev["route"], "earlier visit selected by the metadata rule")
    return (f"The {kind} is {_fmt(top)}. Current visit: {_fmt(ev['current'])}. "
            "(Evidence-only mode: no language model was used.)\nEvidence used: "
            + ", ".join(v["visit_id"] for v in ev["retrieved"]))


def _image_block(path: str) -> dict:
    data = base64.standard_b64encode(Path(path).read_bytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": MEDIA_TYPES[Path(path).suffix.lower()],
                                        "data": data}}


def claude_request(ev: dict) -> dict:
    """Request body for the Anthropic Messages API (built separately so it can be tested without a network)."""
    content = []
    for im in ev["images"]:
        content += [{"type": "text", "text": f"B-scan: {im['visit_id']} ({im['role']} visit)"}, _image_block(im["path"])]
    content.append({"type": "text", "text": evidence_text(ev)})
    return dict(model=MODEL, max_tokens=16000, system=SYSTEM, output_config={"effort": "medium"},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                messages=[{"role": "user", "content": content}])


def redact(msg: str, limit: int = 300) -> str:
    """Error text safe to show: credential-like substrings removed, length capped."""
    return SECRET_RE.sub("[redacted]", str(msg))[:limit]


def check_grounding(text: str, ev: dict) -> list[str]:
    """Warnings where the model's text is not supported by the evidence it was given (empty list = no issue)."""
    visits = [ev["current"]] + ev["retrieved"]
    known = {v["visit_id"] for v in visits}
    warns = []
    unknown = sorted(set(VISIT_ID_RE.findall(text)) - known)
    if unknown:
        warns.append(f"mentions visits that were not in the evidence: {', '.join(unknown)}")
    used = [ln for ln in text.splitlines() if ln.strip().lower().startswith("evidence used:")]
    if not used:
        warns.append('no "Evidence used:" line')
    values = [x for v in visits for x in (v["cst"], v["bcva"]) if x is not None]
    allowed = set(values) | {abs(a - b) for a in values for b in values}
    odd = sorted({m for m in MEASURE_RE.findall(text) if not any(abs(float(m) - x) <= 0.5 for x in allowed)})
    if odd:
        warns.append(f"measurements not found in the evidence (nor as a difference of two evidence values): "
                     f"{', '.join(odd)}")
    return warns


def answer_claude(ev: dict, client=None) -> dict:
    """dict(status, text, warnings, error). status: answered | model_refused | model_incomplete | model_error."""
    fail = "No generated answer: {}. The retrieved evidence is shown unchanged; no answer was substituted."
    try:
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        resp = client.beta.messages.create(**claude_request(ev))
    except ImportError:
        err = "the 'anthropic' package is not installed (pip install anthropic)"
        return dict(status="model_error", text=fail.format(err), warnings=[], error=err)
    except Exception as ex:  # auth, network, rate limit, bad request: report, never fabricate
        err = f"{type(ex).__name__}: {redact(ex)}"
        return dict(status="model_error", text=fail.format(f"the language model call failed ({err})"), warnings=[],
                    error=err)
    text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text").strip()
    if resp.stop_reason == "refusal":
        return dict(status="model_refused", text=fail.format("the language model declined to answer"), warnings=[],
                    error="refusal")
    if not text:
        return dict(status="model_error", text=fail.format("the language model returned no text"), warnings=[],
                    error="empty response")
    warnings = check_grounding(text, ev)
    if resp.stop_reason == "max_tokens":
        return dict(status="model_incomplete", text=text, error="max_tokens",
                    warnings=["the answer was cut off at the output limit and is incomplete"] + warnings)
    return dict(status="answered", text=text, warnings=warnings, error=None)


def answer(ev: dict, backend: str = "evidence", client=None) -> dict:
    if backend not in ("evidence", "claude"):
        raise ValueError(f"unknown backend {backend!r}")
    if ev["status"] != "retrieved":
        res = dict(status="abstained", text=_abstention(ev), warnings=[], error=None)
        backend, used = "none", "abstained: no model call"
    else:
        res = answer_claude(ev, client) if backend == "claude" else dict(
            status="answered", text=answer_evidence_only(ev), warnings=[], error=None)
        used = f"{len(ev['retrieved'])} retrieved visit(s)" + (f", {len(ev['images'])} B-scan(s)" if ev["images"] else "")
    return dict(status=res["status"], answer=res["text"], backend=backend, evidence_used=used,
                warnings=res["warnings"], error=res["error"], disclaimer=DISCLAIMER)
