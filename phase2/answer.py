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
TOP_N = 3            # retrieved visits passed as evidence (similarity-type rankings); on image routes each one's
                     # central B-scan is attached too, so the text evidence and the image payload name the same visits
IMAGE_ROUTES = ("image", "image+proxy_rrf")
# Operations whose answer is one visit (hard conditions: qualifying visits first, then the rest; anchors: previous /
# baseline visit): only rank 1 is passed as evidence.
FILTER_OPS = (I.COMPARE_TO_CURRENT, I.THRESHOLD_FROM_BASELINE, I.LAST_EVENT,
              I.ANCHOR_PREVIOUS, I.ANCHOR_BASELINE)   # anchors: the answer IS one visit; older visits add nothing
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
    "insufficient_labels": "Finding labels (OLIVES per-B-scan annotations) exist only for the first and last visit "
                           "of each eye, so this question cannot be answered for this visit history. The system does "
                           "not infer findings from image similarity.",
    "unsupported_image_finding": "This asks about an OCT finding the system neither extracts from images nor has "
                                 "labels for. Labelled findings: IRF, SRF, DRT/ME, HRF, DRIL, EZ or RPE disruption, "
                                 "serous PED, atrophy, VMT, SHRM (first and last visit only). Image-similarity "
                                 "questions are supported.",
}

CATEGORY_BASIS = {
    "edema_proxy": "Edema question: earlier visits ranked by central subfield thickness (CST) as a disclosed proxy. "
                   "OCT B-scans of the evidence visits are attached for inspection.",
    "pathology": "Finding question: answered only from OLIVES per-B-scan finding labels; image similarity is not "
                 "used to detect findings.",
    "mixed": "Mixed question: image-similarity ranking for the image part and recorded measurements for the "
             "metadata part.",
}

SYSTEM = """You answer a clinician's question about the OCT visit history of one eye, for a research demo.

Use only the evidence in the message: the current visit, the retrieved earlier visits with their measurements \
(CST in micrometres, BCVA in ETDRS letters, study week when recorded) and any attached OCT B-scans, each labelled \
with its visit. The retrieval system has already chosen and ranked the visits and the message states how; the \
visit marked "selected" is the retrieval result, and any "runner-up" visits are context only. Explain what that \
evidence shows for the question. Do not invent visits, dates, measurements, treatments, diagnoses or imaging \
findings, and do not treat a similarity score as a clinical measurement. When you describe an image, describe only \
what is visible in that single B-scan and say that one B-scan does not represent the whole volume. If the evidence \
does not answer the question, say so plainly. The message may also contain disclosures, finding labels \
(OLIVES per-B-scan annotations, given as the fraction of a visit's labelled B-scans) and computed facts such as a \
CST change; use them as stated, and repeat any disclosure that limits the answer.

Keep three kinds of evidence apart and say which one each statement rests on: (1) what you visually observe in the \
attached B-scans (your own observation, not validated), (2) dataset finding labels (OLIVES annotations, given in the \
message), (3) recorded metadata (CST, BCVA, visit order). Never infer disease progression or clinical similarity \
from an image-similarity score. If CST is used as a proxy (for edema or anatomy), say so and do not call it an edema \
or anatomy grade. An image marked UPLOADED SCAN is external: it has no patient, visit, date or history unless the \
message says it matches a dataset visit, so do not attribute any history to it. Say plainly what cannot be \
determined from the evidence.

This is a research prototype, not clinical advice: do not give a diagnosis or a treatment recommendation. Answer \
in 3 to 6 sentences of plain prose. End with one line starting "Evidence used:" that lists the visit IDs you relied \
on (write UPLOADED_SCAN for an uploaded image)."""

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
    send_images = result.route in IMAGE_ROUTES
    ev["current"]["image_attached"] = send_images and result.query_visit_id in images
    if result.final_ranking is None:
        return ev
    n = 1 if (result.route == "metadata" and op in FILTER_OPS) else TOP_N
    if n == 1:
        ev["ranking_basis"] += (" The selected visit is the one meeting the question's condition; other visits are "
                                "not passed on because lower ranks may not meet it.")
    for rank, vid in enumerate(result.final_ranking[:n], 1):
        row = _visit_row(ep, vid)
        row.update(rank=rank, role="selected" if rank == 1 else "runner-up", image_attached=send_images and vid in images,
                   image_cosine=None if vid not in result.image_scores else round(result.image_scores[vid], 4))
        ev["retrieved"].append(row)
    # The image payload is derived from the evidence rows, never from the ranking separately.
    ev["images"] = [dict(visit_id=v["visit_id"], role=role, path=str(images[v["visit_id"]]))
                    for v, role in [(ev["current"], "current")] + [(r, "retrieved") for r in ev["retrieved"]]
                    if v["image_attached"]]
    problems = evidence_inconsistencies(ev)
    assert not problems, f"inconsistent evidence: {problems}"
    return ev


def build_unified_evidence(u, episodes: pd.DataFrame, images: dict[str, Path]) -> dict:
    """One evidence package for a question_router.UnifiedResult (all categories, one schema).

    Rows come from each component in order (metadata, image similarity, finding labels) and are merged by visit, each
    keeping the components that selected it. As in build_evidence, a hard-rule metadata component contributes only
    its selected visit; the image payload is derived from the rows (current visit first) when the category uses
    images, and finding labels are restricted to eligible earlier visits and today."""
    ep = episodes.set_index("visit_id")
    h = u.primary
    ev = dict(question=u.question, status=u.status, category=u.category, route=h.route if h else u.category,
              reason=u.reason or "", intent=u.intent, operation=_operation(u.intent),
              current=_visit_row(ep, u.query_visit_id), retrieved=[], images=[], uses_images=bool(u.uses_images),
              n_candidates=len(u.candidates), eligible=list(u.candidates), eye_id=int(u.eye_id),
              components=[dict(name=c.name, status=c.status, explanation=c.explanation) for c in u.components],
              disclosures=list(u.disclosures), facts=[], biomarkers=list(u.biomarkers),
              guidance=ABSTENTION_GUIDANCE.get(u.status, "") if u.status != "retrieved" else "",
              interpreted_as=getattr(u, "interpreted_as", None))
    ev["ranking_basis"] = CATEGORY_BASIS.get(u.category) or ROUTE_BASIS.get(ev["route"], "")
    ev["current"]["image_attached"] = False
    if u.abstained:
        return ev
    rows: dict[str, dict] = {}
    for c in u.components:
        if not c.ranking:
            continue
        op = _operation(c.hybrid.intent) if c.hybrid is not None else None
        n = 1 if (c.name == "metadata" and op in FILTER_OPS) or c.name == "metadata_change" else TOP_N
        if c.name == "metadata" and n == 1:
            ev["ranking_basis"] += (" The metadata visit shown is the one meeting the question's condition; lower "
                                    "ranks may not meet it and are not passed on.")
        for rank, vid in enumerate(c.ranking[:n], 1):
            tag = f"{c.name} #{rank}"
            if vid in rows:
                rows[vid]["sources"].append(tag)
                continue
            row = _visit_row(ep, vid)
            scores = c.hybrid.image_scores if c.hybrid is not None else {}
            row.update(rank=len(rows) + 1, role="selected" if rank == 1 else "runner-up", sources=[tag],
                       image_cosine=None if vid not in scores else round(scores[vid], 4))
            rows[vid] = row
        if c.name == "metadata_change":
            d = c.details
            unit = " µm" if d["target"] == "cst" else " letters"
            ev["facts"].append(f"{d['target'].upper()} changed from {d['from_value']:.0f}{unit} at {d['from_visit']} "
                               f"to {d['to_value']:.0f}{unit} today ({d['delta']:+.0f}{unit})")
    ev["retrieved"] = list(rows.values())
    ev["current"]["image_attached"] = ev["uses_images"] and u.query_visit_id in images
    for r in ev["retrieved"]:
        r["image_attached"] = ev["uses_images"] and r["visit_id"] in images
    ev["images"] = [dict(visit_id=v["visit_id"], role=role, path=str(images[v["visit_id"]]))
                    for v, role in [(ev["current"], "current")] + [(r, "retrieved") for r in ev["retrieved"]]
                    if v["image_attached"]]
    if ev["uses_images"] and not ev["current"]["image_attached"]:
        ev["disclosures"].append("No B-scan is available for today's visit.")
    missing = [r["visit_id"] for r in ev["retrieved"] if ev["uses_images"] and not r["image_attached"]]
    if missing:
        ev["disclosures"].append(f"No B-scan is available for {', '.join(missing)}.")
    problems = evidence_inconsistencies(ev)
    assert not problems, f"inconsistent evidence: {problems}"
    return ev


def _rows(ev: dict) -> list[dict]:
    """Every evidence item in payload order: uploaded scan (if any), today's visit (if any), retrieved visits."""
    return [x for x in (ev.get("upload"), ev.get("current")) if x] + ev["retrieved"]


def _uses_images(ev: dict) -> bool:
    return ev.get("uses_images", ev["route"] in IMAGE_ROUTES)


def evidence_inconsistencies(ev: dict) -> list[str]:
    """Mismatches between the evidence rows and the image payload (empty list = consistent)."""
    rows = _rows(ev)
    flagged = [v["visit_id"] for v in rows if v.get("image_attached")]
    sent = [i["visit_id"] for i in ev["images"]]
    problems = []
    if sent != flagged:
        problems.append(f"images sent {sent} != visits marked image_attached {flagged}")
    if set(sent) - {v["visit_id"] for v in rows}:
        problems.append(f"images of visits outside the evidence: {sorted(set(sent) - {v['visit_id'] for v in rows})}")
    if not _uses_images(ev) and sent:
        problems.append("images sent on a route that does not use images")
    labelled = {b["visit_id"] for b in ev.get("biomarkers", [])}
    allowed = ({ev["current"]["visit_id"]} | set(ev.get("eligible", []))) if ("eligible" in ev and ev.get("current")) \
        else None
    if allowed is not None and labelled - allowed:
        problems.append(f"finding labels of non-eligible visits: {sorted(labelled - allowed)}")
    return problems


def evidence_summary(ev: dict) -> str:
    """What was actually given to the answer generator, by visit ID."""
    ids = [v["visit_id"] for v in ev["retrieved"]]
    text = f"{len(ids)} earlier visit(s) as text: {', '.join(ids) or 'none'}"
    if _uses_images(ev):
        cur = ev["current"]["visit_id"] if ev.get("current") and ev["current"]["image_attached"] else "none"
        prior = [i["visit_id"] for i in ev["images"] if i["role"] == "retrieved"]
        missing = [v["visit_id"] for v in ev["retrieved"] if not v["image_attached"]]
        up = "uploaded scan, " if ev.get("upload") and ev["upload"].get("image_attached") else ""
        text += (f"; B-scans attached: {up}current visit {cur} and {len(prior)} earlier visit(s)"
                 f"{': ' + ', '.join(prior) if prior else ''}")
        if missing:
            text += f"; no B-scan available for {', '.join(missing)}"
    else:
        text += "; no B-scans (this question type does not use images)"
    if ev.get("biomarkers"):
        text += "; finding labels for " + ", ".join(b["visit_id"] for b in ev["biomarkers"])
    return text


def _fmt(v: dict) -> str:
    wk = f", week {v['week']}" if v["week"] is not None else ""
    cst = "not recorded" if v["cst"] is None else f"{v['cst']:.0f} µm"
    bcva = "not recorded" if v["bcva"] is None else f"{v['bcva']:.0f} letters"
    return f"{v['visit_id']} (visit {v['visit_label']}{wk}): CST {cst}, BCVA {bcva}"


def _fmt_labels(b: dict) -> str:
    parts = [f"{k}: {'present' if x['present'] else 'absent'} ({x['frac'] * 100:.0f}% of "
             f"{b['n_scans'] or '?'} labelled B-scans)" for k, x in b["labels"].items()]
    return f"{b['visit_id']}: " + ("; ".join(parts) or "no label for this finding")


def evidence_text(ev: dict) -> str:
    lines = [f"Question: {ev['question']}"]
    if ev.get("upload"):
        u = ev["upload"]
        lines.append("UPLOADED SCAN: " + (f"byte/pixel-identical to the dataset B-scan of visit {u['matched_visit']} "
                                          f"(eye {u['matched_eye']})" if u.get("matched_visit") else
                                          "external image, not linked to any patient, eye, visit or date"))
    if ev.get("eye_id") is not None:
        lines.append(f"Eye: {ev['eye_id']}")
    if ev.get("current"):
        lines.append(f"Current visit: {_fmt(ev['current'])}")
    if "category" in ev:
        lines.append(f"Question category: {ev['category']}; evidence sources: "
                     + ("; ".join(f"{c['name']} ({c['status']}): {c['explanation']}" for c in ev["components"])
                        or "none"))
    lines += [f"Retrieval route: {ev['route']} (status {ev['status']}{'; ' + ev['reason'] if ev['reason'] else ''})"
              + (f"; interpreted as: {ev['interpreted_as']!r}" if ev.get("interpreted_as") else ""),
              f"How visits were selected: {ev['ranking_basis']}"]
    lines += [f"Disclosure: {d}" for d in ev.get("disclosures", [])]
    if ev["retrieved"]:
        lines.append(f"Retrieved visits ({ev['n_candidates']} candidate visits of this eye), best first:")
        for v in ev["retrieved"]:
            sim = "" if v["image_cosine"] is None else f"; image-embedding cosine {v['image_cosine']:.3f}"
            img = "" if not _uses_images(ev) else (
                "; B-scan attached" if v["image_attached"] else "; no B-scan available")
            src = f" via {', '.join(v['sources'])}" if v.get("sources") else ""
            lines.append(f"  {v['rank']}. [{v['role']}{src}] {_fmt(v)}{sim}{img}")
    elif ev.get("category") != "scan_visual":
        lines.append("No earlier visit was retrieved.")
    lines += [f"Fact: {f}" for f in ev.get("facts", [])]
    if ev.get("biomarkers"):
        lines.append("Finding labels (OLIVES, per visit):")
        lines += [f"  {_fmt_labels(b)}" for b in ev["biomarkers"]]
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
    cat = ev.get("category")
    if cat == "scan_visual":
        labs = "; ".join(_fmt_labels(b) for b in ev.get("biomarkers", []))
        return ("Evidence-only mode cannot interpret images: choose a vision-language model to get a visual "
                "description of this scan." + (f" Dataset finding labels for this scan's visit: {labs}." if labs else
                                               " No dataset finding labels exist for this scan.")
                + "\nEvidence used: " + ", ".join(r["visit_id"] for r in _rows(ev)))
    top = ev["retrieved"][0]
    if cat == "upload_similarity":
        return (f"The embedded visit of eye {ev['eye_id']} most similar to the uploaded scan by image-embedding cosine "
                f"is {_fmt(top)} (cosine {top['image_cosine']:.3f}). This is generic-encoder similarity, not clinical "
                "similarity, and does not show that the scan belongs to this eye. (Evidence-only mode: no language "
                "model was used.)\nEvidence used: UPLOADED_SCAN, " + ", ".join(v["visit_id"] for v in ev["retrieved"]))
    if cat == "pathology":
        lab = next((b for b in ev["biomarkers"] if b["visit_id"] == top["visit_id"]), None)
        return (f"From the OLIVES finding labels, the answer is {top['visit_id']} (visit {top['visit_label']})"
                + (f": {_fmt_labels(lab)}" if lab else "") + ". (Evidence-only mode: no language model was used.)"
                "\nEvidence used: " + ", ".join(v["visit_id"] for v in ev["retrieved"]))
    if cat == "mixed":
        return (f"Image part: the top-ranked earlier visit (image similarity fused with CST closeness) is {_fmt(top)}. "
                + " ".join(f"{f}." for f in ev.get("facts", [])) + f" Current visit: {_fmt(ev['current'])}. "
                "(Evidence-only mode: no language model was used.)\nEvidence used: "
                + ", ".join(v["visit_id"] for v in ev["retrieved"]))
    if cat == "edema_proxy":
        return (f"Using CST as a proxy for edema (not a clinical edema grade), the selected earlier visit is "
                f"{_fmt(top)}. Current visit: {_fmt(ev['current'])}. (Evidence-only mode: no language model was "
                "used.)\nEvidence used: " + ", ".join(v["visit_id"] for v in ev["retrieved"]))
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


def image_label(im: dict) -> str:
    if im["role"] == "uploaded":
        return "B-scan: UPLOADED_SCAN (uploaded by the user)"
    return f"B-scan: {im['visit_id']} ({im['role']} visit)"


def request_parts(ev: dict) -> list[dict]:
    """Provider-neutral user content: each image preceded by its label, then the evidence text. Every VLM backend
    receives exactly this (and SYSTEM), so the evidence package is the only thing a model sees."""
    parts = []
    for im in ev["images"]:
        parts += [dict(type="text", text=image_label(im)), dict(type="image", path=im["path"])]
    return parts + [dict(type="text", text=evidence_text(ev))]


def claude_request(ev: dict) -> dict:
    """Request body for the Anthropic Messages API (built separately so it can be tested without a network)."""
    content = [_image_block(p["path"]) if p["type"] == "image" else {"type": "text", "text": p["text"]}
               for p in request_parts(ev)]
    return dict(model=MODEL, max_tokens=16000, system=SYSTEM, output_config={"effort": "medium"},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                messages=[{"role": "user", "content": content}])


def redact(msg: str, limit: int = 300) -> str:
    """Error text safe to show: credential-like substrings removed, length capped."""
    return SECRET_RE.sub("[redacted]", str(msg))[:limit]


def check_grounding(text: str, ev: dict) -> list[str]:
    """Warnings where the model's text is not supported by the evidence it was given (empty list = no issue)."""
    visits = [v for v in _rows(ev) if "cst" in v]
    known = {v["visit_id"] for v in _rows(ev)} | {b["visit_id"] for b in ev.get("biomarkers", [])}
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


def answer_vlm(ev: dict, vlm) -> dict:
    """Any vision-language model behind an adapter with generate(system, parts) -> object with .text, .finish
    ('stop' | 'length' | 'refusal' | 'error') and .error. Same result handling and grounding as answer_claude."""
    fail = "No generated answer: {}. The retrieved evidence is shown unchanged; no answer was substituted."
    try:
        reply = vlm.generate(SYSTEM, request_parts(ev))
    except Exception as ex:                                   # adapters should not raise; never fabricate if they do
        err = f"{type(ex).__name__}: {redact(ex)}"
        return dict(status="model_error", text=fail.format(f"the model call failed ({err})"), warnings=[], error=err)
    if reply.finish == "error":
        return dict(status="model_error", text=fail.format(f"the model call failed ({redact(reply.error)})"),
                    warnings=[], error=redact(reply.error))
    if reply.finish == "refusal":
        return dict(status="model_refused", text=fail.format("the model declined to answer"), warnings=[],
                    error="refusal")
    text = (reply.text or "").strip()
    if not text:
        return dict(status="model_error", text=fail.format("the model returned no text"), warnings=[],
                    error="empty response")
    warnings = check_grounding(text, ev)
    if reply.finish == "length":
        return dict(status="model_incomplete", text=text, error="max_tokens",
                    warnings=["the answer was cut off at the output limit and is incomplete"] + warnings)
    return dict(status="answered", text=text, warnings=warnings, error=None)


def answer(ev: dict, backend: str = "evidence", client=None, vlm=None) -> dict:
    """backend: 'evidence' (no model), 'claude' (Anthropic SDK; `client` optional) or 'vlm' (adapter in `vlm`).
    An abstaining retrieval is answered with its abstention and never reaches any model."""
    if backend not in ("evidence", "claude", "vlm"):
        raise ValueError(f"unknown backend {backend!r}")
    if backend == "vlm" and vlm is None:
        raise ValueError("backend 'vlm' needs an adapter")
    if ev["status"] != "retrieved":
        res = dict(status="abstained", text=_abstention(ev), warnings=[], error=None)
        backend, used = "none", "abstained: no model call"
    else:
        res = (answer_claude(ev, client) if backend == "claude" else answer_vlm(ev, vlm) if backend == "vlm"
               else dict(status="answered", text=answer_evidence_only(ev), warnings=[], error=None))
        used = evidence_summary(ev)
    return dict(status=res["status"], answer=res["text"], backend=backend, evidence_used=used,
                warnings=res["warnings"], error=res["error"], disclaimer=DISCLAIMER)
