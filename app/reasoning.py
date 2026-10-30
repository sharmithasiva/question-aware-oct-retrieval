"""Question-aware answer generation: retrieval organises the evidence, the vision-language model writes the answer.

  plan_question   what the original question needs: recorded metadata, OCT B-scans, finding labels, the uploaded scan
  computed_facts  exact numbers (rule results, CST/BCVA differences, visit order), computed here, never by the model
  build_prompt    original question + structured evidence (visits named by their visit label, no visit IDs) + the
                  B-scans the plan needs, within the adapter's context budget
  generate        one model call; the prose is checked against the evidence (numbers, visit labels, dates, CST/edema,
                  label coverage, uncertainty) and returned with provenance and computed facts kept OUTSIDE the prose

The retrieval itself (frozen Phase 1 rules, eligibility, Phase 2/3 rankings) is unchanged: this module only decides
which of the retrieved evidence the model receives and how. Abstentions never reach a model; a question that needs
visual interpretation is never answered from metadata alone (without a model it says that a model is needed).
"""
from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase2"))
import answer as A  # noqa: E402

UPLOAD = "UPLOADED_SCAN"
# words that ask about what the retina looks like (as opposed to recorded CST/BCVA values)
VISUAL_RE = re.compile(r"\b(anatom\w*|retina\w*|appear\w*|look(s|ed|ing)?|visible|visibly|see|seen|describ\w*|"
                       r"descript\w*|fluid|o?edema\w*|swell\w*|cyst\w*|lesion\w*|abnormal\w*|structur\w*|"
                       r"morpholog\w*|layers?|scans?|images?|b-?scans?|oct|irf|srf|drusen|atroph\w*|scar\w*)\b", re.I)
FINDING_RE = re.compile(r"\b(fluid|irf|srf|cyst\w*|o?edema\w*|swell\w*|drt|hrf|dril|ellipsoid|ez|rpe|ped|atroph\w*|"
                        r"vmt|shrm|finding\w*|label\w*|lesion\w*|abnormal\w*)\b", re.I)
VISIT_LABEL_RE = re.compile(r"\b(?:V|W)\d{1,3}\b")
DATE_RE = re.compile(r"\b(jan(uary)?|feb(ruary)?|march|apr(il)?|may \d|june?|july?|aug(ust)?|sept?(ember)?|"
                     r"oct(ober)? \d|nov(ember)?|dec(ember)?)\b|\b(19|20)\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", re.I)
HEDGE_RE = re.compile(r"\b(uncertain\w*|may|might|appears?|seems?|suggest\w*|possibl\w*|likely|limited|cannot|"
                      r"can't|unclear|not (be )?(possible|certain|definitive)|single|downscaled|unvalidated)\b", re.I)
MIN_OUTPUT_TOKENS = 256
MAX_CONTEXT_RETRIES = 2              # smaller requests after the server reports a context overflow
# router disclosures that present CST as the answer to an edema/anatomy question (replaced in the prompt)
PROXY_NOTES = ("Edema severity is not recorded", "There is no anatomy grade")

SYSTEM = """You are a research assistant answering a clinician's question about the OCT visit history of one eye \
(research demo on the OLIVES dataset; not a diagnosis or treatment recommendation).

You receive the clinician's original question, OCT B-scans (each introduced by a label naming its visit), recorded \
metadata (CST in micrometres, BCVA in letters), exact computed facts, and for some visits OLIVES dataset finding \
labels.

How to answer:
- Answer the question that was asked, directly, in plain prose (a short paragraph or a few bullet points). Do not \
use a fixed template.
- When B-scans are attached, look at them and describe what you actually see that bears on the question: retinal \
contour and overall thickness impression, hypo-reflective (dark) intraretinal or subretinal spaces, layer \
disruption, other visible features. When the question compares visits, compare the scans with each other.
- Combine your visual observations with the recorded values when the question needs both.
- Keep three kinds of evidence distinct and say which you rely on: what you observe on a scan ("on the V3 scan I \
see ..."), what the dataset labels record ("the OLIVES labels record ..."), and recorded metadata ("the recorded \
CST is ..."). Your observations are unvalidated; dataset labels are annotations; metadata is measured.
- Use only the numbers given. For differences, maxima and visit order use the computed facts. Do not invent values, \
findings, visits or dates. The dataset has no calendar dates: refer to visits by their label (e.g. V3, W12), or as \
"today" and "baseline".
- CST is a thickness measurement. It is not a measure of edema severity or of anatomy; never equate them. It may \
be mentioned as a quantitative cue next to what the scans show.
- An image-similarity score is generic-encoder similarity, not clinical similarity, and does not show progression.
- State your uncertainty: single central B-scans, downscaled images, and finding labels that exist only for the \
first and last visit of each eye. If the evidence cannot answer the question, say so instead of guessing.
- An uploaded scan is external unless the evidence says it matches a dataset visit: never attribute history, dates \
or this eye's labels to it.
- Do not end with a list of evidence or identifiers; the application shows the provenance separately. Do not invent \
anything."""


@dataclass
class Plan:
    kind: str                                # numeric | labels | visual | visual+metadata | none
    needs_metadata: bool = False
    needs_images: bool = False
    needs_labels: bool = False
    uses_upload: bool = False
    image_visits: list[str] = field(default_factory=list)   # in payload order; UPLOAD for the uploaded scan
    min_images: int = 0
    why: str = ""

    @property
    def visual(self) -> bool:
        return self.needs_images and self.min_images > 0


# --- question understanding --------------------------------------------------------------------------------------

def plan_question(ev: dict, images: dict[str, Path] | None = None) -> Plan:
    """Which evidence the ORIGINAL question needs. Uses the router's category (what was retrieved) and the question's
    own words (does it ask what the retina looks like?), never a rewritten question."""
    images = images or {}
    q, cat = ev["question"], ev.get("category")
    cur = (ev.get("current") or {}).get("visit_id")
    rows = [r["visit_id"] for r in ev.get("retrieved", [])]
    visual_words = bool(VISUAL_RE.search(q))
    labels = bool(ev.get("biomarkers")) or bool(FINDING_RE.search(q))

    def scans(*vids, cap=3):
        out = [v for v in dict.fromkeys(v for v in vids if v) if v == UPLOAD or v in images]
        return out[:cap]

    if ev.get("status") != "retrieved":
        return Plan("none", why="the retrieval abstained; no model is asked")
    if cat == "scan_visual":
        v = UPLOAD if ev.get("upload") else cur
        return Plan("visual", needs_images=True, needs_labels=bool(ev.get("biomarkers")), uses_upload=v == UPLOAD,
                    image_visits=scans(v), min_images=1,
                    why="a question about what one scan shows: the model must look at the scan")
    if cat == "upload_similarity":
        return Plan("visual", needs_images=True, uses_upload=True, image_visits=scans(UPLOAD, *rows[:2]), min_images=2,
                    why="which of this eye's visits looks like the uploaded scan: the embedding ranking proposes "
                        "candidates, the model compares the scans")
    if cat == "image_similarity":
        return Plan("visual+metadata", needs_metadata=True, needs_images=True, image_visits=scans(cur, *rows[:2]),
                    min_images=2, why="which earlier scan looks like today's: the embedding ranking proposes "
                                      "candidates, the model compares the scans")
    if cat in ("edema_proxy", "anatomy_proxy"):
        what = "edema" if cat == "edema_proxy" else "retinal anatomy"
        return Plan("visual+metadata", needs_metadata=True, needs_images=True, needs_labels=True,
                    image_visits=scans(cur, *rows[:3], cap=4), min_images=2,
                    why=f"{what} is judged on the scans; recorded CST only shortlists candidate visits and is a "
                        "thickness measurement, not a grade")
    if cat == "trajectory":
        return Plan("visual+metadata", needs_metadata=True, needs_images=True, needs_labels=labels,
                    image_visits=scans(cur, *rows), min_images=2,
                    why="how the retina changed: the model compares today's scan with baseline and the previous "
                        "visit; recorded CST/BCVA changes are given as exact facts")
    if cat == "mixed":
        return Plan("visual+metadata", needs_metadata=True, needs_images=True, needs_labels=labels,
                    image_visits=scans(cur, *rows[:2]), min_images=2,
                    why="a combined question: the model compares the scans and uses the exact recorded changes")
    if cat == "pathology":
        return Plan("labels", needs_metadata=False, needs_images=bool(rows), needs_labels=True,
                    image_visits=scans(*rows[:1], cur), min_images=0,
                    why="a finding question: the OLIVES labels are the recorded answer; the scans are attached so "
                        "the model can say whether they are consistent")
    if cat == "metadata" and (visual_words or ev.get("uses_images")):
        return Plan("visual+metadata", needs_metadata=True, needs_images=True, needs_labels=labels,
                    image_visits=scans(cur, *rows[:2]), min_images=min(2, 1 + len(rows)),
                    why="the question asks about the retina/scans, not only recorded values: the model compares "
                        "the scans and uses the exact recorded values")
    return Plan("numeric", needs_metadata=True, needs_labels=False,
                why="a question about recorded values: the exact result is computed from the metadata and the "
                    "model states it in words; no scans are needed")


# --- deterministic evidence ---------------------------------------------------------------------------------------

def _num(x, unit):
    return "not recorded" if x is None or pd.isna(x) else f"{x:.0f} {unit}"


def visit_labels(ev: dict, ep: pd.DataFrame | None) -> dict[str, str]:
    """visit_id -> label used in the prompt ("V3"); the uploaded scan is "the uploaded scan"."""
    out = {UPLOAD: "the uploaded scan"}
    for r in A._rows(ev) if "retrieved" in ev else []:
        if r.get("visit_label"):
            out[r["visit_id"]] = r["visit_label"]
    if ep is not None and ev.get("eye_id") is not None:
        for vid, lab in ep[ep.eye_id == ev["eye_id"]].visit_label.items():
            out.setdefault(vid, str(lab))
    return out


def delabel(text: str, labels: dict[str, str]) -> str:
    for vid in sorted(labels, key=len, reverse=True):
        if vid != UPLOAD:
            text = text.replace(vid, labels[vid])
    return text


def timeline(ev: dict, ep: pd.DataFrame | None) -> list[dict]:
    """Recorded CST/BCVA of the eligible earlier visits (oldest first) plus today."""
    if ep is None or ev.get("eye_id") is None:
        return []
    vids = list(ev.get("eligible") or [])
    cur = (ev.get("current") or {}).get("visit_id")
    rows = [A._visit_row(ep, v) for v in vids + ([cur] if cur and cur not in vids else []) if v in ep.index]
    return sorted(rows, key=lambda r: r["visit_index"])


def computed_facts(ev: dict, plan: Plan, ep: pd.DataFrame | None) -> list[str]:
    """Exact statements derived from recorded values (no model involved). Visits are named by visit label."""
    lab = visit_labels(ev, ep)
    cur = ev.get("current")
    facts = []
    rows = ev.get("retrieved", [])
    cat = ev.get("category")
    if cat == "upload_similarity":           # the upload has no CST/BCVA/date: only the fixed embedding ranking
        return [f"Embedding ranking of this eye's eligible visits by their best-matching stored B-scan (fixed; not "
                f"for you to change): " + ", ".join(f"{r['visit_label']} cosine {r['image_cosine']:.3f} (B-scan "
                                                    f"{r.get('best_scan_index', '?')} of {r.get('n_scans_compared', '?')}"
                                                    f" compared)" for r in rows) + "."] if rows else []
    if rows and cat == "metadata":
        top = rows[0]
        facts.append(f"Result of the retrieval rule for this question: visit {top['visit_label']} "
                     f"(CST {_num(top['cst'], 'µm')}, BCVA {_num(top['bcva'], 'letters')}).")
    if rows and cat in ("edema_proxy", "anatomy_proxy"):
        order = "highest" if cat == "edema_proxy" else "lowest"
        facts.append(f"Earlier visits with the {order} recorded CST (shortlist, {order} first): "
                     + ", ".join(f"{r['visit_label']} ({_num(r['cst'], 'µm')})" for r in rows) + ".")
    if cur and cur.get("cst") is not None:
        for r in rows:
            if r.get("cst") is not None:
                facts.append(f"CST at {r['visit_label']} was {r['cst']:.0f} µm; today ({cur['visit_label']}) it is "
                             f"{cur['cst']:.0f} µm, a difference of {cur['cst'] - r['cst']:+.0f} µm (today minus "
                             f"{r['visit_label']}).")
    if cur and cur.get("bcva") is not None and plan.kind != "visual":
        for r in rows:
            if r.get("bcva") is not None:
                facts.append(f"BCVA at {r['visit_label']} was {r['bcva']:.0f} letters; today it is "
                             f"{cur['bcva']:.0f} letters ({cur['bcva'] - r['bcva']:+.0f} letters).")
    facts += [delabel(f, lab) + ("" if f.endswith(".") else ".") for f in ev.get("facts", [])]
    tl = timeline(ev, ep)
    if len(tl) >= 2 and any(r["cst"] is not None for r in tl):
        hi = max((r for r in tl if r["cst"] is not None), key=lambda r: r["cst"])
        lo = min((r for r in tl if r["cst"] is not None), key=lambda r: r["cst"])
        facts.append(f"Across the {len(tl)} visits up to today, recorded CST ranged from {lo['cst']:.0f} µm "
                     f"({lo['visit_label']}) to {hi['cst']:.0f} µm ({hi['visit_label']}).")
    return list(dict.fromkeys(facts))


def label_coverage(ev: dict, ep: pd.DataFrame | None) -> str:
    tl = timeline(ev, ep)
    if not tl or ep is None:
        return ""
    labelled = [r["visit_label"] for r in tl if bool(ep.loc[r["visit_id"]].get("has_biomarkers", False))]
    if not labelled:
        return "No visit up to today has OLIVES finding labels."
    return (f"OLIVES finding labels exist for {', '.join(labelled)} only; the other {len(tl) - len(labelled)} visits "
            "up to today have no finding labels.")


# --- prompt -----------------------------------------------------------------------------------------------------------

def image_caption(vid: str, ev: dict, labels: dict[str, str]) -> str:
    if vid == UPLOAD:
        up = ev.get("upload") or {}
        return "B-scan: the uploaded scan" + (f" (identical to the dataset scan of visit {labels.get(up['matched_visit'])})"
                                              if up.get("matched_visit") else " (external image)")
    cur = (ev.get("current") or {}).get("visit_id")
    row = next((r for r in ev.get("retrieved", []) if r["visit_id"] == vid), {})
    scan = f" (B-scan {row['best_scan_index']}, its best match to the upload)" if row.get("best_scan_index") is not None \
        and ev.get("category") == "upload_similarity" else ""
    return f"B-scan: {'today, ' if vid == cur else ''}visit {labels.get(vid, '?')}{scan}"


def evidence_prompt(ev: dict, plan: Plan, ep: pd.DataFrame | None, sent: list[str]) -> str:
    lab = visit_labels(ev, ep)
    L = [f"QUESTION (as asked): {ev['question']}", f"What the question needs: {plan.why}."]
    up = ev.get("upload")
    if up:
        L.append("UPLOADED SCAN: " + (f"identical to the dataset scan of visit {lab.get(up['matched_visit'])}"
                                      if up.get("matched_visit") else
                                      "an external image, not linked to any patient, eye, visit or date"))
    cur = ev.get("current")
    if cur and ev.get("category") == "upload_similarity":
        L.append(f"Selected visit of eye {ev['eye_id']}: {cur['visit_label']}. The uploaded scan was compared only "
                 "with this eye's visits before it; the upload is not this visit's scan.")
    elif cur:
        L.append(f"Today: visit {cur['visit_label']} of eye {ev['eye_id']}; recorded CST {_num(cur['cst'], 'µm')}, "
                 f"BCVA {_num(cur['bcva'], 'letters')}" + ("; B-scan attached" if cur["visit_id"] in sent else "")
                 + ". Only earlier visits of the same eye are eligible as history.")
    if ev.get("ranking_basis"):
        L.append("How the visits below were retrieved: " + delabel(ev["ranking_basis"], lab))
    if ev.get("retrieved"):
        role = {"edema_proxy": "candidate shortlisted by CST", "anatomy_proxy": "candidate shortlisted by CST",
                "image_similarity": "candidate by image similarity", "upload_similarity": "candidate by image similarity",
                "trajectory": "reference visit"}.get(ev.get("category"))
        L.append("Visits provided (in retrieval order):")
        for r in ev["retrieved"]:
            sim = "" if r.get("image_cosine") is None else f"; image-embedding cosine {r['image_cosine']:.3f}"
            if r.get("best_scan_index") is not None:
                sim += f" (best-matching B-scan {r['best_scan_index']} of {r['n_scans_compared']} compared)"
            img = "B-scan attached" if r["visit_id"] in sent else "no B-scan attached"
            L.append(f"  - visit {r['visit_label']} [{role or r['role']}]: CST {_num(r['cst'], 'µm')}, "
                     f"BCVA {_num(r['bcva'], 'letters')}{sim}; {img}")
    facts = computed_facts(ev, plan, ep)
    if facts:
        L.append("Computed facts (exact, from recorded metadata):")
        L += [f"  - {f}" for f in facts]
    tl = timeline(ev, ep)
    if plan.needs_metadata and tl:
        L.append("Recorded CST (µm) / BCVA (letters) by visit, oldest first: "
                 + ", ".join(f"{r['visit_label']} {_num(r['cst'], '').strip()}/{_num(r['bcva'], '').strip()}"
                             for r in tl[-40:]))
    if ev.get("biomarkers"):
        L.append("Dataset finding labels (OLIVES annotations, not your observations):")
        L += [f"  - visit {lab.get(b['visit_id'], '?')}: " + A._fmt_labels(b).split(": ", 1)[1]
              for b in ev["biomarkers"]]
    cov = label_coverage(ev, ep)
    if cov and (plan.needs_labels or ev.get("biomarkers")):
        L.append("Label coverage: " + cov)
    for d in ev.get("disclosures", []):
        if not d.startswith(PROXY_NOTES):                    # the router's proxy wording is replaced below
            L.append("Note: " + delabel(d, lab))
    if ev.get("category") in ("edema_proxy", "anatomy_proxy"):
        what, order = (("edema", "highest") if ev["category"] == "edema_proxy" else ("anatomy", "lowest"))
        L.append(f"Note: no {what} grade is recorded in this dataset. The candidate visits were shortlisted by "
                 f"recorded CST ({order} first) only because thickness is a quantitative cue; CST is not a measure of "
                 f"{what}. Judge the question on the attached scans, and say so if they do not allow a judgement.")
    if sent:
        L.append("Attached B-scans, in order: " + "; ".join(f"Image {i}: {image_caption(v, ev, lab)[8:]}"
                                                          for i, v in enumerate(sent, 1)))
    elif plan.kind == "numeric":
        L.append("No B-scans are attached: this question is about recorded values.")
    return "\n".join(L)


def image_path(vid: str, ev: dict, images: dict[str, Path]) -> str | None:
    """The scan the evidence names for a visit (e.g. its best-matching B-scan), else the visit's central scan."""
    if vid == UPLOAD:
        return next((im["path"] for im in ev.get("images", []) if im.get("role") == "uploaded"), None)
    p = next((im["path"] for im in ev.get("images", []) if im["visit_id"] == vid and im.get("path")), None) \
        or images.get(vid)
    return None if p is None else str(p)


def request_parts(ev: dict, plan: Plan, ep, images, sent: list[str]) -> list[dict]:
    lab = visit_labels(ev, ep)
    parts = []
    for v in sent:
        parts += [dict(type="text", text=image_caption(v, ev, lab)), dict(type="image", path=image_path(v, ev, images))]
    return parts + [dict(type="text", text=evidence_prompt(ev, plan, ep, sent))]


def fit(ev: dict, plan: Plan, ep, images, vlm, max_images: int) -> tuple[list[dict] | None, list[str], str]:
    """Parts within the adapter's context budget: images are shrunk first (down to the adapter's minimum side), then
    optional images beyond plan.min_images are dropped. Returns (parts, images sent, "") or (None, [], reason)."""
    sent, seen = [], set()
    for v in plan.image_visits:                               # skip missing scans and identical files
        p = image_path(v, ev, images)
        d = p and hashlib.sha256(Path(p).read_bytes()).hexdigest()
        if d and d not in seen:
            seen.add(d)
            sent.append(v)
    sent = sent[:max(max_images, plan.min_images)]
    budget = getattr(vlm, "context_tokens", 0) or 0
    estimate = getattr(vlm, "estimate_tokens", None)
    if not budget or estimate is None:
        return request_parts(ev, plan, ep, images, sent), sent, ""
    sides = getattr(vlm, "image_sides", lambda: [None])()
    while True:
        parts = request_parts(ev, plan, ep, images, sent)
        for side in sides:
            need = estimate(SYSTEM, parts, side) + MIN_OUTPUT_TOKENS
            if need <= budget:
                for p in parts:
                    if p["type"] == "image":
                        p["max_side"] = side
                return parts, sent, ""
        if len(sent) > plan.min_images:
            sent = sent[:-1]
            continue
        n = len(sent)
        return None, [], (f"the question needs {n} B-scan{'s' * (n != 1)}, and the request (about {need} tokens with "
                          f"images at {sides[-1]} px, plus {MIN_OUTPUT_TOKENS} answer tokens) exceeds the context "
                          f"budget of {budget} tokens. Raise OCT_VLM_NUM_CTX (Ollama) and OCT_VLM_CONTEXT, e.g. to 8192")


# --- validation of the generated prose -----------------------------------------------------------------------------

def check_answer(text: str, ev: dict, plan: Plan, ep) -> list[str]:
    warns = []
    tl = timeline(ev, ep)
    rows = tl + [r for r in A._rows(ev) if "cst" in r] if "retrieved" in ev else tl
    values = {x for r in rows for x in (r.get("cst"), r.get("bcva")) if x is not None}
    allowed = values | {abs(a - b) for a in values for b in values}
    odd = sorted({m for m in A.MEASURE_RE.findall(text) if not any(abs(float(m) - x) <= 0.5 for x in allowed)})
    if odd:
        warns.append(f"numbers not found in the recorded metadata (nor as a difference of two recorded values): "
                     f"{', '.join(odd)}")
    known = set(visit_labels(ev, ep).values()) | {r["visit_label"] for r in rows}
    if ev.get("eye_id") is not None:
        unknown = sorted(set(VISIT_LABEL_RE.findall(text)) - known)
        if unknown:
            warns.append(f"mentions visits that are not in this eye's evidence: {', '.join(unknown)}")
    if A.VISIT_ID_RE.search(text):
        warns.append("the prose contains visit identifiers; provenance is listed separately")
    if DATE_RE.search(text):
        warns.append("mentions a date or month; the dataset has no calendar dates")
    for sent in re.split(r"(?<=[.!?])\s+", text):
        s = sent.lower()
        if "cst" in s and re.search(r"edema (severity|grade)|severity of (the )?edema|measures? (the )?edema", s) \
                and not re.search(r"\bnot\b|n't|\bno\b|proxy|cue", s):
            warns.append("treats CST as a measure of edema severity")
            break
    cov = label_coverage(ev, ep)
    if (plan.needs_labels or FINDING_RE.search(ev["question"])) and "only" in cov and "label" not in text.lower():
        warns.append("does not acknowledge that finding labels exist only for some visits")
    if plan.visual and not HEDGE_RE.search(text):
        warns.append("describes the scans without acknowledging uncertainty")
    return warns


# --- answer -----------------------------------------------------------------------------------------------------------

def provenance(ev: dict, sent: list[str], ep) -> list[dict]:
    lab = {b["visit_id"]: A._fmt_labels(b).split(": ", 1)[1] for b in ev.get("biomarkers", [])}
    out = []
    if ev.get("upload"):
        up = next((im["path"] for im in ev.get("images", []) if im.get("role") == "uploaded"), None)
        out.append(dict(visit_id=UPLOAD, visit_label="upload", role="uploaded scan", retrieved_by="user upload",
                        similarity=None, scan_file=None if not up else Path(up).name, scan_path=up, cst=None,
                        bcva=None, scan_sent=UPLOAD in sent, labels=None))
    if ev.get("current"):
        c = ev["current"]
        out.append(dict(visit_id=c["visit_id"], visit_label=c["visit_label"], role="today (query visit)",
                        retrieved_by="selected by the user", cst=c["cst"], bcva=c["bcva"],
                        scan_sent=c["visit_id"] in sent, labels=lab.get(c["visit_id"])))
    paths = {im["visit_id"]: im.get("path") for im in ev.get("images", [])}
    for r in ev.get("retrieved", []):
        path = r.get("best_scan_path") or paths.get(r["visit_id"])
        out.append(dict(visit_id=r["visit_id"], visit_label=r["visit_label"], role=r.get("role", ""),
                        retrieved_by=", ".join(r.get("sources", [])), similarity=r.get("image_cosine"),
                        scan_file=None if not path else Path(path).name, scan_path=path, cst=r["cst"],
                        bcva=r["bcva"], scan_sent=r["visit_id"] in sent, labels=lab.get(r["visit_id"])))
    return out


def _result(status, text, backend, ev, plan, ep, sent=(), warnings=(), error=None, prompt="", retries=0):
    return dict(status=status, answer=text, backend=backend, plan=asdict(plan), computed=computed_facts(ev, plan, ep),
                context_retries=retries,
                provenance=provenance(ev, list(sent), ep), images_sent=list(sent), warnings=list(warnings),
                error=error, prompt=prompt, disclaimer=A.DISCLAIMER,
                evidence_used=A.evidence_summary(ev) if "retrieved" in ev else "")


def generate(ev: dict, backend: str = "evidence", vlm=None, ep: pd.DataFrame | None = None,
             images: dict[str, Path] | None = None, max_images: int = 4) -> dict:
    """Answer ev's original question. backend 'evidence' (no model) or 'vlm' (adapter with generate(system, parts))."""
    images = images or {}
    if backend not in ("evidence", "vlm"):
        raise ValueError(f"unknown backend {backend!r}")
    if backend == "vlm" and vlm is None:
        raise ValueError("backend 'vlm' needs an adapter")
    plan = plan_question(ev, images)
    if ev["status"] != "retrieved":
        return _result("abstained", A._abstention(ev), "none", ev, plan, ep)
    if plan.visual and not any(image_path(v, ev, images) for v in plan.image_visits):
        return _result("no_scan", "No answer: this question needs the OCT scans to be inspected, and no B-scan is "
                                  "available for the visits it concerns. Recorded values alone cannot answer it.",
                       "none", ev, plan, ep)
    if backend == "evidence":
        if plan.visual:
            return _result("needs_model", "No answer: this question needs visual interpretation of the OCT scans, which "
                                          "requires a vision-language model (choose one under Answer model). The "
                                          "retrieved visits and exact recorded values are listed with the evidence; "
                                          "they do not answer the question by themselves.", "evidence", ev, plan, ep)
        facts = computed_facts(ev, plan, ep)
        lab = visit_labels(ev, ep)
        lines = facts + [f"Dataset labels, visit {lab.get(b['visit_id'], '?')}: " + A._fmt_labels(b).split(": ", 1)[1]
                         for b in ev.get("biomarkers", [])]
        return _result("answered", "No language model is configured, so this is the computed evidence, not a "
                                   "written answer:\n" + "\n".join(f"- {x}" for x in lines), "evidence", ev, plan, ep)
    fail = "No generated answer: {}. The retrieved evidence and exact values are shown with the evidence; no answer " \
           "was substituted."
    retries = 0
    while True:
        parts, sent, over = fit(ev, plan, ep, images, vlm, max_images)
        if parts is None:
            return _result("model_error", fail.format(f"the model was not called: {over}"), "vlm", ev, plan, ep,
                           error="context budget exceeded", retries=retries)
        prompt = parts[-1]["text"]
        try:
            reply = vlm.generate(SYSTEM, parts)
        except Exception as ex:                               # adapters should not raise; never fabricate if they do
            err = f"{type(ex).__name__}: {A.redact(ex)}"
            return _result("model_error", fail.format(f"the model call failed ({err})"), "vlm", ev, plan, ep, sent,
                           error=err, prompt=prompt, retries=retries)
        # the server rejected the request for its size: learn its count/limit, refit (fewer/smaller scans), retry
        if reply.finish == "error" and retries < MAX_CONTEXT_RETRIES and hasattr(vlm, "learn_overflow") \
                and vlm.learn_overflow(SYSTEM, parts, reply.error or ""):
            retries += 1
            continue
        break
    if reply.finish == "error":
        return _result("model_error", fail.format(f"the model call failed ({A.redact(reply.error, 500)})"), "vlm", ev,
                       plan, ep, sent, error=A.redact(reply.error, 500), prompt=prompt, retries=retries)
    if hasattr(vlm, "learn_usage"):
        vlm.learn_usage(SYSTEM, parts, (reply.usage or {}).get("prompt_tokens"))
    if reply.finish == "refusal":
        return _result("model_refused", fail.format("the model declined to answer"), "vlm", ev, plan, ep, sent,
                       error="refusal", prompt=prompt, retries=retries)
    text = (reply.text or "").strip()
    if not text:
        return _result("model_error", fail.format("the model returned no text"), "vlm", ev, plan, ep, sent,
                       error="empty response", prompt=prompt, retries=retries)
    warns = check_answer(text, ev, plan, ep)
    if reply.finish == "length":
        return _result("model_incomplete", text, "vlm", ev, plan, ep, sent, prompt=prompt, error="max_tokens", retries=retries,
                       warnings=["the answer was cut off at the output limit and is incomplete"] + warns)
    return _result("answered", text, "vlm", ev, plan, ep, sent, warnings=warns, prompt=prompt, retries=retries)
