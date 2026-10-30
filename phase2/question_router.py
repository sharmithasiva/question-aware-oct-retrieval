"""Question-aware routing for the unified hybrid system (Phase 3). Sits OUTSIDE the frozen Phase 1 code.

The frozen parser (`intent.parse_intent`) and retriever (`intent_retrieval.retrieve`) are used read-only, through
`hybrid_retriever.HybridRetriever`. This layer adds what the frozen parser cannot express, and only that:

  category          evidence sources                               when
  metadata          metadata rule (frozen)                         frozen parse = retrieve, no edema wording
  edema_proxy       metadata CST rule (frozen) as a DISCLOSED proxy,  frozen parse = retrieve on CST but the question
                    + B-scans of the evidence visits                  says edema/swelling (not CST/thickness): the
                    + OLIVES DRT/ME and IRF labels where they exist   frozen parser maps these to CST, which is a
                                                                      thickness measurement, not an edema grade
  image_similarity  image embeddings + eligibility metadata        frozen parse = unsupported_image and the question
                    (+ CST-closeness fusion, the frozen S5 design)    asks about appearance / similarity
  pathology         OLIVES per-B-scan biomarker labels             the question asks about a finding (fluid, cysts,
                    (first and last visit of each eye only)          DRT, HRF, DRIL, EZ, ...). Answered only when every
                                                                      eligible visit is labelled; otherwise ABSTAINS
                                                                      (status insufficient_labels). Image similarity is
                                                                      never used as a finding detector.
  mixed             image_similarity component + metadata          a clause asking about similarity joined to a
                    component (a frozen metadata intent, or the       metadata clause; otherwise the frozen
                    change of CST/BCVA from the image-selected        clarification abstention is kept
                    visit to today)
  anatomy_proxy     metadata CST rule (frozen v1 'best/worst anatomy'  "best/worst anatomy": no anatomy grade exists;
                    = min/max CST) as a DISCLOSED proxy + B-scans      Phase 1 defines it by CST, which is disclosed
  trajectory        visit order (baseline, previous visit, today)  "how has the anatomy/retina changed over time":
                    + recorded CST changes + B-scans                  not a ranking; changes are measured or observed
  abstain           none                                           every other frozen abstention, unchanged

Parser gaps handled here, each recorded in UnifiedResult.interpreted_as and disclosed (the frozen parser is not edited):
  * a similarity SEARCH that the frozen parser reads as "previous visit" ("which previous visit is most similar to
    this one?") is routed to image similarity;
  * "largest/biggest increase|rise|drop in CST/BCVA" is mapped to the frozen canonical "largest consecutive ... "
    question; "best/worst anatomy" to the frozen CST extremum (anatomy_proxy);
  * a previous-visit / baseline comparison that mentions the scan or anatomy keeps the frozen metadata selection and
    additionally attaches the B-scans for visual comparison (ranking unchanged).

Hard rules: candidates are always the strictly earlier visits of the same eye (frozen VisitMemory); biomarker labels
are read only for those visits and today's; image similarity never reorders a metadata ranking and never replaces an
abstention.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "phase2"))
import hybrid_retriever as H  # noqa: E402
from oct_retrieval import intent as I  # noqa: E402  (frozen, read-only)
from oct_retrieval import intent_retrieval as R  # noqa: E402

INSUFFICIENT_LABELS = "insufficient_labels"
PARTIAL = "partially_supported"
UNSUPPORTED_FINDING = "unsupported_image_finding"

EDEMA_RE = re.compile(r"\b(o?edema\w*|oedema\w*|swell\w*|swollen)\b", re.I)
EXPLICIT_CST_RE = re.compile(I.CST_STRONG + r"|\bthick\w*|\bthin\w*", re.I)
SIMILARITY_RE = re.compile(r"\blook(s|ed)?\b[^.?!]*\b(like|similar|same)\b|\bsimilar\w*|\bresembl\w*|\bappearance\b"
                           r"|\banatom\w*|\bmorpholog\w*|\bstructur\w*\s+(similar|compar)", re.I)
CHANGE_RE = re.compile(r"\bchang\w*|\bdiffer\w*|\bcompar\w*|\bevolv\w*|\bsince then\b", re.I)
CLAUSE_SPLIT_RE = re.compile(r",?\s*\band\b\s+|;\s*|\?\s+(?=\w)", re.I)
# (pattern, biomarker column stem(s), label). Order matters: specific findings before generic "fluid".
FINDINGS = [
    (r"\bsub-?retinal fluid\b|\bsrf\b", ("fluid_srf",), "subretinal fluid (SRF)"),
    (r"\bintra-?retinal fluid\b|\birf\b|\bcyst\w*", ("fluid_irf",), "intraretinal fluid / cystoid spaces (IRF)"),
    (r"\bdiffuse retinal thickening\b|\bdrt\b", ("drt_me",), "diffuse retinal thickening / macular edema (DRT/ME)"),
    (r"\bhyper-?reflective\b|\bhrf\b|\bfoci\b", ("ir_hrf",), "intraretinal hyperreflective foci (HRF)"),
    (r"\bdril\b|\bdisorgani[sz]ation\b", ("dril",), "disorganisation of retinal inner layers (DRIL)"),
    (r"\bellipsoid\b|\bez\b", ("disruption_of_ez",), "ellipsoid zone disruption"),
    (r"\brpe\b|\bpigment epitheli\w*", ("disruption_of_rpe",), "RPE disruption"),
    (r"\bped\b|\bpigment epithelial detachment\b", ("ped_serous",), "serous PED"),
    (r"\batroph\w*", ("atrophy_thinning_of_retinal_layers",), "atrophy / thinning of retinal layers"),
    (r"\bvmt\b|\bvitreomacular\b", ("vmt",), "vitreomacular traction"),
    (r"\bshrm\b", ("shrm",), "subretinal hyperreflective material"),
    (r"\bfluid\b", ("fluid_irf", "fluid_srf"), "retinal fluid (IRF or SRF)"),
]
SIM_SEARCH_RE = re.compile(r"\b(most|more|very) similar\b|\bsimilar to\b|\blook(s|ed)?( most)? (like|similar)\b"
                           r"|\bresembl\w*", re.I)
SCAN_RE = re.compile(r"\b(scan|scans|oct|b-?scans?|images?|anatom\w*|appearance|morpholog\w*)\b", re.I)
EXTENT_RE = re.compile(r"\b(largest|biggest|greatest|maximum|max|steepest)\b[^.?!]*\b(increase|rise|jump|gain|growth|"
                       r"decrease|drop|fall|decline|reduction|loss)\b", re.I)
DOWN_RE = re.compile(r"\b(decrease|drop|fall|decline|reduction|loss)\b", re.I)
ANATOMY_RE = re.compile(r"\banatom\w*", re.I)
BEST_RE = re.compile(r"\bbest\b|\bhealthiest\b|\bmost normal\b", re.I)
WORST_RE = re.compile(r"\bworst\b|\bleast normal\b", re.I)
TRAJECTORY_RE = re.compile(r"\bover time\b|\bacross (all |the )?visits\b|\bprogress\w*|\btrajector\w*|\bcourse\b"
                           r"|\bevolv\w*|\bhistory\b", re.I)
CANON_SIMILAR = "Which earlier visit's OCT looks most like today's?"
CANON_BASELINE = "How has the retina changed since baseline?"
CANON_PREVIOUS = "Compare with the previous visit."
FIRST_RE = re.compile(r"\bfirst\b|\bearliest\b|\bappear\w*|\bstart\w*|\bbegin\w*|\bonset\b", re.I)
LAST_RE = re.compile(r"\blast\b|\blatest\b|\bmost recent\b", re.I)
MOST_RE = re.compile(r"\bworst\b|\bmost\b|\bhighest\b|\bmaximum\b|\blargest\b|\bmax\b", re.I)
IMPROVE_RE = re.compile(r"\bimprov\w*|\bbetter\b|\bworse\w*|\bresolv\w*|\breduc\w*|\bchang\w*", re.I)


@dataclass
class Component:
    name: str                     # metadata | image_similarity | biomarker_labels | metadata_change
    status: str
    explanation: str
    ranking: list[str] | None = None
    hybrid: object | None = None  # HybridResult for metadata / image components
    details: dict = field(default_factory=dict)


@dataclass
class UnifiedResult:
    question: str
    eye_id: int
    query_visit_id: str
    category: str
    status: str
    reason: str
    intent: str
    candidates: list[str]
    components: list[Component]
    final_ranking: list[str] | None
    uses_images: bool
    disclosures: list[str] = field(default_factory=list)
    biomarkers: list[dict] = field(default_factory=list)
    interpreted_as: str | None = None     # canonical question given to the frozen parser, when the router rewrote it

    @property
    def abstained(self) -> bool:
        return self.status != R.RETRIEVED

    @property
    def primary(self):
        """The HybridResult of the first metadata/image component (for the rankings table), if any."""
        return next((c.hybrid for c in self.components if c.hybrid is not None), None)


def match_finding(text: str):
    for pat, cols, label in FINDINGS:
        if re.search(pat, text, re.I):
            return cols, label
    return None


def pathology_operation(text: str) -> str:
    if IMPROVE_RE.search(text):
        return "change_vs_today"
    if FIRST_RE.search(text):
        return "first_present"
    if LAST_RE.search(text):
        return "last_present"
    if MOST_RE.search(text):
        return "max_extent"
    return "presence"


class QuestionAwareRetriever:
    def __init__(self, episodes: pd.DataFrame, visit_emb: dict, image_proxy=H.CST_PROXY):
        self.episodes = episodes
        self.ep = episodes.set_index("visit_id")
        self.hybrid = H.HybridRetriever(episodes, visit_emb, image_proxy=image_proxy)   # no fusion for metadata ops

    # --- classification ------------------------------------------------------------------------------------------

    def classify(self, question: str) -> tuple[str, dict]:
        p = I.parse_intent(question)
        finding = match_finding(question)
        anchors = (I.ANCHOR_PREVIOUS, I.ANCHOR_BASELINE)
        if p.task == I.RETRIEVE:
            if p.operation in anchors and p.target is None and SIM_SEARCH_RE.search(question):
                return "image_similarity", dict(canonical=CANON_SIMILAR)
            if p.target == "cst" and EDEMA_RE.search(question) and not EXPLICIT_CST_RE.search(question):
                return "edema_proxy", {}
            if p.operation in anchors and SCAN_RE.search(question):
                return "metadata", dict(visual=True)
            return "metadata", {}
        if finding is not None and p.task in (I.UNSUPPORTED_IMAGE, I.CLARIFY):
            return "pathology", dict(columns=finding[0], label=finding[1], op=pathology_operation(question))
        if p.task == I.UNSUPPORTED_IMAGE:
            if not SIMILARITY_RE.search(question):
                return UNSUPPORTED_FINDING, {}
            clauses = self._clauses(question)
            if len(clauses) > 1:                          # never answer only part of a multi-part question
                mixed = self._split_mixed(question)
                if mixed:
                    return "mixed", mixed
                if not all(I.parse_intent(c).task == I.UNSUPPORTED_IMAGE and SIMILARITY_RE.search(c) for c in clauses):
                    return PARTIAL, {}
            return "image_similarity", {}
        if p.task == I.CLARIFY:
            mixed = self._split_mixed(question)
            if mixed:
                return "mixed", mixed
            if len(self._clauses(question)) == 1:
                if p.target in ("cst", "bcva") and EXTENT_RE.search(question):
                    d = "decrease" if DOWN_RE.search(question) else "increase"
                    return "metadata", dict(canonical=f"What was the largest consecutive {p.target.upper()} {d}?")
                if ANATOMY_RE.search(question) and (BEST_RE.search(question) or WORST_RE.search(question)):
                    best = bool(BEST_RE.search(question))
                    return "anatomy_proxy", dict(canonical="When was CST lowest?" if best else "When was CST highest?",
                                                 best=best)
                if TRAJECTORY_RE.search(question) and (SCAN_RE.search(question) or re.search(r"\bretina\w*", question,
                                                                                               re.I)):
                    return "trajectory", {}
        return "abstain", {}

    @staticmethod
    def _clauses(question: str) -> list[str]:
        return [c.strip(" ,.?") for c in CLAUSE_SPLIT_RE.split(question) if c and c.strip(" ,.?")]

    def _split_mixed(self, question: str) -> dict | None:
        clauses = self._clauses(question)
        if len(clauses) != 2:
            return None
        img = [c for c in clauses if I.parse_intent(c).task == I.UNSUPPORTED_IMAGE and SIMILARITY_RE.search(c)
               and match_finding(c) is None]
        if len(img) != 1:
            return None
        other = clauses[1 - clauses.index(img[0])]
        p = I.parse_intent(other)
        if p.task == I.RETRIEVE:
            return dict(image_clause=img[0], metadata_clause=other)
        if CHANGE_RE.search(other) and p.target in ("cst", "bcva"):
            return dict(image_clause=img[0], change_target=p.target, change_clause=other)
        return None

    # --- retrieval -----------------------------------------------------------------------------------------------

    def retrieve(self, question: str, eye_id: int, query_visit_id: str) -> UnifiedResult:
        mem = self.hybrid.memory(eye_id)
        if query_visit_id not in mem.visit_ids:
            raise ValueError(f"visit {query_visit_id} does not belong to eye {eye_id}")
        t = mem.visit_ids.index(query_visit_id)
        cands = list(mem.visit_ids[:t])
        category, info = self.classify(question)
        p = I.parse_intent(question)
        base = dict(question=question, eye_id=int(eye_id), query_visit_id=query_visit_id, category=category,
                    intent=f"{p.task}/{p.operation}/{p.target}", candidates=cands)

        if category in ("metadata", "edema_proxy", "image_similarity", "anatomy_proxy", "abstain"):
            text = info.get("canonical", question)
            h = self.hybrid.retrieve(text, eye_id, query_visit_id)
            name = "image_similarity" if h.route in ("image", "image+proxy_rrf") else "metadata"
            why = h.reason or (f"frozen Phase 1 metadata rule {h.intent} over the eye's earlier visits"
                               if name == "metadata" else "")
            comp = Component(name, h.status, why, h.final_ranking, hybrid=h)
            u = UnifiedResult(**base, status=h.status, reason=h.reason, components=[comp],
                              final_ranking=h.final_ranking, uses_images=name == "image_similarity",
                              interpreted_as=info.get("canonical"))
            if u.interpreted_as:
                u.intent = h.intent
                u.disclosures.append(f"The question was interpreted as: \"{u.interpreted_as}\".")
            if info.get("visual") and not u.abstained:
                u.uses_images = True
                u.disclosures.append("The visit was selected by visit order (frozen metadata rule), not by image "
                                     "similarity. B-scans are attached so visible differences can be described; any "
                                     "description is a visual observation of single central B-scans.")
            if category == "anatomy_proxy":
                u.disclosures.append(
                    f"There is no anatomy grade in this dataset. \"{'Best' if info['best'] else 'Worst'} anatomy\" is "
                    f"approximated, as in the Phase 1 benchmark, by the {'lowest' if info['best'] else 'highest'} "
                    "central subfield thickness (CST); CST is a thickness measurement, not an anatomical assessment. "
                    "B-scans are attached for inspection.")
                u.uses_images = not u.abstained
                u.biomarkers = self._labels(eye_id, [query_visit_id] + (h.final_ranking or [])[:3],
                                            ("drt_me", "fluid_irf"))
            if category == "edema_proxy":
                u.disclosures.append(
                    "Edema severity is not recorded in this dataset. The visit was selected by central subfield "
                    "thickness (CST), used as a quantitative proxy; CST is not a clinical edema grade. B-scans and, "
                    "where available, OLIVES DRT/ME and IRF labels are shown alongside.")
                u.uses_images = not u.abstained
                u.biomarkers = self._labels(eye_id, [query_visit_id] + (h.final_ranking or [])[:3],
                                            ("drt_me", "fluid_irf"))
            if category == "image_similarity" and not u.abstained:
                u.disclosures.append("Visits were ranked by image-embedding similarity (generic pretrained encoder), "
                                     "fused with CST closeness. This is not a detector of fluid or any finding.")
            return u

        if category == PARTIAL:
            return UnifiedResult(**base, status=R.CLARIFY, components=[], final_ranking=None, uses_images=False,
                                 reason="the question has several parts and not all of them are supported; ask the "
                                        "supported part on its own")

        if category == UNSUPPORTED_FINDING:
            return UnifiedResult(**base, status=UNSUPPORTED_FINDING, components=[], final_ranking=None,
                                 uses_images=False, reason="the question asks about an OCT finding this system "
                                 "neither extracts from images nor has labels for")

        if category == "pathology":
            return self._pathology(base, info, eye_id, query_visit_id, cands)

        if category == "trajectory":
            return self._trajectory(base, eye_id, query_visit_id, cands)

        # mixed
        h_img = self.hybrid.retrieve(info["image_clause"], eye_id, query_visit_id)
        comps = [Component("image_similarity", h_img.status, h_img.reason, h_img.final_ranking, hybrid=h_img)]
        if h_img.abstained:
            return UnifiedResult(**base, status=h_img.status, reason=f"image part: {h_img.reason}", components=comps,
                                 final_ranking=None, uses_images=False)
        if "metadata_clause" in info:
            h_meta = self.hybrid.retrieve(info["metadata_clause"], eye_id, query_visit_id)
            comps.append(Component("metadata", h_meta.status, h_meta.reason or f"frozen Phase 1 metadata rule "
                                   f"{h_meta.intent} for: {info['metadata_clause']!r}", h_meta.final_ranking,
                                   hybrid=h_meta))
            if h_meta.abstained:
                return UnifiedResult(**base, status=h_meta.status, reason=f"metadata part: {h_meta.reason}",
                                     components=comps, final_ranking=None, uses_images=False)
        else:
            tgt, anchor = info["change_target"], h_img.final_ranking[0]
            a, b = self.ep.loc[anchor, tgt], self.ep.loc[query_visit_id, tgt]
            if pd.isna(a) or pd.isna(b):
                return UnifiedResult(**base, status=R.INSUFFICIENT, components=comps, final_ranking=None,
                                     uses_images=False, reason=f"{tgt.upper()} missing at {anchor} or today")
            comps.append(Component("metadata_change", R.RETRIEVED, f"{tgt.upper()} change from the image-selected "
                                   f"visit to today", [anchor], details=dict(target=tgt, from_visit=anchor,
                                                                             from_value=float(a), to_value=float(b),
                                                                             delta=float(b - a))))
        return UnifiedResult(**base, status=R.RETRIEVED, reason="mixed question: image and metadata components",
                             components=comps, final_ranking=h_img.final_ranking, uses_images=True,
                             disclosures=["Visits were ranked by image-embedding similarity (generic pretrained "
                                          "encoder), fused with CST closeness; the metadata part was answered from "
                                          "recorded measurements."])

    # --- trajectory ----------------------------------------------------------------------------------------------

    def _trajectory(self, base, eye_id, visit, cands) -> UnifiedResult:
        """Baseline and previous visit (frozen anchor rules) with recorded CST changes to today, B-scans attached."""
        if not cands:
            return UnifiedResult(**base, status=R.INSUFFICIENT, reason="no prior visits", components=[],
                                 final_ranking=None, uses_images=False)
        comps, ranking = [], []
        for label, text in (("baseline", CANON_BASELINE), ("previous visit", CANON_PREVIOUS)):
            h = self.hybrid.retrieve(text, eye_id, visit)
            anchor = h.final_ranking[0]
            if anchor in ranking:
                continue
            ranking.append(anchor)
            a, b = self.ep.loc[anchor, "cst"], self.ep.loc[visit, "cst"]
            details = dict(target="cst", from_visit=anchor, from_value=float(a), to_value=float(b),
                           delta=float(b - a)) if not (pd.isna(a) or pd.isna(b)) else {}
            comps.append(Component("metadata_change" if details else "metadata", R.RETRIEVED,
                                   f"{label}: frozen Phase 1 rule {h.intent}", [anchor], hybrid=h, details=details))
        return UnifiedResult(**base, status=R.RETRIEVED, reason="longitudinal overview: baseline, previous visit and "
                             "today", components=comps, final_ranking=ranking, uses_images=True,
                             disclosures=["Visits were chosen by visit order (baseline, previous visit), not by "
                                          "similarity. CST changes are computed from recorded values; any description "
                                          "of anatomical change is a visual observation of single central B-scans."])

    # --- biomarker labels ----------------------------------------------------------------------------------------

    def _labels(self, eye_id: int, visits: list[str], stems) -> list[dict]:
        """OLIVES visit-level biomarker aggregates for the given (eligible) visits that are labelled."""
        out = []
        for v in dict.fromkeys(visits):
            r = self.ep.loc[v]
            if not bool(r.get("has_biomarkers", False)) or int(r.eye_id) != int(eye_id):
                continue
            n = None if pd.isna(r.get("bm_n_scans")) else int(r.bm_n_scans)
            out.append(dict(visit_id=v, visit_index=int(r.visit_index), n_scans=n,
                            labels={s: dict(frac=float(r[f"bm_frac_{s}"]), present=bool(r[f"bm_any_{s}"]))
                                    for s in stems if f"bm_frac_{s}" in r.index and not pd.isna(r[f"bm_frac_{s}"])}))
        return out

    def _pathology(self, base, info, eye_id, visit, cands) -> UnifiedResult:
        stems, label, op = info["columns"], info["label"], info["op"]
        labels = self._labels(eye_id, cands + [visit], stems)
        labelled = {b["visit_id"] for b in labels}
        unlabelled = [v for v in cands if v not in labelled]
        expl = (f"OLIVES per-B-scan labels for {label}; they exist only for the first and last visit of each eye. "
                f"Operation: {op}.")
        comp = Component("biomarker_labels", R.RETRIEVED, expl, details=dict(stems=stems, label=label, op=op))
        u = UnifiedResult(**base, status=R.RETRIEVED, reason="", components=[comp], final_ranking=None,
                          uses_images=False, biomarkers=labels,
                          disclosures=["Finding labels are OLIVES-provided per-B-scan annotations, aggregated per "
                                       "visit (fraction of labelled B-scans). Image similarity is not used to "
                                       "detect findings."])
        if not cands:
            u.status, u.reason = R.INSUFFICIENT, "no prior visits"
        elif op == "change_vs_today" and visit not in labelled:
            u.status = INSUFFICIENT_LABELS
            u.reason = f"{label} labels do not exist for today's visit, so change since earlier visits cannot be assessed"
        elif unlabelled:
            u.status = INSUFFICIENT_LABELS
            u.reason = (f"{label} labels exist for {len(cands) - len(unlabelled)} of {len(cands)} eligible earlier "
                        f"visits; {len(unlabelled)} are unlabelled, so the question cannot be answered from labels")
        else:
            def extent(v):
                b = next(x for x in labels if x["visit_id"] == v)["labels"]
                return max(b[s]["frac"] for s in stems if s in b)
            present = [v for v in cands if extent(v) > 0]
            if op in ("first_present", "last_present", "presence") and not present:
                u.status, u.reason = R.NO_QUALIFYING, f"no eligible earlier visit has {label} in its labels"
            else:
                order = {"first_present": sorted(present, key=lambda v: cands.index(v)),
                         "last_present": sorted(present, key=lambda v: -cands.index(v)),
                         "presence": sorted(present, key=lambda v: -cands.index(v))}.get(     # max_extent, change_vs_today
                    op, sorted(cands, key=lambda v: (-extent(v), -cands.index(v))))
                comp.ranking = u.final_ranking = order
        comp.status = u.status
        comp.explanation = expl if u.status == R.RETRIEVED else f"{expl} {u.reason}"
        return u
