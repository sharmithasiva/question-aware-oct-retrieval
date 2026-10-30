"""Conversation orchestrator: question (+ eye/visit, + optional uploaded scan) -> retrieval -> evidence -> answer.

Retrieval builds the evidence package (phase2 answer.build_unified_evidence, or the app's own scan/upload evidence);
app/reasoning.py then plans which evidence the original question needs and lets the vision-language model write the
answer from it. Abstentions never reach a model, the scans a model sees are evidence rows' scans, and the model's
prose is checked against the recorded values. Routes:

  history            phase2.question_router.QuestionAwareRetriever on the selected eye and visit (metadata, edema /
                     anatomy proxies, image similarity, finding labels, mixed, trajectory, abstentions), then
                     answer.build_unified_evidence. The uploaded scan, if any, is not used.
  scan_visual        "what is visible in this scan / does this scan show IRF?": one scan, no retrieval. The scan is the
                     upload if there is one, else the current visit's central B-scan. Dataset finding labels are added
                     only for a DATASET scan whose visit is labelled. The model's description is a visual observation.
  upload_similarity  "which visit looks most like this (uploaded) one?": the upload is embedded with the stored
                     embeddings' encoder (ResNet-50) and compared with the selected eye's embedded visits. An upload
                     that exactly matches a dataset visit is answered by the router from that visit instead.
  followup           "what evidence supports your answer?": the previous turn's evidence, no new retrieval or model.
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
import answer as A  # noqa: E402
import hybrid_retriever as H  # noqa: E402
import oct_image as OI  # noqa: E402
import question_router as QR  # noqa: E402
from question_eval import recency_order  # noqa: E402

from app.config import Config  # noqa: E402
from app import reasoning as RS  # noqa: E402
from app.uploads import DatasetImageIndex, ScanIndex, UploadedScan, embed_scan  # noqa: E402

FINDING_STEMS = ("fluid_irf", "fluid_srf", "drt_me", "ir_hrf", "dril", "disruption_of_ez", "disruption_of_rpe",
                 "ped_serous", "atrophy_thinning_of_retinal_layers", "vmt", "shrm")
FOLLOWUP_RE = re.compile(r"^\s*(what|which) evidence\b|\bsupports? (your|that|this|the) answer\b|\bhow do you know\b"
                         r"|\bwhy did you (say|answer|choose)\b|\bexplain (your|that|the) answer\b", re.I)
THIS_SCAN_RE = re.compile(r"\b(this|the uploaded|uploaded|my|current|today'?s)\s+(oct\s+)?(scan|image|b-?scan|picture|"
                          r"oct|one)\b|\bvisible\b|\bin the scan\b", re.I)
HISTORY_RE = re.compile(r"\b(previous|prior|earlier|baseline|last visit|history|over time|before|since|visits)\b", re.I)


@dataclass
class Turn:
    question: str
    route: str                       # history | scan_visual | upload_similarity | followup
    evidence: dict
    answer: dict
    eye: int | None
    visit: str | None
    unified: object | None = None    # question_router.UnifiedResult for history questions
    upload: UploadedScan | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def category(self) -> str:
        return self.evidence.get("category", self.route)


class Assistant:
    def __init__(self, cfg: Config, vlm=None, upload_encoder=None, episodes: pd.DataFrame | None = None,
                 visit_emb: dict | None = None, images: dict | None = None, emb_encoder: str | None = None,
                 scans: ScanIndex | None = None):
        """Normally built from `cfg`; tests pass episodes / visit_emb / images (/ scans) directly."""
        self.cfg, self.vlm = cfg, vlm
        self._encoder = upload_encoder
        if episodes is None:
            episodes = pd.read_csv(ROOT / "data/processed/episodes.csv")
            man = OI.load_manifest(cfg.subset)
            c = man[man.scan_index == 24]
            images = {r.visit_id: cfg.subset / r.local_path for r in c.itertuples() if (cfg.subset / r.local_path).exists()}
            visit_emb = H.load_visit_embeddings(cfg.subset, cfg.embeddings, mode=cfg.visit_mode)
            emb_encoder = str(np.load(cfg.embeddings, allow_pickle=False)["encoder"])
            scans = ScanIndex.load(cfg.subset, cfg.embeddings)
        self.episodes, self.images, self.visit_emb = episodes, images or {}, visit_emb or {}
        self.emb_encoder = emb_encoder
        self.router = QR.QuestionAwareRetriever(episodes, self.visit_emb)
        self.ep = episodes.set_index("visit_id")
        self._index = None
        if scans is None:                    # tests without a scan index: one stored vector per visit
            vids = [v for v in self.visit_emb if v in self.ep.index]
            scans = ScanIndex(keys=[f"{v}__scan24" for v in vids], visit_ids=vids,
                              eye_ids=[int(self.ep.loc[v, "eye_id"]) for v in vids], scan_index=[24] * len(vids),
                              paths=[self.images.get(v) for v in vids],
                              emb=np.stack([self.visit_emb[v] for v in vids]) if vids else np.zeros((0, 1)),
                              encoder=emb_encoder or "resnet50")
        self.scans = scans

    # --- dataset browsing -------------------------------------------------------------------------------------------

    def eyes(self) -> pd.DataFrame:
        e = self.episodes[self.episodes.split.isin(self.cfg.splits)]
        g = e.groupby("eye_id")
        out = g.agg(patient_id=("patient_id", "first"), subject=("subject", "first"), trial=("trial", "first"),
                    laterality=("laterality", "first"), split=("split", "first"), n_visits=("visit_id", "size"))
        out["n_with_oct"] = g.visit_id.apply(lambda v: sum(x in self.images and x in self.visit_emb for x in v))
        out["n_labelled"] = g.has_biomarkers.sum().astype(int)
        return out.reset_index().sort_values(["n_with_oct", "eye_id"], ascending=[False, True])

    def visits(self, eye: int) -> pd.DataFrame:
        v = self.episodes[self.episodes.eye_id == eye].sort_values("visit_index")
        v = v[["visit_id", "visit_label", "visit_index", "week", "cst", "bcva", "has_biomarkers"]].copy()
        v["has_oct"] = v.visit_id.map(lambda x: x in self.images and x in self.visit_emb)
        return v.reset_index(drop=True)

    def scan_path(self, visit_id: str) -> Path | None:
        return self.images.get(visit_id)

    # --- uploads ----------------------------------------------------------------------------------------------------

    def match_upload(self, scan: UploadedScan) -> UploadedScan:
        if self._index is None:
            self._index = DatasetImageIndex(self.cfg.subset, self.cfg.workdir / "cache")
        return self._index.match(scan)

    def _upload_encoder(self):
        if self._encoder is None:
            if not self.cfg.upload_embedding:
                raise RuntimeError("upload embedding is disabled (OCT_UPLOAD_EMBEDDING=0)")
            if (self.scans.encoder, self.scans.pooling) != ("resnet50", "gap"):
                raise RuntimeError(f"the stored B-scan embeddings were made with {self.scans.encoder} "
                                   f"({self.scans.pooling} pooling); uploads are embedded with ResNet-50 (gap) only, so "
                                   f"they cannot be compared. Point OCT_EMBEDDINGS at a resnet50 embeddings.npz")
            import embed_cached as EC                      # the function that produced the stored embeddings
            self._encoder, dim = EC.make_encoder("resnet50", "gap", OI.pick_device())
            if dim != self.scans.dim:
                raise RuntimeError(f"ResNet-50 gives {dim}-d embeddings but the stored ones are {self.scans.dim}-d")
        return self._encoder

    # --- routing ----------------------------------------------------------------------------------------------------

    @staticmethod
    def route(question: str, upload: UploadedScan | None) -> str:
        if FOLLOWUP_RE.search(question):
            return "followup"
        if upload is not None and QR.SIM_SEARCH_RE.search(question) and THIS_SCAN_RE.search(question):
            return "upload_similarity"
        if THIS_SCAN_RE.search(question) and not HISTORY_RE.search(question) and not QR.SIM_SEARCH_RE.search(question):
            return "scan_visual"
        return "history"

    def ask(self, question: str, eye: int | None, visit: str | None, upload: UploadedScan | None = None,
            backend: str = "evidence", previous: Turn | None = None, client=None) -> Turn:
        question = (question or "").strip()
        if not question:
            raise ValueError("empty question")
        route = self.route(question, upload)
        if route == "followup":
            return self._followup(question, eye, visit, previous)
        if route == "upload_similarity" and upload is not None and not upload.external:
            # an exact dataset match is an ordinary dataset visit: use the router from that visit
            eye, visit, route = upload.matched_eye, upload.matched_visit, "history"
            notes = [f"The uploaded scan is identical to the dataset B-scan of visit {visit} (eye {eye}); the "
                     "question was answered from that visit."]
            t = self._history(QR.CANON_SIMILAR, eye, visit, backend, client, shown=question)
            t.notes = notes + t.notes
            t.upload = upload
            return t
        if route in ("history",) or (route == "scan_visual" and upload is None):
            if eye is None or visit is None:
                raise ValueError("select an eye and a current visit first")
        if route == "history":
            t = self._history(question, eye, visit, backend, client)
            if upload is not None:
                t.notes.append("The uploaded scan was not used: this question is about the selected eye's visit "
                               "history.")
            return t
        if route == "scan_visual":
            ev = self._scan_evidence(question, eye, visit, upload)
        else:
            ev = self._upload_similarity_evidence(question, eye, visit, upload)
        ans = self._answer(ev, backend, client)
        return Turn(question, route, ev, ans, eye, visit, upload=upload)

    def _answer(self, ev: dict, backend: str, client=None) -> dict:
        """The vision-language model answers from the evidence (app/reasoning.py); 'claude' uses the Anthropic adapter
        (with `client` when given), 'evidence' answers without a model."""
        vlm = self.vlm
        if backend == "claude":
            from app.vlm import AnthropicVLM
            vlm, backend = AnthropicVLM(client=client), "vlm"
        return RS.generate(ev, backend, vlm, self.ep, self.images, self.cfg.max_images)

    # --- routes -----------------------------------------------------------------------------------------------------

    def _history(self, question, eye, visit, backend, client, shown: str | None = None) -> Turn:
        u = self.router.retrieve(question, eye, visit)
        ev = A.build_unified_evidence(u, self.episodes, self.images)
        if shown:
            ev["question"] = shown
        ans = self._answer(ev, backend, client)
        return Turn(ev["question"], "history", ev, ans, eye, visit, unified=u)

    def _labels(self, eye: int, visit: str) -> list[dict]:
        return self.router._labels(eye, [visit], FINDING_STEMS)

    def _base_ev(self, question, category, route, eye) -> dict:
        return dict(question=question, status="retrieved", category=category, route=route, reason="",
                    intent="app/" + category, operation=None, current=None, upload=None, retrieved=[], images=[],
                    uses_images=True, n_candidates=0, eye_id=eye, components=[], disclosures=[], facts=[],
                    biomarkers=[], guidance="", ranking_basis="")

    def _scan_evidence(self, question, eye, visit, upload) -> dict:
        ev = self._base_ev(question, "scan_visual", "scan_visual", eye)
        ev["ranking_basis"] = "Single-scan question: no retrieval; the scan itself is the evidence."
        if upload is not None:
            ev["upload"] = dict(visit_id="UPLOADED_SCAN", image_attached=True, matched_visit=upload.matched_visit,
                                matched_eye=upload.matched_eye)
            ev["images"] = [dict(visit_id="UPLOADED_SCAN", role="uploaded", path=str(upload.path))]
            ev["components"] = [dict(name="uploaded_scan", status="retrieved", explanation="user-uploaded B-scan")]
            if upload.external:
                ev["eye_id"] = None
                ev["disclosures"].append("The uploaded scan is external: it is not linked to any patient, eye, visit or "
                                         "date, and no dataset labels or history apply to it.")
            else:
                ev["eye_id"] = upload.matched_eye
                ev["biomarkers"] = self._labels(upload.matched_eye, upload.matched_visit)
                ev["disclosures"].append(f"The uploaded scan is identical to the dataset B-scan of visit "
                                         f"{upload.matched_visit} (eye {upload.matched_eye}).")
        else:
            path = self.images.get(visit)
            if path is None:
                ev.update(status="unsupported_image", reason=f"no OCT B-scan is available for visit {visit}",
                          uses_images=False, guidance=A.ABSTENTION_GUIDANCE["unsupported_image"])
                ev["current"] = A._visit_row(self.ep, visit)
                ev["current"]["image_attached"] = False
                return ev
            ev["current"] = A._visit_row(self.ep, visit)
            ev["current"]["image_attached"] = True
            ev["images"] = [dict(visit_id=visit, role="current", path=str(path))]
            ev["biomarkers"] = self._labels(eye, visit)
            ev["components"] = [dict(name="current_visit_scan", status="retrieved",
                                     explanation=f"central B-scan of the selected visit {visit}")]
        ev["disclosures"].append("Any description of the image is the model's own visual observation of a single "
                                 "B-scan; it is not a validated finding." + (
                                     " Dataset finding labels for this visit are listed separately." if ev["biomarkers"]
                                     else " No dataset finding labels exist for this scan."))
        return ev

    def _upload_similarity_evidence(self, question, eye, visit, upload) -> dict:
        """Upload vs the stored INDIVIDUAL B-scan embeddings of the selected eye's eligible visits (strictly before the
        selected visit; all of the eye's visits when none is selected). Per visit the best-matching scan and its
        cosine are kept; visits are ranked by that maximum. The model later compares the scans but never ranks."""
        ev = self._base_ev(question, "upload_similarity", "upload_image_similarity", eye)
        ev["upload"] = dict(visit_id="UPLOADED_SCAN", image_attached=True, matched_visit=None, matched_eye=None)
        if eye is None:
            ev.update(status="insufficient_data", reason="select the eye to compare the uploaded scan with",
                      uses_images=False)
            return ev
        vis = self.visits(eye)
        if visit is not None and visit in set(vis.visit_id):
            vis = vis[vis.visit_index < int(vis.set_index("visit_id").loc[visit, "visit_index"])]
            ev["current"] = A._visit_row(self.ep, visit)
            ev["current"]["image_attached"] = False
        cands = list(vis.visit_id)
        try:
            q = embed_scan(upload, self._upload_encoder())
        except Exception as ex:                       # report the real reason (missing torch, weights download, ...)
            ev.update(status="unsupported_image", uses_images=False, guidance=A.ABSTENTION_GUIDANCE["unsupported_image"],
                      reason=f"the uploaded scan could not be embedded: {type(ex).__name__}: {A.redact(ex)}")
            return ev
        try:
            ranked = self.scans.rank_visits(q, eye, cands)
        except ValueError as ex:
            ev.update(status="unsupported_image", uses_images=False, reason=f"embedding mismatch: {ex}")
            return ev
        if not ranked:
            ev.update(status="insufficient_data", uses_images=False,
                      reason=f"eye {eye} has no stored B-scan embeddings for the {len(cands)} eligible visits"
                             + (f" before {visit}" if visit else ""))
            return ev
        for rank, b in enumerate(ranked[:A.TOP_N], 1):
            row = A._visit_row(self.ep, b["visit_id"])
            path = b["path"] if b["path"] is not None and Path(b["path"]).exists() else None
            row.update(rank=rank, role="selected" if rank == 1 else "runner-up", sources=[f"upload_similarity #{rank}"],
                       image_cosine=round(b["score"], 4), image_attached=path is not None, best_scan=b["image_key"],
                       best_scan_file=None if path is None else Path(path).name, best_scan_index=b["scan_index"],
                       best_scan_path=None if path is None else str(path), n_scans_compared=b["n_scans"])
            ev["retrieved"].append(row)
        ev["images"] = [dict(visit_id="UPLOADED_SCAN", role="uploaded", path=str(upload.path))] + [
            dict(visit_id=r["visit_id"], role="retrieved", path=r["best_scan_path"], scan=r["best_scan"])
            for r in ev["retrieved"] if r["image_attached"]]
        n_scans = sum(b["n_scans"] for b in ranked)
        ev["n_candidates"] = len(ranked)
        ev["eligible"] = [v for v in cands if v in {b["visit_id"] for b in ranked}]
        ev["ranking"] = [b["visit_id"] for b in ranked]
        ev["ranking_basis"] = (f"The uploaded scan was embedded with the same ResNet-50 (global average pooling, "
                               f"{self.scans.dim}-d) and preprocessing as the stored embeddings and compared by cosine "
                               f"with {n_scans} individual stored B-scans of the {len(ranked)} eligible visits of eye "
                               f"{eye}; each visit is scored by its best-matching B-scan.")
        ev["components"] = [dict(name="upload_image_similarity", status="retrieved", explanation=ev["ranking_basis"])]
        missing = [r["visit_id"] for r in ev["retrieved"] if not r["image_attached"]]
        ev["disclosures"] += [
            "The uploaded scan is external: it has no date, and the ranking does not show that it belongs to this eye.",
            "Image-embedding similarity (generic ImageNet encoder) is not clinical similarity and says nothing about "
            "disease progression. The ranking is fixed by the embeddings; the model only compares the scans."]
        if missing:
            ev["disclosures"].append(f"The best-matching B-scan image file is not available locally for "
                                     f"{', '.join(missing)} (its stored embedding was used for ranking).")
        if upload.warnings:
            ev["disclosures"].append("Upload warnings: " + "; ".join(upload.warnings))
        return ev

    def _followup(self, question, eye, visit, previous: Turn | None) -> Turn:
        ev = self._base_ev(question, "followup", "followup", eye)
        ev.update(uses_images=False)
        if previous is None:
            ev.update(status="insufficient_data", reason="there is no previous answer to explain")
            return Turn(question, "followup", ev, A.answer(ev), eye, visit)
        p = previous.evidence
        lines = [f"The previous answer ({previous.question!r}) was based on: {previous.answer['evidence_used']}.",
                 f"Route: {p.get('category')} (status {p['status']}). {p.get('ranking_basis', '')}"]
        lines += [f"Disclosure: {d}" for d in p.get("disclosures", [])]
        lines += [f"Fact: {f}" for f in p.get("facts", [])]
        if previous.answer.get("warnings"):
            lines += [f"Grounding warning: {w}" for w in previous.answer["warnings"]]
        ans = dict(status="answered", answer="\n".join(lines), backend="none", evidence_used="previous turn's evidence",
                   warnings=[], error=None, disclaimer=A.DISCLAIMER)
        return Turn(question, "followup", p, ans, eye, visit, unified=previous.unified, upload=previous.upload)
