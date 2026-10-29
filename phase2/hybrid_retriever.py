"""Metadata + image hybrid retriever (Phase 2). Isolated: imports the frozen Phase 1 parser/retriever read-only.

Order of operations for a question asked at visit t of one eye (nothing later can widen what an earlier step allowed):
  1. Eligibility (hard): candidates are the strictly earlier visits of the SAME eye (hence the same patient),
     taken from the frozen VisitMemory. Image similarity never adds, removes or reorders eligibility.
  2. Metadata: frozen `intent.parse_intent` + `intent_retrieval.retrieve`. Its outcome decides the route:
       retrieved            -> metadata ranking. Image ranks are fused (RRF) ONLY for operations listed in
                               `fuse_ops` (default: none), so a filter/anchor answer ("last visit thicker than now",
                               "baseline", "previous") is never reordered by appearance.
       unsupported_image    -> image route: image ranking, optionally RRF-fused with a metadata proxy
                               (`image_proxy`, default CST closeness = the pre-specified S5 design). If the current
                               visit has no image embedding, or no candidate has one, the retriever ABSTAINS.
       any other abstention -> returned unchanged (no_qualifying_visit, insufficient_data, unavailable_variable,
                               clarification_needed). The image ranking is still computed for inspection only.
  3. Missing images: candidates without an embedding get no image score. In RRF they keep only their metadata
     term; in the image-only route they are appended after the image-ranked visits, most recent first.
Ties everywhere: more recent visit first (question_eval.recency_order), as in the frozen rankings.

Every result keeps the metadata ranking, the image ranking (with cosine scores) and the final ranking;
`HybridResult.table()` shows them side by side per candidate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "phase2"))
from oct_retrieval import intent as I                 # noqa: E402  (frozen, read-only)
from oct_retrieval import intent_retrieval as R       # noqa: E402
import oct_image as OI                                # noqa: E402
from question_eval import RRF_K, recency_order        # noqa: E402

CST_PROXY = I.QueryIntent(task=I.RETRIEVE, operation=I.CLOSEST_TO_CURRENT, target="cst")


@dataclass
class HybridResult:
    status: str
    route: str
    reason: str
    intent: str
    query_visit_id: str
    candidates: list[str]
    metadata_status: str
    metadata_ranking: list[str] | None
    image_ranking: list[str] | None
    image_scores: dict[str, float] = field(default_factory=dict)
    proxy_ranking: list[str] | None = None
    final_ranking: list[str] | None = None
    no_image: list[str] = field(default_factory=list)

    @property
    def abstained(self) -> bool:
        return self.status != R.RETRIEVED

    def table(self) -> pd.DataFrame:
        def rank(lst):
            return {v: i + 1 for i, v in enumerate(lst or [])}
        rm, rp, ri, rf = (rank(x) for x in (self.metadata_ranking, self.proxy_ranking, self.image_ranking,
                                            self.final_ranking))
        return pd.DataFrame([dict(candidate_visit_id=v, chronological_position=i, metadata_rank=rm.get(v),
                                  proxy_rank=rp.get(v), image_rank=ri.get(v), image_cosine=self.image_scores.get(v),
                                  final_rank=rf.get(v)) for i, v in enumerate(self.candidates)])


def rrf(rankings: list[list[str]], candidates: list[str], k: int = RRF_K) -> list[str]:
    """Reciprocal-rank fusion over `candidates`; a candidate absent from a ranking gets no term from it."""
    pos = {v: i for i, v in enumerate(candidates)}
    score = {v: 0.0 for v in candidates}
    for lst in rankings:
        for i, v in enumerate(lst):
            if v in score:
                score[v] += 1.0 / (k + i + 1)
    return recency_order(score, pos)


class HybridRetriever:
    def __init__(self, episodes: pd.DataFrame, visit_emb: dict[str, np.ndarray], rrf_k: int = RRF_K,
                 image_proxy: I.QueryIntent | None = CST_PROXY, fuse_ops: frozenset[str] = frozenset()):
        """visit_emb: visit_id -> L2-normalised embedding (visits without an image are simply absent)."""
        self.episodes = episodes
        self.emb = visit_emb
        self.k = rrf_k
        self.image_proxy = image_proxy
        self.fuse_ops = frozenset(fuse_ops)
        self._mem: dict[int, R.VisitMemory] = {}

    def memory(self, eye_id: int) -> R.VisitMemory:
        if eye_id not in self._mem:
            h = self.episodes[self.episodes.eye_id == eye_id].sort_values("visit_index")
            self._mem[eye_id] = R.build_memories(h)[eye_id]
        return self._mem[eye_id]

    def image_ranking(self, query_visit_id: str, candidates: list[str]) -> tuple[list[str] | None, dict]:
        if query_visit_id not in self.emb:
            return None, {}
        q = self.emb[query_visit_id]
        sc = {v: float(self.emb[v] @ q) for v in candidates if v in self.emb}
        return (recency_order(sc, {v: i for i, v in enumerate(candidates)}) if sc else None), sc

    def retrieve(self, question: str, eye_id: int, query_visit_id: str) -> HybridResult:
        mem = self.memory(eye_id)
        if query_visit_id not in mem.visit_ids:
            raise ValueError(f"visit {query_visit_id} does not belong to eye {eye_id}")
        t = mem.visit_ids.index(query_visit_id)
        cands = list(mem.visit_ids[:t])                                  # step 1: hard eligibility
        intent = I.parse_intent(question)
        meta = R.retrieve(intent, mem, t)                                # step 2: frozen metadata system
        meta_rank = None if meta.abstained else [mem.visit_ids[i] for i in meta.ranked]
        img, scores = self.image_ranking(query_visit_id, cands)
        res = HybridResult(status=meta.status, route="abstain", reason=meta.reason,
                           intent=f"{intent.task}/{intent.operation}/{intent.target}", query_visit_id=query_visit_id,
                           candidates=cands, metadata_status=meta.status, metadata_ranking=meta_rank,
                           image_ranking=img, image_scores=scores, no_image=[v for v in cands if v not in self.emb])

        if meta.status == R.RETRIEVED:
            if intent.operation in self.fuse_ops and img is not None:
                res.route, res.final_ranking = "metadata+image_rrf", rrf([meta_rank, img], cands, self.k)
            else:
                res.route, res.final_ranking = "metadata", meta_rank
        elif meta.status == R.UNSUPPORTED:
            if query_visit_id not in self.emb:
                res.reason = "image question, but the current visit has no image embedding"
            elif img is None:
                res.reason = "image question, but no eligible earlier visit has an image embedding"
            else:
                proxy = None
                if self.image_proxy is not None:
                    p = R.retrieve(self.image_proxy, mem, t)
                    proxy = None if p.abstained else [mem.visit_ids[i] for i in p.ranked]
                res.proxy_ranking = proxy
                res.status = R.RETRIEVED
                if proxy is not None:
                    res.route, res.final_ranking = "image+proxy_rrf", rrf([proxy, img], cands, self.k)
                else:
                    rest = [v for v in reversed(cands) if v not in scores]
                    res.route, res.final_ranking = "image", img + rest
                res.reason = "image question answered from images" + (" + metadata proxy" if proxy else "")
        if res.final_ranking is not None:
            assert sorted(res.final_ranking) == sorted(cands), "final ranking must be a permutation of eligible visits"
        return res


def load_visit_embeddings(subset: Path, emb_path: Path, mode: str = "central", central_scan: int = 24,
                          exclude_duplicates: bool = False) -> dict[str, np.ndarray]:
    """visit_id -> L2-normalised visit embedding from SAVED scan embeddings.
    mode 'central': the central B-scan only (as question_eval); 'mean': mean of the visit's L2-normalised scans,
    re-normalised (oct_image.visit_embeddings). exclude_duplicates drops scans whose bytes (SHA-256) appeared earlier
    in the subset, so a byte-identical scan does not count twice in a visit mean."""
    m = OI.load_manifest(subset).set_index("image_key")
    d = np.load(emb_path, allow_pickle=False)
    keys = list(d["image_key"])
    mm = m.loc[keys].reset_index()
    E = d["embeddings"]
    if exclude_duplicates:
        keep = (~mm.sha256.duplicated()).to_numpy()
        mm, E = mm[keep].reset_index(drop=True), E[keep]
    if mode == "central":
        sel = (mm.scan_index == central_scan).to_numpy()
        return dict(zip(mm.visit_id[sel], OI.l2n(E[sel])))
    if mode == "mean":
        v, vis = OI.visit_embeddings(E, mm)
        return dict(zip(vis.visit_id, v))
    raise ValueError(f"unknown mode {mode!r}")
