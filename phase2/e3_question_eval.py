"""E3: question-conditioned retrieval evaluation across question categories (Phase 3). Saved embeddings only.

Ground truth is never created here. Each category states where its relevance labels come from:
  structural          visit order (previous visit, baseline)
  metadata (frozen)   Phase 1 answer keys (questions.relevant_indices v1, ground_truth_v2.answer_key v2), computed from
                      the SAME CST/BCVA columns the metadata system ranks by: metadata-only reaching 1.0 there is true by
                      construction and shows nothing about clinical understanding
  human (non-expert)  the completed dev B1 visual-similarity judgments (one researcher_non_expert annotator, 3 eyes)
Categories without usable ground truth are reported as N/A with the reason (pathology at visit level, mixed).

Systems (all restricted to the eye's strictly earlier visits):
  latest_prior            most recent first
  metadata_only           frozen Phase 1 parser + retriever (abstains on image questions)
  image_<enc>             cosine similarity of visit embeddings only (ignores the question)
  safe_hybrid_<enc>       hybrid_retriever.HybridRetriever (metadata, or image + CST-proxy RRF for image questions)
  question_aware_<enc>    question_router.QuestionAwareRetriever (the unified system)
  random_expected         exact expectation under a random permutation
Metrics per query: Recall@1, Recall@5, Hit@1, MRR, nDCG@5 (question_eval.score_ranking; Recall@1 = Hit@1 / #relevant).
An abstaining system scores 0 and is counted in `coverage`. Several tied visits are all relevant (binary gains); B1
uses graded gains 2^g - 1 with grade 2 = relevant (protocol v2). The unit of independence is the eye: CIs are
eye-level bootstrap and only with >= 5 eyes.

Usage:
  python phase2/e3_question_eval.py --subset phase2/data/atlas_fullhist \
      --emb resnet50=phase2/results/fullhist/resnet50/embeddings.npz --out phase2/results/e3/fullhist_v1
Outputs (new folder, never overwritten): per_query.csv, summary.csv, summary.md, config.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "phase2"))
import hybrid_retriever as H  # noqa: E402
import question_eval as QE  # noqa: E402
import question_router as QR  # noqa: E402
from oct_retrieval import ground_truth_v2 as GT2  # noqa: E402  (frozen, read-only)
from oct_retrieval import intent as I  # noqa: E402
from oct_retrieval import intent_retrieval as R  # noqa: E402
from oct_retrieval import questions as Q1  # noqa: E402

METRICS = ("recall1", "recall5", "hit1", "mrr", "ndcg5")
MIN_T = QE.MIN_CANDIDATES

# category -> (group, question, ground-truth source, key function(hist, cur) -> list of relevant positions | None)
def _v1(name):
    spec = Q1.QUESTION_TYPES[name][0]
    return lambda h, c: (None if (spec.intent == "similar" and np.isnan(c[spec.target]))
                         else list(Q1.relevant_indices(spec, h[spec.target].to_numpy(float), float(c[spec.target]))))


def _v2(name):
    def f(h, c):
        k = GT2.answer_key(name, h, c)
        return None if k is GT2.INELIGIBLE else list(k or [])
    return f


CATEGORIES = {
    "temporal_previous": ("temporal", "Compare with the previous visit.", "structural (visit order)",
                          lambda h, c: [len(h) - 1]),
    "temporal_baseline": ("temporal", "How has the retina changed since baseline?", "structural (visit order)",
                          lambda h, c: [0]),
    "cst_highest": ("cst_metadata", "When was CST highest?", "metadata (frozen v1 key: max CST)", _v1("worst_edema")),
    "cst_lowest": ("cst_metadata", "When was CST lowest?", "metadata (frozen v1 key: min CST)", _v1("best_anatomy")),
    "cst_most_similar": ("cst_metadata", "When was the CST most similar to today's measurement?",
                         "metadata (frozen v1 key)", _v1("similar_cst")),
    "cst_largest_rise": ("cst_metadata", QE.TIER_A["largest_cst_rise"][0], "metadata (frozen v2 key)",
                         _v2("largest_cst_rise")),
    "cst_last_above_today": ("cst_metadata", QE.TIER_A["last_thicker_than_now"][0], "metadata (frozen v2 key)",
                             _v2("last_thicker_than_now")),
    "edema_worst_proxy": ("pathology_proxy", "When was edema worst?",
                          "metadata (frozen v1 'worst_edema' key = max CST; Phase 1 operationalisation, NOT a clinical "
                          "edema grade)", _v1("worst_edema")),
}
NOT_EVALUABLE = {
    "image_similarity (all eyes)": "no visual relevance labels beyond the 3 dev eyes of B1 (reported separately)",
    "pathology: fluid presence / first appearance / improvement (visit ranking)":
        "OLIVES finding labels exist only for the first and last visit of each eye, so no visit ranking over a "
        "history can be labelled; scan-level finding detection is evaluated in pathology_probe.py",
    "mixed (image + metadata)": "no ground truth exists for the combined answer",
}


def score(ranked, grade: dict, thr: float) -> dict:
    m = QE.score_ranking(ranked, grade, thr)
    n_rel = sum(g >= thr for g in grade.values())
    m["recall1"] = m["hit1"] / n_rel if n_rel else float("nan")
    return m


def systems_for(question, eye, visit, cands, retrievers):
    out = {"latest_prior": cands[::-1]}
    mem = retrievers["mem"](eye)
    t = mem.visit_ids.index(visit)
    res = R.retrieve(I.parse_intent(question), mem, t)
    out["metadata_only"] = None if res.abstained else [mem.visit_ids[i] for i in res.ranked]
    for enc, (hyb, qa) in retrievers["enc"].items():
        img, _ = hyb.image_ranking(visit, cands)
        out[f"image_{enc}"] = None if img is None else img + [v for v in reversed(cands) if v not in img]
        out[f"safe_hybrid_{enc}"] = hyb.retrieve(question, eye, visit).final_ranking
        out[f"question_aware_{enc}"] = qa.retrieve(question, eye, visit).final_ranking
    return out


def metadata_categories(episodes, vembs, eyes) -> pd.DataFrame:
    retr = dict(enc={e: (H.HybridRetriever(episodes, v), QR.QuestionAwareRetriever(episodes, v))
                     for e, v in vembs.items()})
    first = next(iter(retr["enc"].values()))[0]
    retr["mem"] = first.memory
    rows = []
    for cat, (group, q, src, key) in CATEGORIES.items():
        for eye in eyes:
            mem = first.memory(eye)
            h = episodes[episodes.eye_id == eye].sort_values("visit_index").reset_index(drop=True)
            for t in range(MIN_T, len(h)):
                hist, cur = h.iloc[:t], h.iloc[t]
                cands = list(h.visit_id.iloc[:t])
                rel = key(hist, cur)
                base = dict(category=cat, group=group, question=q, gt_source=src, eye_id=int(eye),
                            patient_id=int(cur.patient_id), query_visit_id=cur.visit_id, n_candidates=t)
                if rel is None or not rel:
                    rows.append(dict(base, system="-", status="ineligible" if rel is None else "no_relevant_visit"))
                    continue
                grade = {v: float(i in rel) for i, v in enumerate(cands)}
                for s, rk in systems_for(q, eye, cur.visit_id, cands, retr).items():
                    rows.append(dict(base, system=s, status="evaluated", n_relevant=len(rel), **score(rk, grade, 1.0)))
                r = QE.random_expectation(grade, 1.0)
                r["recall1"] = r["hit1"] / len(rel)
                rows.append(dict(base, system="random_expected", status="evaluated", n_relevant=len(rel), answered=True,
                                 **r))
    return pd.DataFrame(rows)


def b1_category(episodes, vembs) -> pd.DataFrame | None:
    """Image similarity with independent (non-expert, human) labels: the dev B1 judgments, if the key is present."""
    import evaluate_annotations as EA
    import large_eval as LE
    if not (LE.B1_KEY.exists() and LE.B1_SHEET.exists()):
        return None
    frozen, _, _ = EA.load_rankings(LE.B1_RANKINGS)
    qs = pd.read_csv(LE.B1_RANKINGS / "query_set.csv")
    qs = qs[qs.category == "overall_appearance"]
    need = set(qs.query_visit_id) | {v for c in qs.candidate_visit_ids for v in c.split(";")}
    rows = []
    for enc, v in vembs.items():
        if not need <= set(v):
            return None
        hyb, qa = H.HybridRetriever(episodes, v), QR.QuestionAwareRetriever(episodes, v)
        for q in qs.itertuples():
            cands = q.candidate_visit_ids.split(";")
            img, _ = hyb.image_ranking(q.query_visit_id, cands)
            for s, rk in {f"image_{enc}": img, f"safe_hybrid_{enc}": hyb.retrieve(q.question, q.eye_id,
                                                                                q.query_visit_id).final_ranking,
                          f"question_aware_{enc}": qa.retrieve(q.question, q.eye_id, q.query_visit_id).final_ranking
                          }.items():
                rows += [dict(query_id=q.query_id, query_visit_id=q.query_visit_id, category=q.category,
                              eye_id=q.eye_id, system=s, rank=i + 1, candidate_visit_id=c) for i, c in enumerate(rk or [])]
    keep = frozen[(frozen.category == "overall_appearance") &
                  frozen.system.isin(["S1_latest_prior", "S2_metadata", "S2p_metadata_proxy_cst"])].copy()
    keep["system"] = keep.system.map({"S1_latest_prior": "latest_prior", "S2_metadata": "metadata_only",
                                      "S2p_metadata_proxy_cst": "metadata_cst_proxy"})
    rk = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)
    judg = EA.join_sheet_key(EA.load_sheet(LE.B1_SHEET), pd.read_csv(LE.B1_KEY))
    EA.check_pools(judg, rk)
    systems = sorted(set(rk.system)) + ["metadata_only"]
    per = EA.evaluate(judg, rk, pd.DataFrame(dict(system=sorted(set(systems)))), "primary", 2.0)
    per["recall1"] = per.hit1 / per.n_relevant.replace(0, np.nan)
    per = per.drop(columns=["query_code"], errors="ignore")      # never write annotation-code -> visit pairs
    ep = episodes.set_index("visit_id")
    per = per.assign(category="image_similarity_B1", group="image_similarity",
                     question="Which earlier visit's OCT looks most like today's?",
                     gt_source="human (B1, single researcher_non_expert annotator; visual similarity only)",
                     patient_id=per.query_visit_id.map(ep.patient_id).astype(int))
    return per


def summarise(per: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (cat, s), g in per[per.system != "-"].groupby(["category", "system"], sort=False):
        ev = g[g.status == "evaluated"]
        row = dict(category=cat, group=g.group.iloc[0], system=s, gt_source=g.gt_source.iloc[0],
                   n_queries=len(ev), n_eyes=int(ev.eye_id.nunique()), n_patients=int(ev.patient_id.nunique()))
        if ev.empty:
            out.append(row)
            continue
        row["coverage"] = round(float(ev.answered.astype(bool).mean()), 4) if "answered" in ev else None
        for m in METRICS:
            row[m] = round(float(ev[m].mean()), 4)
        primary = "ndcg5" if cat == "image_similarity_B1" else "mrr"
        ci = QE.bootstrap_ci(ev.groupby("eye_id")[primary].mean())
        row[f"primary_metric"] = primary
        row["primary_ci95_eye_bootstrap"] = "n_eyes < 5" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        out.append(row)
    return pd.DataFrame(out)


def to_markdown(summ: pd.DataFrame, cfg: dict) -> str:
    cols = ["system", "n_queries", "n_eyes", "n_patients", "coverage", *METRICS, "primary_ci95_eye_bootstrap"]
    parts = [f"# E3 question-conditioned retrieval: `{cfg['subset']}`", "",
             f"Created {cfg['created_utc']}. Eyes: {cfg['eyes']}. Exploratory; see the caveats in the module docstring.",
             ""]
    for cat, g in summ.groupby("category", sort=False):
        parts += [f"## {cat}  ({g.group.iloc[0]})", "", f"Ground truth: {g.gt_source.iloc[0]}", "",
                  "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in g.iterrows():
            parts.append("| " + " | ".join("" if pd.isna(r.get(c)) else str(r.get(c)) for c in cols) + " |")
        parts.append("")
    parts += ["## Not evaluable (N/A)", ""] + [f"- **{k}**: {v}" for k, v in NOT_EVALUABLE.items()]
    return "\n".join(parts) + "\n"


def run(subset: Path, emb_paths: dict, out: Path, eyes=None, visit_mode="central", b1=True) -> pd.DataFrame:
    episodes = pd.read_csv(ROOT / "data/processed/episodes.csv")
    vembs = {k: H.load_visit_embeddings(subset, Path(p), mode=visit_mode) for k, p in emb_paths.items()}
    have = set.intersection(*(set(v) for v in vembs.values()))
    complete = [int(e) for e, g in episodes.groupby("eye_id") if set(g.visit_id) <= have]
    train = set(episodes[episodes.split == "train"].eye_id)
    eyes = sorted(e for e in (eyes or complete) if e in train and e in complete)
    if not eyes:
        raise SystemExit("no train eye has a complete embedded history in this subset")
    per = metadata_categories(episodes, vembs, eyes)
    if b1:
        b = b1_category(episodes, vembs)
        if b is not None:
            per = pd.concat([per, b], ignore_index=True)
    summ = summarise(per)
    cfg = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), subset=str(subset),
               embeddings={k: str(v) for k, v in emb_paths.items()}, visit_mode=visit_mode, eyes=eyes,
               min_candidates=MIN_T, not_evaluable=NOT_EVALUABLE, bootstrap=QE.BOOTSTRAP)
    out.mkdir(parents=True)
    per.to_csv(out / "per_query.csv", index=False)
    summ.to_csv(out / "summary.csv", index=False)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "summary.md").write_text(to_markdown(summ, cfg), encoding="utf-8")
    return summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--emb", action="append", required=True, help="name=path/to/embeddings.npz (repeatable)")
    ap.add_argument("--eyes", type=int, nargs="*")
    ap.add_argument("--visit-mode", choices=["central", "mean"], default="central")
    ap.add_argument("--no-b1", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    emb = dict(x.split("=", 1) for x in a.emb)
    run(a.subset, emb, a.out, a.eyes, a.visit_mode, not a.no_b1)
    print((a.out / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
