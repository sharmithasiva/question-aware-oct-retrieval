"""Prototype: compare image-based and metadata-based ranking of prior visits on the verified atlas subset.

NOT an evaluation. There is no relevance ground truth for "visually similar visit", so this reports
agreement between rankings and simple associations only. It must not be read as evidence that images
improve retrieval.

Setting (differs from Phase 1 on purpose, and is stated in every output):
  * History = only the visits present in the subset (4 per eye, spaced over the eye's full history),
    not the full history used by Phase 1. Positions 0..3 are subset positions.
  * For every eye and every query position t in {1, 2, 3}, the candidates are subset positions 0..t-1.
  * Metadata rankings come from the FROZEN Phase 1 code (oct_retrieval.intent.parse_intent +
    oct_retrieval.intent_retrieval.retrieve), imported read-only; it is not modified or re-run on its benchmark.
  * Image ranking: cosine similarity between visit embeddings (mean of the visit's scan embeddings).
  * Hybrid router (prototype): parse with the frozen parser; if it abstains with `unsupported_image`,
    answer with the image ranking; otherwise use the metadata ranking.

Usage: python phase2/image_vs_metadata.py --emb phase2/results/<encoder>/embeddings.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))           # frozen Phase 1 package, imported read-only
sys.path.insert(0, str(ROOT / "phase2"))

from oct_retrieval import intent as I            # noqa: E402
from oct_retrieval import intent_retrieval as R  # noqa: E402
import oct_image as OI                           # noqa: E402

QUESTIONS = {
    "closest_cst": "Which earlier visit had a central subfield thickness most similar to today?",
    "closest_bcva": "Which prior visit had visual acuity closest to today?",
    "previous": "Compare with the previous visit.",
    "baseline": "How has the retina changed since baseline?",
    "image_like": "Which earlier scan looks most like today's OCT?",
}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman correlation of x and y controlling for z (first-order partial correlation on ranks)."""
    rxy, rxz, ryz = spearman(x, y), spearman(x, z), spearman(y, z)
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return float((rxy - rxz * ryz) / den) if den > 0 else float("nan")


def run(emb_path: Path, subset_dir: Path, out_dir: Path) -> dict:
    d = np.load(emb_path, allow_pickle=False)
    m = OI.load_manifest(subset_dir)
    if list(d["image_key"]) != list(m.image_key):
        m = m.set_index("image_key").loc[list(d["image_key"])].reset_index()
    vemb, vis = OI.visit_embeddings(d["embeddings"], m)
    order = vis.sort_values(["eye_id", "visit_index"]).index.to_numpy()
    vemb, vis = vemb[order], vis.loc[order].reset_index(drop=True)

    episodes = pd.read_csv(ROOT / "data" / "processed" / "episodes.csv")
    sub_ep = episodes[episodes.visit_id.isin(vis.visit_id)]
    memories = R.build_memories(sub_ep)          # restricted to subset visits; refuses held-out columns
    n_hist = episodes.groupby("eye_id").size()
    complete = bool(all(len(g) == n_hist[e] for e, g in vis.groupby("eye_id")))
    if complete:
        # Full histories: the memory must be exactly the Phase 1 memory for these eyes.
        ref = R.build_memories(episodes[episodes.eye_id.isin(vis.eye_id)])
        for e, mem in memories.items():
            r = ref[e]
            assert mem.visit_ids == r.visit_ids and np.array_equal(mem.cst, r.cst, equal_nan=True)                 and np.array_equal(mem.bcva, r.bcva, equal_nan=True), f"memory mismatch for eye {e}"
    parsed = {k: I.parse_intent(q) for k, q in QUESTIONS.items()}

    rows, pairs = [], []
    for eye, g in vis.groupby("eye_id"):
        mem = memories[eye]
        assert mem.visit_ids == list(g.visit_id), "visit order mismatch between subset and memory"
        idx = g.index.to_numpy()
        sim = vemb[idx] @ vemb[idx].T
        for i in range(len(idx)):
            for j in range(i):
                pairs.append(dict(eye_id=eye, trial=g.trial.iloc[0], visit_a=g.visit_id.iloc[j],
                                  visit_b=g.visit_id.iloc[i], cos=float(sim[i, j]),
                                  abs_d_cst=abs(g.cst.iloc[i] - g.cst.iloc[j]),
                                  abs_d_bcva=abs(g.bcva.iloc[i] - g.bcva.iloc[j]), gap_positions=i - j))
        for t in range(1, len(idx)):
            img_rank = tuple(int(x) for x in np.argsort(-sim[t, :t], kind="stable"))
            row = dict(eye_id=eye, trial=g.trial.iloc[0], split=g.split.iloc[0], query_visit=g.visit_id.iloc[t],
                       t=t, n_candidates=t, image_rank=" ".join(g.visit_id.iloc[list(img_rank)]),
                       image_top1=g.visit_id.iloc[img_rank[0]], image_top1_cos=float(sim[t, img_rank[0]]),
                       image_top1_is_previous=img_rank[0] == t - 1)
            for k, q in parsed.items():
                res = R.retrieve(q, mem, t)
                row[f"{k}_status"] = res.status
                if res.status == R.RETRIEVED:
                    row[f"{k}_top1"] = mem.visit_ids[res.ranked[0]]
                    row[f"{k}_agree_top1"] = res.ranked[0] == img_rank[0]
                    pos_meta = np.empty(t)
                    pos_meta[list(res.ranked)] = np.arange(t)
                    pos_img = np.empty(t)
                    pos_img[list(img_rank)] = np.arange(t)
                    row[f"{k}_spearman"] = spearman(pos_meta, pos_img)
                    # where the metadata top-1 sits in the image ranking: 0 = image top-1, 1 = image last
                    row[f"{k}_top1_norm_rank_in_image"] = pos_img[res.ranked[0]] / (t - 1) if t >= 2 else np.nan
                # Hybrid router: image ranking only where the frozen system abstains for an image question.
                if k == "image_like":
                    row["hybrid_status"] = "image_retrieved" if res.status == R.UNSUPPORTED else res.status
                    row["hybrid_top1"] = row["image_top1"] if res.status == R.UNSUPPORTED else row.get(f"{k}_top1")
            rows.append(row)

    per_q = pd.DataFrame(rows)
    pr = pd.DataFrame(pairs)
    # t = 1 has a single candidate, so every ranking trivially agrees; agreement is reported on t >= 2 only.
    multi = per_q[per_q.n_candidates >= 2]
    full = per_q[per_q.n_candidates >= 3]           # Spearman needs >= 3 candidates

    def pair_rho(p: pd.DataFrame) -> dict:
        cos, gap = p.cos.to_numpy(), -p.gap_positions.to_numpy()
        return {"n_pairs": len(p),
                # rank partial correlations controlling for time gap (CST/BCVA closeness co-vary with recency)
                "partial_cst_given_gap": round(partial_spearman(cos, -p.abs_d_cst.to_numpy(), gap), 3),
                "partial_bcva_given_gap": round(partial_spearman(cos, -p.abs_d_bcva.to_numpy(), gap), 3),
                "cos_vs_minus_abs_d_cst": round(spearman(p.cos.to_numpy(), -p.abs_d_cst.to_numpy()), 3),
                "cos_vs_minus_abs_d_bcva": round(spearman(p.cos.to_numpy(), -p.abs_d_bcva.to_numpy()), 3),
                "cos_vs_minus_gap_positions": round(spearman(p.cos.to_numpy(), -p.gap_positions.to_numpy()), 3)}

    # Identical image content under different visits is a data issue (source or atlas); it is reported, not fixed.
    dup = m[m.sha256.duplicated(keep=False)].groupby("sha256").visit_id.apply(lambda v: sorted(set(v)))
    dup_pairs = [v for v in dup if len(v) > 1]

    summary = dict(
        identical_image_content_across_visits=dup_pairs,
        setting=("full visit history of each eye (memory identical to Phase 1)" if complete else
                 "subset-restricted history (not the full Phase 1 history)")
                + "; agreement only, no ground truth; not an evaluation",
        histories_complete=complete,
        encoder=str(d["encoder"]) if "encoder" in d else "unknown",
        n_eyes=int(vis.eye_id.nunique()), n_visits=len(vis), n_queries_total=len(per_q),
        parsed={k: {"operation": v.operation, "target": v.target, "task": v.task} for k, v in parsed.items()},
        top1_agreement_n_queries=len(multi),
        top1_agreement_chance=round(float(np.mean(1.0 / multi.n_candidates)), 3),
        top1_agreement_image_vs={k: round(float(multi[f"{k}_agree_top1"].astype(float).mean()), 3)
                                 for k in QUESTIONS if f"{k}_agree_top1" in multi},
        spearman_n_queries=len(full),
        mean_spearman_image_vs={k: round(float(full[f"{k}_spearman"].mean()), 3)
                                for k in QUESTIONS if f"{k}_spearman" in full},
        image_like_frozen_status=per_q.image_like_status.value_counts().to_dict(),
        hybrid_answered_by_image=int((per_q.hybrid_status == "image_retrieved").sum()),
        within_eye_pairs_all=pair_rho(pr),
        within_eye_pairs_by_trial={t: pair_rho(p) for t, p in pr.groupby("trial")},
        within_eye_pairs_by_eye={int(e): pair_rho(p) for e, p in pr.groupby("eye_id")},
        # 0 = the metadata top-1 is also the image top-1; 1 = image ranks it last; random ranking -> 0.5
        mean_norm_rank_of_metadata_top1_in_image_ranking={
            k: round(float(multi[f"{k}_top1_norm_rank_in_image"].mean()), 3)
            for k in QUESTIONS if f"{k}_top1_norm_rank_in_image" in multi},
        image_top1_is_previous_visit=round(float(multi.image_top1_is_previous.astype(float).mean()), 3),
        top1_agreement_by_eye={int(e): {k: round(float(g[f"{k}_agree_top1"].astype(float).mean()), 3)
                                        for k in QUESTIONS if f"{k}_agree_top1" in g}
                               | {"n_queries": len(g), "chance": round(float(np.mean(1.0 / g.n_candidates)), 3)}
                               for e, g in multi.groupby("eye_id")},
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    per_q.to_csv(out_dir / "image_vs_metadata_per_query.csv", index=False)
    pr.to_csv(out_dir / "visit_pairs.csv", index=False)
    (out_dir / "image_vs_metadata_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--subset", type=Path, default=ROOT / "phase2" / "data" / "atlas_subset")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    s = run(a.emb, a.subset, a.out or a.emb.parent)
    print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
