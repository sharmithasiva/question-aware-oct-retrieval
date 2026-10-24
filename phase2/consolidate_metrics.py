"""Consolidate Phase 2 metrics from SAVED outputs only (no encoder is run, nothing is re-fetched).

Inputs (read-only):
  phase2/data/{atlas_subset,atlas_fullhist}/manifest.csv
  phase2/results/{,fullhist/}<encoder>/{embeddings.npz,image_vs_metadata_per_query.csv,visit_pairs.csv,
                                         image_vs_metadata_summary.json,run_info.json}
  data/processed/episodes.csv                 (for the tie audit)
  results/fresh_v1/part_a_items.csv           (frozen Phase 1 routing outcomes; read only)
Outputs:
  phase2/results/performance/metrics.csv       one row per (experiment, encoder, metric) with numerator/denominator
  phase2/results/performance/per_eye.csv       per-eye values (dependence / spread)
  phase2/results/performance/tie_audit.csv     metadata top-1 ties for closest-CST/BCVA queries
  phase2/results/performance/consistency.json  recomputed values vs saved summaries

Usage: python phase2/consolidate_metrics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase2"))
import oct_image as OI  # noqa: E402
from image_vs_metadata import partial_spearman, spearman  # noqa: E402

ENCODERS = ("resnet50", "dinov2_vits14")
EXPERIMENTS = {
    "E1_96img_4visits": dict(subset=ROOT / "phase2/data/atlas_subset", res=ROOT / "phase2/results"),
    "E2_73img_fullhist": dict(subset=ROOT / "phase2/data/atlas_fullhist", res=ROOT / "phase2/results/fullhist"),
}
OUT = ROOT / "phase2/results/performance"
META_Q = ("closest_cst", "closest_bcva", "previous", "baseline")


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def row(exp, enc, metric, prop, unit, formula, num, den, value, baseline=None, baseline_how="", source="", note=""):
    return dict(experiment=exp, encoder=enc, metric=metric, measures=prop, unit=unit, formula=formula,
                numerator=num, denominator=den, value=None if value is None else round(float(value), 4),
                baseline=None if baseline is None else round(float(baseline), 4), baseline_how=baseline_how,
                source=source, note=note)


def neighbour_rows(exp, enc, emb_path, m):
    d = np.load(emb_path, allow_pickle=False)
    m = m.set_index("image_key").loc[list(d["image_key"])].reset_index()
    S = OI.cosine_matrix(d["embeddings"])
    n, k = len(m), 5
    eye, visit, trial = m.eye_id.to_numpy(), m.visit_id.to_numpy(), m.trial.to_numpy()
    se = st = sv = 0
    ce = cv = ct = 0.0
    per_eye = {}
    for q in range(n):
        nb = OI.topk(S, q, k)
        a = int(np.sum(eye[nb] == eye[q]))
        se += a
        st += int(np.sum(trial[nb] == trial[q]))
        sv += int(visit[nb[0]] == visit[q])
        ce += (np.sum(eye == eye[q]) - 1) / (n - 1)
        cv += (np.sum(visit == visit[q]) - 1) / (n - 1)
        ct += (np.sum(trial == trial[q]) - 1) / (n - 1)
        per_eye.setdefault(int(eye[q]), []).append(a / k)
    src = f"{rel(emb_path)} + manifest; oct_image.topk (self excluded, ties by index)"
    chance_txt = "expected value under uniformly random ranking: mean over queries of (#other items in group)/(N-1)"
    rows = [
        row(exp, enc, "top5_same_eye_rate", "identity matching (eye)", f"image query (N={n})",
            "sum_q #{top-5 neighbours with same eye_id} / (5*N)", se, k * n, se / (k * n), ce / n, chance_txt, src),
        row(exp, enc, "top5_same_trial_rate", "acquisition/trial similarity", f"image query (N={n})",
            "sum_q #{top-5 neighbours with same trial} / (5*N)", st, k * n, st / (k * n), ct / n, chance_txt, src),
    ]
    if m.groupby("visit_id").size().max() > 1:
        rows.append(row(exp, enc, "top1_same_visit_rate", "identity matching (visit)", f"image query (N={n})",
                        "#{queries whose top-1 neighbour is another scan of the same visit} / N", sv, n, sv / n,
                        cv / n, chance_txt, src))
    else:
        rows.append(row(exp, enc, "top1_same_visit_rate", "identity matching (visit)", f"image query (N={n})",
                        "n/a", None, None, None, source=src,
                        note="Not applicable: one scan per visit, so no other scan of the same visit exists"))
    pe = [dict(experiment=exp, encoder=enc, eye_id=e, metric="top5_same_eye_rate", n=len(v), value=round(np.mean(v), 4))
          for e, v in per_eye.items()]
    return rows, pe


def ranking_rows(exp, enc, pq_path, summ):
    pq = pd.read_csv(pq_path)
    multi = pq[pq.n_candidates >= 2]
    full = pq[pq.n_candidates >= 3]
    src = rel(pq_path)
    chance = float((1.0 / multi.n_candidates).sum())
    rows = []
    num = int(multi.image_top1_is_previous.sum())
    prev_note = ("'previous' = previous visit in the 4-visit subset (~8 real visits earlier), not the true previous visit"
                 if exp.startswith("E1") else "'previous' = true previous visit (full history)")
    rows.append(row(exp, enc, "image_top1_is_previous_visit_rate", "temporal similarity (recency)",
                    f"query visit with >=2 candidates (N={len(multi)})",
                    "#{queries whose image top-1 is visit t-1} / #{queries with t>=2}", num, len(multi),
                    num / len(multi), chance / len(multi), "random ranking: mean over queries of 1/(#candidates)", src,
                    prev_note))
    for q in META_Q:
        a = multi[f"{q}_agree_top1"].astype(bool)
        rows.append(row(exp, enc, f"top1_agreement_image_vs_{q}", "agreement between rankings (no ground truth)",
                        f"query visit with >=2 candidates (N={len(multi)})",
                        f"#{{image top-1 == frozen-metadata '{q}' top-1}} / #{{queries with t>=2}}", int(a.sum()),
                        len(multi), a.mean(), chance / len(multi),
                        "random ranking: mean over queries of 1/(#candidates)", src))
        rows.append(row(exp, enc, f"norm_rank_of_{q}_top1_in_image_ranking", "agreement between rankings (no ground truth)",
                        f"query visit with >=2 candidates (N={len(multi)})",
                        f"mean_q position of metadata '{q}' top-1 in image ranking / (#candidates-1); 0=best",
                        None, len(multi), multi[f"{q}_top1_norm_rank_in_image"].mean(), 0.5,
                        "random ranking: expected normalised position 0.5", src))
        rows.append(row(exp, enc, f"mean_rank_spearman_image_vs_{q}", "agreement between rankings (no ground truth)",
                        f"query visit with >=3 candidates (N={len(full)})",
                        "mean_q Spearman(rank positions of candidates under image vs metadata ranking)", None, len(full),
                        full[f"{q}_spearman"].mean(), 0.0, "random ranking: expected 0", src,
                        "NaN rows (constant ranks) excluded: " + str(int(full[f"{q}_spearman"].isna().sum()))))
    cov = int((pq.hybrid_status == "image_retrieved").sum())
    rows.append(row(exp, enc, "hybrid_image_route_coverage", "routing coverage (NOT accuracy, NOT retrieval)",
                    f"query visit (N={len(pq)}); ONE fixed question wording",
                    "#{queries where the frozen parser returns unsupported_image and the image ranking is used} / N",
                    cov, len(pq), cov / len(pq), source=src,
                    note="Same sentence repeated for every query; measures that the route fires, not how well "
                         "varied image questions are recognised"))
    return rows


def pair_rows(exp, enc, pairs_path):
    pr = pd.read_csv(pairs_path)
    src = rel(pairs_path)
    unit = f"within-eye visit pair (N={len(pr)} pairs, {pr.eye_id.nunique()} eyes; pairs not independent)"
    gapnote = ("gap in subset positions (0..3), not weeks or true visit count" if exp.startswith("E1")
               else "gap in visit positions (visit_index difference), not weeks")
    cos, gap = pr.cos.to_numpy(), -pr.gap_positions.to_numpy()
    rows = [
        row(exp, enc, "spearman_cos_vs_temporal_closeness_pooled", "temporal similarity", unit,
            "Spearman(cosine, -gap_positions) over all within-eye pairs pooled across eyes", None, len(pr),
            spearman(cos, gap), 0.0, "no association: 0", src, gapnote),
        row(exp, enc, "partial_spearman_cos_vs_cst_closeness_given_gap_pooled", "association with a clinical variable",
            unit, "rank partial corr of cosine and -|dCST| controlling for -gap (first-order, on ranks), pooled",
            None, len(pr), partial_spearman(cos, -pr.abs_d_cst.to_numpy(), gap), 0.0, "no association: 0", src,
            "Not clinical relevance; CST is one scalar, measured in the central subfield"),
        row(exp, enc, "partial_spearman_cos_vs_bcva_closeness_given_gap_pooled", "association with a clinical variable",
            unit, "rank partial corr of cosine and -|dBCVA| controlling for -gap, pooled", None, len(pr),
            partial_spearman(cos, -pr.abs_d_bcva.to_numpy(), gap), 0.0, "no association: 0", src),
    ]
    pe = []
    for e, g in pr.groupby("eye_id"):
        c, gg = g.cos.to_numpy(), -g.gap_positions.to_numpy()
        pe += [dict(experiment=exp, encoder=enc, eye_id=int(e), metric="spearman_cos_vs_temporal_closeness",
                    n=len(g), value=round(spearman(c, gg), 4)),
               dict(experiment=exp, encoder=enc, eye_id=int(e), metric="partial_spearman_cos_vs_cst_given_gap",
                    n=len(g), value=round(partial_spearman(c, -g.abs_d_cst.to_numpy(), gg), 4))]
    return rows, pe


def tie_audit(exp, enc, pq_path, episodes):
    """For closest-CST/BCVA queries, is the metadata top-1 uniquely best, or tied (resolved by recency)?"""
    pq = pd.read_csv(pq_path)
    multi = pq[pq.n_candidates >= 2]
    out = []
    for r in multi.itertuples():
        hist = episodes[episodes.eye_id == r.eye_id].sort_values("visit_index")
        if exp.startswith("E1"):
            hist = hist[hist.visit_id.isin(pd.read_csv(pq_path.parent / "visit_pairs.csv")
                                           .query("eye_id == @r.eye_id")[["visit_a", "visit_b"]].stack())]
        hist = hist.reset_index(drop=True)
        t = int(hist.index[hist.visit_id == r.query_visit][0])
        for var, q in (("cst", "closest_cst"), ("bcva", "closest_bcva")):
            dist = np.abs(hist[var].to_numpy()[:t] - hist[var].to_numpy()[t])
            best = set(hist.visit_id[:t][dist == np.nanmin(dist)])
            out.append(dict(experiment=exp, encoder=enc, query_visit=r.query_visit, question=q, n_tied_best=len(best),
                            meta_top1=getattr(r, f"{q}_top1"), image_top1=r.image_top1,
                            strict_agree=getattr(r, f"{q}_top1") == r.image_top1,
                            tie_aware_agree=r.image_top1 in best))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = pd.read_csv(ROOT / "data/processed/episodes.csv")
    rows, per_eye, ties, consistency = [], [], [], {}
    for exp, cfg in EXPERIMENTS.items():
        m = OI.load_manifest(cfg["subset"])
        for enc in ENCODERS:
            res = cfg["res"] / enc
            summ = json.loads((res / "image_vs_metadata_summary.json").read_text())
            r1, p1 = neighbour_rows(exp, enc, res / "embeddings.npz", m)
            r2 = ranking_rows(exp, enc, res / "image_vs_metadata_per_query.csv", summ)
            r3, p3 = pair_rows(exp, enc, res / "visit_pairs.csv")
            rows += r1 + r2 + r3
            per_eye += p1 + p3
            ties += tie_audit(exp, enc, res / "image_vs_metadata_per_query.csv", episodes)
            # consistency: recomputed vs values saved by the earlier runs
            got = {x["metric"]: x["value"] for x in r1 + r2 + r3}
            chk = {f"top1_agreement_image_vs_{q}": summ["top1_agreement_image_vs"][q] for q in META_Q}
            chk["spearman_cos_vs_temporal_closeness_pooled"] = summ["within_eye_pairs_all"]["cos_vs_minus_gap_positions"]
            chk["partial_spearman_cos_vs_cst_closeness_given_gap_pooled"] = summ["within_eye_pairs_all"]["partial_cst_given_gap"]
            ri = res / "run_info.json"
            info = json.loads(ri.read_text()) if ri.exists() else {}
            if "neighbour_stats" in info:
                chk["top5_same_eye_rate"] = info["neighbour_stats"]["same_eye_at_k"]
                chk["top1_same_visit_rate"] = info["neighbour_stats"]["same_visit_at_1"]
                chk["top5_same_trial_rate"] = info["neighbour_stats"]["same_trial_at_k"]
            consistency[f"{exp}/{enc}"] = {
                # saved summaries hold 3-dp values; allow the rounding half-step
                k: {"saved": v, "recomputed": got[k], "match": abs(v - got[k]) <= 5e-4 + 1e-9}
                for k, v in chk.items()}
    # Sensitivity: E2 without the duplicated-content visit 01-013_W12 as a query or in any pair.
    # (W12 stays a candidate for later queries; its cosine ties W8 exactly and the stable sort picks W8.)
    dup = "01-013_W12"
    for enc in ENCODERS:
        res = EXPERIMENTS["E2_73img_fullhist"]["res"] / enc
        pq = pd.read_csv(res / "image_vs_metadata_per_query.csv")
        multi = pq[(pq.n_candidates >= 2) & (pq.query_visit != dup)]
        pr = pd.read_csv(res / "visit_pairs.csv")
        pr = pr[(pr.visit_a != dup) & (pr.visit_b != dup)]
        exp = "E2_sensitivity_excl_01-013_W12"
        src = f"{rel(res)}/image_vs_metadata_per_query.csv, visit_pairs.csv"
        num = int(multi.image_top1_is_previous.sum())
        gap = -pr.gap_positions.to_numpy()
        rows += [
            row(exp, enc, "image_top1_is_previous_visit_rate", "temporal similarity (recency)",
                f"query visit, t>=2, W12 query dropped (N={len(multi)})", "as E2", num, len(multi), num / len(multi),
                float((1.0 / multi.n_candidates).mean()), "random ranking: mean 1/(#candidates)", src),
            row(exp, enc, "spearman_cos_vs_temporal_closeness_pooled", "temporal similarity",
                f"within-eye pair, pairs with W12 dropped (N={len(pr)})", "as E2", None, len(pr),
                spearman(pr.cos.to_numpy(), gap), 0.0, "no association: 0", src),
            row(exp, enc, "partial_spearman_cos_vs_cst_closeness_given_gap_pooled", "association with a clinical variable",
                f"within-eye pair, pairs with W12 dropped (N={len(pr)})", "as E2", None, len(pr),
                partial_spearman(pr.cos.to_numpy(), -pr.abs_d_cst.to_numpy(), gap), 0.0, "no association: 0", src),
        ]
        d = np.load(res / "embeddings.npz", allow_pickle=False)
        idx = pd.read_csv(res / "embedding_index.csv")
        consistency[f"E2_73img_fullhist/{enc}"]["embedding_order_matches_index_csv"] = {
            "saved": True, "recomputed": list(d["image_key"]) == list(idx.image_key),
            "match": list(d["image_key"]) == list(idx.image_key)}
    # Frozen Phase 1 routing on image questions (fresh_v1 Part A; read only)
    a = pd.read_csv(ROOT / "results/fresh_v1/part_a_items.csv")
    img = a[a.gold_routing == "requires_images"]
    non = a[a.gold_routing != "requires_images"]
    src = "results/fresh_v1/part_a_items.csv (frozen; reports/FRESH_V1_REPORT.md)"
    unit = "fresh_v1 question (AI-generated benchmark, limited human review)"
    tp = int((img.system_task == "unsupported_image").sum())
    fp = int((non.system_task == "unsupported_image").sum())
    rows += [
        row("Phase1_fresh_v1", "n/a (parser)", "image_question_routing_recall_exact", "routing accuracy", unit,
            "#{gold requires_images routed to unsupported_image} / #{gold requires_images}", tp, len(img), tp / len(img),
            source=src),
        row("Phase1_fresh_v1", "n/a (parser)", "image_question_any_refusal", "routing safety", unit,
            "#{gold requires_images not retrieved} / #{gold requires_images}", int((~img.system_retrieves.astype(bool)).sum()),
            len(img), (~img.system_retrieves.astype(bool)).mean(), source=src),
        row("Phase1_fresh_v1", "n/a (parser)", "image_route_precision", "routing accuracy", unit,
            "#{routed unsupported_image with gold requires_images} / #{routed unsupported_image}", tp, tp + fp,
            tp / (tp + fp), source=src, note=f"false image routes: {fp} of {len(non)} non-image items"),
    ]
    # Standard retrieval metrics: no relevance labels exist for image questions
    for exp in EXPERIMENTS:
        for name in ("Hit@1", "Hit@5/Recall@5", "MRR", "nDCG@5", "QA retrieval accuracy"):
            rows.append(row(exp, "both", name, "retrieval accuracy", "-", "-", None, None, None,
                            note="Not evaluated - relevance ground truth unavailable"))
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(per_eye).to_csv(OUT / "per_eye.csv", index=False)
    td = pd.DataFrame(ties)
    td.to_csv(OUT / "tie_audit.csv", index=False)
    (OUT / "consistency.json").write_text(json.dumps(consistency, indent=2))
    bad = [(k, m) for k, v in consistency.items() for m, x in v.items() if not x["match"]]
    print(f"metrics: {len(rows)} rows | consistency mismatches: {bad or 'none'}")
    print(td.groupby(["experiment", "encoder", "question"]).agg(
        n=("n_tied_best", "size"), tied=("n_tied_best", lambda s: int((s > 1).sum())),
        strict=("strict_agree", "sum"), tie_aware=("tie_aware_agree", "sum")).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
