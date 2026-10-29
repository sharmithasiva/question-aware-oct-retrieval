"""Evaluate image, metadata and hybrid retrieval on a (larger) verified subset from SAVED embeddings. No encoder runs.

Every metric reuses existing definitions; nothing here changes them:
  E1-def  scan-level identity sanity: top-5 same-eye rate, top-1 same-visit rate (oct_image.neighbour_stats).
          Identity, NOT relevance.
  E2-def  within-eye temporal sanity on visit embeddings, eyes with a complete history in the subset only:
          image top-1 == previous visit; Spearman(cosine, temporal closeness); partial Spearman(cosine, CST
          closeness | gap) (image_vs_metadata.spearman / partial_spearman). Reported pooled AND as a mean over
          eyes (pairs of one eye are not independent), with an eye-level bootstrap CI when >= 5 eyes.
  Tier A  CST questions, relevance from the FROZEN Phase 1 answer keys (question_eval.tier_a_labels); systems
          S1 latest_prior, S2 metadata (frozen), S3S4 image, S5 prototype RRF (question_eval) and
          H_fuse_closest = hybrid_retriever with RRF only for closest_to_current (all other operations and all
          abstentions stay metadata). Metadata relevance labels favour the metadata system by construction.
  B1      (--b1, dev eyes only) the completed non-expert B1 visual-similarity judgments (evaluate_annotations.py
          maths), for systems built from the given embeddings. These rankings are NOT the frozen pre-label
          rankings: every B1 number here is POST-HOC and exploratory (3 eyes, 18 queries, one annotator).

Queries are never pooled across categories. Byte-identical images (`duplicate_of`) are dropped from E1-def; Tier A/B1
use question_eval's duplicate-query exclusion. Outputs go to a NEW folder (never overwritten).

Usage:
  python phase2/large_eval.py --subset phase2/data/scan_subset_v1 \
      --emb resnet50=phase2/results/scan_subset_v1/resnet50/embeddings.npz \
      --emb dinov2_vits14=phase2/results/scan_subset_v1/dinov2_vits14/embeddings.npz \
      --visit-mode central --b1 --out phase2/results/scan_subset_v1/eval_central
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "phase2"))
import evaluate_annotations as EA  # noqa: E402
import hybrid_retriever as H  # noqa: E402
import oct_image as OI  # noqa: E402
import question_eval as QE  # noqa: E402
from image_vs_metadata import partial_spearman, spearman  # noqa: E402
from oct_retrieval import intent as I  # noqa: E402

EPISODES = ROOT / "data/processed/episodes.csv"
B1_SHEET = ROOT / "phase2/annotations/v2/dev/B1/completed_1_dev-B1-v2.csv"
B1_KEY = ROOT / "phase2/annotations/keys/v2/dev_B1_key.csv"
B1_RANKINGS = ROOT / "phase2/results/question_eval_v2/dev/rankings_tierB_24a90d2768c6"


def eye_ci(per_eye: pd.Series):
    ci = QE.bootstrap_ci(per_eye.dropna())
    return None if ci is None else [round(ci[0], 4), round(ci[1], 4)]


def not_duplicate(m: pd.DataFrame) -> np.ndarray:
    """False for every image whose bytes (SHA-256) already appeared earlier in manifest order."""
    return (~m.sha256.duplicated()).to_numpy()


# --- E1 definition: scan-level identity ---------------------------------------------------------------------

def scan_level(E: np.ndarray, m: pd.DataFrame) -> dict:
    keep = not_duplicate(m)
    E, m = E[keep], m[keep].reset_index(drop=True)
    S = OI.cosine_matrix(E)
    ns = OI.neighbour_stats(S, m, k=5)
    multi = m.groupby("visit_id").size().max() > 1
    eye = m.eye_id.to_numpy()
    per_eye = pd.Series([np.mean(eye[OI.topk(S, q, 5)] == eye[q]) for q in range(len(m))]).groupby(eye).mean()
    return dict(n_scans=len(m), n_eyes=int(m.eye_id.nunique()), dropped_duplicates=int((~keep).sum()),
                top5_same_eye_rate=round(ns.same_eye_at_k, 4), top5_same_eye_chance=round(ns.same_eye_chance, 4),
                top5_same_eye_ci95_eye_bootstrap=eye_ci(per_eye),
                top1_same_visit_rate=round(ns.same_visit_at_1, 4) if multi else "n/a (one scan per visit)",
                top1_same_visit_chance=round(ns.same_visit_chance_at_1, 4) if multi else "n/a",
                measures="identity (same eye / same visit), not relevance")


# --- E2 definition: within-eye temporal structure -------------------------------------------------------------

def complete_eyes(visits: set[str], episodes: pd.DataFrame) -> list[int]:
    g = episodes.groupby("eye_id").visit_id.apply(set)
    return sorted(int(e) for e, vs in g.items() if vs <= visits and len(vs) >= 3)


def within_eye(vemb: dict[str, np.ndarray], episodes: pd.DataFrame, eyes: list[int]) -> tuple[dict, pd.DataFrame]:
    rows, pairs = [], []
    for e in eyes:
        h = episodes[episodes.eye_id == e].sort_values("visit_index").reset_index(drop=True)
        Z = np.stack([vemb[v] for v in h.visit_id])
        sim = Z @ Z.T
        for i in range(len(h)):
            for j in range(i):
                pairs.append(dict(eye_id=e, cos=sim[i, j], gap=i - j, abs_d_cst=abs(h.cst[i] - h.cst[j])))
        for t in range(2, len(h)):
            top = int(np.argsort(-sim[t, :t], kind="stable")[0])
            rows.append(dict(eye_id=e, t=t, top1_is_previous=float(top == t - 1), chance=1.0 / t))
    q, p = pd.DataFrame(rows), pd.DataFrame(pairs)

    def rho(g):
        return pd.Series(dict(temporal=spearman(g.cos.to_numpy(), -g.gap.to_numpy()),
                              partial_cst=partial_spearman(g.cos.to_numpy(), -g.abs_d_cst.to_numpy(),
                                                           -g.gap.to_numpy())))
    pe = p.groupby("eye_id").apply(rho, include_groups=False)
    prev_e = q.groupby("eye_id").top1_is_previous.mean()
    pooled = rho(p)
    out = dict(n_eyes=len(eyes), n_queries=len(q), n_pairs=len(p),
               image_top1_is_previous_rate=round(float(q.top1_is_previous.mean()), 4),
               image_top1_is_previous_chance=round(float(q.chance.mean()), 4),
               image_top1_is_previous_ci95_eye_bootstrap=eye_ci(prev_e),
               spearman_cos_vs_temporal_closeness_pooled=round(float(pooled.temporal), 4),
               spearman_cos_vs_temporal_closeness_mean_over_eyes=round(float(pe.temporal.mean()), 4),
               spearman_temporal_ci95_eye_bootstrap=eye_ci(pe.temporal),
               partial_spearman_cos_vs_cst_given_gap_pooled=round(float(pooled.partial_cst), 4),
               partial_spearman_cos_vs_cst_given_gap_mean_over_eyes=round(float(pe.partial_cst.mean()), 4),
               partial_cst_ci95_eye_bootstrap=eye_ci(pe.partial_cst),
               measures="temporal / CST association within an eye (no relevance labels)")
    return out, pe.reset_index()


# --- Tier A and hybrid systems ----------------------------------------------------------------------------------

class SystemsWithHybrid(QE.Systems):
    """question_eval systems (unchanged) plus the isolated hybrid retriever, on injected embeddings."""

    def __init__(self, episodes, embeddings):
        super().__init__(episodes, "dev", embeddings=embeddings)
        self.hyb = {enc: H.HybridRetriever(episodes, e, fuse_ops=frozenset({I.CLOSEST_TO_CURRENT}))
                    for enc, e in embeddings.items()}

    def rank_all(self, q):
        out = super().rank_all(q)
        for enc, h in self.hyb.items():
            out[f"H_fuse_closest_{enc}"] = h.retrieve(q.question, q.eye_id, q.query_visit_id).final_ranking
        return out


def tier_a(episodes, embeddings, eyes, hashes) -> tuple[pd.DataFrame, pd.DataFrame]:
    name = "large_eval"
    QE.SETS[name] = dict(eyes=tuple(eyes), subset=None, emb=None)   # in-memory registration only
    qs = QE.build_query_set(episodes, name, "A")
    QE.assert_no_candidate_leakage(qs, episodes)
    labels = QE.tier_a_labels(qs, episodes)
    per, _ = QE.evaluate(qs, labels, SystemsWithHybrid(episodes, embeddings), hashes, 1.0)
    return QE.summarise(per, "A"), per


# --- B1 (post-hoc) ----------------------------------------------------------------------------------------------

def b1_posthoc(episodes, embeddings) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if not (B1_KEY.exists() and B1_SHEET.exists()):
        return None
    frozen, systems, _ = EA.load_rankings(B1_RANKINGS)
    qs = pd.read_csv(B1_RANKINGS / "query_set.csv")
    qs = qs[qs.category == "overall_appearance"]
    need = set(qs.query_visit_id) | {v for c in qs.candidate_visit_ids for v in c.split(";")}
    rows = []
    for enc, e in embeddings.items():
        if not need <= set(e):
            return None
        variants = {f"S3S4_image_{enc}": H.HybridRetriever(episodes, e, image_proxy=None),
                    f"S5_hybrid_rrf_{enc}": H.HybridRetriever(episodes, e)}
        for s, h in variants.items():
            for q in qs.itertuples():
                for i, v in enumerate(h.retrieve(q.question, q.eye_id, q.query_visit_id).final_ranking):
                    rows.append(dict(query_id=q.query_id, query_visit_id=q.query_visit_id, category=q.category,
                                     eye_id=q.eye_id, system=f"posthoc_{s}", rank=i + 1, candidate_visit_id=v))
    ref = frozen[(frozen.category == "overall_appearance") &
                 frozen.system.isin(["S1_latest_prior", "S2p_metadata_proxy_cst"])]
    rk = pd.concat([ref, pd.DataFrame(rows)], ignore_index=True)
    judg = EA.join_sheet_key(EA.load_sheet(B1_SHEET), pd.read_csv(B1_KEY))
    EA.check_pools(judg, rk)
    sysdf = pd.DataFrame(dict(system=sorted(rk.system.unique())))
    per = EA.evaluate(judg, rk, sysdf, "primary", 2.0)
    return EA.summarise(per)[0], per


# --- driver -------------------------------------------------------------------------------------------------------

def run(subset: Path, emb_paths: dict[str, Path], visit_mode: str, out: Path, do_b1: bool) -> dict:
    episodes = pd.read_csv(EPISODES)
    m_all = OI.load_manifest(subset)
    res: dict = dict(subset=str(subset), visit_mode=visit_mode, embeddings={k: str(v) for k, v in emb_paths.items()})
    vembs, tables = {}, {}
    for enc, p in emb_paths.items():
        d = np.load(p, allow_pickle=False)
        m = m_all.set_index("image_key").loc[list(d["image_key"])].reset_index()
        res.setdefault("scan_level", {})[enc] = scan_level(d["embeddings"], m)
        vembs[enc] = H.load_visit_embeddings(subset, p, mode=visit_mode, exclude_duplicates=visit_mode == "mean")
        # (central mode keeps duplicate central scans: they are real visits; Tier A/B1 exclude duplicate queries)
    eyes = complete_eyes(set.intersection(*(set(v) for v in vembs.values())), episodes)
    res["complete_history_eyes"] = eyes
    if not eyes:
        res["within_eye"] = res["tier_a"] = "NOT EVALUATED: no eye has its complete history in the subset"
    else:
        for enc, v in vembs.items():
            res.setdefault("within_eye", {})[enc], tables[f"within_eye_per_eye_{enc}"] = within_eye(v, episodes, eyes)
        c = m_all[m_all.scan_index == 24]
        hashes = dict(zip(c.visit_id, c.sha256))
        summ, per = tier_a(episodes, vembs, eyes, hashes)
        tables["tierA_summary"], tables["tierA_per_query"] = summ, per
    if do_b1:
        b1 = b1_posthoc(episodes, vembs)
        if b1 is None:
            res["b1_posthoc"] = "NOT EVALUATED: B1 key/sheet missing or dev visits lack embeddings"
        else:
            tables["b1_posthoc_summary"], tables["b1_posthoc_per_query"] = b1
    out.mkdir(parents=True)
    for k, t in tables.items():
        t.to_csv(out / f"{k}.csv", index=False)
    (out / "summary.json").write_text(json.dumps(res, indent=2, default=str))
    (out / "report.md").write_text(report(res, tables), encoding="utf-8")
    return res


def md(df: pd.DataFrame, cols) -> str:
    cols = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def report(res: dict, tables: dict) -> str:
    parts = [f"# Retrieval evaluation: `{res['subset']}` (visit embedding: {res['visit_mode']})", "",
             "Exploratory. Image relevance here is either identity/temporal structure (no labels), frozen metadata "
             "answer keys (Tier A) or one non-expert's visual-similarity judgments (B1, post-hoc). None is clinical "
             "ground truth.", "", "## Inputs", "", "```json", json.dumps(res["embeddings"], indent=2), "```", "",
             "## E1 definition: scan-level identity (not relevance)", "", "```json",
             json.dumps(res["scan_level"], indent=2), "```", "",
             f"## E2 definition: within-eye structure ({len(res['complete_history_eyes'])} complete-history eyes)", "",
             "```json", json.dumps(res.get("within_eye"), indent=2), "```", ""]
    cols = ["category", "system", "n_evaluated", "coverage", "hit1", "hit5", "recall5", "mrr", "ndcg5", "n_eyes",
            "mrr_ci95_eye_bootstrap", "mrr_minus_latest_prior", "eyes_better_equal_worse_vs_latest_prior",
            "mrr_minus_S2_metadata"]
    if "tierA_summary" in tables:
        parts += ["## Tier A: CST questions, frozen metadata answer keys (primary metric MRR)", "",
                  "`S2_metadata` is the frozen Phase 1 system. `S5_hybrid_rrf_*` is the v0.1 prototype "
                  "(fuses every metadata ranking and falls back to images when metadata abstains). "
                  "`H_fuse_closest_*` is phase2/hybrid_retriever.py: fuses only `closest_to_current`, never "
                  "overrides abstentions.", "", md(tables["tierA_summary"], cols), ""]
    if "b1_posthoc_summary" in tables:
        parts += ["## B1 visual similarity (dev, POST-HOC systems; relevant = grade 2; primary nDCG@5)", "",
                  "Single non-expert annotator (`researcher_non_expert`), 3 eyes, 18 queries. The pre-registered "
                  "result is the frozen-rankings evaluation (`question_eval_v2/dev/eval_B1_cda4ea5b_*`).", "",
                  md(tables["b1_posthoc_summary"], ["system", "n_evaluated", "hit1", "hit5", "recall5", "mrr", "ndcg5",
                                                    "ndcg5_minus_S1_latest_prior",
                                                    "eyes_better_equal_worse_vs_S1_latest_prior"]), ""]
    elif "b1_posthoc" in res:
        parts += ["## B1", "", res["b1_posthoc"], ""]
    return "\n".join(parts) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--emb", action="append", required=True, help="name=path/to/embeddings.npz (repeatable)")
    ap.add_argument("--visit-mode", choices=["central", "mean"], default="central")
    ap.add_argument("--b1", action="store_true", help="also score the dev B1 judgments (post-hoc)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    emb = dict(x.split("=", 1) for x in a.emb)
    res = run(a.subset, {k: Path(v) for k, v in emb.items()}, a.visit_mode, a.out, a.b1)
    print((a.out / "report.md").read_text(encoding="utf-8"))
    return 0 if res else 1


if __name__ == "__main__":
    sys.exit(main())
