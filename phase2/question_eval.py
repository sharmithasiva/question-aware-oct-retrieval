"""Question-conditioned retrieval evaluation harness (protocol: phase2/QUESTION_EVAL_PROTOCOL.md).

Compares, on one shared query set and candidate pool per question:
  S1 latest_prior | S2 metadata (frozen Phase 1 parser + retriever) | S2p metadata proxy (Tier B only)
  S3 image ResNet-50 | S4 image DINOv2 | S5 hybrid RRF(k=60) of metadata and image ranks (per encoder)

Labels:
  Tier A  metadata_rule labels from FROZEN Phase 1 answer keys (questions.relevant_indices, ground_truth_v2.answer_key)
  Tier B  human grades from an annotation sheet + key (phase2/annotations/); blank grade = missing, never 0

Nothing here modifies Phase 1 code or data; Phase 1 modules are imported read-only. Saved embeddings are reused;
no encoder is run. Missing labels are reported as NOT EVALUATED, never as zero.

Usage:
  python phase2/question_eval.py --set dev --tier A
  python phase2/question_eval.py --set dev --tier B --sheet phase2/annotations/tierB_dev_sheet.csv \
      --key phase2/annotations/keys/tierB_dev_key.csv
  add --drop-visits 01-013_W12 for the duplicate-image sensitivity analysis
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "phase2"))

from oct_retrieval import ground_truth_v2 as GT2      # noqa: E402  (frozen, read-only)
from oct_retrieval import intent as I                 # noqa: E402
from oct_retrieval import intent_retrieval as R       # noqa: E402
from oct_retrieval import questions as Q1             # noqa: E402
import oct_image as OI                                # noqa: E402

MIN_CANDIDATES = 6
TIER_B_QUERIES_PER_EYE = 6
RRF_K = 60
BOOTSTRAP = dict(n=10_000, seed=20261024, min_eyes=5)
ENCODERS = ("resnet50", "dinov2_vits14")

SETS = {
    "dev": dict(eyes=(25, 11, 61), subset=ROOT / "phase2/data/atlas_fullhist",
                emb=ROOT / "phase2/results/fullhist/{enc}/embeddings.npz"),
    "heldout": dict(eyes=(93, 74, 40, 14), subset=ROOT / "phase2/data/atlas_heldout",
                    emb=ROOT / "phase2/results/heldout/{enc}/embeddings.npz"),
}
TIER_A = {   # category -> (question text, answer-key source)
    "similar_cst": ("When was the CST most similar to today's measurement?", "v1"),
    "worst_edema": ("Compared with when the edema was worst, has it improved?", "v1"),
    "best_anatomy": ("Compared with the visit with the lowest CST, has thickness increased?", "v1"),
    "last_thicker_than_now": ("Most recent prior visit with CST above today's value?", "v2"),
    "largest_cst_rise": ("Which prior visit showed the biggest jump in central subfield thickness compared with "
                         "the visit right before it?", "v2"),
}
TIER_B = {
    "overall_appearance": "Which earlier visit's OCT looks most like today's?",
    "fluid_similarity": "At which earlier visit was the retinal fluid/thickening most similar to today?",
}
PRIMARY = {"A": "mrr", "B": "ndcg5"}
METRICS = ("hit1", "hit5", "recall5", "mrr", "ndcg5")


# --- query set -----------------------------------------------------------------------------------------------

def eye_history(episodes: pd.DataFrame, eye: int, drop_visits=()) -> pd.DataFrame:
    h = episodes[(episodes.eye_id == eye) & ~episodes.visit_id.isin(drop_visits)]
    return h.sort_values("visit_index").reset_index(drop=True)


def tier_b_positions(n_visits: int) -> list[int]:
    if n_visits - 1 < MIN_CANDIDATES:
        return []
    return sorted(set(int(round(x)) for x in np.linspace(MIN_CANDIDATES, n_visits - 1, TIER_B_QUERIES_PER_EYE)))


def build_query_set(episodes: pd.DataFrame, set_name: str, tier: str, drop_visits=()) -> pd.DataFrame:
    rows = []
    cats = TIER_A if tier == "A" else TIER_B
    for eye in SETS[set_name]["eyes"]:
        h = eye_history(episodes, eye, drop_visits)
        positions = range(MIN_CANDIDATES, len(h)) if tier == "A" else tier_b_positions(len(h))
        for t in positions:
            cur = h.iloc[t]
            for cat in cats:
                text = TIER_A[cat][0] if tier == "A" else TIER_B[cat]
                rows.append(dict(
                    query_id=f"{set_name}-{tier}-{cat}-{cur.visit_id}", set=set_name, tier=tier, category=cat,
                    question=text, eye_id=int(eye), patient_id=int(cur.patient_id), trial=cur.trial, split=cur.split,
                    query_visit_id=cur.visit_id, t=t, n_candidates=t,
                    candidate_visit_ids=";".join(h.visit_id.iloc[:t])))
    return pd.DataFrame(rows)


def assert_no_candidate_leakage(qs: pd.DataFrame, episodes: pd.DataFrame) -> None:
    """Candidates must be exactly the strictly earlier visits of the query's own eye."""
    vi = episodes.set_index("visit_id")
    for r in qs.itertuples():
        cands = r.candidate_visit_ids.split(";")
        assert r.query_visit_id not in cands, f"{r.query_id}: query visit among candidates"
        assert (vi.loc[cands, "eye_id"] == r.eye_id).all(), f"{r.query_id}: candidate from another eye"
        assert (vi.loc[cands, "visit_index"] < vi.loc[r.query_visit_id, "visit_index"]).all(), \
            f"{r.query_id}: non-earlier candidate"
        assert len(set(cands)) == len(cands) == r.n_candidates, f"{r.query_id}: duplicate/miscounted candidates"


# --- labels --------------------------------------------------------------------------------------------------

def tier_a_labels(qs: pd.DataFrame, episodes: pd.DataFrame, drop_visits=()) -> pd.DataFrame:
    """Long table (query_id, candidate_visit_id, grade, label_source) from the frozen answer keys.
    A query with INELIGIBLE gets status 'ineligible'; an empty relevant set gets all-zero grades."""
    out = []
    for r in qs.itertuples():
        h = eye_history(episodes, r.eye_id, drop_visits)
        hist, cur = h.iloc[:r.t], h.iloc[r.t]
        src = TIER_A[r.category][1]
        if src == "v1":
            spec = Q1.QUESTION_TYPES[r.category][0]
            rel = Q1.relevant_indices(spec, hist[spec.target].to_numpy(float), float(cur[spec.target]))
            ineligible = spec.intent == "similar" and np.isnan(cur[spec.target])
            fn = "questions.relevant_indices"
        else:
            rel = GT2.answer_key(r.category, hist, cur)
            ineligible = rel is GT2.INELIGIBLE
            rel = rel or []
            fn = "ground_truth_v2.answer_key"
        for i, vid in enumerate(hist.visit_id):
            out.append(dict(query_id=r.query_id, candidate_visit_id=vid,
                            grade=np.nan if ineligible else float(i in rel),
                            ineligible=ineligible, label_source=f"metadata_rule:{fn}"))
    return pd.DataFrame(out)


def human_labels(sheet: Path, key: Path) -> pd.DataFrame:
    """Merge an annotator sheet with its blinding key. Blank grade -> NaN (missing), never 0."""
    s = pd.read_csv(sheet, dtype=str, keep_default_na=False)
    k = pd.read_csv(key, dtype=str, keep_default_na=False)
    m = s.merge(k[["query_visit_id", "candidate_code", "candidate_visit_id"]],
                on=["query_visit_id", "candidate_code"], how="left", validate="many_to_one")
    if m.candidate_visit_id.eq("").any() or m.candidate_visit_id.isna().any():
        raise ValueError("sheet rows without a key entry")
    g = pd.to_numeric(m.grade.replace("", np.nan), errors="coerce")
    bad = g.notna() & ~g.isin([0, 1, 2])
    if bad.any():
        raise ValueError(f"grades must be 0, 1 or 2; got {sorted(set(m.grade[bad]))}")
    ungradable = m.ungradable.str.lower().isin(["1", "true", "yes", "y"])
    unanswerable = m.query_answerable.str.lower().isin(["0", "false", "no", "n"])
    return pd.DataFrame(dict(query_id=m.query_id, candidate_visit_id=m.candidate_visit_id,
                             grade=np.where(ungradable, 0.0, g), ungradable=ungradable,
                             unanswerable=unanswerable, label_source="human"))


# --- systems -------------------------------------------------------------------------------------------------

def recency_order(scores: dict[str, float], position: dict[str, int]) -> list[str]:
    """Descending score; exact ties -> more recent visit first (protocol §2)."""
    return sorted(scores, key=lambda v: (-scores[v], -position[v]))


class Systems:
    def __init__(self, episodes: pd.DataFrame, set_name: str, drop_visits=(), embeddings: dict | None = None):
        self.episodes = episodes
        self.drop = tuple(drop_visits)
        self.mem = {}
        self.emb = embeddings if embeddings is not None else load_visit_embeddings(set_name)

    def memory(self, eye: int) -> R.VisitMemory:
        if eye not in self.mem:
            self.mem[eye] = R.build_memories(eye_history(self.episodes, eye, self.drop))[eye]
        return self.mem[eye]

    def rank_all(self, q) -> dict[str, list[str] | None]:
        cands = q.candidate_visit_ids.split(";")
        pos = {v: i for i, v in enumerate(cands)}
        mem = self.memory(q.eye_id)
        assert mem.visit_ids[:q.t] == cands and mem.visit_ids[q.t] == q.query_visit_id
        out = {"S1_latest_prior": cands[::-1]}

        def meta(intent) -> list[str] | None:
            res = R.retrieve(intent, mem, q.t)
            return None if res.status != R.RETRIEVED else [mem.visit_ids[i] for i in res.ranked]

        parsed = I.parse_intent(q.question)
        out["S2_metadata"] = meta(parsed)
        meta_for_hybrid = out["S2_metadata"]
        if q.tier == "B":
            out["S2p_metadata_proxy_cst"] = meta(I.QueryIntent(task=I.RETRIEVE, operation=I.CLOSEST_TO_CURRENT,
                                                                target="cst"))
            meta_for_hybrid = out["S2p_metadata_proxy_cst"]
        for enc in ENCODERS:
            e = self.emb.get(enc)
            if e is None or q.query_visit_id not in e:
                out[f"S3S4_image_{enc}"] = None
                out[f"S5_hybrid_rrf_{enc}"] = None
                continue
            have = [v for v in cands if v in e]           # visits without an image are not image candidates
            sc = {v: float(e[v] @ e[q.query_visit_id]) for v in have}
            img = recency_order(sc, pos)
            out[f"S3S4_image_{enc}"] = img
            if meta_for_hybrid is None:
                out[f"S5_hybrid_rrf_{enc}"] = img
            else:
                rm = {v: i + 1 for i, v in enumerate(meta_for_hybrid)}
                ri = {v: i + 1 for i, v in enumerate(img)}
                fused = {v: 1 / (RRF_K + rm[v]) + (1 / (RRF_K + ri[v]) if v in ri else 0.0) for v in cands}
                out[f"S5_hybrid_rrf_{enc}"] = recency_order(fused, pos)
        out["_parsed"] = f"{parsed.task}/{parsed.operation}/{parsed.target}"
        return out


def load_visit_embeddings(set_name: str) -> dict[str, dict[str, np.ndarray]]:
    """visit_id -> L2-normalised central-scan (index 24) embedding, per encoder, from SAVED embeddings."""
    cfg = SETS[set_name]
    if not (cfg["subset"] / "manifest.csv").exists():
        return {}
    m = OI.load_manifest(cfg["subset"]).set_index("image_key")
    out = {}
    for enc in ENCODERS:
        p = Path(str(cfg["emb"]).format(enc=enc))
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=False)
        keys = list(d["image_key"])
        mm = m.loc[keys]
        sel = (mm.scan_index == 24).to_numpy()
        z = OI.l2n(d["embeddings"][sel])
        out[enc] = dict(zip(mm.visit_id[sel], z))
    return out


def image_hashes(set_name: str) -> dict[str, str]:
    cfg = SETS[set_name]
    if not (cfg["subset"] / "manifest.csv").exists():
        return {}
    m = OI.load_manifest(cfg["subset"])
    m = m[m.scan_index == 24]
    return dict(zip(m.visit_id, m.sha256))


# --- metrics -------------------------------------------------------------------------------------------------

def dcg(gains) -> float:
    return float(sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains)))


def score_ranking(ranked: list[str] | None, grade: dict[str, float], rel_threshold: float) -> dict:
    """Metrics for one ranking. `grade`: candidate -> 0/1/2 (or 0/1). Abstention (None) scores 0 everywhere."""
    rel = {v for v, g in grade.items() if g >= rel_threshold}
    ideal = sorted(grade.values(), reverse=True)[:5]
    if ranked is None:
        return dict(hit1=0.0, hit5=0.0, recall5=0.0, mrr=0.0, ndcg5=0.0, answered=False)
    top5 = ranked[:5]
    first = next((i + 1 for i, v in enumerate(ranked) if v in rel), None)
    idcg = dcg(ideal)
    return dict(hit1=float(ranked[0] in rel), hit5=float(any(v in rel for v in top5)),
                recall5=len(rel & set(top5)) / len(rel), mrr=0.0 if first is None else 1.0 / first,
                ndcg5=dcg([grade.get(v, 0.0) for v in top5]) / idcg if idcg > 0 else float("nan"), answered=True)


def random_expectation(grade: dict[str, float], rel_threshold: float) -> dict:
    """Exact expected metrics of a uniformly random permutation of the candidates."""
    n = len(grade)
    r = sum(g >= rel_threshold for g in grade.values())
    k = min(5, n)
    hit5 = 1 - math.comb(n - r, 5) / math.comb(n, 5) if n >= 5 else 1.0
    mrr = sum((1 / j) * (math.comb(n - j, r - 1) / math.comb(n, r)) for j in range(1, n - r + 2))
    mean_gain = np.mean([2 ** g - 1 for g in grade.values()])
    idcg = dcg(sorted(grade.values(), reverse=True)[:5])
    ndcg = mean_gain * sum(1 / math.log2(i + 2) for i in range(k)) / idcg if idcg > 0 else float("nan")
    return dict(hit1=r / n, hit5=hit5, recall5=k / n, mrr=mrr, ndcg5=ndcg)


# --- evaluation ----------------------------------------------------------------------------------------------

def evaluate(qs: pd.DataFrame, labels: pd.DataFrame | None, systems: Systems, hashes: dict[str, str],
             rel_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-query-system metrics, long rankings). Status per query is explicit."""
    per, ranks = [], []
    lab = {} if labels is None else {k: g for k, g in labels.groupby("query_id")}
    for q in qs.itertuples():
        cands = q.candidate_visit_ids.split(";")
        rankings = systems.rank_all(q)
        parsed = rankings.pop("_parsed")
        for s, rk in rankings.items():
            if rk is not None:
                assert set(rk) <= set(cands), f"{q.query_id}/{s}: ranking contains a non-candidate"
                for i, v in enumerate(rk):
                    ranks.append(dict(query_id=q.query_id, system=s, rank=i + 1, candidate_visit_id=v))
        g = lab.get(q.query_id)
        if q.query_visit_id in hashes and any(hashes.get(c) == hashes[q.query_visit_id] for c in cands):
            status = "excluded_duplicate_image"
        elif g is None or g.grade.isna().any() or set(g.candidate_visit_id) != set(cands):
            status = "ineligible" if g is not None and g.get("ineligible", pd.Series([False])).any() \
                else "not_evaluated_missing_labels"
        elif "unanswerable" in g and g.unanswerable.any():
            status = "unanswerable"
        elif (g.grade >= rel_threshold).sum() == 0:
            status = "no_relevant_candidate"
        else:
            status = "evaluated"
        base = dict(query_id=q.query_id, set=q.set, tier=q.tier, category=q.category, eye_id=q.eye_id,
                    query_visit_id=q.query_visit_id, n_candidates=q.n_candidates, status=status, parsed_intent=parsed)
        if status != "evaluated":
            for s in rankings:
                per.append(dict(base, system=s))
            continue
        grade = dict(zip(g.candidate_visit_id, g.grade))
        base["n_relevant"] = int(sum(v >= rel_threshold for v in grade.values()))
        for s, rk in rankings.items():
            per.append(dict(base, system=s, **score_ranking(rk, grade, rel_threshold)))
        per.append(dict(base, system="random_expected", answered=True, **random_expectation(grade, rel_threshold)))
    return pd.DataFrame(per), pd.DataFrame(ranks)


def bootstrap_ci(per_eye: pd.Series) -> tuple[float, float] | None:
    if per_eye.size < BOOTSTRAP["min_eyes"]:
        return None
    rng = np.random.default_rng(BOOTSTRAP["seed"])
    v = per_eye.to_numpy()
    means = v[rng.integers(0, v.size, (BOOTSTRAP["n"], v.size))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(per: pd.DataFrame, tier: str) -> pd.DataFrame:
    rows = []
    primary = PRIMARY[tier]
    for (cat, s), g in per.groupby(["category", "system"], sort=False):
        ev = g[g.status == "evaluated"]
        counts = g.status.value_counts().to_dict()
        row = dict(category=cat, system=s, n_queries=len(g), n_evaluated=len(ev),
                   **{f"n_{k}": v for k, v in counts.items() if k != "evaluated"})
        if ev.empty:
            row.update({m: "NOT EVALUATED" for m in METRICS})
            row["note"] = "no evaluable queries (labels missing or excluded)"
            rows.append(row)
            continue
        for m in METRICS:
            row[m] = round(float(ev[m].mean()), 4)
        row["coverage"] = round(float(ev.answered.astype(bool).mean()), 4)
        pe = ev.groupby("eye_id")[primary].mean()
        row["n_eyes"] = int(pe.size)
        row[f"{primary}_per_eye"] = "; ".join(f"{e}:{v:.3f}" for e, v in pe.items())
        ci = bootstrap_ci(pe)
        row[f"{primary}_ci95_eye_bootstrap"] = "not reported (n_eyes < 5)" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        if s not in ("S1_latest_prior", "random_expected"):
            ref = per[(per.category == cat) & (per.system == "S1_latest_prior") & (per.status == "evaluated")]
            d = ev.set_index("query_id")[primary] - ref.set_index("query_id")[primary]
            de = d.groupby(ev.set_index("query_id").eye_id).mean()
            row[f"{primary}_minus_latest_prior"] = round(float(d.mean()), 4)
            row["eyes_better_equal_worse_vs_latest_prior"] = \
                f"{int((de > 1e-12).sum())}/{int((de.abs() <= 1e-12).sum())}/{int((de < -1e-12).sum())}"
            # Hybrids inherit the metadata ranking, so they are also compared with their metadata component.
            mref = "S2_metadata" if tier == "A" else "S2p_metadata_proxy_cst"
            if s.startswith("S5_"):
                ref2 = per[(per.category == cat) & (per.system == mref) & (per.status == "evaluated")]
                row[f"{primary}_minus_{mref}"] = round(float(
                    (ev.set_index("query_id")[primary] - ref2.set_index("query_id")[primary]).mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=list(SETS), required=True)
    ap.add_argument("--tier", choices=["A", "B"], required=True)
    ap.add_argument("--sheet", type=Path)
    ap.add_argument("--key", type=Path)
    ap.add_argument("--rel-threshold", type=float, default=None,
                    help="binary relevance threshold (default: 1 for Tier A, 2 for Tier B)")
    ap.add_argument("--drop-visits", nargs="*", default=[])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--confirm-heldout-freeze", action="store_true",
                    help="required for --set heldout: confirms the freeze in phase2/annotations/ANNOTATION_LOG.md")
    a = ap.parse_args(argv)
    if a.set == "heldout" and not a.confirm_heldout_freeze:
        ap.error("held-out evaluation runs once, after the freeze (protocol §7); pass --confirm-heldout-freeze")
    thr = a.rel_threshold if a.rel_threshold is not None else (1.0 if a.tier == "A" else 2.0)
    tag = f"tier{a.tier}" + (f"_thr{thr:g}" if a.rel_threshold is not None else "") + \
        ("_drop_" + "_".join(a.drop_visits) if a.drop_visits else "")
    out = a.out or ROOT / "phase2/results/question_eval" / a.set / tag
    episodes = pd.read_csv(ROOT / "data/processed/episodes.csv")

    qs = build_query_set(episodes, a.set, a.tier, a.drop_visits)
    assert_no_candidate_leakage(qs, episodes)
    if a.tier == "A":
        labels = tier_a_labels(qs, episodes, a.drop_visits)
    elif a.sheet and a.key and a.sheet.exists() and a.key.exists():
        labels = human_labels(a.sheet, a.key)
    else:
        labels = None
    systems = Systems(episodes, a.set, a.drop_visits)
    if not systems.emb:
        print(f"WARNING: no saved embeddings for set {a.set!r}; image and hybrid systems will abstain", file=sys.stderr)
    per, ranks = evaluate(qs, labels, systems, image_hashes(a.set), thr)
    summ = summarise(per, a.tier)

    out.mkdir(parents=True, exist_ok=True)
    qs.to_csv(out / "query_set.csv", index=False)
    if labels is not None:
        labels.to_csv(out / "labels_used.csv", index=False)
    ranks.to_csv(out / "rankings.csv", index=False)
    per.to_csv(out / "per_query_metrics.csv", index=False)
    summ.to_csv(out / "summary.csv", index=False)
    meta = dict(set=a.set, tier=a.tier, rel_threshold=thr, drop_visits=a.drop_visits, primary_metric=PRIMARY[a.tier],
                label_source=None if labels is None else sorted(labels.label_source.unique()),
                labels_present=labels is not None, n_queries=int(qs.query_id.nunique()),
                status_counts=per[per.system == "S1_latest_prior"].status.value_counts().to_dict(),
                encoders_with_embeddings=sorted(systems.emb), rrf_k=RRF_K, min_candidates=MIN_CANDIDATES,
                protocol="phase2/QUESTION_EVAL_PROTOCOL.md v0.1")
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
