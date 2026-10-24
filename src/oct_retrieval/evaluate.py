"""Run the Phase 1 metadata-retrieval benchmark and write results + a markdown report.

Usage:  python -m oct_retrieval.evaluate
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import config
from .episodes import build_all
from .metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from .questions import HELDOUT_PARAPHRASES, QUESTION_TYPES, generate_instances, parse_question
from .retrievers import (ClinicalOnly, Combined, EarliestFirst, FullHistory, History, LatestPrior, RandomHistory,
                         RuleRouter, TemporalOnly)

KS = (1, 2, 3)
ALPHA_GRID = [round(a, 2) for a in np.arange(0, 1.0001, 0.05)]
BOOTSTRAP_REPS = 1000


def eye_histories(ep: pd.DataFrame) -> dict[int, History]:
    """Full chronological visit sequence per eye; slice [:t] to get the history of visit t."""
    out = {}
    for e, g in ep.groupby("eye_id"):
        g = g.sort_values("visit_index")
        out[e] = History(list(g.visit_id), g.cst.to_numpy(float), g.bcva.to_numpy(float), g.n_images.to_numpy())
    return out


def run_retriever(retriever, instances: pd.DataFrame, eyes: dict, seed: int = config.SEED) -> pd.DataFrame:
    rows = []
    for inst in instances.itertuples():
        eye = eyes[inst.eye_id]
        t = inst.current_visit_index
        hist = History(eye.visit_ids[:t], eye.cst[:t], eye.bcva[:t], eye.n_images[:t])
        cur_images = int(eye.n_images[t])
        current = {"cst": eye.cst[t], "bcva": eye.bcva[t], "visit_id": eye.visit_ids[t]}
        assert current["visit_id"] == inst.current_visit_id
        assert current["visit_id"] not in hist.visit_ids  # current visit is never a candidate
        spec = parse_question(inst.question)
        rng = np.random.default_rng([seed, int(inst.Index)])
        t0 = time.perf_counter()
        order = retriever.rank(spec, hist, current, rng)
        latency_ms = (time.perf_counter() - t0) * 1000
        ranked = [hist.visit_ids[i] for i in order]
        rel = set(inst.relevant_visit_ids.split(";"))
        row = dict(instance_id=inst.instance_id, retriever=retriever.name, eye_id=inst.eye_id,
                   patient_id=inst.patient_id, split=inst.split, qtype=inst.qtype, n_history=inst.n_history,
                   mrr=reciprocal_rank(ranked, rel), latency_ms=latency_ms,
                   ranked_visit_ids=";".join(ranked[: max(KS)]))
        for k in KS:
            row[f"hit@{k}"] = hit_at_k(ranked, rel, k)
            row[f"recall@{k}"] = recall_at_k(ranked, rel, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, rel, k)
            topk = order[:k] if retriever.name != "full_history" else order
            row[f"visits@{k}"] = 1 + len(topk)
            row[f"images@{k}"] = int(cur_images + hist.n_images[topk].sum())
        rows.append(row)
    return pd.DataFrame(rows)


METRIC_COLS = ["hit@1", "hit@3", "recall@1", "recall@3", "mrr", "ndcg@3"]
PIPELINE_CHECK_NOTE = (
    "**Relabelled (v2 review):** the 1.000 scores of `combined` and `rule_router` are **pipeline checks, not "
    "evidence of retrieval superiority**. These strategies implement the same rules that define this ground truth. "
    "See `reports/RETRIEVAL_REPORT_V2.md` for non-circular question types.")


def eye_macro(df: pd.DataFrame, cols=METRIC_COLS) -> pd.Series:
    """Average within eye first, then across eyes, so long histories do not dominate."""
    return df.groupby("eye_id")[cols].mean().mean()


def bootstrap_ci(df: pd.DataFrame, col: str, reps=BOOTSTRAP_REPS, seed=config.SEED) -> tuple[float, float]:
    """95% CI resampling patients (clusters), metric macro-averaged over eyes."""
    per_eye = df.groupby(["patient_id", "eye_id"])[col].mean().reset_index()
    by_patient = {p: g[col].to_numpy() for p, g in per_eye.groupby("patient_id")}
    pids = np.array(list(by_patient))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(reps):
        sample = rng.choice(pids, size=len(pids), replace=True)
        stats.append(np.concatenate([by_patient[p] for p in sample]).mean())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def tune_alpha(instances: pd.DataFrame, eyes: dict) -> tuple[float, pd.DataFrame]:
    train = instances[instances.split == "train"]
    scores = [(a, eye_macro(run_retriever(Combined(a), train, eyes), ["mrr"])["mrr"]) for a in ALPHA_GRID]
    curve = pd.DataFrame(scores, columns=["alpha", "train_mrr"])
    best = curve.sort_values(["train_mrr", "alpha"], ascending=[False, True]).iloc[0]
    return float(best.alpha), curve


def parser_eval() -> pd.DataFrame:
    rows = []
    for qtype, (gold, templates) in QUESTION_TYPES.items():
        for q in templates:
            rows.append(dict(set="benchmark_template", qtype=qtype, question=q, gold=asdict(gold),
                             parsed=asdict(parse_question(q)), correct=parse_question(q) == gold))
    for q, gold in HELDOUT_PARAPHRASES:
        rows.append(dict(set="heldout_paraphrase", qtype="", question=q, gold=asdict(gold),
                         parsed=asdict(parse_question(q)), correct=parse_question(q) == gold))
    return pd.DataFrame(rows)


def fmt(x: float) -> str:
    return f"{x:.3f}"


def write_report(ep, instances, summary, by_qtype, eff, parser, alpha, curve, motivation, path):
    test = instances[instances.split == "test"]
    L = []
    L.append("# Phase 1 Retrieval Evaluation Report (metadata only)\n")
    L.append("_Auto-generated by `python -m oct_retrieval.evaluate`. All numbers below are actual results on the "
             "OLIVES label files; nothing is illustrative unless stated._\n")
    L.append("## 1. Setup\n")
    L.append(f"- Episodes: {len(ep)} visits, {ep.eye_id.nunique()} eyes, {ep.patient_id.nunique()} patients "
             f"(see `docs/DATA_AUDIT.md`).")
    sp = ep.groupby("split").agg(patients=("patient_id", "nunique"), eyes=("eye_id", "nunique"),
                                 visits=("visit_id", "size"))
    L.append("- Patient-level split (seed {}, stratified by trial): ".format(config.SEED) + ", ".join(
        f"{s}: {r.patients} patients / {r.eyes} eyes / {r.visits} visits" for s, r in sp.iterrows()) + ".")
    L.append(f"- Instances: one per (current visit, question type) with ≥2 prior visits and defined ground truth: "
             f"{len(instances)} total, {len(test)} in test ({test.eye_id.nunique()} eyes).")
    L.append("- Candidates are only the visits *before* the current visit of the same eye; the current visit is "
             "never a candidate (asserted).")
    L.append("- Retrievers see the parsed question text, not the gold question type.")
    L.append(f"- Combined retriever weight α (clinical) tuned on **train** by eye-macro MRR over "
             f"{ALPHA_GRID[0]}–{ALPHA_GRID[-1]}: **α = {alpha}**. Val/test untouched during tuning.")
    L.append("- Metrics: binary relevance; ties in ground truth are all relevant. Means are macro-averaged over eyes "
             f"(instances → eye → mean); 95% CIs from a {BOOTSTRAP_REPS}-rep patient-cluster bootstrap.\n")

    L.append("## 2. Question types\n")
    L.append("| Type | Gold target / intent | Ground truth | Test instances | Latest prior relevant | Baseline relevant |")
    L.append("|---|---|---|---|---|---|")
    gt_desc = {"baseline": "first visit of the eye", "previous": "immediately preceding visit",
               "extreme": "visit(s) with the {d} recorded {t} in history", "similar": "visit(s) with {t} closest to current"}
    for qtype, (gold, _) in QUESTION_TYPES.items():
        d = gt_desc[gold.intent].format(d="highest" if gold.direction == "max" else "lowest", t=gold.target.upper())
        L.append(f"| `{qtype}` | {gold.target.upper()} / {gold.intent}{'-' + gold.direction if gold.direction else ''} "
                 f"| {d} | {(test.qtype == qtype).sum()} | {motivation.latest_prior_is_relevant[qtype]:.1%} "
                 f"| {motivation.baseline_is_relevant[qtype]:.1%} |")
    L.append("\nThe last two columns (all splits) are the share of instances where simply taking the latest prior "
             "visit, or the baseline visit, already hits a relevant visit. They measure how often the motivating "
             "problem (\"a fixed heuristic picks the wrong visit\") actually occurs in OLIVES.\n")

    L.append("## 3. Question parser\n")
    for s, g in parser.groupby("set"):
        L.append(f"- `{s}`: {g.correct.sum()}/{len(g)} parsed exactly to the gold spec.")
    wrong = parser[~parser.correct]
    for r in wrong.itertuples():
        L.append(f"  - ✗ \"{r.question}\" → parsed {r.parsed}, gold {r.gold}")
    L.append("- Caveat: the parser lexicon and all paraphrases were written by the same author, so this is an "
             "optimistic check of the rules, not an estimate of accuracy on real clinician questions.\n")

    L.append("## 4. Retrieval results — test split\n")
    L.append("| Retriever | Hit@1 | Hit@3 | Recall@1 | Recall@3 | MRR | nDCG@3 |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        L.append(f"| {r['retriever']} | {fmt(r['hit@1'])} [{fmt(r['hit1_lo'])}, {fmt(r['hit1_hi'])}] | "
                 f"{fmt(r['hit@3'])} | {fmt(r['recall@1'])} | {fmt(r['recall@3'])} | "
                 f"{fmt(r['mrr'])} [{fmt(r['mrr_lo'])}, {fmt(r['mrr_hi'])}] | {fmt(r['ndcg@3'])} |")
    L.append("\nBrackets: 95% patient-bootstrap CI. `full_history` is not listed: it passes every prior visit, so its "
             "recall over the whole history is 1 by construction; see the cost table.\n")
    L.append(PIPELINE_CHECK_NOTE + "\n")

    L.append("### Hit@1 by question type (test)\n")
    piv = by_qtype.pivot(index="qtype", columns="retriever", values="hit@1")
    cols = [c for c in summary.retriever if c in piv.columns]
    L.append("| Type | " + " | ".join(cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for q, r in piv[cols].iterrows():
        L.append(f"| `{q}` | " + " | ".join(fmt(r[c]) for c in cols) + " |")

    L.append("\n## 5. Cost (test split, mean per question)\n")
    L.append("Images = B-scans the downstream model would receive (current visit + selected history). "
             "Latency is retrieval-only on this laptop; VLM latency and token cost are not measurable in Phase 1.\n")
    L.append("| Strategy | K | Visits passed | Images passed | Retrieval latency (ms) |")
    L.append("|---|---|---|---|---|")
    for r in eff.itertuples():
        L.append(f"| {r.retriever} | {r.K} | {r.visits:.1f} | {r.images:.1f} | {r.latency_ms:.3f} |")

    L.append("\n## 6. Combined-retriever tuning curve (train MRR)\n")
    L.append(", ".join(f"α={r.alpha}: {fmt(r.train_mrr)}" for r in curve.itertuples()))

    L.append("\n\n## 7. How to read these results (limitations)\n")
    L.append("- **Ground truth and the question-aware rules share their definition.** For metadata questions, once the "
             "question is parsed correctly, `rule_router` / `clinical_only` reproduce the ground-truth rule, so near-"
             "perfect scores are expected by construction. Phase 1 therefore (a) validates the pipeline, (b) quantifies "
             "how far common history-selection heuristics (latest, random, fixed window) fall from the visit a question "
             "actually refers to, and (c) checks that each signal alone is insufficient across question types. It does "
             "**not** show that question-aware retrieval improves downstream answers — that needs Phase 2/3.")
    L.append(f"- **The motivating example is often solved by a fixed heuristic.** Under treatment, CST mostly falls, so the "
             f"highest-CST visit is the baseline in {motivation.baseline_is_relevant['worst_edema']:.1%} of "
             f"`worst_edema` instances (and the lowest-BCVA visit in "
             f"{motivation.baseline_is_relevant['worst_vision']:.1%} of `worst_vision`). The advantage of question-aware "
             "selection comes from covering *all* question types with one system, not from any single type.")
    L.append("- CST is used as a proxy for edema severity, and BCVA for visual function; neither is a disease-severity label.")
    L.append("- Biomarker labels exist only for the first and last visit of each eye, so no biomarker questions are "
             "included.")
    L.append("- Instances from the same eye are strongly correlated; only "
             f"{test.eye_id.nunique()} test eyes ({test.patient_id.nunique()} patients) contribute, hence the wide CIs.")
    L.append("- TREX visit numbers are ordinal; real time intervals are not used.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ep, images = build_all()
    ep.to_csv(config.PROCESSED_DIR / "episodes.csv", index=False)
    images.to_csv(config.PROCESSED_DIR / "visit_images.csv", index=False)
    eyes = eye_histories(ep)

    instances = generate_instances(ep)
    instances.to_csv(config.RESULTS_DIR / "instances.csv", index=False)
    motivation = instances.groupby("qtype")[["latest_prior_is_relevant", "baseline_is_relevant"]].mean()

    alpha, curve = tune_alpha(instances, eyes)
    curve.to_csv(config.RESULTS_DIR / "combined_alpha_curve.csv", index=False)

    retrievers = [LatestPrior(), EarliestFirst(), RandomHistory(), TemporalOnly(), ClinicalOnly(), Combined(alpha),
                  RuleRouter(), FullHistory()]
    per = pd.concat([run_retriever(r, instances, eyes) for r in retrievers], ignore_index=True)
    per.to_csv(config.RESULTS_DIR / "per_instance_metrics.csv", index=False)

    ranked_names = [r.name for r in retrievers if r.name != "full_history"]
    summary_rows, qtype_rows = [], []
    for split in ("train", "val", "test"):
        for name in ranked_names:
            d = per[(per.retriever == name) & (per.split == split)]
            m = eye_macro(d)
            lo, hi = bootstrap_ci(d, "hit@1")
            mlo, mhi = bootstrap_ci(d, "mrr")
            summary_rows.append(dict(split=split, retriever=name, **m.to_dict(),
                                     hit1_lo=lo, hit1_hi=hi, mrr_lo=mlo, mrr_hi=mhi, n_instances=len(d)))
            for q, dq in d.groupby("qtype"):
                qtype_rows.append(dict(split=split, retriever=name, qtype=q, **eye_macro(dq).to_dict(), n=len(dq)))
    summary = pd.DataFrame(summary_rows)
    by_qtype = pd.DataFrame(qtype_rows)
    summary.to_csv(config.RESULTS_DIR / "summary.csv", index=False)
    by_qtype.to_csv(config.RESULTS_DIR / "summary_by_qtype.csv", index=False)

    test = per[per.split == "test"]
    eff = []
    for name in ["latest_prior", "earliest_first", "rule_router", "combined", "full_history"]:
        d = test[test.retriever == name]
        for k in KS if name != "full_history" else (max(KS),):
            eff.append(dict(retriever=name, K=k if name != "full_history" else "all",
                            visits=d[f"visits@{k}"].mean(), images=d[f"images@{k}"].mean(),
                            latency_ms=d.latency_ms.mean()))
    eff = pd.DataFrame(eff)
    eff.to_csv(config.RESULTS_DIR / "efficiency_test.csv", index=False)

    parser = parser_eval()
    parser.to_csv(config.RESULTS_DIR / "parser_eval.csv", index=False)

    test_summary = summary[summary.split == "test"][
        ["retriever", "hit@1", "hit1_lo", "hit1_hi", "hit@3", "recall@1", "recall@3", "mrr", "mrr_lo", "mrr_hi",
         "ndcg@3"]].reset_index(drop=True)
    write_report(ep, instances, test_summary, by_qtype[by_qtype.split == "test"], eff, parser, alpha, curve,
                 motivation, config.REPORTS_DIR / "RETRIEVAL_REPORT.md")
    (config.RESULTS_DIR / "run_config.json").write_text(json.dumps(
        dict(seed=config.SEED, alpha=alpha, ks=KS, bootstrap_reps=BOOTSTRAP_REPS, min_history=2,
             n_instances=len(instances)), indent=2))
    print(test_summary.round(3).to_string(index=False))
    print(eff.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
