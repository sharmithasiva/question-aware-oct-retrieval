"""Phase 1 v2 benchmark: harder metadata question types, frozen v1 strategies, independent locked paraphrases.

Usage:  python -m oct_retrieval.evaluate_v2      (writes results/v2/ and reports/RETRIEVAL_REPORT_V2.md)
v1 outputs (results/*.csv, results/v1_snapshot/) are never written here.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import config
from .episodes import build_all
from .evaluate import BOOTSTRAP_REPS, bootstrap_ci, eye_histories, eye_macro, run_retriever
from .ground_truth_v2 import PRIMARY_TYPES, SECONDARY_TYPES, TYPES_V2, generate_instances_v2, last_injection_lookup
from .paraphrases import locked_by_type, read_jsonl, DEV, LOCKED, verify
from .questions import parse_question
from .retrievers import (ClinicalOnly, Combined, EarliestFirst, LatestPrior, RandomHistory, Retriever, RuleRouter,
                         TemporalOnly, _order)
from .side_tables import build_heldout_prime, prime_alignment, trex_alignment

OUT = config.RESULTS_DIR / "v2"
SNAPSHOT = config.RESULTS_DIR / "v1_snapshot"
METRICS = ["hit@1", "hit@3", "recall@1", "mrr", "ndcg@3"]
PRIME_TYPES = [q for q, (_, _, cohort) in TYPES_V2.items() if cohort == "Prime"]


class FixedHeuristic(Retriever):
    """Single fixed visit choice ranked first, rest by recency. Uses CST/BCVA/order only."""

    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def rank(self, spec, hist, current, rng):
        return _order(self.fn(hist))


def _pos(h):
    return np.arange(len(h), dtype=float)


HEURISTICS = [
    FixedHeuristic("h_second_visit", lambda h: (_pos(h) == 1).astype(float)),
    FixedHeuristic("h_max_cst", lambda h: h.cst),
    FixedHeuristic("h_min_cst", lambda h: -h.cst),
    FixedHeuristic("h_max_bcva", lambda h: h.bcva),
    FixedHeuristic("h_min_bcva", lambda h: -h.bcva),
]


def strategies(alpha: float) -> list[Retriever]:
    return [LatestPrior(), EarliestFirst(), RandomHistory(), TemporalOnly(), ClinicalOnly(), Combined(alpha),
            RuleRouter()] + HEURISTICS


def per_type_table(per: pd.DataFrame, inst: pd.DataFrame, types, split: str | None) -> pd.DataFrame:
    """Metrics per (type, strategy), eye-macro with patient-cluster bootstrap CIs."""
    rows = []
    sub = per if split is None else per[per.split == split]
    for q in types:
        iq = inst[(inst.qtype == q) & inst.answerable & ((inst.split == split) if split else True)]
        for name, d in sub[sub.qtype == q].groupby("retriever", sort=False):
            m = eye_macro(d, METRICS)
            h_lo, h_hi = bootstrap_ci(d, "hit@1")
            m_lo, m_hi = bootstrap_ci(d, "mrr")
            rows.append(dict(qtype=q, retriever=name, **m.to_dict(), hit1_lo=h_lo, hit1_hi=h_hi, mrr_lo=m_lo,
                             mrr_hi=m_hi, n=len(d), eyes=d.eye_id.nunique(), patients=d.patient_id.nunique(),
                             mean_acceptable=iq.n_relevant.mean(), multi_answer_rate=iq.multi_answer.mean()))
    return pd.DataFrame(rows)


def parser_behaviour() -> pd.DataFrame:
    """How the FROZEN v1 parser reads the independent paraphrases (no gold spec exists for v2 types)."""
    rows = []
    for split_name, path in (("dev", DEV), ("locked", LOCKED)):
        for r in read_jsonl(path):
            s = parse_question(r["text"])
            rows.append(dict(set=split_name, qtype=r["qtype"], text=r["text"], **asdict(s)))
    return pd.DataFrame(rows)


def f3(x):
    return "–" if pd.isna(x) else f"{x:.3f}"


def _metric_table(L, t: pd.DataFrame, q: str, order: list[str]):
    d = t[t.qtype == q].set_index("retriever")
    if d.empty:
        return
    r0 = d.iloc[0]
    L.append(f"\n#### `{q}` — n = {int(r0.n)} answerable questions, {int(r0.eyes)} eyes, {int(r0.patients)} patients; "
             f"acceptable visits per question: mean {r0.mean_acceptable:.2f}, multi-answer rate "
             f"{r0.multi_answer_rate:.1%}\n")
    L.append("| Strategy | Hit@1 (visit-level accuracy) [95% CI] | Hit@3 | Recall@1 | MRR [95% CI] | nDCG@3 |")
    L.append("|---|---|---|---|---|---|")
    for name in order:
        if name in d.index:
            r = d.loc[name]
            L.append(f"| {name} | {f3(r['hit@1'])} [{f3(r.hit1_lo)}, {f3(r.hit1_hi)}] | {f3(r['hit@3'])} | "
                     f"{f3(r['recall@1'])} | {f3(r.mrr)} [{f3(r.mrr_lo)}, {f3(r.mrr_hi)}] | {f3(r['ndcg@3'])} |")
    h = d.loc[[n for n in d.index if n.startswith("h_") or n in ("latest_prior", "earliest_first")]]
    best = h["hit@1"].idxmax()
    L.append(f"\nFixed-heuristic ceiling (best single fixed choice, picked on this same data, so optimistic): "
             f"**{best} = {f3(h.loc[best, 'hit@1'])}**. Combined (v1) − ceiling = "
             f"{d.loc['combined', 'hit@1'] - h.loc[best, 'hit@1']:+.3f}.")


def write_report(ctx: dict, path):
    inst, prim, sec, expl, ie, parser, inj, order = (ctx[k] for k in
                                                     ("inst", "primary", "secondary", "exploratory", "ie", "parser",
                                                      "injection", "order"))
    L = ["# Phase 1 v2 Retrieval Report — harder metadata questions\n",
         "_Auto-generated by `python -m oct_retrieval.evaluate_v2`. All numbers are actual results. Retrieval "
         "metrics only; no VLM answers are evaluated. No novelty claim is made._\n",
         "## 1. What was evaluated\n",
         f"- Spec: `docs/PHASE1_V2_QUESTION_SPEC.md`. Episodes: {ctx['n_visits']} visits, 96 eyes, 87 patients; "
         "same patient-level split as v1 (seed 42).",
         f"- **Frozen v1 strategies** (code unchanged; Combined α = {ctx['alpha']} from the v1 snapshot). The v1 "
         "parser and retrievers implement none of the v2 definitions, so ground truth is not shared with any "
         "strategy (non-circular).",
         "- Fixed heuristics (`h_*`, `latest_prior`, `earliest_first`) are reported so the "
         "\"solved by a fixed choice\" question can be checked per type.",
         "- **Every strategy is scored against the same answer key** (`ground_truth_v2.py`, which does not import "
         "the retrievers). Ties: every acceptable visit counts.",
         f"- Question text: **locked independent paraphrases** (SHA-256 `{ctx['locked_sha'][:16]}…`), 5 per type, "
         "rotated across instances. The parser was not changed or tuned on them.",
         "- Held-out variables (leakage index, DRSS, injection) are never in retriever inputs. This is enforced by "
         "`test_retriever_inputs_carry_no_heldout` and `test_heldout_never_in_retriever_memory`.",
         f"- Alignment checks: Prime {ctx['prime_align']}, TREX {ctx['trex_align']}.",
         f"- Uncertainty: 95% CIs from a {BOOTSTRAP_REPS}-rep bootstrap resampling **patients**; metrics are "
         "macro-averaged per eye.\n",
         "## 2. Sample counts per type\n",
         "| Type | Role | Tier | Cohort | Test answerable (eyes / patients) | Test IE | All eligible | All IE | "
         "Mean acceptable visits | Multi-answer rate |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for q, (role, tier, cohort) in TYPES_V2.items():
        a = inst[inst.qtype == q]
        ta = a[(a.split == "test") & a.answerable]
        L.append(f"| `{q}` | {role} | {tier} | {cohort} | {len(ta)} ({ta.eye_id.nunique()} / {ta.patient_id.nunique()}) "
                 f"| {int(((a.split == 'test') & ~a.answerable).sum())} | {len(a)} | {int((~a.answerable).sum())} | "
                 f"{a[a.answerable].n_relevant.mean():.2f} | {a[a.answerable].multi_answer.mean():.1%} |")

    L.append("\n## 3. Primary benchmark — frozen patient-level TEST split\n")
    L.append("Hit@1 = the strategy's top-ranked visit is an acceptable visit (visit-level retrieval accuracy).")
    for q in PRIMARY_TYPES:
        _metric_table(L, prim, q, order)

    L.append("\n## 4. Secondary benchmark — TEST split (recency already strong; reported separately)\n")
    for q in SECONDARY_TYPES:
        _metric_table(L, sec, q, order)

    L.append("\n## 5. Exploratory — Prime types on ALL 40 Prime patients (train + val + test)\n")
    L.append("**Not independent test performance.** No v1 component was fitted on these question types. However, "
             "train/val patients were used in v1 (α tuning on other question types), and these numbers are pooled "
             "over the same 40 patients. They are shown only because the Prime test split has 7–8 patients.")
    for q in PRIME_TYPES:
        _metric_table(L, expl, q, order)

    L.append("\n## 6. Insufficient-evidence (IE) questions\n")
    L.append("The correct output is \"no qualifying visit\". IE questions are excluded from the ranking metrics above. "
             "No frozen v1 strategy can abstain; each returns a ranking for every question, so all of them score "
             "0% here.\n")
    L.append("| Type | IE questions (test) | IE questions (all) | Abstention rate of every v1 strategy |")
    L.append("|---|---|---|---|")
    for r in ie.itertuples():
        rate = "n/a (no IE questions)" if pd.isna(r.abstain) else f"{r.abstain:.0%}"
        L.append(f"| `{r.qtype}` | {r.ie_test} | {r.ie_all} | {rate} |")

    L.append("\n## 7. How the frozen v1 parser reads the independent paraphrases\n")
    L.append("There is no v2 gold parse for v1 (it has no v2 intents), so this shows *what v1 does*, not "
             "accuracy. Counts are over dev + locked (10 per type).\n")
    L.append("| Type | Parsed intents | Parsed targets | evidence = image |")
    L.append("|---|---|---|---|")
    for q, g in parser.groupby("qtype"):
        intents = ", ".join(f"{k}:{v}" for k, v in g.intent.value_counts().items())
        targets = ", ".join(f"{k}:{v}" for k, v in g.target.fillna("none").value_counts().items())
        L.append(f"| `{q}` | {intents} | {targets} | {int((g.evidence == 'image').sum())}/{len(g)} |")

    L.append("\n## 8. `last_injection` — structured lookup, NOT a retrieval benchmark\n")
    L.append(f"- Prime only; source OCT-DR `Data` sheet. {len(inj)} eligible current visits; "
             f"{int((inj.answer_visit_id != '').sum())} have a prior injected visit. When one exists, the latest prior "
             f"visit is that visit in {inj[inj.answer_visit_id != ''].latest_prior_is_answer.mean():.1%} of cases.")
    L.append("- Injection status is not in the retriever memory, so no retrieval metrics are reported. With "
             "the field in memory the answer is a deterministic lookup.")

    L.append("\n## 9. v1 results relabelled\n")
    L.append("The v1 report's 1.000 scores for `combined` and `rule_router` are **pipeline checks**: those "
             "strategies implement the same rules that define the v1 ground truth. They are not evidence that "
             "question-aware retrieval is better. v1 outputs are preserved unchanged in `results/v1_snapshot/` "
             "(checksums verified), and the test suite reproduces the v1 test-split metrics exactly.")

    L.append("\n## 10. Deviations from the spec\n")
    for d in ctx["deviations"]:
        L.append(f"- {d}")

    L.append("\n## 11. Limitations\n")
    L.extend([
        "- Prime-only types (`one_year_ago`, `six_months_ago`, `max_leakage`) have 7–8 test patients, so their CIs "
        "are wide.",
        "- The leakage index is an angiography measure, not OCT. Failure to retrieve it from CST/BCVA/time does "
        "not show that OCT images would or would not find it.",
        "- Tier-B ground truth comes from the same CST/BCVA/order data the retrievers see. A future retriever that "
        "implements these rules would score ~1.0 by construction; such results must be reported as rule coverage.",
        "- The independent paraphrases were written by another instance of the same model family, not by "
        "clinicians. They reduce, but do not remove, parser-test leakage.",
        "- The fixed-heuristic ceiling is chosen on the evaluation data itself, so it is an optimistic reference.",
        "- No abstention mechanism exists yet, so IE questions are not handled by any strategy.",
    ])
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = verify()
    alpha = json.loads((SNAPSHOT / "run_config.json").read_text())["alpha"]

    ep, _ = build_all()
    heldout = build_heldout_prime(ep)
    heldout.to_csv(config.PROCESSED_DIR / "heldout_prime.csv", index=False)
    inst = generate_instances_v2(ep, heldout, locked_by_type())
    inst.to_csv(OUT / "instances_v2.csv", index=False)

    eyes = eye_histories(ep)
    answerable = inst[inst.answerable]
    order = [s.name for s in strategies(alpha)]
    per = pd.concat([run_retriever(s, answerable, eyes) for s in strategies(alpha)], ignore_index=True)
    per = per.merge(inst[["instance_id", "n_relevant", "multi_answer", "role", "tier"]], on="instance_id")
    per.to_csv(OUT / "per_instance_metrics_v2.csv", index=False)

    primary = per_type_table(per, inst, PRIMARY_TYPES, "test")
    secondary = per_type_table(per, inst, SECONDARY_TYPES, "test")
    exploratory = per_type_table(per, inst, PRIME_TYPES, None)
    primary.to_csv(OUT / "summary_primary_test.csv", index=False)
    secondary.to_csv(OUT / "summary_secondary_test.csv", index=False)
    exploratory.to_csv(OUT / "summary_prime_all_patients_EXPLORATORY.csv", index=False)

    # IE: run every strategy; it abstains only if it returns an empty ranking. Computed, not assumed.
    ie_inst = inst[~inst.answerable]
    per_ie = pd.concat([run_retriever(s, ie_inst, eyes) for s in strategies(alpha)], ignore_index=True)
    per_ie["abstained"] = per_ie.ranked_visit_ids.fillna("") == ""
    ie = pd.DataFrame([dict(qtype=q, ie_test=int(((ie_inst.qtype == q) & (ie_inst.split == "test")).sum()),
                            ie_all=int((ie_inst.qtype == q).sum()),
                            abstain=per_ie.loc[per_ie.qtype == q, "abstained"].mean() if (ie_inst.qtype == q).any()
                            else float("nan"))
                       for q in TYPES_V2])
    ie.to_csv(OUT / "insufficient_evidence.csv", index=False)

    parser = parser_behaviour()
    parser.to_csv(OUT / "parser_behaviour_v1_on_v2_paraphrases.csv", index=False)
    inj = last_injection_lookup(ep, heldout)
    inj.to_csv(OUT / "last_injection_structured_lookup.csv", index=False)

    deviations = [
        "None in definitions, eligibility, tie handling or splits. Counts match spec §4 exactly "
        "(`test_v2_counts_match_spec`).",
        "Added beyond the spec: `h_*` fixed heuristics run as strategies (the spec's heuristic ceiling was "
        "computed analytically). They use CST/BCVA/order only.",
        "The v1 report text was relabelled (pipeline-check wording); v1 metrics and files are unchanged.",
    ]
    ctx = dict(inst=inst, primary=primary, secondary=secondary, exploratory=exploratory, ie=ie, parser=parser,
               injection=inj, order=order, alpha=alpha, n_visits=len(ep), locked_sha=manifest["locked"]["sha256"],
               prime_align=prime_alignment(ep), trex_align=trex_alignment(ep), deviations=deviations)
    write_report(ctx, config.REPORTS_DIR / "RETRIEVAL_REPORT_V2.md")
    (OUT / "run_config_v2.json").write_text(json.dumps(dict(
        seed=config.SEED, alpha_frozen_from_v1=alpha, bootstrap_reps=BOOTSTRAP_REPS,
        paraphrases_locked_sha256=manifest["locked"]["sha256"], n_instances=len(inst),
        n_answerable=int(inst.answerable.sum())), indent=2))
    cols = ["qtype", "retriever", "hit@1", "hit1_lo", "hit1_hi", "mrr", "n", "patients"]
    print(primary[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
