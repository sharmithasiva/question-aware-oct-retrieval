"""Phase 1.3 DEVELOPMENT evaluation: new intent parser + retrieval vs baselines, dev data only.

Usage:  python -m oct_retrieval.evaluate_v3_dev
Development data = patients in the `train` and `val` splits x the DEV paraphrases. The `test` split and the
locked paraphrases are never used here (asserted). Writes results/v3_dev/ and reports/RETRIEVAL_REPORT_V3_DEV.md.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config
from . import intent as I
from .episodes import build_all
from .evaluate import BOOTSTRAP_REPS, bootstrap_ci, eye_histories, eye_macro, run_retriever
from .evaluate_v2 import strategies
from .ground_truth_v2 import PRIMARY_TYPES, SECONDARY_TYPES, TYPES_V2, generate_instances_v2
from .intent_retrieval import NO_QUALIFYING, RETRIEVED, answer, build_memories
from .metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from .paraphrases import dev_by_type, verify
from .questions import HELDOUT_PARAPHRASES, QUESTION_TYPES, QuestionSpec, generate_instances
from .side_tables import build_heldout_prime

OUT = config.RESULTS_DIR / "v3_dev"
SNAPSHOT = config.RESULTS_DIR / "v1_snapshot"
DEV_SPLITS = ("train", "val")
NEW = "intent_v3"
FIXED_RULES = ["latest_prior", "earliest_first", "h_second_visit", "h_max_cst", "h_min_cst", "h_max_bcva",
               "h_min_bcva"]
METRICS = ["hit@1", "hit@3", "recall@1", "mrr", "ndcg@3"]

# Gold intents, derived from the frozen definitions in docs/PHASE1_V2_QUESTION_SPEC.md (not from any text).
GOLD_V2 = {
    "largest_cst_rise": dict(task=I.RETRIEVE, operation=I.CONSECUTIVE_CHANGE, target="cst", direction="increase"),
    "largest_bcva_drop": dict(task=I.RETRIEVE, operation=I.CONSECUTIVE_CHANGE, target="bcva", direction="decrease"),
    "first_vision_gain_1line": dict(task=I.RETRIEVE, operation=I.THRESHOLD_FROM_BASELINE, target="bcva",
                                    direction="increase", threshold=5.0),
    "one_year_ago": dict(task=I.RETRIEVE, operation=I.TIME_OFFSET, offset_weeks=52),
    "six_months_ago": dict(task=I.RETRIEVE, operation=I.TIME_OFFSET, offset_weeks=26),
    "max_leakage": dict(task=I.UNAVAILABLE, missing_variable="leakage_index"),
    "last_thicker_than_now": dict(task=I.RETRIEVE, operation=I.COMPARE_TO_CURRENT, target="cst", comparator="gt",
                                  recency="last"),
    "last_vision_as_good": dict(task=I.RETRIEVE, operation=I.COMPARE_TO_CURRENT, target="bcva", comparator="ge",
                                recency="last"),
    "last_injection": dict(task=I.UNAVAILABLE, missing_variable="injection"),
}
REFUSAL_TASKS = (I.UNSUPPORTED_IMAGE, I.UNAVAILABLE)   # acceptable routes for "unsupported" questions


def gold_from_v1(spec: QuestionSpec) -> dict:
    if spec.evidence == "image":
        return dict(task=I.UNSUPPORTED_IMAGE)
    op = {"baseline": I.ANCHOR_BASELINE, "previous": I.ANCHOR_PREVIOUS, "extreme": I.EXTREMUM,
          "similar": I.CLOSEST_TO_CURRENT}[spec.intent]
    g = dict(task=I.RETRIEVE, operation=op, target=spec.target)
    if op == I.EXTREMUM:
        g["direction"] = spec.direction
    return g


def intent_matches(q: I.QueryIntent, gold: dict) -> bool:
    return all(getattr(q, k) == v for k, v in gold.items())


def route_matches(q: I.QueryIntent, qtype: str) -> bool:
    if qtype == "unsupported":
        return q.task in REFUSAL_TASKS
    return q.task == GOLD_V2[qtype]["task"]


def dev_episodes(ep: pd.DataFrame) -> pd.DataFrame:
    dev = ep[ep.split.isin(DEV_SPLITS)]
    assert not (dev.split == "test").any()
    # Patient-level separation: no patient may appear in both dev and test.
    assert not set(dev.patient_id) & set(ep.loc[ep.split == "test", "patient_id"])
    return dev


def run_new_system(instances: pd.DataFrame, memories: dict) -> pd.DataFrame:
    rows = []
    for inst in instances.itertuples():
        mem = memories[inst.eye_id]
        t = inst.current_visit_index
        assert mem.visit_ids[t] == inst.current_visit_id
        q, res = answer(inst.question, mem, t)
        ranked = [mem.visit_ids[i] for i in res.ranked]
        assert inst.current_visit_id not in ranked
        rel = set(inst.relevant_visit_ids.split(";")) if inst.relevant_visit_ids else set()
        row = dict(instance_id=inst.instance_id, retriever=NEW, eye_id=inst.eye_id, patient_id=inst.patient_id,
                   split=inst.split, qtype=inst.qtype, answerable=inst.answerable, n_relevant=inst.n_relevant,
                   status=res.status, abstained=res.abstained, reason=res.reason, task=q.task,
                   operation=q.operation, ranked_visit_ids=";".join(ranked[:3]))
        if inst.answerable:
            ok = res.status == RETRIEVED
            row.update({"hit@1": hit_at_k(ranked, rel, 1) if ok else 0.0,
                        "hit@3": hit_at_k(ranked, rel, 3) if ok else 0.0,
                        "recall@1": recall_at_k(ranked, rel, 1) if ok else 0.0,
                        "mrr": reciprocal_rank(ranked, rel) if ok else 0.0,
                        "ndcg@3": ndcg_at_k(ranked, rel, 3) if ok else 0.0})
        rows.append(row)
    return pd.DataFrame(rows)


def retrieval_table(per: pd.DataFrame, types, names) -> pd.DataFrame:
    rows = []
    for q in types:
        for name in names:
            d = per[(per.qtype == q) & (per.retriever == name)]
            if d.empty:
                continue
            m = eye_macro(d, METRICS)
            h_lo, h_hi = bootstrap_ci(d, "hit@1")
            m_lo, m_hi = bootstrap_ci(d, "mrr")
            by_split = {f"hit@1_{s}": eye_macro(d[d.split == s], ["hit@1"])["hit@1"] for s in DEV_SPLITS}
            rows.append(dict(qtype=q, retriever=name, **m.to_dict(), hit1_lo=h_lo, hit1_hi=h_hi, mrr_lo=m_lo,
                             mrr_hi=m_hi, **by_split, n=len(d), eyes=d.eye_id.nunique(),
                             patients=d.patient_id.nunique()))
    return pd.DataFrame(rows)


def parser_table() -> pd.DataFrame:
    rows = []
    for qtype, texts in dev_by_type().items():
        for text in texts:
            q = I.parse_intent(text)
            exact = route_matches(q, qtype) if qtype == "unsupported" else intent_matches(q, GOLD_V2[qtype])
            rows.append(dict(set="v2_dev", qtype=qtype, text=text, task=q.task, operation=q.operation,
                             target=q.target, direction=q.direction, comparator=q.comparator, recency=q.recency,
                             offset_weeks=q.offset_weeks, threshold=q.threshold, reason=q.reason,
                             exact=exact, route_ok=route_matches(q, qtype)))
    for qtype, (gold, templates) in QUESTION_TYPES.items():
        for text in templates:
            q = I.parse_intent(text)
            g = gold_from_v1(gold)
            rows.append(dict(set="v1_template", qtype=qtype, text=text, task=q.task, operation=q.operation,
                             target=q.target, direction=q.direction, reason=q.reason,
                             exact=intent_matches(q, g), route_ok=q.task == g["task"]))
    for text, gold in HELDOUT_PARAPHRASES:
        q = I.parse_intent(text)
        g = gold_from_v1(gold)
        rows.append(dict(set="v1_heldout", qtype="", text=text, task=q.task, operation=q.operation, target=q.target,
                         direction=q.direction, reason=q.reason, exact=intent_matches(q, g),
                         route_ok=q.task == g["task"]))
    return pd.DataFrame(rows)


def f3(x):
    return "–" if pd.isna(x) else f"{x:.3f}"


def write_report(ctx, path):
    L = ["# Phase 1.3 — DEVELOPMENT evaluation of the intent parser + retrieval (no locked data)\n",
         "_Auto-generated by `python -m oct_retrieval.evaluate_v3_dev`. Development numbers only. They were used to "
         "build and check the method, so they are **optimistic** and are **not** evidence of improvement. The "
         "locked test set was not run._\n",
         "## 1. Protocol\n",
         f"- Data: patients in `train` + `val` ({ctx['dev_patients']} patients, {ctx['dev_eyes']} eyes); `test` "
         "patients excluded by assertion. Question text: **dev** paraphrases only (5 per type), rotated across "
         "instances; the locked file is not read by this runner.",
         "- Same answer key for every strategy: `ground_truth_v2.answer_key` (not imported by the new retriever).",
         "- New system: `intent.parse_intent` → `intent_retrieval.retrieve` (`intent_v3`). An abstention on an "
         "answerable question scores 0 on every retrieval metric.",
         "- Baselines: frozen v1 strategies (`combined` α frozen from the v1 snapshot, `rule_router`, `random`) and "
         "fixed rules. **The best fixed rule per type is selected here on dev** and frozen in "
         "`results/v3_dev/selected_baselines.json` for any later locked evaluation.",
         f"- Uncertainty: 95% patient-cluster bootstrap ({BOOTSTRAP_REPS} reps), eye-macro averaging. It does **not** "
         "include paraphrase variability: only 5 dev paraphrases per type.",
         "- **Circularity caveat:** for Tier-B types the new operations implement the frozen definitions, so a correct "
         "parse gives ≈1.0 by construction (rule coverage). The informative quantities are parser/routing "
         "accuracy on *independent* text and abstention behaviour, and dev text is not independent of the "
         "parser that was built on it.\n",
         "## 2. Parser and routing (dev paraphrases; v1 regression)\n",
         "| Set / type | Exact intent | Route correct |", "|---|---|---|"]
    p = ctx["parser"]
    for (s, q), g in p.groupby(["set", "qtype"], sort=False):
        L.append(f"| {s} / `{q or 'all'}` | {int(g.exact.sum())}/{len(g)} | {int(g.route_ok.sum())}/{len(g)} |")
    L.append("\n**Before (frozen v1 parser on the same dev paraphrases):** see `docs/RETRIEVAL_IMPROVEMENT_PLAN.md` "
             "§2. No v2 type was parsed to a correct operation, because v1 has no such operations.\n")

    sel = ctx["selected"]
    for title, types in (("3. Retrieval — primary types (dev)", PRIMARY_TYPES),
                         ("4. Retrieval — secondary types (dev)", SECONDARY_TYPES)):
        L.append(f"## {title}\n")
        L.append("Hit@1 = top-ranked visit is acceptable (ties: any acceptable visit counts). MRR = 1/rank of the first "
                 "acceptable visit. Recall@1 ≤ 1/#acceptable when several visits tie.\n")
        for q in types:
            d = ctx["table"][ctx["table"].qtype == q].set_index("retriever")
            if d.empty:
                continue
            r0 = d.iloc[0]
            L.append(f"\n#### `{q}`: {int(r0.n)} answerable dev questions, {int(r0.eyes)} eyes, "
                     f"{int(r0.patients)} patients; dev-selected fixed rule: **{sel[q]}**\n")
            L.append("| Strategy | Hit@1 [95% CI] | Hit@1 train / val | MRR [95% CI] | Recall@1 | nDCG@3 |")
            L.append("|---|---|---|---|---|---|")
            for name in [NEW, sel[q], "combined", "rule_router", "random"]:
                if name in d.index:
                    r = d.loc[name]
                    label = f"{name} (selected fixed rule)" if name == sel[q] else name
                    L.append(f"| {label} | {f3(r['hit@1'])} [{f3(r.hit1_lo)}, {f3(r.hit1_hi)}] | "
                             f"{f3(r['hit@1_train'])} / {f3(r['hit@1_val'])} | {f3(r.mrr)} [{f3(r.mrr_lo)}, "
                             f"{f3(r.mrr_hi)}] | {f3(r['recall@1'])} | {f3(r['ndcg@3'])} |")

    L.append("\n## 5. Abstention (dev)\n")
    L.append("- *Correct abstention*: on insufficient-evidence (IE) questions the system returns `no_qualifying_visit`.")
    L.append("- *Any abstention on IE*: any non-retrieval outcome.")
    L.append("- *False abstention*: the system abstains although an acceptable visit exists.")
    L.append("- Frozen v1 strategies and fixed rules never abstain: 0 % correct abstention and 0 % false abstention, "
             "by construction.\n")
    L.append("| Type | Answerable | False abstention | IE questions | Correct abstention | Any abstention on IE |")
    L.append("|---|---|---|---|---|---|")
    for r in ctx["abstention"].itertuples():
        L.append(f"| `{r.qtype}` | {r.n_ans} | {r.false_abstain:.1%} | {r.n_ie} | "
                 f"{'–' if pd.isna(r.correct_abstain) else f'{r.correct_abstain:.1%}'} | "
                 f"{'–' if pd.isna(r.any_abstain_ie) else f'{r.any_abstain_ie:.1%}'} |")
    L.append("\nOutcome counts per type (all dev instances):\n")
    L.append("| Type | " + " | ".join(ctx["status_cols"]) + " |")
    L.append("|---|" + "---|" * len(ctx["status_cols"]))
    for q, r in ctx["status"].iterrows():
        L.append(f"| `{q}` | " + " | ".join(str(int(r[c])) for c in ctx["status_cols"]) + " |")

    L.append("\n## 6. v1 question types on dev patients — rule coverage, not evidence\n")
    L.append("| Type | intent_v3 Hit@1 | frozen rule_router Hit@1 |")
    L.append("|---|---|---|")
    for r in ctx["v1cov"].itertuples():
        L.append(f"| `{r.qtype}` | {f3(r.new)} | {f3(r.old)} |")

    L.append("\n## 7. Limitations\n")
    L.extend([
        "- All numbers are development numbers on text the parser was built against. They do not estimate "
        "performance on new phrasings.",
        "- 5 paraphrases per type: parser accuracy is a count, not a rate with a defensible CI.",
        "- `max_leakage` is answered with `unavailable_variable` (the variable is not in memory by design), so its "
        "Hit@1 is 0. This is the intended behaviour unless a proxy policy is approved (decision D1).",
        "- The developer had prior exposure to the locked paraphrases (see `docs/RETRIEVAL_IMPROVEMENT_PLAN.md`); "
        "lexicon terms recognised as locked-only were removed, but contamination cannot be excluded.",
        "- Prime-only types rest on 25–32 dev patients; TREX eyes have no week data, so time questions there "
        "return `insufficient_data` by design.",
    ])
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    verify()
    alpha = json.loads((SNAPSHOT / "run_config.json").read_text())["alpha"]
    ep, _ = build_all()
    dev = dev_episodes(ep)
    heldout = build_heldout_prime(ep)
    inst = generate_instances_v2(dev, heldout, dev_by_type())
    assert set(inst.split) <= set(DEV_SPLITS)
    inst.to_csv(OUT / "instances_dev.csv", index=False)

    memories = build_memories(dev)
    new = run_new_system(inst, memories)
    eyes = eye_histories(dev)
    answerable = inst[inst.answerable]
    base = pd.concat([run_retriever(s, answerable, eyes) for s in strategies(alpha)], ignore_index=True)
    per = pd.concat([new[new.answerable], base], ignore_index=True)
    per.to_csv(OUT / "per_instance_dev.csv", index=False)
    new.to_csv(OUT / "intent_v3_outcomes_dev.csv", index=False)

    # Select the best fixed rule per type on dev (eye-macro Hit@1), then freeze it.
    selected = {}
    for q in TYPES_V2:
        d = base[(base.qtype == q) & base.retriever.isin(FIXED_RULES)]
        scores = {name: eye_macro(g, ["hit@1"])["hit@1"] for name, g in d.groupby("retriever")}
        selected[q] = max(FIXED_RULES, key=lambda n: (scores.get(n, -1), -FIXED_RULES.index(n)))
    (OUT / "selected_baselines.json").write_text(json.dumps(dict(
        selected_fixed_rule=selected, selected_on="dev (train+val patients, dev paraphrases), eye-macro Hit@1",
        candidates=FIXED_RULES), indent=2))

    names = [NEW] + sorted(set(base.retriever))
    table = retrieval_table(per, list(TYPES_V2), names)
    table.to_csv(OUT / "summary_dev.csv", index=False)

    ab = []
    for q in TYPES_V2:
        d = new[new.qtype == q]
        a, ie = d[d.answerable], d[~d.answerable]
        ab.append(dict(qtype=q, n_ans=len(a), false_abstain=a.abstained.mean() if len(a) else np.nan, n_ie=len(ie),
                       correct_abstain=(ie.status == NO_QUALIFYING).mean() if len(ie) else np.nan,
                       any_abstain_ie=ie.abstained.mean() if len(ie) else np.nan))
    abstention = pd.DataFrame(ab)
    abstention.to_csv(OUT / "abstention_dev.csv", index=False)
    status = new.pivot_table(index="qtype", columns="status", values="instance_id", aggfunc="count", fill_value=0)

    parser = parser_table()
    parser.to_csv(OUT / "parser_dev.csv", index=False)

    v1inst = generate_instances(dev)
    v1inst = v1inst.assign(answerable=True, n_relevant=v1inst.relevant_visit_ids.str.count(";") + 1)
    v1new = run_new_system(v1inst, memories)
    v1old = run_retriever(strategies(alpha)[6], v1inst, eyes)   # frozen rule_router
    v1cov = pd.DataFrame([dict(qtype=q, new=eye_macro(v1new[v1new.qtype == q], ["hit@1"])["hit@1"],
                               old=eye_macro(v1old[v1old.qtype == q], ["hit@1"])["hit@1"])
                          for q in QUESTION_TYPES])
    v1cov.to_csv(OUT / "v1_rule_coverage_dev.csv", index=False)

    ctx = dict(parser=parser, table=table, selected=selected, abstention=abstention, status=status,
               status_cols=list(status.columns), v1cov=v1cov, dev_patients=dev.patient_id.nunique(),
               dev_eyes=dev.eye_id.nunique())
    write_report(ctx, config.REPORTS_DIR / "RETRIEVAL_REPORT_V3_DEV.md")
    (OUT / "run_config_v3_dev.json").write_text(json.dumps(dict(
        seed=config.SEED, splits=DEV_SPLITS, paraphrases="dev", alpha_frozen_from_v1=alpha,
        bootstrap_reps=BOOTSTRAP_REPS, temporal_tolerance_weeks=I.TEMPORAL_TOLERANCE_WEEKS,
        n_instances=len(inst), n_answerable=int(inst.answerable.sum())), indent=2))
    show = table[table.retriever.isin([NEW, "combined", "rule_router"]) | table.apply(
        lambda r: r.retriever == selected[r.qtype], axis=1)]
    print(show[["qtype", "retriever", "hit@1", "hit1_lo", "hit1_hi", "mrr", "n", "patients"]].round(3)
          .to_string(index=False))
    print(abstention.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
