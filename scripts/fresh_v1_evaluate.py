"""fresh_v1 step 9: one-shot evaluation harness (results/fresh_v1_prereg/ANALYSIS_PLAN.md).

  python scripts/fresh_v1_evaluate.py --dev-test
      Plumbing test on development data ONLY: dev paraphrases as questions, train+val patients for Part C.
      Never reads sealed/ and never uses test-split patients. Writes results/fresh_v1_harness_devtest/.

  python scripts/fresh_v1_evaluate.py --open-sealed --i-have-explicit-approval
      The one-shot run. Refuses unless the code freeze, gold manifest and this harness's recorded hash all verify,
      and results/fresh_v1/ does not yet exist. Reads sealed/fresh_v1/gold.jsonl; test-split patients for Part C.

Written after the code freeze; imports frozen modules read-only; modifies nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fresh_v1_build_gold import EXPECTED_TASK, TYPE_INTENT, TYPE_ROUTING  # noqa: E402
from oct_retrieval import config  # noqa: E402
from oct_retrieval import intent as I  # noqa: E402
from oct_retrieval.episodes import build_all  # noqa: E402
from oct_retrieval.evaluate import bootstrap_ci, eye_histories, eye_macro, run_retriever  # noqa: E402
from oct_retrieval.evaluate_v2 import strategies  # noqa: E402
from oct_retrieval.evaluate_v3_dev import run_new_system  # noqa: E402
from oct_retrieval.ground_truth_v2 import TYPES_V2, generate_instances_v2  # noqa: E402
from oct_retrieval.intent_retrieval import NO_QUALIFYING, build_memories  # noqa: E402
from oct_retrieval.side_tables import build_heldout_prime  # noqa: E402

PREREG = ROOT / "results" / "fresh_v1_prereg"
SEALED_GOLD = ROOT / "sealed" / "fresh_v1" / "gold.jsonl"
HARNESS_HASH = PREREG / "HARNESS.sha256"
DISCLAIMER = (
    "> **AI-generated preliminary benchmark with limited human review. NOT clinician-validated.** Questions and labels "
    "were produced by language-model agents; a human researcher reviewed only 2 disagreements and a 12-item sample. "
    "The annotator's independence could not be verified, and the original inter-agent agreement results are marked "
    "**potentially compromised** (`results/fresh_v1_prereg/STAGE_LOG.md`). 14 of 120 items were viewed by the "
    "developer during the guided review (after the code freeze). Results do not establish research novelty or "
    "clinical usefulness.")
INTENT_FIELDS = ("operation", "target", "direction", "threshold", "offset_weeks", "tolerance_weeks", "comparator",
                 "recency")
PART_C_SYSTEMS = ("intent_v3", "global_fixed_rule", "per_type_rule_UPPER_REF", "combined", "rule_router")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt_w(k, n):
    p, lo, hi = wilson(k, n)
    return "–" if n == 0 else f"{k}/{n} = {p:.3f} [{lo:.3f}, {hi:.3f}]"


# --------------------------------------------------------------------------------------------- gold inputs
def dev_gold() -> list[dict]:
    """Gold-format items built from the DEV paraphrases (plumbing test only; developer-seen text)."""
    rows = [json.loads(x) for x in (ROOT / "tests" / "data" / "paraphrases_v2_dev.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    gold = []
    for n, r in enumerate(rows, 1):
        q, text = r["qtype"], r["text"]
        if q in TYPE_INTENT:
            cat, typ, routing = "supported", q, TYPE_ROUTING[q]
        elif q == "last_injection":
            cat, typ, routing = "images_or_unavailable", "none", "data_unavailable"
        else:   # dev "unsupported": the one severity-grade question is unavailable data, the rest need images
            cat, typ = "images_or_unavailable", "none"
            routing = "data_unavailable" if "grade" in text.lower() else "requires_images"
        gold.append(dict(id=f"D{n:03d}", text=text, category=cat, routing=routing, intended_type=typ,
                         expected_system_task=EXPECTED_TASK[routing],
                         gold_intent=TYPE_INTENT.get(typ) if routing == "retrieve" else None,
                         developer_viewed=True))
    return gold


def sealed_gold() -> list[dict]:
    return [json.loads(x) for x in SEALED_GOLD.read_text(encoding="utf-8").splitlines() if x.strip()]


def preflight_sealed():
    import fresh_v1
    problems = []
    if not fresh_v1.cmd_verify_freeze():
        problems.append("code freeze")
    for line in (PREREG / "GOLD_MANIFEST.sha256").read_text().splitlines():
        d, p = line.split("  ", 1)
        if sha256(ROOT / p) != d:
            problems.append(f"gold manifest: {p}")
    if not HARNESS_HASH.exists():
        problems.append("HARNESS.sha256 missing")
    else:
        for line in HARNESS_HASH.read_text().splitlines():
            d, p = line.split("  ", 1)
            if sha256(ROOT / p) != d:
                problems.append(f"harness hash: {p}")
    if (ROOT / "results" / "fresh_v1").exists():
        problems.append("results/fresh_v1 already exists (one-shot rule)")
    if problems:
        sys.exit("REFUSING TO OPEN: " + "; ".join(problems))


# ---------------------------------------------------------------------------------------------- Part A
def part_a(gold: list[dict]) -> pd.DataFrame:
    rows = []
    for g in gold:
        q = I.parse_intent(g["text"])
        row = dict(id=g["id"], category=g["category"], intended_type=g["intended_type"], gold_routing=g["routing"],
                   expected_task=g["expected_system_task"], system_task=q.task, developer_viewed=g["developer_viewed"],
                   routing_correct=q.task == g["expected_system_task"],
                   system_retrieves=q.task == I.RETRIEVE, reason=q.reason)
        for f in INTENT_FIELDS:
            row[f"parsed_{f}"] = getattr(q, f, None)
        gi = g.get("gold_intent")
        if gi:
            per = {f: getattr(q, f, None) == v for f, v in gi.items()}
            row.update({f"field_ok_{f}": ok for f, ok in per.items()})
            row["intent_exact"] = all(per.values()) and q.task == I.RETRIEVE
        rows.append(row)
    return pd.DataFrame(rows)


def part_a_summary(a: pd.DataFrame) -> dict:
    no_retrieve = a[a.gold_routing != "retrieve"]
    retrieve = a[a.gold_routing == "retrieve"]
    limit = a[a.gold_routing.isin(["requires_images", "data_unavailable"])]
    r = retrieve[retrieve.intent_exact.notna()] if "intent_exact" in a else retrieve.iloc[0:0]
    out = dict(
        n_items=len(a), n_distinct_texts=int(a.id.nunique()),
        primary_routing_accuracy=(int(a.routing_correct.sum()), len(a)),
        primary_unsafe_retrieval=(int(no_retrieve.system_retrieves.sum()), len(no_retrieve)),
        over_refusal=(int((~retrieve.system_retrieves).sum()), len(retrieve)),
        limitation_exact=(int(limit.routing_correct.sum()), len(limit)),
        limitation_any_refusal=(int((~limit.system_retrieves).sum()), len(limit)),
        intent_exact=(int(r.intent_exact.astype(bool).sum()), len(r)),
        field_accuracy={f: (int(r[f"field_ok_{f}"].dropna().astype(bool).sum()), int(r[f"field_ok_{f}"].notna().sum()))
                        for f in INTENT_FIELDS if f"field_ok_{f}" in r},
    )
    return out


# ---------------------------------------------------------------------------------------------- Part C
def part_c(gold: list[dict], splits: tuple[str, ...]) -> dict:
    alpha = json.loads((config.RESULTS_DIR / "v1_snapshot" / "run_config.json").read_text())["alpha"]
    global_rule = json.loads((PREREG / "global_fixed_rule.json").read_text())["global_fixed_rule"]
    per_type_rule = json.loads((config.RESULTS_DIR / "v3_dev" / "selected_baselines.json").read_text())[
        "selected_fixed_rule"]
    texts = {t: [g["text"] for g in gold if g["routing"] == "retrieve" and g["intended_type"] == t] for t in TYPES_V2}
    types = [t for t in TYPES_V2 if texts[t]]
    ep, _ = build_all()
    sub = ep[ep.split.isin(splits)]
    assert set(sub.split) <= set(splits)
    heldout = build_heldout_prime(ep)
    paraphrases = {t: (texts[t] or ["(unused placeholder)"]) for t in TYPES_V2}
    inst = generate_instances_v2(sub, heldout, paraphrases)
    inst = inst[inst.qtype.isin(types)].reset_index(drop=True)
    new = run_new_system(inst, build_memories(sub))
    ans = inst[inst.answerable]
    strat = {s.name: s for s in strategies(alpha)}
    eyes = eye_histories(sub)
    per = {"intent_v3": new[new.answerable].copy()}
    per["global_fixed_rule"] = run_retriever(strat[global_rule], ans, eyes)
    pt = pd.concat([run_retriever(strat[per_type_rule[t]], ans[ans.qtype == t], eyes) for t in types
                    if (ans.qtype == t).any()], ignore_index=True)
    per["per_type_rule_UPPER_REF"] = pt
    per["combined"] = run_retriever(strat["combined"], ans, eyes)
    per["rule_router"] = run_retriever(strat["rule_router"], ans, eyes)

    table = []
    for t in types:
        for name in PART_C_SYSTEMS:
            d = per[name][per[name].qtype == t]
            if d.empty:
                continue
            m = eye_macro(d, ["hit@1", "mrr"])
            lo, hi = bootstrap_ci(d, "hit@1")
            table.append(dict(qtype=t, system=name, hit1=m["hit@1"], hit1_lo=lo, hit1_hi=hi, mrr=m["mrr"], n=len(d),
                              eyes=d.eye_id.nunique(), patients=d.patient_id.nunique()))
    diffs = []
    g = per["global_fixed_rule"].set_index("instance_id")["hit@1"]
    for t in types:
        d = per["intent_v3"][per["intent_v3"].qtype == t].copy()
        if d.empty:
            continue
        d["diff"] = d["hit@1"].astype(float).values - g.loc[d.instance_id].values
        lo, hi = bootstrap_ci(d, "diff")
        diffs.append(dict(qtype=t, diff=eye_macro(d, ["diff"])["diff"], lo=lo, hi=hi, n=len(d),
                          patients=d.patient_id.nunique()))
    abst = []
    for t in types:
        d = new[new.qtype == t]
        a_, ie = d[d.answerable], d[~d.answerable]
        abst.append(dict(qtype=t, answerable=len(a_), false_abstention=a_.abstained.mean() if len(a_) else np.nan,
                         ie=len(ie), correct_abstention=(ie.status == NO_QUALIFYING).mean() if len(ie) else np.nan))
    return dict(table=pd.DataFrame(table), diffs=pd.DataFrame(diffs), abstention=pd.DataFrame(abst),
                global_rule=global_rule, per_type_rule=per_type_rule, texts_per_type={t: len(texts[t]) for t in types},
                instances=inst, new=new, splits=splits)


# ---------------------------------------------------------------------------------------------- report
def write_report(path: Path, mode: str, gold: list[dict], a: pd.DataFrame, s: dict, c: dict):
    L = [f"# fresh_v1 evaluation report ({mode})\n", DISCLAIMER + "\n"]
    if mode == "DEV-TEST":
        L.append("> **HARNESS PLUMBING TEST ONLY.** Questions are the developer-seen DEV paraphrases and Part C uses "
                 "train+val patients. These numbers are NOT results and must not be reported as such.\n")
    L.append("## 1. Items (unit = distinct formulation)\n")
    L.append(f"- Distinct formulations: **{len({g['text'] for g in gold})}** ({len(gold)} items). By category: "
             + ", ".join(f"{k} {v}" for k, v in Counter(g['category'] for g in gold).items()) + ".")
    L.append("- By gold routing: " + ", ".join(f"{k} {v}" for k, v in Counter(g['routing'] for g in gold).items()) + ".")
    L.append(f"- Developer-viewed items: {sum(g['developer_viewed'] for g in gold)}.\n")
    L.append("## 2. Part A1 — routing and availability (Wilson 95 % CI over formulations)\n")
    L.append("| Endpoint | Result |\n|---|---|")
    L.append(f"| **Primary 1: routing accuracy** | {fmt_w(*s['primary_routing_accuracy'])} |")
    L.append(f"| **Primary 2: unsafe-retrieval rate** (gold ≠ retrieve, system retrieves; lower is better) | "
             f"{fmt_w(*s['primary_unsafe_retrieval'])} |")
    L.append(f"| Secondary: over-refusal rate (gold retrieve, system does not; lower is better) | {fmt_w(*s['over_refusal'])} |")
    L.append(f"| Secondary: correct limitation handling, exact outcome (images / data unavailable) | "
             f"{fmt_w(*s['limitation_exact'])} |")
    L.append(f"| Secondary: limitation items not retrieved (any refusal) | {fmt_w(*s['limitation_any_refusal'])} |")
    L.append("\n**Confusion matrix** (rows: gold routing; columns: system outcome)\n")
    cm = pd.crosstab(a.gold_routing, a.system_task)
    L.append("| gold \\ system | " + " | ".join(cm.columns) + " |")
    L.append("|---|" + "---|" * len(cm.columns))
    for idx, r in cm.iterrows():
        L.append(f"| {idx} | " + " | ".join(str(int(v)) for v in r) + " |")
    L.append("\n**Routing accuracy by category and by developer-viewed stratum (descriptive)**\n")
    L.append("| Stratum | Routing accuracy |\n|---|---|")
    for k, d in a.groupby("category"):
        L.append(f"| category = {k} | {fmt_w(int(d.routing_correct.sum()), len(d))} |")
    for k, d in a.groupby("developer_viewed"):
        L.append(f"| developer_viewed = {k} | {fmt_w(int(d.routing_correct.sum()), len(d))} |")
    L.append("\n## 3. Part A2 — intent accuracy on gold-`retrieve` items (gold parameters from type definitions, "
             "clarification 01 (a))\n")
    L.append(f"- Exact intent: {fmt_w(*s['intent_exact'])}")
    for f, (k, n) in s["field_accuracy"].items():
        L.append(f"- {f}: {fmt_w(k, n)}")
    L.append("\n| Type (descriptive) | Routing correct | Exact intent |\n|---|---|---|")
    for t, d in a[a.category == "supported"].groupby("intended_type"):
        ex = d.intent_exact.dropna().astype(bool) if "intent_exact" in d else pd.Series(dtype=bool)
        L.append(f"| {t} | {fmt_w(int(d.routing_correct.sum()), len(d))} | "
                 f"{fmt_w(int(ex.sum()), len(ex)) if len(ex) else 'n/a (routing only)'} |")
    L.append(f"\n## 4. Part C — visit retrieval (gold-`retrieve` items only; patients: {', '.join(c['splits'])})\n")
    L.append(f"- Global fixed rule (fair baseline): **{c['global_rule']}**. Per-type rule = UPPER REFERENCE (it is given "
             "the true question type). Items with gold routing ≠ retrieve (incl. max_leakage, image questions) are "
             "never scored here.")
    L.append(f"- Distinct question texts per type used: {c['texts_per_type']}. Hit@1 = top-ranked visit acceptable; "
             "an abstention on an answerable instance scores 0. 95 % patient-cluster bootstrap CIs.\n")
    L.append("| Type | System | Hit@1 [95 % CI] | MRR | Instances | Patients |\n|---|---|---|---|---|---|")
    for r in c["table"].itertuples():
        L.append(f"| {r.qtype} | {r.system} | {r.hit1:.3f} [{r.hit1_lo:.3f}, {r.hit1_hi:.3f}] | {r.mrr:.3f} | {r.n} | "
                 f"{r.patients} |")
    L.append("\n**Paired difference: intent_v3 − global fixed rule (Hit@1)**\n")
    L.append("| Type | Difference [95 % CI] | Instances | Patients |\n|---|---|---|---|")
    for r in c["diffs"].itertuples():
        L.append(f"| {r.qtype} | {r.diff:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] | {r.n} | {r.patients} |")
    L.append("\n**Abstention (intent_v3; fixed rules cannot abstain)**\n")
    L.append("| Type | Answerable | False abstention | IE instances | Correct abstention |\n|---|---|---|---|---|")
    for r in c["abstention"].itertuples():
        fa = "–" if pd.isna(r.false_abstention) else f"{r.false_abstention:.3f}"
        ca = "–" if pd.isna(r.correct_abstention) else f"{r.correct_abstention:.3f}"
        L.append(f"| {r.qtype} | {r.answerable} | {fa} | {r.ie} | {ca} |")
    L.append("\n## 5. Known issues and limitations\n")
    L.extend([
        "- KI-01 (`results/fresh_v1_prereg/KNOWN_ISSUES.md`): the retriever does not enforce the ≥ 2-earlier-visits "
        "rule; benchmark instances always have ≥ 2 earlier visits, so it cannot affect Part C.",
        "- With ~10 formulations per type, per-type figures are descriptive; inference uses pooled A1 endpoints.",
        "- Part C rests on the test split (18 patients; 7–8 for Prime-only types); intervals are wide.",
        "- Questions were written by an AI agent from the developer's definitions; real clinician wording may differ.",
    ])
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dev-test", action="store_true")
    g.add_argument("--open-sealed", action="store_true")
    ap.add_argument("--i-have-explicit-approval", action="store_true")
    args = ap.parse_args()

    if args.open_sealed:
        if not args.i_have_explicit_approval:
            sys.exit("REFUSING: --open-sealed requires --i-have-explicit-approval (researcher's explicit go-ahead).")
        preflight_sealed()
        gold, splits, out, mode = sealed_gold(), ("test",), ROOT / "results" / "fresh_v1", "SEALED, ONE-SHOT"
        report = ROOT / "reports" / "FRESH_V1_REPORT.md"
    else:
        gold, splits, out, mode = dev_gold(), ("train", "val"), ROOT / "results" / "fresh_v1_harness_devtest", "DEV-TEST"
        report = out / "DEVTEST_REPORT.md"
        assert not any("sealed" in str(p) for p in (out, report))
    out.mkdir(parents=True, exist_ok=False if args.open_sealed else True)

    a = part_a(gold)
    s = part_a_summary(a)
    c = part_c(gold, splits)
    a.to_csv(out / "part_a_items.csv", index=False)
    json.dump({k: v for k, v in s.items()}, open(out / "part_a_summary.json", "w"), indent=2, default=str)
    c["table"].to_csv(out / "part_c_table.csv", index=False)
    c["diffs"].to_csv(out / "part_c_paired_diff.csv", index=False)
    c["abstention"].to_csv(out / "part_c_abstention.csv", index=False)
    c["new"].to_csv(out / "part_c_intent_v3_outcomes.csv", index=False)
    write_report(report, mode, gold, a, s, c)
    print(json.dumps(dict(mode=mode, out=str(out.relative_to(ROOT)), report=str(report.relative_to(ROOT)),
                          **{k: v for k, v in s.items() if k != "field_accuracy"}), indent=2, default=str))


if __name__ == "__main__":
    main()
