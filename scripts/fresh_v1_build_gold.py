"""fresh_v1 step 6: build and hash the gold set (docs/FRESH_EVALUATION_PROTOCOL.md §2, step 6).

  python scripts/fresh_v1_build_gold.py

Written AFTER the code freeze. It only imports frozen code (read-only) and does not modify any frozen file.
Gold rules:
  - agreed items: the shared category / routing / intended_type / answerability;
  - disagreements: the researcher's adjudication (generator | annotator | ambiguous | drop);
  - review-sample decisions: ok keeps, ambiguous moves to needs_clarification, drop excludes;
  - protocol exclusions: near-duplicates and dev/v1 overlap (from agreement_summary.json);
  - gold intent parameters: from the frozen type definitions (PROPOSED_CLARIFICATION_01, option (a), approved).
Prints counts, IDs and hashes only; never question text.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fresh_v1 import COMPARE_FIELDS, SEALED, read_jsonl, sha256  # noqa: E402  (frozen, read-only)

PREREG = ROOT / "results" / "fresh_v1_prereg"
GOLD = SEALED / "gold.jsonl"
GOLD_MANIFEST = PREREG / "GOLD_MANIFEST.sha256"

# Routing -> accepted system outcome (ANALYSIS_PLAN.md, "Routing mapping (fixed)").
EXPECTED_TASK = {"retrieve": "retrieve", "data_unavailable": "unavailable_variable",
                 "requires_images": "unsupported_image", "needs_clarification": "clarify",
                 "unsupported_operation": "clarify"}

# Gold intent per type (PROPOSED_CLARIFICATION_01 option (a); frozen definitions in PHASE1_V2_QUESTION_SPEC.md).
TYPE_INTENT = {
    "largest_cst_rise": dict(operation="max_consecutive_change", target="cst", direction="increase"),
    "largest_bcva_drop": dict(operation="max_consecutive_change", target="bcva", direction="decrease"),
    "first_vision_gain_1line": dict(operation="first_threshold_crossing", target="bcva", direction="increase",
                                    threshold=5.0),
    "one_year_ago": dict(operation="time_offset", offset_weeks=52, tolerance_weeks=4),
    "six_months_ago": dict(operation="time_offset", offset_weeks=26, tolerance_weeks=4),
    "last_thicker_than_now": dict(operation="compare_to_current", target="cst", comparator="gt", recency="last"),
    "last_vision_as_good": dict(operation="compare_to_current", target="bcva", comparator="ge", recency="last"),
    "max_leakage": None,   # gold routing data_unavailable under the restricted input (D1); scored in A1 only
}
TYPE_ROUTING = {t: ("data_unavailable" if t == "max_leakage" else "retrieve") for t in TYPE_INTENT}
DEVELOPER_VIEWED = {"F065", "F080", "F002", "F089", "F082", "F059", "F056", "F097", "F011", "F075", "F017", "F014",
                    "F060", "F118"}


def main():
    if GOLD.exists() or GOLD_MANIFEST.exists():
        sys.exit("gold set already built; it is final. Delete deliberately to rebuild.")
    qs = {r["id"]: r["text"] for r in read_jsonl(SEALED / "questions.jsonl")}
    gen = {r["id"]: r for r in read_jsonl(SEALED / "generator_labels.jsonl")}
    ann = {r["id"]: r for r in read_jsonl(SEALED / "annotator_labels.jsonl")}
    summary = json.loads((SEALED / "agreement_summary.json").read_text())
    adj = {r["id"]: r for r in csv.DictReader(open(SEALED / "adjudication.csv", newline="", encoding="utf-8"))}
    assert all(r["decision"] for r in adj.values()), "adjudication incomplete"

    excluded = {}
    for pair in summary["near_duplicate_pairs"]:
        excluded[pair[1]] = "near_duplicate"
    for i in summary["overlap_with_dev_or_v1"]:
        excluded[i] = "overlap_dev_or_v1"

    gold, notes = [], Counter()
    for i in sorted(qs):
        g, a = gen[i], ann[i]
        agreed = all(g.get(f) == a.get(f) for f in COMPARE_FIELDS)
        decision = adj.get(i, {}).get("decision", "")
        if decision == "drop":
            excluded[i] = "researcher_drop"
        if i in excluded:
            continue
        if agreed:
            src, base = "agreed", g
        elif decision in ("generator", "annotator"):
            src, base = f"adjudicated_{decision}", (g if decision == "generator" else a)
        elif decision == "ambiguous":
            src, base = "adjudicated_ambiguous", None
        else:
            sys.exit(f"{i}: disagreement without a valid decision")
        if decision == "ambiguous" or base is None:
            lab = dict(category="ambiguous_underspecified_offtopic", routing="needs_clarification",
                       intended_type="none", answerability="not_answerable_as_posed")
            notes["moved_to_needs_clarification"] += 1
        else:
            lab = {f: base[f] for f in COMPARE_FIELDS}
        intent = TYPE_INTENT.get(lab["intended_type"]) if lab["routing"] == "retrieve" else None
        consistent = (lab["intended_type"] == "none") == (lab["category"] != "supported") and \
            (lab["intended_type"] == "none" or lab["routing"] == TYPE_ROUTING[lab["intended_type"]])
        if not consistent:
            notes["inconsistent_" + i] += 1
        gold.append(dict(
            id=i, text=qs[i], **lab, expected_system_task=EXPECTED_TASK[lab["routing"]], gold_intent=intent,
            gold_source=src, review_decision=decision or None, developer_viewed=i in DEVELOPER_VIEWED,
            consistent_with_type_definition=consistent,
            agent_required_fields=dict(generator=g.get("required_fields"), annotator=a.get("required_fields")),
            agent_parameters=dict(generator=g.get("parameters"), annotator=a.get("parameters")),
        ))

    GOLD.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in gold), encoding="utf-8")
    files = ["questions.jsonl", "generator_labels.jsonl", "annotator_labels.jsonl", "agreement_summary.json",
             "disagreement_report.md", "review_sample.md", "adjudication.csv", "gold.jsonl"]
    lines = [f"{sha256(SEALED / f)}  sealed/fresh_v1/{f}" for f in files]
    lines += [f"{sha256(PREREG / f)}  results/fresh_v1_prereg/{f}"
              for f in ("CODE_FREEZE.sha256", "PROPOSED_CLARIFICATION_01.md")]
    lines.append(f"{sha256(Path(__file__))}  scripts/{Path(__file__).name}")
    GOLD_MANIFEST.write_text("\n".join(lines) + "\n")

    out = dict(
        n_gold=len(gold), excluded=excluded,
        by_category=dict(Counter(r["category"] for r in gold)),
        by_routing=dict(Counter(r["routing"] for r in gold)),
        by_expected_system_task=dict(Counter(r["expected_system_task"] for r in gold)),
        supported_by_type=dict(Counter(r["intended_type"] for r in gold if r["category"] == "supported")),
        retrieve_items_with_gold_intent=sum(r["gold_intent"] is not None for r in gold),
        by_gold_source=dict(Counter(r["gold_source"] for r in gold)),
        developer_viewed=sum(r["developer_viewed"] for r in gold),
        inconsistent_ids=[r["id"] for r in gold if not r["consistent_with_type_definition"]],
        notes=dict(notes), gold_sha256=sha256(GOLD), gold_manifest_sha256=sha256(GOLD_MANIFEST),
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
