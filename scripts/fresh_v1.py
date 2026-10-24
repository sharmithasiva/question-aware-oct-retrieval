"""fresh_v1 protocol utilities (docs/FRESH_EVALUATION_PROTOCOL.md).

  python scripts/fresh_v1.py freeze         step 1: verify checksums, pick global fixed rule, write CODE_FREEZE
  python scripts/fresh_v1.py verify-freeze  check that no frozen file changed
  python scripts/fresh_v1.py seal-info      counts / ids / schema validity / hashes of sealed files (never text)
  python scripts/fresh_v1.py compare        agreement summary (counts + ids, safe for the developer) and the
                                            user-only disagreement report + review sample (these contain text)

Nothing in this script prints question text to stdout.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import platform
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PREREG = ROOT / "results" / "fresh_v1_prereg"
SEALED = ROOT / "sealed" / "fresh_v1"
FREEZE = PREREG / "CODE_FREEZE.sha256"
SEED = 42

CATEGORIES = {"supported", "plausible_unsupported", "images_or_unavailable", "ambiguous_underspecified_offtopic"}
TYPES = {"largest_cst_rise", "largest_bcva_drop", "first_vision_gain_1line", "one_year_ago", "six_months_ago",
         "max_leakage", "last_thicker_than_now", "last_vision_as_good", "none"}
ROUTING = {"retrieve", "data_unavailable", "requires_images", "unsupported_operation", "needs_clarification"}
ANSWERABILITY = {"metadata_answerable", "needs_unavailable_data", "needs_images", "not_answerable_as_posed"}
FIELDS = {"cst", "bcva", "visit_order", "week", "leakage_index", "injection", "severity_grade", "oct_images",
          "biomarker_labels", "other"}
QUOTA = {"supported": 80, "plausible_unsupported": 15, "images_or_unavailable": 15,
         "ambiguous_underspecified_offtopic": 10}
COMPARE_FIELDS = ("category", "routing", "intended_type", "answerability")
FIXED_RULES = ["latest_prior", "earliest_first", "h_second_visit", "h_max_cst", "h_min_cst", "h_max_bcva",
               "h_min_bcva"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------------------------- freeze
def verify_existing_checksums() -> list[str]:
    problems = []
    snap = ROOT / "results" / "v1_snapshot"
    for line in (snap / "MANIFEST.md5").read_text().splitlines():
        digest, name = line.split(" *", 1)
        if md5(snap / name) != digest:
            problems.append(f"v1_snapshot/{name}")
    for line in (ROOT / "results" / "v3_dev" / "frozen_artifacts.md5").read_text().splitlines():
        digest, path = line.split(" *", 1)
        if md5(ROOT / path) != digest:
            problems.append(path)
    return problems


def frozen_files() -> list[Path]:
    files = sorted((ROOT / "src" / "oct_retrieval").glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    files += [ROOT / "tests" / "data" / "paraphrases_v2_dev.jsonl", ROOT / "tests" / "data" / "paraphrases_v2_manifest.json",
              ROOT / "pyproject.toml", ROOT / "scripts" / "fresh_v1.py", ROOT / "docs" / "FRESH_EVALUATION_PROTOCOL.md",
              PREREG / "ANALYSIS_PLAN.md", PREREG / "global_fixed_rule.json", PREREG / "environment.json"]
    files += sorted((ROOT / "results" / "v3_dev").glob("*"))
    return files


def select_global_rule() -> dict:
    """Single type-agnostic fixed rule: highest mean (over the 7 retrievable types, equal weight) of dev eye-macro
    Hit@1. max_leakage is excluded because its gold routing is data_unavailable under the restricted input."""
    import pandas as pd
    per = pd.read_csv(ROOT / "results" / "v3_dev" / "per_instance_dev.csv", low_memory=False)
    per = per[per.retriever.isin(FIXED_RULES) & (per.qtype != "max_leakage")]
    by_type = (per.groupby(["retriever", "qtype", "eye_id"])["hit@1"].mean()
               .groupby(["retriever", "qtype"]).mean().unstack())
    score = by_type.mean(axis=1)
    best = max(FIXED_RULES, key=lambda r: (score.get(r, -1), -FIXED_RULES.index(r)))
    return dict(global_fixed_rule=best, criterion="max mean over 7 retrievable types of dev eye-macro Hit@1",
                data="results/v3_dev/per_instance_dev.csv (train+val patients, dev paraphrases)",
                scores={r: round(float(score[r]), 4) for r in FIXED_RULES},
                per_type={r: {q: round(float(v), 4) for q, v in by_type.loc[r].items()} for r in FIXED_RULES})


def environment() -> dict:
    import numpy
    import openpyxl
    import pandas
    import pytest
    return dict(python=sys.version.split()[0], platform=platform.platform(), pandas=pandas.__version__,
                numpy=numpy.__version__, openpyxl=openpyxl.__version__, pytest=pytest.__version__)


def cmd_freeze():
    if FREEZE.exists():
        sys.exit(f"{FREEZE} already exists; the freeze is final. Use verify-freeze.")
    problems = verify_existing_checksums()
    if problems:
        sys.exit("STOP: existing checksums failed: " + ", ".join(problems))
    PREREG.mkdir(parents=True, exist_ok=True)
    (PREREG / "global_fixed_rule.json").write_text(json.dumps(select_global_rule(), indent=2))
    (PREREG / "environment.json").write_text(json.dumps(environment(), indent=2))
    lines = [f"{sha256(p)}  {p.relative_to(ROOT).as_posix()}" for p in frozen_files()]
    FREEZE.write_text("\n".join(lines) + "\n")
    print(f"frozen {len(lines)} files -> {FREEZE.relative_to(ROOT).as_posix()}")
    print(f"CODE_FREEZE.sha256 = {sha256(FREEZE)}")
    print("global fixed rule:", json.loads((PREREG / "global_fixed_rule.json").read_text())["global_fixed_rule"])


def cmd_verify_freeze() -> bool:
    bad = []
    for line in FREEZE.read_text().splitlines():
        digest, path = line.split("  ", 1)
        if sha256(ROOT / path) != digest:
            bad.append(path)
    bad += verify_existing_checksums()
    print("freeze OK" if not bad else "FREEZE BROKEN: " + ", ".join(bad))
    return not bad


# ------------------------------------------------------------------------------------------ seal info
def validate(labels: list[dict], ids: set[str]) -> Counter:
    issues = Counter()
    for r in labels:
        if r.get("id") not in ids:
            issues["unknown_id"] += 1
        for f, allowed in (("category", CATEGORIES), ("intended_type", TYPES), ("routing", ROUTING),
                           ("answerability", ANSWERABILITY)):
            if r.get(f) not in allowed:
                issues[f"invalid_{f}"] += 1
        if not set(r.get("required_fields") or []) <= FIELDS:
            issues["invalid_required_fields"] += 1
    if len({r.get("id") for r in labels}) != len(labels):
        issues["duplicate_id"] += 1
    if {r.get("id") for r in labels} != ids:
        issues["id_set_mismatch"] += 1
    return issues


def cmd_seal_info():
    out = {}
    q = SEALED / "questions.jsonl"
    if q.exists():
        qs = read_jsonl(q)
        ids = {r["id"] for r in qs}
        out["questions"] = dict(n=len(qs), distinct_ids=len(ids), distinct_texts=len({r["text"] for r in qs}),
                                keys=sorted({k for r in qs for k in r}), sha256=sha256(q))
        for name in ("generator_labels", "annotator_labels"):
            p = SEALED / f"{name}.jsonl"
            if p.exists():
                lab = read_jsonl(p)
                out[name] = dict(n=len(lab), sha256=sha256(p), schema_issues=dict(validate(lab, ids)),
                                 category_counts=dict(Counter(r.get("category") for r in lab)),
                                 type_counts=dict(Counter(r.get("intended_type") for r in lab)),
                                 routing_counts=dict(Counter(r.get("routing") for r in lab)))
    print(json.dumps(out, indent=2))


# --------------------------------------------------------------------------------------------- compare
def kappa(a: list, b: list) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", t.lower()).strip()


def _near(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio() >= 0.9


def cmd_compare():
    import numpy as np
    from oct_retrieval.paraphrases import DEV, read_jsonl as rj
    from oct_retrieval.questions import HELDOUT_PARAPHRASES, QUESTION_TYPES

    qs = {r["id"]: r["text"] for r in read_jsonl(SEALED / "questions.jsonl")}
    gen = {r["id"]: r for r in read_jsonl(SEALED / "generator_labels.jsonl")}
    ann = {r["id"]: r for r in read_jsonl(SEALED / "annotator_labels.jsonl")}
    ids = sorted(qs)
    common = [i for i in ids if i in gen and i in ann]

    agree = {f: [gen[i].get(f) == ann[i].get(f) for i in common] for f in COMPARE_FIELDS}
    fields_agree = [set(gen[i].get("required_fields") or []) == set(ann[i].get("required_fields") or []) for i in common]
    params_agree = [(gen[i].get("parameters") or {}) == (ann[i].get("parameters") or {}) for i in common]
    disagree_ids = [i for i in common if not all(gen[i].get(f) == ann[i].get(f) for f in COMPARE_FIELDS)]

    dev_texts = [r["text"] for r in rj(DEV)] + [t for _, ts in QUESTION_TYPES.values() for t in ts] + \
                [t for t, _ in HELDOUT_PARAPHRASES]
    near_internal = sorted({tuple(sorted((a, b))) for a in ids for b in ids if a < b and _near(qs[a], qs[b])})
    overlap_dev = sorted(i for i in ids if any(_near(qs[i], d) for d in dev_texts))

    agreed = [i for i in common if i not in disagree_ids]
    rng = np.random.default_rng(SEED)
    sample = []
    for t in sorted(TYPES - {"none"}):
        pool = [i for i in agreed if gen[i]["category"] == "supported" and gen[i]["intended_type"] == t]
        if pool:
            sample.append(str(rng.choice(pool)))
    for c in sorted(CATEGORIES - {"supported"}):
        pool = [i for i in agreed if gen[i]["category"] == c and i not in sample]
        if pool:
            sample.append(str(rng.choice(pool)))
    rest = [i for i in agreed if i not in sample]
    if rest:
        sample.append(str(rng.choice(rest)))

    summary = dict(
        n_questions=len(ids), n_labelled_by_both=len(common),
        agreement_rate={f: round(float(np.mean(v)), 3) for f, v in agree.items()},
        cohen_kappa={f: round(kappa([gen[i].get(f) for i in common], [ann[i].get(f) for i in common]), 3)
                     for f in ("category", "routing", "intended_type")},
        required_fields_exact_agreement=round(float(np.mean(fields_agree)), 3),
        parameters_exact_agreement=round(float(np.mean(params_agree)), 3),
        n_disagreements=len(disagree_ids), disagreement_ids=disagree_ids,
        disagreements_by_field={f: int(len(common) - sum(v)) for f, v in agree.items()},
        quota_generator=dict(Counter(gen[i]["category"] for i in ids)),
        quota_annotator=dict(Counter(ann[i]["category"] for i in ids if i in ann)),
        quota_target=QUOTA,
        supported_type_counts_generator=dict(Counter(gen[i]["intended_type"] for i in ids
                                                     if gen[i]["category"] == "supported")),
        near_duplicate_pairs=[list(p) for p in near_internal], overlap_with_dev_or_v1=overlap_dev,
        review_sample_ids=sample,
        note="Agreement between two AI agents from the same model family is not evidence of clinical correctness.",
    )
    (SEALED / "agreement_summary.json").write_text(json.dumps(summary, indent=2))

    def block(i):
        g, a = gen[i], ann[i]
        rows = "\n".join(f"| {f} | {g.get(f)} | {a.get(f)} |" for f in
                         COMPARE_FIELDS + ("required_fields", "parameters", "rationale"))
        return f"### {i}\n\n> {qs[i]}\n\n| Field | Generator | Annotator |\n|---|---|---|\n{rows}\n"

    header = ("> AI-generated preliminary benchmark with limited human review. Not clinician-validated.\n"
              "> USER-ONLY FILE: contains question text. The developer must not open it.\n\n")
    (SEALED / "disagreement_report.md").write_text(
        "# fresh_v1 — disagreement report\n\n" + header +
        "For each item record ONE decision in `adjudication.csv` as `id,decision` with decision in "
        "{generator, annotator, ambiguous, drop}.\n\n" + "\n".join(block(i) for i in disagree_ids), encoding="utf-8")
    (SEALED / "review_sample.md").write_text(
        "# fresh_v1 — review sample (agreed items)\n\n" + header +
        "For each item record `id,decision` in `adjudication.csv` with decision in {ok, ambiguous, wrong, drop}. "
        "If more than 2 of these are `wrong`, labelling is redone and the set is not opened.\n\n" +
        "\n".join(block(i) for i in sample), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"freeze": cmd_freeze, "verify-freeze": cmd_verify_freeze, "seal-info": cmd_seal_info,
     "compare": cmd_compare}.get(cmd, lambda: sys.exit(__doc__))()
