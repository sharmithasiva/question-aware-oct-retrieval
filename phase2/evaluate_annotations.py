"""Evaluate frozen Tier B rankings against completed BLINDED human judgments (protocol: annotations/ANNOTATION_PROTOCOL_v2.md).

Join: completed sheet --(query_code, candidate_code)--> key --(query_visit_id, candidate_visit_id)--> frozen rankings
(phase2/freeze_rankings.py; SHA-256 verified). Metric maths is reused from question_eval.py (score_ranking,
random_expectation) so Tier A and Tier B use identical definitions.

Label handling (protocol §5): primary relevance = grade 2, secondary = grade >= 1, nDCG gains 2^g - 1.
U (uncertain) and X (ungradable) candidates are REMOVED from the pool for every system (primary) or scored 0
(pessimistic sensitivity); never silently 0 in the primary analysis. Blank -> NOT EVALUATED.

Outputs go to a NEW folder phase2/results/question_eval_v2/<set>/eval_<B1|B2>_<sheet sha8>_<rankings id>/; an existing
folder is never overwritten.

Usage:
  python phase2/evaluate_annotations.py --set dev --category B1 --sheet phase2/annotations/v2/dev/B1/completed_A1.csv
  [--second-sheet <another annotator's completed csv>]  [--rankings <frozen rankings dir>]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase2"))
import question_eval as QE  # noqa: E402
from freeze_rankings import OUT_ROOT, git_head, sha256  # noqa: E402
from make_annotation_packet_v2 import CATEGORIES, LABELS, SHEET_COLUMNS  # noqa: E402

GRADES = {"0": 0.0, "1": 1.0, "2": 2.0}
REF_SYSTEMS = ("S1_latest_prior", "S2p_metadata_proxy_cst")
PRIMARY = "ndcg5"
METRIC_NAMES = {"hit1": "Hit@1", "hit5": "Hit@5", "recall5": "Recall@5", "mrr": "MRR", "ndcg5": "nDCG@5"}


# --- loading and joining ---------------------------------------------------------------------------------------

def load_sheet(path: Path) -> pd.DataFrame:
    s = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in ("query_code", "candidate_code", "label", "query_gradable") if c not in s.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    s["label"] = s.label.str.strip().str.upper()
    s["query_gradable"] = s.query_gradable.str.strip().str.lower()
    bad = ~s.label.isin(list(LABELS) + [""])
    if bad.any():
        raise ValueError(f"{path}: labels must be one of {LABELS} or blank; got {sorted(set(s.label[bad]))}")
    badq = ~s.query_gradable.isin(["yes", "no", ""])
    if badq.any():
        raise ValueError(f"{path}: query_gradable must be yes/no/blank")
    if s.duplicated(["query_code", "candidate_code"]).any():
        raise ValueError(f"{path}: duplicate (query_code, candidate_code) rows")
    for c in ("annotator_id", "annotator_role", "rationale"):
        if c not in s.columns:
            s[c] = ""
    return s


def join_sheet_key(sheet: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """One row per judgment with visit identities. Sheet and key must cover exactly the same (query, candidate) pairs."""
    k = key.astype({"query_code": str, "candidate_code": str})
    sk, kk = set(zip(sheet.query_code, sheet.candidate_code)), set(zip(k.query_code, k.candidate_code))
    if sk != kk:
        raise ValueError(f"sheet/key mismatch: {len(sk - kk)} sheet rows not in key, {len(kk - sk)} key rows missing "
                         "from sheet")
    j = sheet.merge(k, on=["query_code", "candidate_code"], how="inner", validate="one_to_one",
                    suffixes=("", "_key"))
    if "packet_key" in j and (j.packet != j.packet_key).any():
        raise ValueError("sheet rows belong to a different packet than the key")
    return j


def load_rankings(rdir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    man = json.loads((rdir / "MANIFEST.json").read_text())
    if sha256(rdir / "rankings.csv") != man["rankings_sha256"]:
        raise ValueError(f"{rdir}: rankings.csv does not match its manifest hash (frozen rankings were modified)")
    return pd.read_csv(rdir / "rankings.csv"), pd.read_csv(rdir / "systems.csv"), man


def default_rankings_dir(set_name: str) -> Path:
    cands = sorted((OUT_ROOT / set_name).glob("rankings_tierB_*"))
    if len(cands) != 1:
        raise SystemExit(f"expected exactly one frozen Tier B rankings folder, found {len(cands)}; pass --rankings")
    return cands[0]


def check_pools(judg: pd.DataFrame, rankings: pd.DataFrame) -> None:
    """Every ranking of a query must be a permutation of exactly the judged candidate pool (no leakage, no gaps)."""
    pools = judg.groupby("query_id").candidate_visit_id.apply(set).to_dict()
    for (qid, s), g in rankings[rankings.query_id.isin(pools)].groupby(["query_id", "system"]):
        if set(g.candidate_visit_id) != pools[qid] or len(g) != len(pools[qid]):
            raise ValueError(f"{qid}/{s}: frozen ranking does not match the judged candidate pool")
        if sorted(g["rank"]) != list(range(1, len(g) + 1)):
            raise ValueError(f"{qid}/{s}: ranks are not 1..n")
    missing = set(pools) - set(rankings.query_id)
    if missing:
        raise ValueError(f"no frozen rankings for queries {sorted(missing)[:3]}")


# --- per-query evaluation --------------------------------------------------------------------------------------

def query_status(g: pd.DataFrame, mode: str, thr: float) -> tuple[str, dict]:
    """(status, grade dict over the evaluation pool) for one query's judgments."""
    if (g.query_gradable == "no").any():
        return "excluded_query_ungradable", {}
    if (g.label == "").any():
        return "not_evaluated_missing_labels", {}
    if (g.query_source_sha256 == g.candidate_source_sha256).any():
        return "excluded_duplicate_image", {}
    if mode == "primary":
        pool = g[g.label.isin(GRADES)]
        grade = {v: GRADES[l] for v, l in zip(pool.candidate_visit_id, pool.label)}
    else:  # pessimistic: U and X scored 0
        grade = {v: GRADES.get(l, 0.0) for v, l in zip(g.candidate_visit_id, g.label)}
    if len(grade) < 2:
        return "excluded_too_few_gradable", grade
    if not any(x >= thr for x in grade.values()):
        return "no_relevant_candidate", grade
    return "evaluated", grade


def evaluate(judg: pd.DataFrame, rankings: pd.DataFrame, systems: pd.DataFrame, mode="primary",
             thr=2.0) -> pd.DataFrame:
    ranked = {k: list(g.sort_values("rank").candidate_visit_id) for k, g in rankings.groupby(["query_id", "system"])}
    sysnames = sorted(systems.system.unique())
    rows = []
    for qid, g in judg.groupby("query_id", sort=True):
        status, grade = query_status(g, mode, thr)
        base = dict(query_id=qid, query_code=g.query_code.iloc[0], eye_id=int(g.eye_id.iloc[0]),
                    query_visit_id=g.query_visit_id.iloc[0], status=status, n_candidates=len(g),
                    n_pool=len(grade), n_U=int((g.label == "U").sum()), n_X=int((g.label == "X").sum()),
                    n_relevant=int(sum(x >= thr for x in grade.values())))
        for s in sysnames:
            if status != "evaluated":
                rows.append(dict(base, system=s))
                continue
            full = ranked.get((qid, s))
            rk = None if full is None else [v for v in full if v in grade]   # same filtering for every system
            rows.append(dict(base, system=s, **QE.score_ranking(rk, grade, thr)))
        if status == "evaluated":
            rows.append(dict(base, system="random_expected", answered=True, **QE.random_expectation(grade, thr)))
    return pd.DataFrame(rows)


def summarise(per: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out, eyes = [], []
    for s, g in per.groupby("system", sort=False):
        ev = g[g.status == "evaluated"]
        row = dict(system=s, n_queries=len(g), n_evaluated=len(ev),
                   **{f"n_{k}": int(v) for k, v in g.status.value_counts().items() if k != "evaluated"})
        if ev.empty:
            row.update({m: "NOT EVALUATED" for m in METRIC_NAMES})
            out.append(row)
            continue
        row["coverage"] = round(float(ev.answered.astype(bool).mean()), 4)
        for m in METRIC_NAMES:
            row[m] = round(float(ev[m].mean()), 4)
        pe = ev.groupby("eye_id")[PRIMARY].mean()
        row["n_eyes"] = int(pe.size)
        ci = QE.bootstrap_ci(pe)
        row[f"{PRIMARY}_ci95_eye_bootstrap"] = "not reported (n_eyes < 5)" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        for ref in REF_SYSTEMS:
            r = per[(per.system == ref) & (per.status == "evaluated")].set_index("query_id")[PRIMARY]
            if s == ref or r.empty or s == "random_expected":
                continue
            d = ev.set_index("query_id")[PRIMARY] - r
            de = d.groupby(ev.set_index("query_id").eye_id).mean()
            row[f"{PRIMARY}_minus_{ref}"] = round(float(d.mean()), 4)
            row[f"eyes_better_equal_worse_vs_{ref}"] = \
                f"{int((de > 1e-12).sum())}/{int((de.abs() <= 1e-12).sum())}/{int((de < -1e-12).sum())}"
        out.append(row)
        for e, ge in ev.groupby("eye_id"):
            eyes.append(dict(system=s, eye_id=int(e), n_queries=len(ge),
                             **{m: round(float(ge[m].mean()), 4) for m in METRIC_NAMES}))
    return pd.DataFrame(out), pd.DataFrame(eyes)


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Quadratic-weighted Cohen's kappa on items both annotators graded 0/1/2; agreement on grade 2; U/X counts."""
    m = a.merge(b, on=["query_code", "candidate_code"], suffixes=("_a", "_b"))
    both = m[m.label_a.isin(GRADES) & m.label_b.isin(GRADES)]
    res = dict(n_common_rows=len(m), n_both_graded=len(both),
               n_U_or_X_disagreement=int(((m.label_a.isin(["U", "X"])) != (m.label_b.isin(["U", "X"]))).sum()))
    if len(both) == 0:
        return dict(res, weighted_kappa=None, pct_agree_grade2=None)
    x, y = both.label_a.map(GRADES).astype(int).to_numpy(), both.label_b.map(GRADES).astype(int).to_numpy()
    O = np.zeros((3, 3))
    np.add.at(O, (x, y), 1)
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    Wt = np.array([[(i - j) ** 2 / 4 for j in range(3)] for i in range(3)])
    denom = (Wt * E).sum()
    kappa = None if denom == 0 else float(1 - (Wt * O).sum() / denom)
    return dict(res, weighted_kappa=None if kappa is None else round(kappa, 4),
                pct_agree_grade2=round(float(((x == 2) == (y == 2)).mean()), 4),
                pct_exact_agree=round(float((x == y).mean()), 4))


# --- report ----------------------------------------------------------------------------------------------------

def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    cols = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def report(cfg: dict, counts: dict, primary: pd.DataFrame, eyes: pd.DataFrame, sens: pd.DataFrame,
           second: pd.DataFrame, agree: dict | None) -> str:
    cols = ["system", "n_evaluated", "coverage"] + list(METRIC_NAMES) + \
        [f"{PRIMARY}_minus_{r}" for r in REF_SYSTEMS] + [f"eyes_better_equal_worse_vs_{r}" for r in REF_SYSTEMS] + \
        [f"{PRIMARY}_ci95_eye_bootstrap"]
    ev = int(primary.n_evaluated.max()) if len(primary) else 0
    head = "**NOT EVALUATED: no complete, gradable query with a relevant candidate.**" if ev == 0 else \
        f"**{ev} evaluated queries.** Exploratory (development eyes); no significance testing."
    warn = ""
    if cfg["category"] == "B2" and not set(counts["annotator_roles"]) & {"ophthalmologist", "optometrist",
                                                                         "certified_grader"}:
        warn = ("\n> **B2 graded without a trained OCT grader**: a workflow pilot only, NOT clinical relevance.\n")
    parts = [
        f"# Tier B evaluation: {cfg['set']} / {cfg['category']} ({cfg['category_name']})", "",
        f"Run {cfg['run_id']}, {cfg['created_utc']}. Protocol: `phase2/annotations/ANNOTATION_PROTOCOL_v2.md`.",
        warn, head, "",
        "## Inputs", "", "```json", json.dumps({k: cfg[k] for k in ("sheet", "sheet_sha256", "key", "key_sha256",
                                                                   "rankings", "rankings_sha256", "second_sheet",
                                                                   "git_head")}, indent=2), "```", "",
        "## Counts", "", "```json", json.dumps(counts, indent=2), "```", "",
        "## Primary analysis (relevant = grade 2; U/X removed from pool; nDCG gains 0/1/3)", "",
        md_table(primary, cols), "",
        "S2_metadata (frozen Phase 1) abstains on every Tier B question (`unsupported_image`): coverage 0, scored 0 "
        "when abstaining. `random_expected` = exact expectation under random ranking of the same pool.", "",
        "## Per eye (primary)", "", md_table(eyes, ["system", "eye_id", "n_queries"] + list(METRIC_NAMES))
        if len(eyes) else "NOT EVALUATED", "",
        "## Sensitivity: U/X scored 0 (pessimistic)", "", md_table(sens, ["system", "n_evaluated"] + list(METRIC_NAMES)), "",
        "## Secondary: relevant = grade >= 1", "", md_table(second, ["system", "n_evaluated"] + list(METRIC_NAMES)), "",
        "## Inter-annotator agreement", "", json.dumps(agree, indent=2) if agree else "No second sheet supplied.", "",
        "## Caveats", "",
        "- Development eyes only (3 eyes): used for rubric calibration and the S5 encoder choice. No CI with < 5 eyes.",
        "- Queries of one eye share candidates; the eye, not the query, is the unit of independence.",
        "- B1 measures visual similarity, not clinical relevance. B2 is clinical only if graded by a trained grader.",
    ]
    return "\n".join(parts) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=list(QE.SETS), required=True)
    ap.add_argument("--category", choices=list(CATEGORIES), required=True)
    ap.add_argument("--sheet", type=Path, required=True)
    ap.add_argument("--key", type=Path)
    ap.add_argument("--rankings", type=Path)
    ap.add_argument("--second-sheet", type=Path)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--confirm-heldout-freeze", action="store_true")
    a = ap.parse_args(argv)
    if a.set == "heldout" and not a.confirm_heldout_freeze:
        ap.error("held-out evaluation runs once, after the freeze (protocol v2 §6)")
    key_path = a.key or ROOT / "phase2/annotations/keys/v2" / f"{a.set}_{a.category}_key.csv"
    rdir = a.rankings or default_rankings_dir(a.set)
    sheet, key = load_sheet(a.sheet), pd.read_csv(key_path)
    judg = join_sheet_key(sheet, key)
    rankings, systems, rman = load_rankings(rdir)
    cat = CATEGORIES[a.category]
    rankings = rankings[rankings.category == cat]
    systems = systems[systems.query_id.isin(judg.query_id)]
    check_pools(judg, rankings)

    sheet_sha = sha256(a.sheet)
    run_id = f"eval_{a.category}_{sheet_sha[:8]}_{rdir.name.split('_')[-1]}"
    out = a.out_root / a.set / run_id
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing evaluation {out}")

    per = evaluate(judg, rankings, systems, "primary", 2.0)
    prim, eyes = summarise(per)
    sens, _ = summarise(evaluate(judg, rankings, systems, "pessimistic", 2.0))
    sec, _ = summarise(evaluate(judg, rankings, systems, "primary", 1.0))
    agree = None
    if a.second_sheet:
        agree = agreement(sheet, load_sheet(a.second_sheet))
    qstat = per[per.system == "S1_latest_prior"].status.value_counts().to_dict()
    counts = dict(n_queries=int(judg.query_id.nunique()), n_judgments=len(judg),
                  n_labelled=int((judg.label != "").sum()), n_blank=int((judg.label == "").sum()),
                  labels={k: int(v) for k, v in judg.label.replace("", "blank").value_counts().items()},
                  query_status=qstat, annotators=sorted(set(judg.annotator_id) - {""}),
                  annotator_roles=sorted(set(judg.annotator_role) - {""}),
                  n_eyes=int(judg.eye_id.nunique()))
    cfg = dict(run_id=run_id, created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               set=a.set, category=a.category, category_name=cat, sheet=str(a.sheet), sheet_sha256=sheet_sha,
               key=str(key_path), key_sha256=sha256(key_path), rankings=str(rdir),
               rankings_sha256=rman["rankings_sha256"],
               second_sheet=None if not a.second_sheet else f"{a.second_sheet} sha256={sha256(a.second_sheet)}",
               relevance_primary="grade 2", relevance_secondary="grade >= 1", ndcg_gains="2^g - 1",
               u_x_primary="removed from pool for all systems", u_x_sensitivity="scored 0",
               bootstrap=QE.BOOTSTRAP, git_head=git_head(), evaluator_sha256=sha256(Path(__file__)))
    out.mkdir(parents=True)
    judg.to_csv(out / "judgments_joined.csv", index=False)
    per.to_csv(out / "per_query_primary.csv", index=False)
    prim.to_csv(out / "summary_primary.csv", index=False)
    eyes.to_csv(out / "per_eye_primary.csv", index=False)
    sens.to_csv(out / "summary_sensitivity_UX_as_0.csv", index=False)
    sec.to_csv(out / "summary_secondary_grade_ge1.csv", index=False)
    (out / "config.json").write_text(json.dumps(dict(cfg, counts=counts, agreement=agree), indent=2))
    (out / "report.md").write_text(report(cfg, counts, prim, eyes, sens, sec, agree), encoding="utf-8")
    print(f"{run_id}: query status {qstat}")
    print(f"report -> {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
