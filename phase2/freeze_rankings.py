"""Freeze every system's ranking for a query set BEFORE any human label exists (annotation protocol v2 §5).

Reuses question_eval.py (query set, candidate pools, systems, recency tie-breaks) read-only and reuses SAVED
embeddings; no encoder is run. Writes a content-addressed, never-overwritten folder:

  phase2/results/question_eval_v2/<set>/rankings_tier<T>_<sha12>/
      rankings.csv     query_id, query_visit_id, category, eye_id, system, rank, candidate_visit_id
      systems.csv      per query x system: status (ranked / abstained) and the frozen parser's reading
      query_set.csv    the shared query set and candidate pools
      MANIFEST.json    SHA-256 of rankings and of every input (embeddings, manifests, episodes, code), git HEAD

Usage: python phase2/freeze_rankings.py --set dev --tier B
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase2"))
import question_eval as QE  # noqa: E402

OUT_ROOT = ROOT / "phase2" / "results" / "question_eval_v2"


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def freeze(set_name: str, tier: str, out_root: Path = OUT_ROOT) -> Path:
    episodes = pd.read_csv(ROOT / "data/processed/episodes.csv")
    qs = QE.build_query_set(episodes, set_name, tier)
    QE.assert_no_candidate_leakage(qs, episodes)
    systems = QE.Systems(episodes, set_name)
    if not systems.emb:
        raise SystemExit(f"no saved embeddings for set {set_name!r}; cannot freeze image rankings")
    rows, status = [], []
    for q in qs.itertuples():
        rk = systems.rank_all(q)
        parsed = rk.pop("_parsed")
        cands = set(q.candidate_visit_ids.split(";"))
        for s, lst in rk.items():
            status.append(dict(query_id=q.query_id, system=s, status="ranked" if lst is not None else "abstained",
                               parsed_intent=parsed))
            if lst is None:
                continue
            if set(lst) != cands or len(lst) != len(cands):
                raise AssertionError(f"{q.query_id}/{s}: ranking is not a permutation of the candidate pool")
            rows += [dict(query_id=q.query_id, query_visit_id=q.query_visit_id, category=q.category,
                          eye_id=q.eye_id, system=s, rank=i + 1, candidate_visit_id=v) for i, v in enumerate(lst)]
    rk = pd.DataFrame(rows)
    blob = rk.to_csv(index=False).encode()
    digest = hashlib.sha256(blob).hexdigest()
    out = out_root / set_name / f"rankings_tier{tier}_{digest[:12]}"
    if out.exists():
        if sha256(out / "rankings.csv") == digest:
            print(f"already frozen (identical): {out}")
            return out
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    (out / "rankings.csv").write_bytes(blob)
    pd.DataFrame(status).to_csv(out / "systems.csv", index=False)
    qs.to_csv(out / "query_set.csv", index=False)
    cfg = QE.SETS[set_name]
    inputs = {ROOT / "data/processed/episodes.csv", cfg["subset"] / "manifest.csv",
              ROOT / "phase2/question_eval.py", ROOT / "phase2/oct_image.py", Path(__file__)}
    inputs |= {Path(str(cfg["emb"]).format(enc=e)) for e in QE.ENCODERS}
    manifest = dict(
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), git_head=git_head(),
        set=set_name, tier=tier, n_queries=int(qs.query_id.nunique()), n_rows=len(rk),
        systems=sorted(rk.system.unique()), abstaining={s: int(n) for s, n in
                                                        pd.DataFrame(status).query("status == 'abstained'")
                                                        .system.value_counts().items()},
        tie_rule="exact score ties -> more recent visit first (question_eval.recency_order)",
        rrf_k=QE.RRF_K, min_candidates=QE.MIN_CANDIDATES, rankings_sha256=digest,
        query_set_sha256=sha256(out / "query_set.csv"),
        inputs_sha256={str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in sorted(inputs) if p.exists()},
        note="Frozen before any Tier B human label existed. Evaluation verifies rankings_sha256.")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"frozen {len(rk)} ranking rows, {manifest['n_queries']} queries -> {out}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(QE.SETS), required=True)
    ap.add_argument("--tier", choices=["A", "B"], default="B")
    ap.add_argument("--confirm-heldout-freeze", action="store_true")
    a = ap.parse_args(argv)
    if a.set == "heldout" and not a.confirm_heldout_freeze:
        ap.error("held-out rankings are frozen only after the dev choices are logged (protocol v2 §6)")
    freeze(a.set, a.tier)
    return 0


if __name__ == "__main__":
    sys.exit(main())
