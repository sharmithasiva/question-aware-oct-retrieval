"""Pathology probe: do image embeddings carry information about labelled OCT findings? (Phase 3, Step 4)

Labels: OLIVES per-B-scan biomarker annotations (full_labels/Biomarker_Clinical_Data_Images.csv), joined to the
subset's images by EXACT archive path (never by a derived index). They exist only for the first and last visit of each
eye. Tasks: intraretinal fluid (IRF), subretinal fluid (SRF), diffuse retinal thickening / macular edema (DRT/ME),
each binary per B-scan. No label is created or inferred here; CST is never used as a label.

Methods (all evaluated out-of-sample with folds grouped by PATIENT, so no patient's scans are in both train and test):
  knn_<enc>        no training: score = fraction of positives among the k most cosine-similar scans of OTHER patients
  probe_<enc>      L2 logistic regression on the embedding (PCA fitted inside each training fold)
  metadata_cst     L2 logistic regression on the visit's CST only (a thickness measurement, not a finding label)
  position_only    L2 logistic regression on the B-scan position (one-hot) only: a CONTROL, because findings
                   cluster near the fovea and an encoder may simply recognise the scan position
  combined_<enc>   probe features + CST
Metrics: AUROC, AUPRC (chance = prevalence) with patient-level bootstrap 95 % CIs (>= 5 patients), and the visit-level
Spearman correlation between a visit's mean out-of-fold score and its labelled fraction over all 49 B-scans.
A task is N/A when it has fewer than --min-pos positives or negatives, or fewer than 4 patients.

Usage:
  python phase2/pathology_probe.py --subset phase2/data/atlas_subset \
      --emb resnet50=phase2/results/resnet50/embeddings.npz --out phase2/results/pathology/atlas_subset_v1
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
import oct_image as OI  # noqa: E402

LABELS = ROOT / "OLIVES_Dataset_Labels/full_labels/Biomarker_Clinical_Data_Images.csv"
TASKS = {"irf": "Fluid (IRF)", "srf": "Fluid (SRF)", "drt_me": "DRT/ME"}
VISIT_FRAC = {"irf": "bm_frac_fluid_irf", "srf": "bm_frac_fluid_srf", "drt_me": "bm_frac_drt_me"}
SEED = 20261030
N_BOOT = 2000


# --- data ---------------------------------------------------------------------------------------------------------

def labelled_scans(subset: Path, labels_csv: Path = LABELS) -> pd.DataFrame:
    m = OI.load_manifest(subset)
    b = pd.read_csv(labels_csv).rename(columns={"Path (Trial/Arm/Folder/Visit/Eye/Image Name)": "archive_path"})
    j = m.merge(b[["archive_path", "Scan (n/49)", *TASKS.values()]], on="archive_path", how="inner",
                validate="one_to_one")
    j = j.rename(columns={v: k for k, v in TASKS.items()})
    ep = pd.read_csv(ROOT / "data/processed/episodes.csv").set_index("visit_id")
    j["visit_cst"] = j.visit_id.map(ep.cst)
    for k, c in VISIT_FRAC.items():
        j[f"visit_frac_{k}"] = j.visit_id.map(ep[c])
    return j.reset_index(drop=True)


# --- small numpy toolkit (no sklearn dependency) ------------------------------------------------------------------------

def auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = pd.Series(np.r_[pos, neg]).rank().to_numpy()          # average ranks handle ties
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision (step-wise, as in sklearn.average_precision_score)."""
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s, kind="stable")
    yt = y[o]
    tp = np.cumsum(yt)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * yt).sum() / yt.sum())


def spearman(a, b) -> float:
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def fit_logistic(X: np.ndarray, y: np.ndarray, lam: float = 1.0, iters: int = 50) -> np.ndarray:
    """L2-penalised logistic regression by Newton/IRLS; returns weights with the intercept first (not penalised)."""
    Xb = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xb.shape[1])
    P = lam * np.eye(Xb.shape[1])
    P[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) + P @ w
        Hm = Xb.T @ (Xb * (p * (1 - p))[:, None]) + P + 1e-8 * np.eye(len(w))
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w


def predict_logistic(w, X):
    return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(X)), X] @ w, -30, 30)))


def patient_folds(patients: np.ndarray, n_folds: int = 5) -> list[np.ndarray]:
    """Test-index arrays; leave-one-patient-out when there are <= 20 patients, else seeded grouped k-fold."""
    u = np.unique(patients)
    if len(u) <= 20:
        groups = [[p] for p in u]
    else:
        perm = np.random.default_rng(SEED).permutation(u)
        groups = [perm[i::n_folds] for i in range(n_folds)]
    return [np.flatnonzero(np.isin(patients, g)) for g in groups]


def pca_fit(X: np.ndarray, k: int):
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:k]


# --- methods -----------------------------------------------------------------------------------------------------------

def knn_scores(E: np.ndarray, y: np.ndarray, patients: np.ndarray, k: int = 5) -> np.ndarray:
    z = OI.l2n(E)
    S = z @ z.T
    out = np.empty(len(y))
    for i in range(len(y)):
        s = np.where(patients == patients[i], -np.inf, S[i])      # other patients only
        nb = np.argsort(-s, kind="stable")[:k]
        out[i] = y[nb].mean()
    return out


def probe_scores(X: np.ndarray | None, extra: np.ndarray | None, y: np.ndarray, patients: np.ndarray,
                 n_comp: int = 16, lam: float = 1.0) -> np.ndarray:
    """Out-of-fold probabilities from PCA(X) and/or `extra` (n, k) features, folds grouped by patient."""
    out = np.full(len(y), np.nan)
    for test in patient_folds(patients):
        train = np.setdiff1d(np.arange(len(y)), test)
        if y[train].min() == y[train].max():
            out[test] = y[train].mean()
            continue
        feats_tr, feats_te = [], []
        if X is not None:
            mu, V = pca_fit(X[train], min(n_comp, len(train) - 1))
            feats_tr.append((X[train] - mu) @ V.T)
            feats_te.append((X[test] - mu) @ V.T)
        if extra is not None:
            feats_tr.append(extra[train])
            feats_te.append(extra[test])
        Ftr, Fte = np.hstack(feats_tr), np.hstack(feats_te)
        m, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
        w = fit_logistic((Ftr - m) / sd, y[train], lam)
        out[test] = predict_logistic(w, (Fte - m) / sd)
    return out


def bootstrap(y, s, patients, fn) -> list | None:
    u = np.unique(patients)
    if len(u) < 5:
        return None
    rng = np.random.default_rng(SEED)
    idx = {p: np.flatnonzero(patients == p) for p in u}
    vals = []
    for _ in range(N_BOOT):
        sel = np.concatenate([idx[p] for p in rng.choice(u, len(u))])
        if y[sel].min() == y[sel].max():
            continue
        vals.append(fn(y[sel], s[sel]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def evaluate(scans: pd.DataFrame, embs: dict[str, np.ndarray], min_pos: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    patients = scans.patient_id.to_numpy()
    cst = scans.visit_cst.to_numpy(float)[:, None]
    position = pd.get_dummies(scans.scan_index.astype(str)).to_numpy(float)
    rows, preds = [], scans[["image_key", "visit_id", "eye_id", "patient_id", *TASKS]].copy()
    for task in TASKS:
        y = scans[task].to_numpy(float)
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        base = dict(task=task, n_scans=len(y), n_pos=n_pos, n_neg=n_neg, prevalence=round(y.mean(), 4),
                    n_patients=len(np.unique(patients)), n_eyes=int(scans.eye_id.nunique()),
                    n_visits=int(scans.visit_id.nunique()),
                    n_patients_with_pos=int(len(np.unique(patients[y == 1]))))
        if n_pos < min_pos or n_neg < min_pos or base["n_patients"] < 4 or base["n_patients_with_pos"] < 2:
            rows.append(dict(base, method="-", status=f"N/A: insufficient labels (need >= {min_pos} positives and "
                                                     f"negatives, >= 4 patients, >= 2 with positives)"))
            continue
        methods = {"metadata_cst": probe_scores(None, cst, y, patients),
                   "position_only": probe_scores(None, position, y, patients)}
        for name, E in embs.items():
            methods[f"knn_{name}"] = knn_scores(E, y, patients)
            methods[f"probe_{name}"] = probe_scores(E, None, y, patients)
            methods[f"combined_{name}"] = probe_scores(E, cst, y, patients)
        for meth, s in methods.items():
            preds[f"{task}__{meth}"] = s
            vis = pd.DataFrame(dict(v=scans.visit_id, s=s, f=scans[f"visit_frac_{task}"])).groupby("v").agg(
                s=("s", "mean"), f=("f", "first"))
            rows.append(dict(base, method=meth, status="evaluated", auroc=round(auroc(y, s), 4),
                             auroc_ci95_patient=bootstrap(y, s, patients, auroc), auprc=round(auprc(y, s), 4),
                             auprc_ci95_patient=bootstrap(y, s, patients, auprc),
                             # every visit has the same scan positions, so position_only is constant per visit
                             visit_spearman_vs_label_fraction=(float("nan") if meth == "position_only"
                                                               else round(spearman(vis.s, vis.f), 4)),
                             n_visits_for_spearman=len(vis)))
    return pd.DataFrame(rows), preds


def to_markdown(summ: pd.DataFrame, cfg: dict) -> str:
    cols = ["method", "auroc", "auroc_ci95_patient", "auprc", "auprc_ci95_patient", "visit_spearman_vs_label_fraction"]
    parts = [f"# Pathology probe: `{cfg['subset']}`", "", f"Created {cfg['created_utc']}. Labels: OLIVES per-B-scan "
             "biomarkers (first and last visit of each eye only), joined by exact archive path. Folds grouped by "
             "patient. Exploratory.", ""]
    for task, g in summ.groupby("task", sort=False):
        r0 = g.iloc[0]
        parts += [f"## {task}: {r0.n_pos} positive / {r0.n_neg} negative B-scans (prevalence {r0.prevalence}); "
                  f"{r0.n_patients} patients, {r0.n_eyes} eyes, {r0.n_visits} visits", ""]
        if (g.status != "evaluated").all():
            parts += [g.status.iloc[0], ""]
            continue
        parts += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in g.iterrows():
            parts.append("| " + " | ".join("" if (not isinstance(r.get(c), list) and pd.isna(r.get(c)))
                                         else str(r.get(c)) for c in cols) + " |")
        parts += ["", f"AUPRC chance level = prevalence ({r0.prevalence}); AUROC chance = 0.5. CIs only with >= 5 "
                  "patients.", ""]
    return "\n".join(parts) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--emb", action="append", required=True, help="name=path/to/embeddings.npz (repeatable)")
    ap.add_argument("--min-pos", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    scans = labelled_scans(a.subset)
    held = {93, 74, 40, 14}
    if held & set(scans.eye_id):
        raise SystemExit("subset contains protocol held-out eyes; refusing")
    embs = {}
    for name, p in (x.split("=", 1) for x in a.emb):
        d = np.load(p, allow_pickle=False)
        pos = {k: i for i, k in enumerate(d["image_key"].tolist())}
        missing = set(scans.image_key) - set(pos)
        if missing:
            raise SystemExit(f"{name}: {len(missing)} labelled scans have no embedding")
        embs[name] = d["embeddings"][[pos[k] for k in scans.image_key]]
    summ, preds = evaluate(scans, embs, a.min_pos)
    cfg = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), subset=str(a.subset),
               embeddings=dict(x.split("=", 1) for x in a.emb), min_pos=a.min_pos, seed=SEED, n_boot=N_BOOT,
               splits=scans.groupby("split").eye_id.nunique().to_dict(), labels=str(LABELS.relative_to(ROOT)))
    a.out.mkdir(parents=True)
    summ.to_csv(a.out / "summary.csv", index=False)
    preds.to_csv(a.out / "per_scan_out_of_fold_scores.csv", index=False)
    (a.out / "config.json").write_text(json.dumps(cfg, indent=2))
    (a.out / "summary.md").write_text(to_markdown(summ, cfg), encoding="utf-8")
    print((a.out / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
