# Question-Aware Episodic Retrieval for Longitudinal Retinal OCT — Phase 1

Phase 1 covers retrieval from metadata only: it uses the OLIVES label files and no images or models.

## Layout

```
OLIVES_Dataset_Labels/         raw label files (input; not modified)
docs/DATA_AUDIT.md             data audit, identifier mapping, assumptions, aggregation strategy
src/oct_retrieval/
  load.py                      read raw tables, parse trial-specific image paths
  episodes.py                  visit-level episodic memory table + patient-level splits
  questions.py                 question types, rule-based question parser, metadata ground truth
  retrievers.py                latest / earliest / random / full-history / temporal / clinical / combined / rule router
  metrics.py                   Hit@K, Recall@K, MRR, nDCG@K
  evaluate.py                  benchmark runner -> results/ + reports/RETRIEVAL_REPORT.md
  audit.py                     prints the numbers quoted in the audit
tests/                         unit tests (synthetic) + integration tests on the real label files
data/processed/                episodes.csv (1 row per visit), visit_images.csv (1 row per B-scan path)
results/                       per-instance metrics, summaries, parser check, run config
reports/RETRIEVAL_REPORT.md    generated evaluation report
```

## Run

```bash
pip install pandas numpy openpyxl pytest
```

```bash
python -m pytest -q
```

```bash
PYTHONPATH=src python -m oct_retrieval.evaluate
```

v2 adds harder question types (spec: `docs/PHASE1_V2_QUESTION_SPEC.md`) and writes to `results/v2/` and `reports/RETRIEVAL_REPORT_V2.md`. It never overwrites v1, which is preserved in `results/v1_snapshot/` with checksums.

```bash
PYTHONPATH=src python -m oct_retrieval.evaluate_v2
```

On PowerShell, set the path with `$env:PYTHONPATH="src"` instead. The full run takes about 2 minutes on a CPU laptop and is deterministic (seed 42).

## Key data facts

These are verified facts; see the audit for details.

- 1,594 visits, 96 eyes and 87 patients. A visit is (`Eye_ID`, visit folder). CST and BCVA are constant within every visit.
- Biomarker labels exist only for the first and last visit of each eye.
- Splits are made at patient level. Both eyes of a patient, and all visits of an eye, always land in the same split.

Phase 1.3 (development only; the locked test is not run) adds a structured intent parser and retrieval with explicit abstention:

```bash
PYTHONPATH=src python -m oct_retrieval.evaluate_v3_dev
```

## Data attribution

This repository includes OLIVES data and derived material (CC BY 4.0). See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) for the required citation, the image source and what was changed.
