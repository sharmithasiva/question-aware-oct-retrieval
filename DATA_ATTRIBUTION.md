# Data attribution

This repository contains data from, and material derived from, the **OLIVES** dataset, used under the
**Creative Commons Attribution 4.0 International licence (CC BY 4.0)**: <https://creativecommons.org/licenses/by/4.0/>.

## Required citation

> M. Prabhushankar, K. Kokilepersaud, Y. Logan, S. Trejo Corona, G. AlRegib and C. Wykoff, "OLIVES Dataset:
> Ophthalmic Labels for Investigating Visual Eye Semantics", *Advances in Neural Information Processing Systems
> (NeurIPS) Datasets and Benchmarks Track*, 2022.

- **Official source:** Zenodo record 7105232, doi:[10.5281/zenodo.7105232](https://doi.org/10.5281/zenodo.7105232). The Retina OCT atlas cites the related DOI [10.5281/zenodo.6622145](https://doi.org/10.5281/zenodo.6622145).
- **Dataset repository:** <https://github.com/olivesgatech/OLIVES_Dataset>.

If you use any data or derived material from this repository, cite the OLIVES dataset above, not this repository and not the atlas.

## Image source

The OCT B-scan images in this repository were **not** taken from the official archive. They were downloaded from the
**Retina OCT atlas** (<https://retinaoct.com>; source code <https://github.com/NicoleMulla/retinaoct>), a third-party, searchable index of OLIVES.
- **Format:** the atlas serves **downsampled, lossy WebP derivatives** (504 × 496) of the original TIFF/PNG scans.
- **Not checked against the original:** these images have not been compared with the official Zenodo files.
- **Known data issue:** the atlas serves byte-identical images for subject `01-013`, visits W8 and W12 (see `phase2/PERFORMANCE_REPORT.md`).

## What is included, and what was changed

CC BY 4.0 requires indicating changes. The table lists each item and the change made.

| Path | Origin | Changes |
|---|---|---|
| `OLIVES_Dataset_Labels/` | OLIVES label archive `OLIVES_Dataset_Labels.zip` (Zenodo record 7105232) | **None.** Unmodified copies (MD5 of the archive in `docs/PHASE_2_HANDOFF.md`). |
| `data/processed/episodes.csv`, `visit_images.csv`, `heldout_prime.csv` | derived from the label files above | Restructured into one row per eye-visit / image path, with project-local IDs, patient-level splits and QC flags (`src/oct_retrieval/episodes.py`, `side_tables.py`; `docs/DATA_AUDIT.md`). |
| `phase2/data/atlas_subset/images/`, `phase2/data/atlas_fullhist/images/` (169 files) | Retina OCT atlas WebP derivatives of OLIVES B-scans | **None** to the image bytes (SHA-256 in each folder's `SHA256SUMS`). Renamed to `<visit_id>__scan<NN>.webp`; per-image provenance in `manifest.csv` and `provenance.json`. |
| `phase2/annotations/packets/` | copies of the images above | Renamed to opaque candidate codes for blinded annotation. Contact sheets are **adaptations**: images converted to grayscale, resized and arranged in a grid. |
| `phase2/results/**` (figures, embeddings, CSVs), `phase2/results/executed_local_*.ipynb` | computed from the images and labels above | **Derived:** resized/normalised images, pretrained-encoder embeddings, similarity statistics, plots. |
| `sealed/`, `tests/data/`, `results/` (Phase 1) | questions and evaluation outputs created in this project from the OLIVES metadata | Derived; AI-generated benchmark questions with limited human review (see `docs/PHASE_1_HANDOFF.md`). |

## Notices

- **Research use only.** Nothing in this repository is clinically validated or reviewed by a clinician, and it must not inform diagnosis or treatment.
- **No endorsement.** Neither the OLIVES authors nor the Retina OCT atlas endorse this project.
- **De-identified data.** The data are de-identified research data as distributed by the OLIVES authors. This repository adds no patient-identifiable information. `patient_id` and `eye_id` are the dataset's/project's own identifiers.
- **Code licence.** This attribution covers the **data**. No licence has been chosen yet for the code in this repository.
