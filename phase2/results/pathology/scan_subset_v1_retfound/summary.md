# Pathology probe: `phase2\data\scan_subset_v1`

Created 2026-10-30T09:18:59+00:00. Labels: OLIVES per-B-scan biomarkers (first and last visit of each eye only), joined by exact archive path. Folds grouped by patient. Exploratory.

## irf: 149 positive / 208 negative B-scans (prevalence 0.4174); 36 patients, 37 eyes, 73 visits

| method | auroc | auroc_ci95_patient | auprc | auprc_ci95_patient | visit_spearman_vs_label_fraction |
|---|---|---|---|---|---|
| metadata_cst | 0.7354 | [0.6465, 0.815] | 0.6944 | [0.5519, 0.8124] | 0.5521 |
| position_only | 0.4924 | [0.4298, 0.5548] | 0.3993 | [0.3044, 0.5299] |  |
| knn_retfound_oct | 0.6882 | [0.5856, 0.7717] | 0.6552 | [0.5023, 0.7814] | 0.4258 |
| probe_retfound_oct | 0.7566 | [0.673, 0.8317] | 0.6978 | [0.5563, 0.8366] | 0.4855 |
| combined_retfound_oct | 0.7865 | [0.7043, 0.8587] | 0.725 | [0.5767, 0.8634] | 0.6053 |

AUPRC chance level = prevalence (0.4174); AUROC chance = 0.5. CIs only with >= 5 patients.

## srf: 6 positive / 351 negative B-scans (prevalence 0.0168); 36 patients, 37 eyes, 73 visits

N/A: insufficient labels (need >= 10 positives and negatives, >= 4 patients, >= 2 with positives)

## drt_me: 99 positive / 258 negative B-scans (prevalence 0.2773); 36 patients, 37 eyes, 73 visits

| method | auroc | auroc_ci95_patient | auprc | auprc_ci95_patient | visit_spearman_vs_label_fraction |
|---|---|---|---|---|---|
| metadata_cst | 0.8624 | [0.7579, 0.9373] | 0.7333 | [0.5356, 0.8755] | 0.6613 |
| position_only | 0.423 | [0.3598, 0.4874] | 0.2366 | [0.1569, 0.3417] |  |
| knn_retfound_oct | 0.8156 | [0.7096, 0.8955] | 0.7356 | [0.5435, 0.8525] | 0.5073 |
| probe_retfound_oct | 0.876 | [0.8203, 0.9231] | 0.749 | [0.6127, 0.8496] | 0.578 |
| combined_retfound_oct | 0.9044 | [0.8478, 0.948] | 0.7867 | [0.6341, 0.897] | 0.6598 |

AUPRC chance level = prevalence (0.2773); AUROC chance = 0.5. CIs only with >= 5 patients.

