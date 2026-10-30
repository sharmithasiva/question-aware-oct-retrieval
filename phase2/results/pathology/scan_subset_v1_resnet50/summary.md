# Pathology probe: `phase2\data\scan_subset_v1`

Created 2026-10-30T08:40:02+00:00. Labels: OLIVES per-B-scan biomarkers (first and last visit of each eye only), joined by exact archive path. Folds grouped by patient. Exploratory.

## irf: 149 positive / 208 negative B-scans (prevalence 0.4174); 36 patients, 37 eyes, 73 visits

| method | auroc | auroc_ci95_patient | auprc | auprc_ci95_patient | visit_spearman_vs_label_fraction |
|---|---|---|---|---|---|
| metadata_cst | 0.7354 | [0.6465, 0.815] | 0.6944 | [0.5519, 0.8124] | 0.5521 |
| position_only | 0.4924 | [0.4298, 0.5548] | 0.3993 | [0.3044, 0.5299] |  |
| knn_resnet50 | 0.6948 | [0.6113, 0.7723] | 0.6099 | [0.4923, 0.772] | 0.4778 |
| probe_resnet50 | 0.7838 | [0.697, 0.8577] | 0.7543 | [0.6125, 0.8631] | 0.663 |
| combined_resnet50 | 0.7854 | [0.6964, 0.8621] | 0.7671 | [0.6333, 0.8676] | 0.6813 |

AUPRC chance level = prevalence (0.4174); AUROC chance = 0.5. CIs only with >= 5 patients.

## srf: 6 positive / 351 negative B-scans (prevalence 0.0168); 36 patients, 37 eyes, 73 visits

N/A: insufficient labels (need >= 10 positives and negatives, >= 4 patients, >= 2 with positives)

## drt_me: 99 positive / 258 negative B-scans (prevalence 0.2773); 36 patients, 37 eyes, 73 visits

| method | auroc | auroc_ci95_patient | auprc | auprc_ci95_patient | visit_spearman_vs_label_fraction |
|---|---|---|---|---|---|
| metadata_cst | 0.8624 | [0.7579, 0.9373] | 0.7333 | [0.5356, 0.8755] | 0.6613 |
| position_only | 0.423 | [0.3598, 0.4874] | 0.2366 | [0.1569, 0.3417] |  |
| knn_resnet50 | 0.8174 | [0.7598, 0.8694] | 0.6398 | [0.5114, 0.7904] | 0.653 |
| probe_resnet50 | 0.89 | [0.8335, 0.9377] | 0.7896 | [0.6683, 0.8855] | 0.6705 |
| combined_resnet50 | 0.9082 | [0.8522, 0.951] | 0.8018 | [0.6401, 0.9101] | 0.7412 |

AUPRC chance level = prevalence (0.2773); AUROC chance = 0.5. CIs only with >= 5 patients.

