# Pathology probe: `phase2/data/atlas_subset`

Created 2026-10-30T07:59:10+00:00. Labels: OLIVES per-B-scan biomarkers (first and last visit of each eye only), joined by exact archive path. Folds grouped by patient. Exploratory.

## irf: 19 positive / 29 negative B-scans (prevalence 0.3958); 8 patients, 8 eyes, 16 visits

| method | auroc | auroc_ci95_patient | auprc | auprc_ci95_patient | visit_spearman_vs_label_fraction |
|---|---|---|---|---|---|
| metadata_cst | 0.6225 | [0.3801, 0.8215] | 0.6208 | [0.2759, 0.8199] | 0.4183 |
| position_only | 0.4946 | [0.3697, 0.6727] | 0.3856 | [0.202, 0.6759] |  |
| knn_resnet50 | 0.6561 | [0.525, 0.7739] | 0.5384 | [0.3719, 0.7881] | 0.3861 |
| probe_resnet50 | 0.7786 | [0.598, 0.9286] | 0.7409 | [0.4812, 0.9452] | 0.5758 |
| combined_resnet50 | 0.7495 | [0.5597, 0.9083] | 0.7378 | [0.4729, 0.9159] | 0.5022 |
| knn_dinov2_vits14 | 0.7505 | [0.5703, 0.8911] | 0.6524 | [0.4681, 0.9118] | 0.5966 |
| probe_dinov2_vits14 | 0.8131 | [0.6543, 0.9391] | 0.6787 | [0.4728, 0.9095] | 0.5714 |
| combined_dinov2_vits14 | 0.8022 | [0.6835, 0.9231] | 0.6669 | [0.4894, 0.8871] | 0.5729 |

AUPRC chance level = prevalence (0.3958); AUROC chance = 0.5. CIs only with >= 5 patients.

## srf: 1 positive / 47 negative B-scans (prevalence 0.0208); 8 patients, 8 eyes, 16 visits

N/A: insufficient labels (need >= 10 positives and negatives, >= 4 patients, >= 2 with positives)

## drt_me: 12 positive / 36 negative B-scans (prevalence 0.25); 8 patients, 8 eyes, 16 visits

| method | auroc | auroc_ci95_patient | auprc | auprc_ci95_patient | visit_spearman_vs_label_fraction |
|---|---|---|---|---|---|
| metadata_cst | 0.7431 | [0.4092, 0.9105] | 0.5088 | [0.1864, 0.859] | 0.5314 |
| position_only | 0.4583 | [0.3688, 0.6528] | 0.2332 | [0.1107, 0.465] |  |
| knn_resnet50 | 0.8113 | [0.6852, 0.9464] | 0.5962 | [0.4482, 0.8806] | 0.4912 |
| probe_resnet50 | 0.9005 | [0.8182, 0.9773] | 0.7414 | [0.6414, 0.9425] | 0.5075 |
| combined_resnet50 | 0.9028 | [0.8107, 0.993] | 0.7346 | [0.6697, 0.9844] | 0.6478 |
| knn_dinov2_vits14 | 0.9005 | [0.7725, 0.9875] | 0.7939 | [0.4822, 0.9743] | 0.667 |
| probe_dinov2_vits14 | 0.794 | [0.6189, 0.9891] | 0.5912 | [0.4244, 0.9319] | 0.5433 |
| combined_dinov2_vits14 | 0.7708 | [0.5859, 0.9716] | 0.5421 | [0.4232, 0.9188] | 0.5672 |

AUPRC chance level = prevalence (0.25); AUROC chance = 0.5. CIs only with >= 5 patients.

