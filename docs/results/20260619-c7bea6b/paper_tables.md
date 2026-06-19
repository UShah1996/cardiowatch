# CardioWatch Paper Result Tables

## Clinical Models (Kaggle cohort, distinct from ECG cohorts)

| Model | Dataset | n | AUC | Recall | Specificity | Balanced Acc | PPV | NPV | F1 | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_forest | Kaggle clinical test split | 92 | 0.954 | 0.941 | 0.854 | 0.897 | 0.889 | 0.921 | 0.914 | 0.50 |
| xgboost | Kaggle clinical test split | 92 | 0.927 | 0.961 | 0.634 | 0.797 | 0.766 | 0.929 | 0.852 | 0.21 |

## ECG Controlled And Deployment Results (CPSC holdout)

| Model | Dataset | n | AUC | Recall | Specificity | Balanced Acc | PPV | NPV | F1 | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rr_rf | CPSC pre-registered holdout | 1376 | 0.935 | 0.803 | 0.879 | 0.841 | 0.589 | 0.954 | 0.679 | 0.40 |
| random_baseline | CPSC pre-registered holdout | 1376 | 0.502 | 0.508 | 0.511 | 0.509 | 0.183 | 0.828 | 0.269 | 0.50 |
| cnn_cpsc | CPSC pre-registered holdout | 1376 | 0.949 | 0.959 | 0.832 | 0.896 | 0.552 | 0.989 | 0.701 | 0.40 |
| cnn_combined_deploy | CPSC pre-registered holdout | 1376 | 0.975 | 0.918 | 0.941 | 0.929 | 0.770 | 0.982 | 0.837 | 0.40 |

## Statistical Comparisons (CPSC holdout)

Primary endpoint: `rr_rf_vs_cnn_cpsc`. Holm family size: 6 (DeLong + fixed-threshold McNemar over model-vs-model pairs; matched-specificity and random-baseline rows are raw sensitivity/sanity).

| Comparison | ΔAUC | 95% CI | DeLong p | DeLong p(Holm) | McNemar b/c | McNemar p | McNemar p(Holm) | In family |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| cnn_combined_deploy_vs_random_baseline | +0.473 | [0.433, 0.514] | 1.07e-114 | — | 633/46 | 5.36e-112 | — | no |
| cnn_cpsc_vs_cnn_combined_deploy | -0.027 | [-0.040, -0.013] | 9.46e-05 | 0.000284 | 26/139 | 2.8e-18 | 1.68e-17 | yes |
| cnn_cpsc_vs_random_baseline | +0.447 | [0.403, 0.491] | 6.06e-89 | — | 580/106 | 6.68e-73 | — | no |
| rr_rf_vs_cnn_combined_deploy | -0.040 | [-0.053, -0.027] | 3.79e-09 | 1.52e-08 | 58/156 | 3.34e-11 | 1.67e-10 | yes |
| rr_rf_vs_cnn_cpsc | -0.013 | [-0.032, 0.006] | 0.165 | 0.33 | 147/132 | 0.402 | 0.402 | yes |
| rr_rf_vs_random_baseline | +0.433 | [0.392, 0.475] | 8.54e-92 | — | 578/89 | 1.24e-79 | — | no |

## Cross-Device: MIT-BIH AFib (zero-shot)

| Model | Dataset | n | AUC | Recall | Specificity | Balanced Acc | PPV | NPV | F1 | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rr_rf | MIT-BIH AFib (afdb, Holter 250 Hz, zero-shot) | 28104 | 0.907 | 0.906 | 0.759 | 0.833 | 0.714 | 0.924 | 0.798 | 0.40 |

## Apple Watch (exploratory — device-classifier labels)

| Model | Dataset | n | n AFib | Accuracy | 95% CI (Wilson) | Threshold |
|---|---|---:|---:|---:|---:|---:|
| cnn_lstm (best available checkpoint) | Apple Watch personal exports | 54 | 1 | 0.907 | 0.801–0.960 | 0.50 |

_Labels are Apple Watch's on-device classifier output, not independent clinical adjudication._

## Detection Latency (bootstrap CPSC transitions)

| Metric | Value |
|---|---:|
| Transitions | 100 |
| Detected | 100 |
| Threshold | 0.4 |
| Stride (s) | 5 |
| Median latency (min) | 0.04999999999999716 |
| Latency IQR (min) | [0.04999999999999716, 0.04999999999999716] |
| Normal-phase FP rate (median) | 0.083 |

_Transitions are sampled with replacement from a shared record pool; treat as robustness analysis, not independent prospective episodes._

## Validity Notes

- Clinical-only and ECG-only models use different cohorts and are reported separately (not a cross-modal ablation).
- Apple Watch agreement is not a clinical gold standard unless independently adjudicated.
- Fusion rows are exploratory unless based on truly paired clinical + ECG data.
- Detection latency is measured after AFib onset; the old 30-minute lead-time framing is invalid.
- Corrected p-values come from stat_tests.json (pre-registered Holm family); matched-specificity and random-baseline tests are raw.
