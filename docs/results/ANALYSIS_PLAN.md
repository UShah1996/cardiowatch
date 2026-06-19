# CardioWatch Pre-Registered Analysis Plan

This analysis plan must be committed before final HPC runs. Results used in the PSB 2027 manuscript should trace to a single `docs/results/<run_id>/` folder and to the CPSC holdout manifest checksum recorded there.

## Venue And Submission Constraints

- Target: Pacific Symposium on Biocomputing 2027.
- Deadline: August 3, 2026 at 11:59 PM PT.
- Length: 12 pages excluding cover letter, title page with author names/addresses, and references.
- Supplement: linked URL only; PSB does not host supplemental material.
- Preprint: allowed/encouraged with World Scientific acknowledgement language.

## Frozen Split Policy

- CPSC holdout is created before training using `src/experiments/make_cpsc_holdout.py`.
- Default holdout fraction: `0.20`.
- Seed: `42`.
- Split unit: patient/subject ID if multiple records per patient can be inferred; otherwise record ID after an explicit one-record-per-patient check.
- Holdout manifest must include:
  - sorted eligible record list hash,
  - holdout record hash,
  - label counts,
  - split unit counts,
  - seed,
  - git SHA,
  - creation timestamp.

## Primary Endpoint

Controlled same-data ECG comparison on the pre-registered CPSC holdout:

- RR+RF trained only on CPSC complement.
- CNN-LSTM trained only on CPSC complement.
- Both models scored on the identical CPSC holdout records.
- Primary discrimination comparison: paired DeLong test for AUC.
- Primary fixed-threshold comparison: McNemar test on correctness vectors.

## Secondary Endpoints

- Deployment comparison: best combined CNN-LSTM trained on CPSC complement plus PhysioNet 2017 vs best RR+RF. This is explicitly data-asymmetric and should not be used as the primary method superiority claim.
- Cross-device evaluation: MIT-BIH and Apple Watch exports.
- Detection latency: bootstrap of many normal-to-AFib CPSC transitions, reporting median latency, IQR, and normal-phase false-positive-rate distribution.
- Exploratory fusion: only on truly paired ECG + clinical data; Apple Watch fusion is exploratory because labels and positives are limited.

## Threshold Policy

- Frozen fixed thresholds:
  - RR+RF: `0.40`.
  - CNN-LSTM: `0.40` for CPSC validation and paired CPSC testing.
  - Fusion: `0.50`.
- Thresholds must be selected from training/complement validation data only, never from the CPSC holdout.
- Matched-operating-point McNemar will also be reported at 90% specificity, with thresholds estimated from non-holdout validation scores.

## Statistical Tests And Corrections

- DeLong paired AUC test for RR+RF vs CNN-LSTM probabilities on identical holdout labels.
- McNemar exact/binomial test when discordant counts are small; continuity-corrected chi-square otherwise.
- Holm correction for multiple pairwise tests.
- Effect sizes:
  - Delta AUC and DeLong/bootstrap confidence interval.
  - McNemar odds ratio `b/c` with continuity-adjusted confidence interval.
  - Paired accuracy difference where applicable.

## Metric Reporting

Report all applicable metrics with dataset, threshold, and sample count:

- AUC.
- Recall/sensitivity.
- Specificity.
- Balanced accuracy.
- PPV.
- NPV.
- F1.
- Confusion matrix.
- Wilson CIs for proportions.
- Bootstrap CIs for F1 and other non-smooth metrics.

## Validity Caveats

- Apple Watch labels are agreement with Apple’s classifier unless independently adjudicated; state how the AFib example was confirmed.
- Clinical-only and ECG-only models use different cohorts and are not a clean cross-modal ablation.
- Fusion is exploratory unless trained and evaluated on truly paired clinical + ECG data.
- Detection latency is measured after AFib onset; the old 30-minute lead-time framing is invalid for the normal-to-AFib splice.
- Bootstrap latency transitions reuse records from a shared pool and are not fully independent.
