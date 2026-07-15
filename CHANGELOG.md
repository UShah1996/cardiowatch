# Changelog

All notable changes to CardioWatch are documented here.

## [Unreleased] — PSB 2027 cross-device generalization & manuscript reframe (2026-07-11)

Branch: `psb2027/cross-device-and-reframe` (local only; not pushed).

Prepares the paper for **Pacific Symposium on Biocomputing (PSB) 2027**
(full-paper track, *AI and Machine Learning in Clinical Medicine* session). Two
goals: (1) reframe the paper around the clinical / population-screening angle,
and (2) strengthen the cross-device generalization claim from a single point
estimate into a statistically **provable**, pre-registered, multi-cohort result.

> **Status note.** The deep model's zero-shot cross-device AUCs are produced by
> the HPC run and are **not yet filled**: the manuscript's cross-device table has
> 14 `\XX` placeholders keyed to `crossdevice_stats.json`. Run
> `scripts/hpc/submit_pipeline.sh` on HPC, then fill them (see
> `paper/SUBMISSION_CHECKLIST.md`). No results were fabricated.

### Added

- **`src/evaluation/crossdevice_eval.py`** — zero-shot cross-device scorer. Scores
  RR+RF and both CNN-LSTMs on the **same** 10 s Lead-I windows across `afdb`,
  `ltafdb`, `cinc2017`, and `apple_watch`, one cohort per invocation. The combined
  CNN-LSTM is omitted for `cinc2017` (it was trained on CinC-2017 → avoids
  train-on-test). Has a `--self-test`.
- **`src/evaluation/crossdevice_stats.py`** — provable cross-device statistics:
  per-cohort **paired DeLong** (RR+RF vs CNN-LSTM-CPSC) as a single Holm family,
  plus **patient/record-clustered bootstrap 95% CIs** on every external AUC.
  Has a `--self-test`.
- **`src/evaluation/seed_robustness.py`** — secondary sensitivity analysis:
  `extract` (one seed's paired JSON → per-seed result) and `reduce` (aggregate 20
  seeds → distribution of ΔAUC and DeLong p). Has a `--self-test`.
- **`scripts/hpc/14_crossdevice_eval.sbatch`** — SLURM **array over 4 cohorts**
  (parallel); each task scores all models on one cohort. GPU.
- **`scripts/hpc/15_crossdevice_stats.sbatch`** — runs the cross-device DeLong +
  clustered-CI stats after job 14.
- **`scripts/hpc/16_seed_robustness.sbatch`** — SLURM **array over 20 seeds**
  (parallel), each in an isolated `CARDIOWATCH_RUN_ID` so nothing clobbers the
  primary run; only the tiny per-seed result is written back.
- **`paper/cover_letter.md`** — PSB cover letter, including the **required
  disclosure of the role LLMs played in authoring**.
- **`paper/SUBMISSION_CHECKLIST.md`** — remaining author steps (run pipeline, fill
  `\XX`, port to `ws-procs`, format, DOI).
- **`docs/PSB_2027_PLAN.md`** — the approved plan (with an implementation note on
  the refined sbatch layout).
- **`docs/ARCHIVAL_DOI.md`** — step-by-step Zenodo/DOI archival for the
  reproducibility snapshot.

### Changed

- **`docs/results/ANALYSIS_PLAN.md`** — pre-registration **amendment (committed
  before the run)**: cross-device degradation as a pre-registered comparison
  (zero-shot, identical windows, separate DeLong Holm family, clustered CIs);
  20-seed robustness as labeled secondary sensitivity; onset neurokit variant
  fixed as canonical.
- **`paper/main.tex`** — reframed for the clinical-screening angle: clinical
  motivation intro, contributions reordered (protocol → cross-device → same-data
  comparison → onset), RawECGNet-aware Related Work rebuttal, new cross-device
  head-to-head table (real RR MIT-BIH AUC 0.907 + `\XX` deep cells), seed-robustness
  reported, onset demoted to one paragraph, tabular clinical + fusion trimmed to a
  short separate-cohort paragraph.
- **`paper/references.bib`** — added `benmoshe2024rawecgnet` (RawECGNet) and
  `kapoor2023leakage` (leakage/reproducibility). All cite keys resolve.
- **`src/evaluation/confidence_intervals.py`** — added `cluster_bootstrap_auc()`
  (resamples whole patients/records, not windows).
- **`src/evaluation/result_tables.py`** — consumes `crossdevice_stats.json` and
  `seed_robustness.json`; renders the cross-device degradation table and the
  seed-robustness summary.
- **`src/evaluation/make_figures.py`** — added `fig_crossdevice()`: grouped
  RR+RF vs zero-shot CNN-LSTM(CPSC) AUC bars with clustered-CI error bars, chance
  line, and Holm DeLong p annotations; single-class cohorts (Apple Watch) skipped.
- **`scripts/hpc/submit_pipeline.sh`** — rewired DAG with parallel fan-out; adds
  jobs 14/15/16 and **un-orphans** onset (10/11/12) and mechanism (13); job 08 now
  depends on the new jobs.
- **`scripts/hpc/08_stats_tables.sbatch`** — runs the seed-robustness reduce and
  feeds the cross-device + seed JSONs into `result_tables.py`.
- **`scripts/hpc/09_figures.sbatch`** — passes `--crossdevice-json` to
  `make_figures.py`.
- **`src/experiments/make_cpsc_holdout.py`** — added `split_unit_note` clarifying
  that `split_unit="record"` is patient-grouped in CPSC (one ECG per patient).
- **`README.md`** — checkpoint-comment AUCs distinguish pre-holdout validation
  (≈0.968/0.974) from the pre-registered holdout (0.949/0.975); onset number
  reconciled to the neurokit primary (0.663) with sensitivity variants noted.

### Fixed

- Onset-prediction number inconsistency (0.606 base vs 0.663 neurokit): the
  neurokit variant is now the pre-registered primary everywhere, with the RR-only
  and P-wave variants reported as labeled sensitivity analyses.
- `fig_crossdevice` skips single-class / plausibility-only cohorts (`None` **and**
  `NaN` AUC) so Apple Watch no longer leaves an empty axis tick.

### Verification

- New Python modules pass their `--self-test` (DeLong Holm family, clustered
  bootstrap, seed extract/reduce, aggregation).
- `result_tables.py` and `make_figures.py` verified end-to-end on synthetic
  fixtures; the cross-device figure was rendered and inspected.
- All `scripts/hpc/*.sbatch` and `submit_pipeline.sh` pass `bash -n`.
- All cross-device claims carry a paired significance test and clustered CIs — no
  bare point estimates.

### Notes

- Executed locally with a scratch numpy/scipy/scikit-learn/matplotlib venv; the
  full HPC pipeline (raw ECG data + GPU checkpoints) was not run here.
- Nothing pushed to any remote.
