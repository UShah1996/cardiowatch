# CardioWatch → PSB 2027 Submission Plan

> **Implementation note (added when the plan was executed).** The sbatch layout
> was refined during implementation vs. the original plan below: instead of four
> jobs (14 deep-eval, 15 RR-eval, 16 stats, 17 seeds), the cross-device eval is a
> single **SLURM array over cohorts** — `14_crossdevice_eval.sbatch` (array 0–3:
> afdb, ltafdb, cinc2017, apple_watch) scores *all* models on one cohort per task,
> which guarantees window alignment across models with no cross-job merge —
> followed by `15_crossdevice_stats.sbatch` and `16_seed_robustness.sbatch`
> (array 0–19). Everything else matches the plan. See `paper/SUBMISSION_CHECKLIST.md`
> for the remaining author steps.

---

## Context

The goal is to get `cardiowatch_paper.pdf` accepted to the **Pacific Symposium on
Biocomputing (PSB) 2027**. This is a full-paper submission targeting the **Aug 3, 2026
deadline** (today is 2026-07-11 → **~3 weeks**). User decisions locked in:

- **Track:** Full paper, Aug 3, 2026 (12-page max, World Scientific `ws-procs` format).
- **Framing:** Lead with the **clinical / population-screening** angle (device-agnostic
  AFib screening on any wearable → stroke prevention), with the model comparison as
  supporting evidence — *not* "simple beats deep" as the headline.
- **Scope:** Willing to **add experiments** to strengthen the generalization claim.

I did a code+results audit and a 5-year literature scan. **Bottom line: the paper's
methodology is its strongest asset and is genuinely well-executed** (pre-registered
analysis plan, correctly implemented paired DeLong + McNemar + Holm, leakage-safe
patient/record-grouped hold-out, reproducible SLURM pipeline, provenance tracking). The
two real risks are (1) **PSB topical fit** — PSB skews molecular/genomics; wearable AFib
must be argued into the *"AI and ML in Clinical Medicine"* session — and (2) **thin
cross-device evidence** for the "generalizes better across hardware" claim (only one real
external AUC + an n=1-positive Apple Watch plausibility check).

---

## Expert review of the current draft

### What is strong (keep and foreground)
- **Pre-registration + leakage safety.** `docs/results/ANALYSIS_PLAN.md` freezes split
  (seed 42, 20% hold-out), thresholds (0.40), and a 6-test Holm family *before* training.
  `src/experiments/make_cpsc_holdout.py` writes a checksummed manifest; SLURM DAG
  (`scripts/hpc/submit_pipeline.sh`) enforces holdout-before-training ordering. This
  directly answers the medical-AI reproducibility/leakage crisis (Kapoor & Narayanan,
  *Patterns* 2023) and is rare at PSB. **This is the paper's moat — make it the spine.**
- **Correct statistics.** `src/evaluation/stat_tests.py` implements Sun & Xu fast DeLong,
  exact/continuity-corrected McNemar, and Holm — verified against `stat_tests.json`.
- **A clean, honest null result** (RR+RF vs CNN-LSTM CPSC: ΔAUC −0.013, DeLong p=0.33,
  McNemar p=0.40 Holm-corrected) that is *not* overstated.
- Honest limitations section; scope discipline (AFib not MI; fusion/onset flagged
  exploratory).

### What is weak / risky (must fix before submission)
1. **PSB fit is not yet argued.** The draft reads as an ML/signal-processing paper. PSB
   reviewers in the Clinical Medicine session need a *biomedicine* through-line:
   population-scale undiagnosed-AFib burden → stroke prevention → why device-agnostic
   screening changes who gets caught. Currently this is one sentence in the intro.
2. **Headline contradicts recent literature.** The abstract/conclusion claim simple
   timing features "are competitive with deep models and generalize better across
   hardware." **RawECGNet (Ben-Moshe et al., 2024)** and cross-dataset domain-
   generalization work argue *raw-ECG deep models generalize better* than RR-interval
   features. A reviewer citing this can sink the paper. Per the chosen framing, demote
   this to a supporting, carefully-bounded claim ("on matched hospital data, indistinguishable;
   timing features additionally transfer zero-shot") and pre-empt RawECGNet explicitly.
3. **Cross-device evidence is thin for its claim.** "Generalizes better across hardware"
   rests on **one** real external AUC (MIT-BIH Holter, 0.907) plus Apple Watch (n=54, **1**
   confirmed positive, device-classifier labels). The draft *asserts* the CNN-LSTM drops
   to ~0.50 on Apple Watch but **never shows the CNN-LSTM's zero-shot number head-to-head
   with RR+RF in a table.** The comparative degradation is the whole point and is missing.
4. **Onset-prediction number inconsistency (factual bug).** `onset_prediction.json` has
   window AUC **0.606** (sens 0.22, FA/h 1.06); the paper Table VI reports **0.663** (sens
   0.28, FA/h 0.48) from the neurokit variant `onset_prediction_nk.json`. The pipeline
   already treats the neurokit variant as canonical (`08_stats_tables.sbatch` feeds
   `--onset-json .../onset_prediction_nk.json` into `result_tables.py`), so **0.663 is the
   intended number**; the README's "≈0.66" is loose and the base 0.606 run is a secondary
   variant. Fix = pre-register nk as primary, report base RR-only as a labeled sensitivity
   analysis (no shortcut — both are reported), and correct the README.
5. **Filler that invites attack.** The tabular clinical model (separate Kaggle cohort,
   AUC 0.954) and the exploratory late-fusion add ~a page, are not part of any validated
   claim, and draw "why is this here / is this leakage?" questions. In a 12-page clinical
   paper they cost more than they earn.
6. **Minor:** README checkpoint comments say CNN AUC 0.968/0.974 (pre-holdout val) vs the
   correct holdout 0.949/0.975 — cosmetic but a reviewer who clones the repo will notice;
   fix the comments. Manifest `split_unit:"record"` should be labeled "record = patient
   (one ECG/patient in CPSC)" to avoid a false leakage flag.

### Shortcuts the author took that reviewers will probe
- Apple Watch "accuracy 0.907" is specificity in disguise (1 positive). Draft already
  caveats this well — **keep it a plausibility check, never a headline number.**
- Detection-latency IQR is degenerate (all 0.05 min due to 5 s stride). Report as
  "first-window detection," drop the IQR pretense.
- Combined CNN-LSTM (0.975) is a data-asymmetric deployment model. Draft handles this
  honestly; keep the "statement about data, not architecture" line.

---

## Recommended framing (the reframe)

**New title direction:** keep "device-agnostic" but lead clinical, e.g.
*"Device-Agnostic Screening for Atrial Fibrillation on Consumer Wearables: A
Pre-Registered, Leakage-Safe Cross-Device Evaluation for Stroke Prevention."*

**New contribution ordering (was: comparison → cross-device → onset):**
1. **Clinical problem & why device-agnosticism matters for screening** (new, ~½ page):
   undiagnosed-AFib → stroke burden (cite Perez AppleHeart 2019; a 2023–2025 screening/
   guideline ref); the *deployment reality* that a screening model must run unchanged
   across the heterogeneous device fleet consumers actually own.
2. **A pre-registered, leakage-safe evaluation protocol** as a methodological
   contribution in its own right (this is what makes it PSB-caliber and citable).
3. **Cross-device generalization result** — the strengthened head-to-head degradation
   table (see experiments) is the empirical centerpiece.
4. **Controlled same-data comparison** (RR vs deep) as *supporting* evidence, bounded and
   RawECGNet-aware.
5. **Onset prediction** — one tight main-text paragraph (sub-clinical, exploratory) with
   the full results + sensitivity variants in the supplement (no shortcut = fully reported,
   per E5). Do **not** let it dilute the screening story.

---

## Execution plan (prioritized, parallel, no shortcuts)

Principle: **stronger, statistically *provable* results, no shortcuts.** Every cross-device
claim gets a paired significance test and patient-clustered bootstrap CIs — not point
estimates. All work produces **one new pre-registered `run_id`** (`common.sh` stamps
`YYYYMMDD-<gitSHA>`, git SHA + checksums via `provenance.py`) so every number in the paper
traces to a single reproducible folder. HPC work is expressed as **new sbatch jobs wired
into the existing parallel SLURM DAG** (`scripts/hpc/submit_pipeline.sh`); the literature
and writing tracks run **concurrently** with the HPC jobs, not after them.

### P0 — Science that de-risks acceptance (no shortcuts, HPC-parallel)

**E1. Head-to-head zero-shot cross-device degradation (empirical centerpiece).**
Score **both** deep models — CNN-LSTM(CPSC) and CNN-LSTM(combined) — *zero-shot* on every
external set (MIT-BIH `afdb`, Long-Term AF `ltafdb`, PhysioNet/CinC 2017 hold-out, Apple
Watch), alongside RR+RF on the identical records. Turns the asserted "deep → ~0.50" into a
shown, per-cohort table. Models already exist (no retraining).
*(Built as `src/evaluation/crossdevice_eval.py`.)*

**E2. Multiple external cohorts, not one (breadth for the generalization claim).**
Evaluate RR+RF and the deep models across **≥3 external cohorts** (`afdb`, `ltafdb`,
CinC-2017 hold-out). Reuse `extract_rr_features()` and the windowing harness.

**E3. Make the "timing generalizes, deep degrades" claim *provable* (no shortcut).**
On each external cohort where both models scored the same records, run **paired DeLong
(RR+RF vs CNN-LSTM-CPSC)** and report **patient-clustered bootstrap 95% CIs** on every
external AUC (cluster by patient, not window — closes the pseudo-replication caveat that
MIT-BIH windows are correlated within 25 patients). Converts "0.907 vs ~0.50" into a
signed, CI-bounded, p-valued degradation result.
*(Built as `src/evaluation/crossdevice_stats.py` + `confidence_intervals.cluster_bootstrap_auc`.)*

**E4. Seed-robustness sensitivity for the primary null (pre-registration-safe).**
Keep the pre-registered primary on seed 42. **Additionally** report a secondary robustness
analysis over N≥20 alternate hold-out seeds (distribution of ΔAUC and DeLong p), clearly
labeled *sensitivity, not the pre-registered primary*. Runs the existing train+eval per
seed in a **SLURM array**, aggregated into `seed_robustness.json`.
*(Built as `src/evaluation/seed_robustness.py`.)*

**E5. Onset consistency + full reporting (no shortcut).** Pre-register `onset_prediction_nk`
(0.663) as primary; report the base RR-only variant (0.606) and the P-wave variant (job 12)
as labeled sensitivity analyses. Reconcile README. Wire orphaned onset jobs (10/11/12) +
mechanism (13) into the DAG so the whole run is reproducible from one `submit_pipeline.sh`.

### HPC parallelization — sbatch scripts (as built)

- **`14_crossdevice_eval.sbatch`** — `#SBATCH --array=0-3` over cohorts
  {afdb, ltafdb, cinc2017, apple_watch}; each task scores RR+RF + both CNN-LSTMs on one
  cohort → `crossdevice/<cohort>.json`. `--dependency=afterok:j03:j04:j05`. GPU.
- **`15_crossdevice_stats.sbatch`** — paired DeLong Holm family + clustered CIs →
  `crossdevice_stats.json`. `--dependency=afterok:j14`.
- **`16_seed_robustness.sbatch`** — `#SBATCH --array=0-19`; each task reruns
  holdout→train→paired-eval on one seed in an **isolated run id** (no clobber), writing
  `seed_robustness/seed_<S>.json` into the primary run. `--dependency=afterok:j01`.
- `submit_pipeline.sh` rewired: after `j01` (holdout) the training tier fans out
  02/03/04/05 + onset 10/11/12 + seed array 16 in parallel; then 06/07/13/14 in parallel;
  15 after 14; **08 tables** depends on 06:07:10:11:12:13:15:16 and folds in the new JSONs;
  09 figures after 08.

### P1 — Reframe & rewrite
- **W1.** Port to World Scientific `ws-procs` template (header comment in `main.tex` shows how); ≤12 pages excluding refs.
- **W2.** Rewrite Intro with clinical/screening framing; reorder contributions.
- **W3.** Related-Work paragraph naming and rebutting RawECGNet / domain-generalization; add 2024–2026 refs.
- **W4.** Trim tabular clinical + fusion to a short separate-cohort paragraph.
- **W5.** Tighten abstract/conclusion to the bounded, now-provable claim.

### P2 — Polish & submission logistics
- **F1.** Regenerate figures/tables from the new `run_id`; add a cross-device degradation figure.
- **F2.** Reproducibility statement + archived DOI (Zenodo at submission commit); README/manifest fixups.
- **F3.** Cover letter (`paper/cover_letter.md`): corresponding-author email, session, originality, co-author approval, **LLM-authoring disclosure**.
- **F4.** Final checks: 12-page limit, PDF-only, filename `shah.pdf`, supplement-by-URL.

### Explicitly out of scope this cycle (future work)
- Deployable clinical+ECG fusion (needs same-patient paired data).
- Purpose-built onset dataset with P-wave morphology + longer pre-onset context.
- Prospective / multi-generation Apple Watch validation with adjudicated labels.

---

## Critical files
- **Writing:** `paper/main.tex`, `paper/references.bib`, `paper/cover_letter.md`, `paper/SUBMISSION_CHECKLIST.md`.
- **New experiments:** `src/evaluation/crossdevice_eval.py`, `src/evaluation/crossdevice_stats.py`,
  `src/evaluation/seed_robustness.py`, `src/evaluation/confidence_intervals.py` (clustered bootstrap),
  `src/evaluation/result_tables.py` (new tables).
- **New sbatch jobs:** `scripts/hpc/14_crossdevice_eval.sbatch`, `15_crossdevice_stats.sbatch`,
  `16_seed_robustness.sbatch`; `scripts/hpc/submit_pipeline.sh`, `scripts/hpc/08_stats_tables.sbatch`.
- **Protocol / reproducibility:** `docs/results/ANALYSIS_PLAN.md` (cross-device amendment),
  `src/experiments/make_cpsc_holdout.py` (split_unit note), `README.md` (AUC/onset fixups).

## Verification
1. **Amend the pre-registration first** — done and committed before any new job.
2. **Reproduce end-to-end, parallel** via `scripts/hpc/submit_pipeline.sh` → fresh
   `docs/results/<run_id>/` with `crossdevice/*.json`, `crossdevice_stats.json`,
   `seed_robustness.json`, updated `stat_tests.json` + `paper_tables.json`.
3. **Provable-claim gate:** every external cohort shows RR+RF AUC with a patient-clustered
   95% CI and a paired-DeLong p vs the deep model; the primary null survives the
   seed-robustness distribution. No cross-device claim rests on a bare point estimate.
4. **Consistency gate:** every number in `main.tex` traces to the new `run_id` JSON; the 14
   `\XX` placeholders in the cross-device table are all filled (`grep -c '\\XX' paper/main.tex`
   → 0); README, paper, and results agree.
5. **Build the PDF** in `ws-procs`; ≤12 pages excluding references; all figures render.
6. **Submission dry-run:** cover letter complete (session + LLM disclosure), filename
   `shah.pdf`, PDF-only, supplement-by-URL.
