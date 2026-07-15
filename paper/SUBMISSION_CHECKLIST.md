# PSB 2027 Submission Checklist

Deadline: **Aug 3, 2026, 11:59 PM PT**. Notifications Sep 8, 2026; camera-ready
Oct 1, 2026. Publisher: World Scientific; proceedings indexed in PubMed.

## Before submission — content
- [ ] **Run the pre-registered HPC pipeline** (`bash scripts/hpc/submit_pipeline.sh`)
      to produce a fresh `docs/results/<run_id>/`.
- [ ] **Fill every `\XX` cell** in `paper/main.tex` Table~\ref{tab:crossdevice}
      from `crossdevice_stats.json` (RR+RF and CNN-LSTM(CPSC) AUC + clustered CIs,
      ΔAUC, Holm DeLong p per cohort). There are 14 `\XX` markers — grep to confirm
      none remain: `grep -c '\\XX' paper/main.tex` must print `0`.
- [ ] **Consistency gate:** every number in `main.tex` traces to the new run's
      JSON (`paper_tables.json`, `stat_tests.json`, `crossdevice_stats.json`,
      `seed_robustness.json`). README, paper, and results agree.
- [ ] Fill the seed-robustness sentence in Table~\ref{tab:ecg} footnote from
      `seed_robustness.json` (n significant / 20, ΔAUC range).
- [ ] Insert real figures (ROC + cross-device degradation) from
      `docs/results/<run_id>/figures/` (add a cross-device figure to
      `make_figures.py` if desired).

## Before submission — format (World Scientific ws-procs)
- [ ] Port `main.tex` body into the **ws-procs** template (see header comment in
      `main.tex`). Section/table/figure/cite commands transfer unchanged.
- [ ] **≤ 12 pages** excluding title page, author list, and references.
- [ ] Color figures OK at no cost. Supplemental material by **URL only** (PSB does
      not host it) — point to the GitHub repo + archived DOI snapshot.
- [ ] Output **PDF only**; filename **`shah.pdf`** (last name of first author).

## Cover letter (`paper/cover_letter.md`)
- [ ] Corresponding-author email present.
- [ ] Names the specific session: *AI and Machine Learning in Clinical Medicine*.
- [ ] States originality / unpublished / not under review.
- [ ] Confirms co-author approval.
- [ ] **Discloses the role LLMs played in authoring** (PSB 2027 new requirement).

## Reproducibility / archival
- [ ] Tag the submission commit and create a **Zenodo DOI** snapshot of the repo.
- [ ] Add the DOI + repo URL to the manuscript's Reproducibility section.
- [ ] Confirm `docs/results/ANALYSIS_PLAN.md` (with the cross-device amendment) is
      committed **before** the final run, preserving pre-registration.

## Notes / open decisions for the author
- The manuscript keeps the tabular clinical model as a short "separate cohort"
  paragraph; cut entirely if page budget is tight.
- Apple Watch stays a plausibility check (1 positive) — do not promote to a
  headline sensitivity number.
- If the deep-model zero-shot cells come back *not* showing clear degradation, the
  honest move is to soften the abstract/conclusion accordingly rather than force
  the narrative — the pre-registration makes that defensible.
