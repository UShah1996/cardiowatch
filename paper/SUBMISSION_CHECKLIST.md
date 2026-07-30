# PSB 2027 Submission Checklist

Deadline: **Aug 3, 2026, 11:59 PM PT**. Notifications Sep 8, 2026; camera-ready
Oct 1, 2026. Publisher: World Scientific; proceedings indexed in PubMed.

## Before submission — content
- [x] **Cross-device run completed** (jobs 14/15 → `crossdevice_stats.json`,
      run `20260619-c7bea6b`): afdb, cinc2017, apple_watch.
- [x] **All `\XX` cells filled** from `crossdevice_stats.json`; placeholder
      machinery removed. Verify: `grep -c '\\XX' paper/main.tex` → `0`.
- [x] **Corrected MIT-BIH 0.907 → 0.871** (0.907 came from the standalone
      MIT-BIH script, not the pre-registered paired protocol). One number per cohort.
- [x] **Claims matched to evidence:** inversion holds on Holter; NULL on AliveCor
      (p=0.93); Apple Watch untestable (1 positive). Clustered CIs primary, DeLong
      secondary with a non-independence caveat (23 patients / 84,295 windows).
- [x] **ltafdb** declared pre-registered-but-excluded (compute budget), not
      silently subsampled.
- [x] **Seed-robustness claim removed** — job 16 never completed; declared as not run.
- [ ] **Consistency gate:** re-read `main.tex` against `paper_tables.md`,
      `stat_tests.json`, `crossdevice_stats.json`. README, paper, results agree.
- [ ] Insert figures from `docs/results/<run_id>/figures/`; consider adding a
      cross-device **inversion figure** (in-domain vs Holter AUC, lines crossing) —
      it is the paper's single most persuasive visual.
- [ ] *(Optional strengthener)* Re-run job 16 (seed sensitivity) if compute frees
      up; it would materially firm up the 23-patient Holter result.

## Before submission — format (World Scientific ws-procs)
- [x] `paper/main_wsprocs.tex` created (body identical to `main.tex` apart from
      `bibliographystyle`). Add PSB's `ws-procs*.cls` / `.bst` to the project;
      if the class name differs, change only the `\documentclass` line.
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

## Venue fit (do early — affects framing)
- [ ] Check the **PSB 2027 session list** when posted. PSB 2026's clinical-ML
      session was themed "Bridging or Separating Model Intelligence and Human
      Expertise"; confirm a 2027 session fits a benchmark/robustness paper.
- [ ] Fallbacks if no session fits: IEEE EMBC, JMIR / JMIR Cardio, npj Digital
      Medicine (brief). The manuscript needs only reformatting for these.

## Notes / open decisions for the author
- **Framing:** the paper is a *cautionary benchmark* (in-domain ranking inverts
  off-device), not a model proposal. PSB 2026 accepted several papers of this
  genre ("…Remains Unsolved", "The Intention-Execution Disconnect"), so keep the
  critical framing — do not soften it into a "our model is better" claim.
- The tabular clinical model is a short "separate cohort" paragraph; cut entirely
  if the page budget is tight.
- Apple Watch stays a plausibility check (1 positive) — never a headline
  sensitivity number.
- **Known reviewer targets, all already disclosed in the text:** (i) the Holter
  result rests on 23 patients; (ii) the inversion is cohort-dependent (null on
  AliveCor); (iii) two pre-registered analyses were not completed. Keeping these
  explicit is the defensible position — do not quietly drop them.
