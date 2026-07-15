# Reproducibility Archival — Zenodo DOI (PSB 2027)

PSB requires supplemental material to be referenced **by URL only**. We satisfy
this with a permanent, versioned snapshot of the repository at the exact commit
that produced the submitted numbers, archived on Zenodo with a citable DOI. This
is the URL the manuscript's Reproducibility section points to.

Do this **after** the final pre-registered HPC run and after all `\XX` cells in
`paper/main.tex` are filled, so the archived snapshot matches the submitted PDF.

## One-time setup
1. Sign in to <https://zenodo.org> with your GitHub account (or ORCID).
2. Zenodo → **Settings → GitHub**, and toggle the CardioWatch repository **On**.
   Zenodo now watches the repo and archives any GitHub *Release* automatically.
   (If the repo is private, either make it public first or use the manual
   upload path in "Alternative" below.)

## Freeze the submission snapshot
1. Ensure the working tree is clean and the submitted `run_id` results are
   committed (`docs/results/<run_id>/`, `paper/`, `ANALYSIS_PLAN.md`).
2. Tag and create a GitHub Release from the submission commit:
   ```bash
   git tag -a psb2027-submission -m "PSB 2027 submission snapshot (run_id=<run_id>)"
   git push origin psb2027-submission        # only when you are ready to publish
   gh release create psb2027-submission \
     --title "CardioWatch — PSB 2027 submission" \
     --notes "Frozen snapshot for the PSB 2027 submission. run_id=<run_id>, git SHA=<sha>."
   ```
   > Per the current instruction, do **not** push/publish until you decide to.
   > Everything above the `git push` line can be prepared locally first.
3. Zenodo mints a DOI for the release within a minute or two. Two DOIs appear:
   a **version DOI** (this exact snapshot) and a **concept DOI** (all versions).
   Cite the **version DOI** in the paper.

## Wire the DOI into the manuscript
Replace the placeholder sentence in the Reproducibility section of
`paper/main.tex`:
```
a versioned snapshot will be archived with a DOI.
```
with the concrete DOI + badge, e.g.:
```
A versioned snapshot of the code, analysis plan, hold-out manifest checksum, and
compact result files is archived at \url{https://doi.org/10.5281/zenodo.XXXXXXX}.
```
Also add the DOI badge to `README.md` near the other badges:
```
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

## What the archive must contain (and must not)
- **Include:** all code (`src/`, `scripts/`), `configs/`, `docs/results/<run_id>/`
  compact JSON/MD summaries + `manifests/` checksums + `provenance.json` +
  `requirements.lock`, `paper/`, `docs/*.md`, `README.md`.
- **Exclude (stays gitignored):** raw ECG corpora (CPSC/MIT-BIH/CinC/PhysioNet),
  model checkpoints (`data/processed/*.pt`, `*.pkl`), and the personal Apple Watch
  exports (PHI). The manifest checksums + `ANALYSIS_PLAN.md` let a third party
  reproduce from the public source datasets without redistributing them.

## Alternative — manual upload (private repo / no GitHub integration)
1. `git archive --format=zip -o cardiowatch-psb2027.zip psb2027-submission`
2. Zenodo → **New upload** → attach the zip, set authors/title/license (MIT),
   description, and the related-identifier = GitHub repo URL → **Publish** → DOI.

## Checklist
- [ ] Final run complete; all `\XX` filled; numbers consistent across paper/README/results.
- [ ] Submission commit tagged `psb2027-submission`.
- [ ] Zenodo release archived; **version DOI** obtained.
- [ ] DOI inserted into `paper/main.tex` Reproducibility + `README.md` badge.
- [ ] Archive excludes raw data / checkpoints / PHI; includes manifests + provenance.
