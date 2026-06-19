# SJSU COE HPC Runbook

This project is set up to run long CardioWatch experiments through the San Jose State University College of Engineering HPC JupyterHub terminal and SLURM.

## 1. Log In

- Open the COE JupyterHub URL: <https://coe-hpc2.sjsu.edu:8000/>
- Log in with your COE HPC account credentials.
- Open a Terminal from the JupyterLab launcher.
- Do not run long training jobs inside a notebook. Use `sbatch`.

## 2. Clone Or Update The Repo

```bash
git clone https://github.com/UShah1996/cardiowatch.git
cd cardiowatch
git checkout fix/clinical-methodology-and-dashboard
git pull
```

## 3. Prepare Python

Use a repo-local virtual environment unless the cluster admins provide a preferred environment.

```bash
bash scripts/hpc/sjsu_setup.sh
```

If the cluster requires a specific Python path:

```bash
export CARDIOWATCH_PYTHON=/path/to/python
bash scripts/hpc/sjsu_setup.sh
```

## 4. Stage Data On HPC Storage

Place large datasets on the HPC filesystem, not in git. The expected layout is:

```text
data/raw/heart.csv                                                                 # tracked in repo
data/raw/classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2/training/cpsc_2018/
data/raw/challenge_2017/training2017/
data/raw/mit_afib/files/
data/apple_health_export/        # optional/private Apple Watch validation
```

`data/raw/`, `data/processed/`, checkpoints, CSVs, and Apple exports are ignored by git.

### Automated staging (recommended)

Run the helper **on a login node** (the JupyterLab terminal — compute nodes have
no outbound network). It downloads CPSC 2018 (Kaggle), PhysioNet 2017, and
MIT-BIH AFib to a scratch directory, symlinks them into `data/raw/`, checks file
counts, and runs the preflight validator:

```bash
source .venv/bin/activate                  # the env from sjsu_setup.sh
# CPSC needs Kaggle credentials at ~/.kaggle/kaggle.json (or set KAGGLE_JSON=...)
CARDIOWATCH_DATA=/scratch/$USER/cardiowatch_data bash scripts/hpc/stage_data.sh
```

The script is idempotent — datasets already meeting the expected counts are
skipped. If the Kaggle CPSC folder unzips under a different top-level name, or the
PhysioNet 2017 mirror URL changes, the helper prints the exact fallback step.

### Manual check

To verify staging without the helper:

```bash
find -L data/raw/classification-*/training/cpsc_2018 -name '*.hea' | wc -l   # expect >= 6800
find -L data/raw/challenge_2017 -name '*.mat'        | wc -l                  # expect >= 8200
find -L data/raw/mit_afib       -name '*.dat'        | wc -l                  # expect >= 20
python -m src.experiments.validate_data --require-p17 --require-mit
```

## 5. Submit The Full Pipeline

```bash
bash scripts/hpc/submit_pipeline.sh
```

The helper creates one shared `CARDIOWATCH_RUN_ID`, submits jobs with `sbatch`, and chains dependencies with `--dependency=afterok`.

If you want to submit every job manually with `sbatch`, use this exact sequence:

```bash
export CARDIOWATCH_RUN_ID="$(date -u +%Y%m%d)-$(git rev-parse --short HEAD)"
mkdir -p "docs/results/${CARDIOWATCH_RUN_ID}" "logs/${CARDIOWATCH_RUN_ID}"
echo "${CARDIOWATCH_RUN_ID}" > docs/results/latest_run_id.txt

j00=$(sbatch --parsable --export=ALL,CARDIOWATCH_RUN_ID="${CARDIOWATCH_RUN_ID}" scripts/hpc/00_validate_data.sbatch)
j01=$(sbatch --parsable --dependency=afterok:${j00} --export=ALL,CARDIOWATCH_RUN_ID="${CARDIOWATCH_RUN_ID}" scripts/hpc/01_make_holdout.sbatch)
j02=$(sbatch --parsable --dependency=afterok:${j01} --export=ALL,CARDIOWATCH_RUN_ID="${CARDIOWATCH_RUN_ID}" scripts/hpc/02_train_clinical.sbatch)
j03=$(sbatch --parsable --dependency=afterok:${j01} --export=ALL,CARDIOWATCH_RUN_ID="${CARDIOWATCH_RUN_ID}" scripts/hpc/03_train_rr_complement.sbatch)
j04=$(sbatch --parsable --dependency=afterok:${j01} --export=ALL,CARDIOWATCH_RUN_ID="${CARDIOWATCH_RUN_ID}" scripts/hpc/04_train_cnn_cpsc_complement.sbatch)
j05=$(sbatch --parsable --dependency=afterok:${j01} --export=ALL,CARDIOWATCH_RUN_ID="${CARDIOWATCH_RUN_ID}" scripts/hpc/05_train_cnn_combined_deploy.sbatch)
j06=$(sbatch --parsable --dependency=afterok:${j03}:${j04}:${j05} --export=ALL,CARDIOWATCH_RUN_ID="${CARDIOWATCH_RUN_ID}" scripts/hpc/06_cross_device_eval.sbatch)
j07=$(sbatch --parsable --dependency=afterok:${j04} --export=ALL,CARDIOWATCH_RUN_ID="${CARDIOWATCH_RUN_ID}" scripts/hpc/07_latency_bootstrap.sbatch)
j08=$(sbatch --parsable --dependency=afterok:${j06}:${j07} --export=ALL,CARDIOWATCH_RUN_ID="${CARDIOWATCH_RUN_ID}" scripts/hpc/08_stats_tables.sbatch)

printf "Submitted jobs:\n00 %s\n01 %s\n02 %s\n03 %s\n04 %s\n05 %s\n06 %s\n07 %s\n08 %s\n" \
  "$j00" "$j01" "$j02" "$j03" "$j04" "$j05" "$j06" "$j07" "$j08"
```

Monitor jobs:

```bash
squeue -u $USER
```

Review logs:

```bash
ls -lh logs/
tail -n 100 logs/cardiowatch-*.out
```

## 6. Commit Only Compact Results

After jobs finish, inspect:

```bash
cat docs/results/latest_run_id.txt
ls docs/results/$(cat docs/results/latest_run_id.txt)
```

Commit only:

- `docs/results/ANALYSIS_PLAN.md`
- `docs/results/<run_id>/*.json`
- `docs/results/<run_id>/*.md`
- selected final figures (`*.png`)

Do not commit raw ECG, Apple exports, `.csv`, `.pkl`, `.pt`, or full logs.

## 7. If A Job Fails

- Check `squeue -u $USER`.
- Read the matching SLURM output/error file.
- Fix the issue, then resubmit from the failed stage manually, preserving the same run ID:

```bash
export CARDIOWATCH_RUN_ID=$(cat docs/results/latest_run_id.txt)
sbatch scripts/hpc/03_train_rr_complement.sbatch
```

For dependent downstream stages, resubmit with `--dependency=afterok:<jobid>` or rerun `submit_pipeline.sh` with a fresh run ID.
