#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-10-onset
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=cpuqs
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# EXPLORATORY: AFib onset (early-warning) prediction, pooled across afdb +
# ltafdb (both standard WFDB rhythm-annotated). Trains its own RR/HRV model on
# pre-onset SINUS windows with patient(record)-grouped CV; does NOT use or
# retrain the detection models. Missing data dirs are skipped automatically.
# Stage ltafdb first (login node):
#   python -c "import wfdb; wfdb.dl_database('ltafdb', 'data/raw/ltafdb/files')"
source scripts/hpc/common.sh

"${CARDIOWATCH_PYTHON}" -m src.evaluation.afib_onset_prediction \
  --data-dirs afdb:data/raw/mit_afib/files ltafdb:data/raw/ltafdb/files \
  --horizon-min 30 --far-min 60 --window-sec 30 --stride-sec 30 \
  --k-sustained 2 --target-specificity 0.95 \
  --out "${CARDIOWATCH_RESULTS_DIR}/onset_prediction.json" \
  --fig "${CARDIOWATCH_RESULTS_DIR}/figures/onset_prediction_lead_time.png"
