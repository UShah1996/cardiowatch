#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-10-onset
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpuqs
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G

# EXPLORATORY: AFib onset (early-warning) prediction on MIT-BIH afdb.
# Trains its own RR/HRV model on pre-onset SINUS windows with record-grouped CV;
# does NOT use or retrain the detection models. Safe to run standalone after
# data is staged (no dependency on the main pipeline outputs).
source scripts/hpc/common.sh

"${CARDIOWATCH_PYTHON}" -m src.evaluation.afib_onset_prediction \
  --afdb-dir data/raw/mit_afib/files \
  --horizon-min 30 --far-min 60 --window-sec 30 --stride-sec 30 \
  --k-sustained 2 --target-specificity 0.95 \
  --out "${CARDIOWATCH_RESULTS_DIR}/onset_prediction.json" \
  --fig "${CARDIOWATCH_RESULTS_DIR}/figures/onset_prediction_lead_time.png"
