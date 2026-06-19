#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-11-onset-nk
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=cpuqs
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# EXPLORATORY: AFib onset prediction with the RICHER feature path —
# neurokit2 ectopy + expanded HRV markers, plus per-patient baseline
# normalization, pooled over afdb + ltafdb. neurokit features are slow, so
# stride is coarsened to 60 s (60 s lead-time resolution) to bound runtime.
#
# RECOMMENDED: pilot on afdb ALONE first to gauge per-window cost before the
# full pooled run, e.g. (on a login node or a short job):
#   python -m src.evaluation.afib_onset_prediction \
#     --data-dirs afdb:data/raw/mit_afib/files \
#     --features both --normalize-per-patient --stride-sec 60 \
#     --out /tmp/onset_pilot.json
# Add --pwave only if the pilot finishes comfortably (P-wave delineation is
# the slowest step).
source scripts/hpc/common.sh

"${CARDIOWATCH_PYTHON}" -m src.evaluation.afib_onset_prediction \
  --data-dirs afdb:data/raw/mit_afib/files ltafdb:data/raw/ltafdb/files \
  --features both --normalize-per-patient \
  --horizon-min 30 --far-min 60 --window-sec 30 --stride-sec 60 \
  --k-sustained 2 --target-specificity 0.95 \
  --out "${CARDIOWATCH_RESULTS_DIR}/onset_prediction_nk.json" \
  --fig "${CARDIOWATCH_RESULTS_DIR}/figures/onset_prediction_nk_lead_time.png"
