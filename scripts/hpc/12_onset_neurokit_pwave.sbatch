#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-12-onset-pwave
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=cpuqs
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# EXPLORATORY: onset prediction with the FULL feature set including P-wave
# indices (--pwave; delineation is the slowest step). Same pool/stride/config
# as job 11 so the only difference vs onset_prediction_nk.json is the P-wave
# features — isolates their contribution. Writes a separate output file.
# If this times out at 12 h, re-run with a coarser --stride-sec 90.
source scripts/hpc/common.sh

"${CARDIOWATCH_PYTHON}" -m src.evaluation.afib_onset_prediction \
  --data-dirs afdb:data/raw/mit_afib/files ltafdb:data/raw/ltafdb/files \
  --features both --pwave --normalize-per-patient \
  --horizon-min 30 --far-min 60 --window-sec 30 --stride-sec 60 \
  --k-sustained 2 --target-specificity 0.95 \
  --out "${CARDIOWATCH_RESULTS_DIR}/onset_prediction_nk_pwave.json" \
  --fig "${CARDIOWATCH_RESULTS_DIR}/figures/onset_pwave_lead_time.png"
