#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-13-mechanism
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpuqs
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Mechanism analysis (why device-agnostic features transfer): device-separability
# of the RR/HRV feature vector vs the CNN-LSTM penultimate embedding, per device
# pair. Reuses the trained CNN checkpoint + the CPSC hold-out manifest; no GPU,
# no retraining. Writes representation_shift.json into the run folder.
source scripts/hpc/common.sh

"${CARDIOWATCH_PYTHON}" -m src.evaluation.representation_shift \
  --manifest "${CARDIOWATCH_RESULTS_DIR}/manifests/cpsc_holdout.json" \
  --apple-dir data/apple_health_export \
  --mit-dir data/raw/mit_afib/files \
  --max-per-device 500 \
  --out "${CARDIOWATCH_RESULTS_DIR}/representation_shift.json"
