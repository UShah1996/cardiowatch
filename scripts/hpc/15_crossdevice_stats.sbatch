#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-15-crossdev-stats
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpuqs
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# E3 — provable cross-device result: paired DeLong (RR+RF vs CNN-LSTM-CPSC) per
# cohort as one Holm family, plus patient-clustered bootstrap 95% CIs on every
# external AUC. Reads docs/results/<run_id>/crossdevice/*.json from job 14.
source scripts/hpc/common.sh

"${CARDIOWATCH_PYTHON}" -m src.evaluation.crossdevice_stats \
  --run-dir "${CARDIOWATCH_RESULTS_DIR}" \
  --n-boot 2000 --seed 42 \
  --out "${CARDIOWATCH_RESULTS_DIR}/crossdevice_stats.json"
