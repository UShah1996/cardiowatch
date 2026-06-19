#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-09-figures
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

source scripts/hpc/common.sh

"${CARDIOWATCH_PYTHON}" -m src.evaluation.make_figures \
  --paired-json "${CARDIOWATCH_RESULTS_DIR}/paired_cpsc_predictions.json" \
  --latency-json "${CARDIOWATCH_RESULTS_DIR}/latency_bootstrap.json" \
  --out-dir "${CARDIOWATCH_RESULTS_DIR}/figures"
