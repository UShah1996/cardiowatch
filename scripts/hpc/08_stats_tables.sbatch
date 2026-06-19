#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-08-tables
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpuqs
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

source scripts/hpc/common.sh

"${CARDIOWATCH_PYTHON}" -m src.evaluation.stat_tests \
  --paired-json "${CARDIOWATCH_RESULTS_DIR}/paired_cpsc_predictions.json" \
  --out "${CARDIOWATCH_RESULTS_DIR}/stat_tests.json"

"${CARDIOWATCH_PYTHON}" -m src.evaluation.result_tables \
  --paired-cpsc-json "${CARDIOWATCH_RESULTS_DIR}/paired_cpsc_predictions.json" \
  --latency-json "${CARDIOWATCH_RESULTS_DIR}/latency_bootstrap.json" \
  --out-prefix "${CARDIOWATCH_RESULTS_DIR}/paper_tables"
