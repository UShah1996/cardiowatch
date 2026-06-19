#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-07-latency
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpuqs
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source scripts/hpc/common.sh
"${CARDIOWATCH_PYTHON}" -m src.evaluation.latency_bootstrap \
  --out "${CARDIOWATCH_RESULTS_DIR}/latency_bootstrap.json"

