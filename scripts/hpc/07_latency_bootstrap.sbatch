#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-07-latency
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpuqs
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source scripts/hpc/common.sh
# Fine stride + off-grid normal length so AFib onset does not land exactly on a
# window boundary (which forces latency==0). Gives a real latency distribution.
"${CARDIOWATCH_PYTHON}" -m src.evaluation.latency_bootstrap \
  --stride-sec 5 --normal-minutes 34.7 --afib-minutes 30.3 \
  --out "${CARDIOWATCH_RESULTS_DIR}/latency_bootstrap.json"

