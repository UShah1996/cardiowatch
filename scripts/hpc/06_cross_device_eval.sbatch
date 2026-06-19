#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-06-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

source scripts/hpc/common.sh

"${CARDIOWATCH_PYTHON}" -m src.evaluation.paired_cpsc_eval \
  --manifest "${CARDIOWATCH_MANIFEST}" \
  --out "${CARDIOWATCH_RESULTS_DIR}/paired_cpsc_predictions.json"

if [[ -f data/processed/rr_rf_model.pkl || -f data/processed/rr_rf_cpsc_complement.pkl ]]; then
  "${CARDIOWATCH_PYTHON}" src/evaluation/evaluate_mitbih_afib.py || true
fi

if [[ -d data/apple_health_export ]]; then
  "${CARDIOWATCH_PYTHON}" src/models/build_fusion_apple_watch.py || true
fi

