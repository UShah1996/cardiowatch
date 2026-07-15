#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-08-tables
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpuqs
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

source scripts/hpc/common.sh

R="${CARDIOWATCH_RESULTS_DIR}"

"${CARDIOWATCH_PYTHON}" -m src.evaluation.stat_tests \
  --paired-json "${R}/paired_cpsc_predictions.json" \
  --out "${R}/stat_tests.json"

# Reduce the seed-robustness array (job 16) into one summary. Tolerate absence
# so the tables still build if the sensitivity array was not run.
if [[ -d "${R}/seed_robustness" ]]; then
  "${CARDIOWATCH_PYTHON}" -m src.evaluation.seed_robustness reduce \
    --dir "${R}/seed_robustness" --out "${R}/seed_robustness.json" || true
fi

"${CARDIOWATCH_PYTHON}" -m src.evaluation.result_tables \
  --paired-cpsc-json "${R}/paired_cpsc_predictions.json" \
  --latency-json "${R}/latency_bootstrap.json" \
  --clinical-json "${R}/clinical_eval.json" \
  --mitbih-json "${R}/mitbih_eval.json" \
  --apple-watch-json "${R}/apple_watch_eval.json" \
  --stats-json "${R}/stat_tests.json" \
  --onset-json "${R}/onset_prediction_nk.json" \
  --mechanism-json "${R}/representation_shift.json" \
  --crossdevice-json "${R}/crossdevice_stats.json" \
  --seed-robustness-json "${R}/seed_robustness.json" \
  --out-prefix "${R}/paper_tables"
