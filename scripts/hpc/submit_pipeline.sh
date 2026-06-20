#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${CARDIOWATCH_RUN_ID:-$(date -u +%Y%m%d)-$(git rev-parse --short HEAD)}"
export CARDIOWATCH_RUN_ID="${RUN_ID}"

mkdir -p "logs/${RUN_ID}" "docs/results/${RUN_ID}"
printf '%s\n' "${RUN_ID}" > docs/results/latest_run_id.txt

echo "Submitting CardioWatch pipeline with RUN_ID=${RUN_ID}"

j00=$(sbatch --parsable --export=ALL,CARDIOWATCH_RUN_ID="${RUN_ID}" scripts/hpc/00_validate_data.sbatch)
j01=$(sbatch --parsable --dependency=afterok:${j00} --export=ALL,CARDIOWATCH_RUN_ID="${RUN_ID}" scripts/hpc/01_make_holdout.sbatch)
j02=$(sbatch --parsable --dependency=afterok:${j01} --export=ALL,CARDIOWATCH_RUN_ID="${RUN_ID}" scripts/hpc/02_train_clinical.sbatch)
j03=$(sbatch --parsable --dependency=afterok:${j01} --export=ALL,CARDIOWATCH_RUN_ID="${RUN_ID}" scripts/hpc/03_train_rr_complement.sbatch)
j04=$(sbatch --parsable --dependency=afterok:${j01} --export=ALL,CARDIOWATCH_RUN_ID="${RUN_ID}" scripts/hpc/04_train_cnn_cpsc_complement.sbatch)
j05=$(sbatch --parsable --dependency=afterok:${j01} --export=ALL,CARDIOWATCH_RUN_ID="${RUN_ID}" scripts/hpc/05_train_cnn_combined_deploy.sbatch)
j06=$(sbatch --parsable --dependency=afterok:${j03}:${j04}:${j05} --export=ALL,CARDIOWATCH_RUN_ID="${RUN_ID}" scripts/hpc/06_cross_device_eval.sbatch)
j07=$(sbatch --parsable --dependency=afterok:${j04} --export=ALL,CARDIOWATCH_RUN_ID="${RUN_ID}" scripts/hpc/07_latency_bootstrap.sbatch)
j08=$(sbatch --parsable --dependency=afterok:${j06}:${j07} --export=ALL,CARDIOWATCH_RUN_ID="${RUN_ID}" scripts/hpc/08_stats_tables.sbatch)
j09=$(sbatch --parsable --dependency=afterok:${j08} --export=ALL,CARDIOWATCH_RUN_ID="${RUN_ID}" scripts/hpc/09_figures.sbatch)

cat <<EOF
Submitted:
  00 validate      ${j00}
  01 holdout       ${j01}
  02 clinical      ${j02}
  03 rr            ${j03}
  04 cnn cpsc      ${j04}
  05 cnn combined  ${j05}
  06 eval          ${j06}
  07 latency       ${j07}
  08 tables        ${j08}
  09 figures       ${j09}
EOF

