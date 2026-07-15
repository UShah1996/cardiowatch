#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${CARDIOWATCH_RUN_ID:-$(date -u +%Y%m%d)-$(git rev-parse --short HEAD)}"
export CARDIOWATCH_RUN_ID="${RUN_ID}"

mkdir -p "logs/${RUN_ID}" "docs/results/${RUN_ID}"
printf '%s\n' "${RUN_ID}" > docs/results/latest_run_id.txt

echo "Submitting CardioWatch pipeline with RUN_ID=${RUN_ID}"

sb() { sbatch --parsable --export=ALL,CARDIOWATCH_RUN_ID="${RUN_ID}" "$@"; }

# ── Tier 0: validate, then create the pre-registered holdout ────────────────
j00=$(sb scripts/hpc/00_validate_data.sbatch)
j01=$(sb --dependency=afterok:${j00} scripts/hpc/01_make_holdout.sbatch)

# ── Tier A: everything that only needs the holdout, fanned out in parallel ──
j02=$(sb --dependency=afterok:${j01} scripts/hpc/02_train_clinical.sbatch)
j03=$(sb --dependency=afterok:${j01} scripts/hpc/03_train_rr_complement.sbatch)
j04=$(sb --dependency=afterok:${j01} scripts/hpc/04_train_cnn_cpsc_complement.sbatch)
j05=$(sb --dependency=afterok:${j01} scripts/hpc/05_train_cnn_combined_deploy.sbatch)
# Exploratory onset (nk = pre-registered primary) + mechanism, previously orphaned.
j10=$(sb --dependency=afterok:${j01} scripts/hpc/10_onset_prediction.sbatch)
j11=$(sb --dependency=afterok:${j01} scripts/hpc/11_onset_neurokit.sbatch)
j12=$(sb --dependency=afterok:${j01} scripts/hpc/12_onset_neurokit_pwave.sbatch)
# E4 secondary sensitivity — 20-seed array, isolated run ids (parallel).
j16=$(sb --dependency=afterok:${j01} scripts/hpc/16_seed_robustness.sbatch)

# ── Tier B: evaluations, fanned out off their training deps ─────────────────
j06=$(sb --dependency=afterok:${j03}:${j04}:${j05} scripts/hpc/06_cross_device_eval.sbatch)
j07=$(sb --dependency=afterok:${j04} scripts/hpc/07_latency_bootstrap.sbatch)
j13=$(sb --dependency=afterok:${j04} scripts/hpc/13_mechanism.sbatch)
# E1/E2 head-to-head cross-device eval — array over 4 cohorts (parallel).
j14=$(sb --dependency=afterok:${j03}:${j04}:${j05} scripts/hpc/14_crossdevice_eval.sbatch)

# ── Tier C: cross-device stats (needs all cohort files from the j14 array) ──
j15=$(sb --dependency=afterok:${j14} scripts/hpc/15_crossdevice_stats.sbatch)

# ── Tier D: canonical stats + tables (folds in cross-device, onset, seeds) ──
j08=$(sb --dependency=afterok:${j06}:${j07}:${j10}:${j11}:${j12}:${j13}:${j15}:${j16} \
        scripts/hpc/08_stats_tables.sbatch)
j09=$(sb --dependency=afterok:${j08} scripts/hpc/09_figures.sbatch)

cat <<EOF
Submitted (RUN_ID=${RUN_ID}):
  00 validate        ${j00}
  01 holdout         ${j01}
  02 clinical        ${j02}
  03 rr complement   ${j03}
  04 cnn cpsc        ${j04}
  05 cnn combined    ${j05}
  06 cross-dev (RR)  ${j06}
  07 latency         ${j07}
  10 onset base      ${j10}
  11 onset nk*       ${j11}   (* pre-registered primary onset variant)
  12 onset pwave     ${j12}
  13 mechanism       ${j13}
  14 crossdev eval   ${j14}   (array 0-3: afdb,ltafdb,cinc2017,apple_watch)
  15 crossdev stats  ${j15}
  16 seed robustness ${j16}   (array 0-19: secondary sensitivity)
  08 stats+tables    ${j08}
  09 figures         ${j09}
EOF
