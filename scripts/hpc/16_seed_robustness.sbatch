#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-16-seedrobust
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpuqs
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --array=0-19

# E4 — SECONDARY sensitivity: rerun holdout -> train(RR, CNN-CPSC) -> paired-eval
# over 20 alternate hold-out seeds, one per array task (parallel). Each task runs
# in an ISOLATED run id so its manifests/checkpoints/summaries never clobber the
# primary run; only the tiny per-seed result is written back into the primary
# run's seed_robustness/ folder. The seed-42 primary is unchanged by this job.
source scripts/hpc/common.sh

PRIMARY_RESULTS_DIR="${CARDIOWATCH_RESULTS_DIR}"
SEED=$((1000 + SLURM_ARRAY_TASK_ID))

# Isolate every incidental write (provenance, training summaries) under a
# per-seed run id so parallel tasks and the primary run do not collide.
export CARDIOWATCH_RUN_ID="${CARDIOWATCH_RUN_ID}__seed${SEED}"
export CARDIOWATCH_RESULTS_DIR="docs/results/${CARDIOWATCH_RUN_ID}"
SDIR="${CARDIOWATCH_RESULTS_DIR}"
MANIFEST="${SDIR}/manifests/cpsc_holdout.json"
PROC="${SDIR}/processed"
mkdir -p "${SDIR}" "${PROC}"

echo "seed-robustness SEED=${SEED} -> ${SDIR}"

"${CARDIOWATCH_PYTHON}" -m src.experiments.make_cpsc_holdout --seed "${SEED}" --out "${MANIFEST}"
"${CARDIOWATCH_PYTHON}" -m src.experiments.train_rr_complement \
  --manifest "${MANIFEST}" --processed-dir "${PROC}" --seed "${SEED}"
"${CARDIOWATCH_PYTHON}" -m src.experiments.train_cnn_holdout \
  --manifest "${MANIFEST}" --mode cpsc --processed-dir "${PROC}" --seed "${SEED}"
"${CARDIOWATCH_PYTHON}" -m src.evaluation.paired_cpsc_eval \
  --manifest "${MANIFEST}" \
  --rr-model "${PROC}/rr_rf_cpsc_complement.pkl" \
  --cnn-cpsc-checkpoint "${PROC}/cnn_lstm_cpsc_complement.pt" \
  --cnn-combined-checkpoint "${PROC}/__none__.pt" \
  --out "${SDIR}/paired_cpsc_predictions.json"

# Write the single per-seed result back into the PRIMARY run folder.
mkdir -p "${PRIMARY_RESULTS_DIR}/seed_robustness"
"${CARDIOWATCH_PYTHON}" -m src.evaluation.seed_robustness extract \
  --paired-json "${SDIR}/paired_cpsc_predictions.json" \
  --seed "${SEED}" \
  --out "${PRIMARY_RESULTS_DIR}/seed_robustness/seed_${SEED}.json"
