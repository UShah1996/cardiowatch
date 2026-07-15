#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-14-crossdev
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpuqs
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --array=0-3

# E1/E2 — zero-shot cross-device evaluation, ONE cohort per array task so all
# four cohorts run in parallel. Each task scores RR+RF and both CNN-LSTMs on
# the SAME 10 s windows (combined CNN is auto-omitted for cinc2017 to avoid
# train-on-test). GPU because the CNN scores every window.
source scripts/hpc/common.sh

COHORTS=(afdb ltafdb cinc2017 apple_watch)
COHORT="${COHORTS[$SLURM_ARRAY_TASK_ID]}"
echo "cross-device cohort = ${COHORT}"

"${CARDIOWATCH_PYTHON}" -m src.evaluation.crossdevice_eval \
  --cohort "${COHORT}" \
  --rr-model data/processed/rr_rf_cpsc_complement.pkl \
  --cnn-cpsc-checkpoint data/processed/cnn_lstm_cpsc_complement.pt \
  --cnn-combined-checkpoint data/processed/cnn_lstm_combined_deploy.pt
