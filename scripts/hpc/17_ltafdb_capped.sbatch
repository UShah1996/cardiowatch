#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-17-ltafdb
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=cpuqs
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G

# Recovers the pre-registered LTAFDB cohort, which could not be scored
# exhaustively (84 records x ~24 h ~= 725k windows exceeded the wall clock on
# three attempts, on both gpuqs and cpuqs).
#
# We instead score a bounded, seeded, uniformly-random subsample of at most 300
# windows per record (~25k windows total). Random-without-replacement sampling
# avoids the aliasing failure mode of fixed-stride sampling; record identity is
# preserved so the patient-clustered CIs stay valid — the subsample only widens
# them. The cap and seed are recorded in the output JSON.
#
# CPU by design: this is inference only, and the GPU queue was the source of
# repeated "CUDA device busy" failures when other jobs held the GPUs.
source scripts/hpc/common.sh

"${CARDIOWATCH_PYTHON}" -m src.evaluation.crossdevice_eval \
  --cohort ltafdb \
  --max-windows-per-record 300 --seed 42 \
  --rr-model data/processed/rr_rf_cpsc_complement.pkl \
  --cnn-cpsc-checkpoint data/processed/cnn_lstm_cpsc_complement.pt \
  --cnn-combined-checkpoint data/processed/cnn_lstm_combined_deploy.pt
