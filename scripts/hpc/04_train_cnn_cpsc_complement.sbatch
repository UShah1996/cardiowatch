#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-04-cnn-cpsc
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpuqs
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G

source scripts/hpc/common.sh
"${CARDIOWATCH_PYTHON}" -m src.experiments.train_cnn_holdout --manifest "${CARDIOWATCH_MANIFEST}" --mode cpsc
