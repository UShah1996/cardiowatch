#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-05-cnn-combined
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

source scripts/hpc/common.sh
"${CARDIOWATCH_PYTHON}" -m src.experiments.train_cnn_holdout --manifest "${CARDIOWATCH_MANIFEST}" --mode combined

