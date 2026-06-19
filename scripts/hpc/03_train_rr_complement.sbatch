#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-03-rr
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpuqs
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G

source scripts/hpc/common.sh
"${CARDIOWATCH_PYTHON}" -m src.experiments.train_rr_complement --manifest "${CARDIOWATCH_MANIFEST}"

