#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-01-holdout
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

source scripts/hpc/common.sh
"${CARDIOWATCH_PYTHON}" -m src.experiments.make_cpsc_holdout --out "${CARDIOWATCH_MANIFEST}"

