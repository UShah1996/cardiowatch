#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-02-clinical
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpuqs
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

source scripts/hpc/common.sh
"${CARDIOWATCH_PYTHON}" src/models/random_forest.py
"${CARDIOWATCH_PYTHON}" src/models/xgboost_model.py

