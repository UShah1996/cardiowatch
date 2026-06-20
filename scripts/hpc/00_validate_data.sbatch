#!/usr/bin/env bash
#SBATCH --job-name=cardiowatch-00-validate
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=cpuqs
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

source scripts/hpc/common.sh

if [[ -d data/processed && -n "$(find data/processed -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  quarantine="data/processed_stale_${CARDIOWATCH_RUN_ID}"
  echo "Quarantining stale data/processed -> ${quarantine}"
  mv data/processed "${quarantine}"
fi
mkdir -p data/processed

"${CARDIOWATCH_PYTHON}" -m src.experiments.validate_data --require-p17 --require-mit

