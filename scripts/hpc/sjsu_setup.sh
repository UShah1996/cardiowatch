#!/usr/bin/env bash
set -euo pipefail

# Run this from the SJSU COE JupyterHub terminal after cloning/pulling.
# It prepares a repo-local Python environment without touching raw data
# or generated results.

PYTHON_BIN="${CARDIOWATCH_PYTHON:-python3}"
VENV_DIR="${CARDIOWATCH_VENV:-.venv}"

echo "Using Python: $(${PYTHON_BIN} --version 2>&1)"
echo "Creating/updating virtualenv: ${VENV_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

mkdir -p data/raw data/processed docs/results logs

cat <<EOF
SJSU COE HPC setup complete.

Next:
  1. Stage datasets under data/raw/ on HPC storage.
  2. Submit the chained pipeline:
       bash scripts/hpc/submit_pipeline.sh
  3. Monitor:
       squeue -u \$USER
EOF

