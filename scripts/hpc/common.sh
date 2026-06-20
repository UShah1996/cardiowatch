#!/usr/bin/env bash
set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module purge
fi

export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export CARDIOWATCH_RUN_ID="${CARDIOWATCH_RUN_ID:-$(date -u +%Y%m%d)-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"
export CARDIOWATCH_RESULTS_DIR="docs/results/${CARDIOWATCH_RUN_ID}"
export CARDIOWATCH_MANIFEST="${CARDIOWATCH_MANIFEST:-${CARDIOWATCH_RESULTS_DIR}/manifests/cpsc_holdout.json}"

mkdir -p "${CARDIOWATCH_RESULTS_DIR}" "logs/${CARDIOWATCH_RUN_ID}"
printf '%s\n' "${CARDIOWATCH_RUN_ID}" > docs/results/latest_run_id.txt

if [[ -z "${CARDIOWATCH_VENV:-}" ]]; then
  if [[ -d ".venv" ]]; then
    export CARDIOWATCH_VENV=".venv"
  elif [[ -d ".venv311" ]]; then
    export CARDIOWATCH_VENV=".venv311"
  fi
fi

if [[ -n "${CARDIOWATCH_VENV:-}" && -f "${CARDIOWATCH_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${CARDIOWATCH_VENV}/bin/activate"
fi

export CARDIOWATCH_PYTHON="${CARDIOWATCH_PYTHON:-python}"

echo "CARDIOWATCH_RUN_ID=${CARDIOWATCH_RUN_ID}"
echo "git_sha=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python=$(${CARDIOWATCH_PYTHON} --version 2>&1)"
echo "host=$(hostname)"
