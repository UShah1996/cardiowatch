#!/usr/bin/env bash
#
# stage_data.sh — download + symlink the CardioWatch datasets into data/raw/.
#
# Run this ON A LOGIN NODE (e.g. the SJSU COE JupyterLab terminal), NOT inside
# an sbatch job — compute nodes typically have no outbound network. It downloads
# large data to a scratch directory and symlinks it into data/raw/ so the repo's
# relative paths resolve, then runs the same preflight the pipeline enforces.
#
# Datasets / sources:
#   CPSC 2018      — Kaggle mirror gamalasran/physionet-challenge-2020 (~3 GB)
#                    needs Kaggle API credentials (kaggle.json)
#   PhysioNet 2017 — physionet.org open access (~600 MB), no credentials
#   MIT-BIH AFib   — physionet.org afdb open access (~440 MB), via wfdb
#
# Usage:
#   source .venv/bin/activate            # the env created by sjsu_setup.sh
#   bash scripts/hpc/stage_data.sh                       # uses default scratch dir
#   CARDIOWATCH_DATA=/scratch/$USER/cw bash scripts/hpc/stage_data.sh
#   bash scripts/hpc/stage_data.sh --data-dir /path/to/scratch
#
# Env / flags:
#   CARDIOWATCH_DATA / --data-dir   where large data is stored (default
#                                   /scratch/$USER/cardiowatch_data)
#   KAGGLE_JSON                     path to kaggle.json (default ~/.kaggle/kaggle.json)
#   CARDIOWATCH_PYTHON              python to use (default: python)
#
# Idempotent: datasets that already meet the expected file counts are skipped.

set -euo pipefail

# ── Resolve repo root and config ──────────────────────────────────────
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

DATA="${CARDIOWATCH_DATA:-/scratch/${USER}/cardiowatch_data}"
KAGGLE_JSON="${KAGGLE_JSON:-${HOME}/.kaggle/kaggle.json}"
PY="${CARDIOWATCH_PYTHON:-python}"

# Minimum file counts (must match src/experiments/validate_data.py)
CPSC_MIN=6800     # .hea
P17_MIN=8200      # .mat
MIT_MIN=20        # .dat

CPSC_EXPECTED_NAME="classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2"
P17_ZIP_URL="https://archive.physionet.org/challenge/2017/training2017.zip"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA="$2"; shift 2 ;;
    --data-dir=*) DATA="${1#*=}"; shift ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${DATA}"
echo "Repo root : ${REPO_ROOT}"
echo "Data dir  : ${DATA}"
echo "Python    : $(${PY} --version 2>&1)"
echo

# ── Helpers ───────────────────────────────────────────────────────────
count_files() {  # count_files <dir> <suffix>; follows symlinks
  local dir="$1" suffix="$2"
  [[ -e "$dir" ]] || { echo 0; return; }
  find -L "$dir" -type f -name "*${suffix}" 2>/dev/null | wc -l | tr -d ' '
}

# ── CPSC 2018 (Kaggle) ────────────────────────────────────────────────
stage_cpsc() {
  local existing
  existing="$(count_files "${DATA}" .hea)"
  if [[ "${existing}" -ge "${CPSC_MIN}" ]]; then
    echo "CPSC: ${existing} .hea files already present — skipping download."
    return
  fi

  if [[ ! -f "${KAGGLE_JSON}" ]]; then
    echo "ERROR: Kaggle credentials not found at ${KAGGLE_JSON}." >&2
    echo "  Create a token at https://www.kaggle.com/settings and place it there," >&2
    echo "  or set KAGGLE_JSON=/path/to/kaggle.json, then re-run." >&2
    exit 1
  fi
  mkdir -p "${HOME}/.kaggle"
  cp "${KAGGLE_JSON}" "${HOME}/.kaggle/kaggle.json"
  chmod 600 "${HOME}/.kaggle/kaggle.json"

  if ! command -v kaggle >/dev/null 2>&1; then
    echo "Installing kaggle CLI into the active environment..."
    "${PY}" -m pip install --quiet kaggle
  fi

  echo "Downloading CPSC 2018 (Kaggle physionet-challenge-2020, ~3 GB)..."
  kaggle datasets download gamalasran/physionet-challenge-2020 -p "${DATA}" --unzip

  existing="$(count_files "${DATA}" .hea)"
  echo "CPSC: ${existing} .hea files after download."
}

# ── PhysioNet 2017 ────────────────────────────────────────────────────
stage_p17() {
  local existing
  existing="$(count_files "${DATA}/challenge_2017" .mat)"
  if [[ "${existing}" -ge "${P17_MIN}" ]]; then
    echo "PhysioNet 2017: ${existing} .mat files already present — skipping download."
    return
  fi

  mkdir -p "${DATA}/challenge_2017"
  local zip="${DATA}/training2017.zip"
  if [[ ! -f "${zip}" ]]; then
    echo "Downloading PhysioNet 2017 (~600 MB)..."
    if ! wget -c "${P17_ZIP_URL}" -O "${zip}"; then
      echo "ERROR: could not download ${P17_ZIP_URL}." >&2
      echo "  Get training2017.zip from https://physionet.org/content/challenge-2017/1.0.0/" >&2
      echo "  and place it at ${zip}, then re-run." >&2
      exit 1
    fi
  fi
  echo "Extracting PhysioNet 2017..."
  unzip -q -o "${zip}" -d "${DATA}/challenge_2017/"

  existing="$(count_files "${DATA}/challenge_2017" .mat)"
  echo "PhysioNet 2017: ${existing} .mat files after extraction."
}

# ── MIT-BIH AFib (afdb via wfdb) ──────────────────────────────────────
stage_mit() {
  local existing
  existing="$(count_files "${DATA}/mit_afib" .dat)"
  if [[ "${existing}" -ge "${MIT_MIN}" ]]; then
    echo "MIT-BIH AFib: ${existing} .dat files already present — skipping download."
    return
  fi

  echo "Downloading MIT-BIH AFib (afdb, ~440 MB) via wfdb..."
  "${PY}" -c "import wfdb; wfdb.dl_database('afdb', '${DATA}/mit_afib/files')"

  existing="$(count_files "${DATA}/mit_afib" .dat)"
  echo "MIT-BIH AFib: ${existing} .dat files after download."
}

# ── Symlink scratch data into data/raw/ ───────────────────────────────
link_all() {
  echo
  echo "Linking datasets into data/raw/ ..."

  # CPSC: locate the dir containing training/cpsc_2018 (robust to the
  # unzipped top-level folder name) and link the expected name to its root.
  local cpsc_hea_dir cpsc_root
  cpsc_hea_dir="$(find -L "${DATA}" -type d -path '*/training/cpsc_2018' 2>/dev/null | head -1)"
  if [[ -n "${cpsc_hea_dir}" ]]; then
    cpsc_root="$(dirname "$(dirname "${cpsc_hea_dir}")")"
    ln -sfn "${cpsc_root}" "data/raw/${CPSC_EXPECTED_NAME}"
    echo "  CPSC  -> data/raw/${CPSC_EXPECTED_NAME} (${cpsc_root})"
  else
    echo "  WARNING: training/cpsc_2018 not found under ${DATA}" >&2
  fi

  # PhysioNet 2017: link data/raw/challenge_2017 to the dir holding training2017
  local p17_dir
  p17_dir="$(find -L "${DATA}" -type d -name training2017 2>/dev/null | head -1)"
  if [[ -n "${p17_dir}" ]]; then
    ln -sfn "$(dirname "${p17_dir}")" "data/raw/challenge_2017"
    echo "  P17   -> data/raw/challenge_2017 ($(dirname "${p17_dir}"))"
  else
    echo "  WARNING: training2017 not found under ${DATA}" >&2
  fi

  # MIT-BIH AFib
  if [[ -d "${DATA}/mit_afib" ]]; then
    ln -sfn "${DATA}/mit_afib" "data/raw/mit_afib"
    echo "  MIT   -> data/raw/mit_afib (${DATA}/mit_afib)"
  fi
}

# ── Run ───────────────────────────────────────────────────────────────
stage_cpsc
stage_p17
stage_mit
link_all

echo
echo "File counts (via symlinks in data/raw/):"
printf '  CPSC .hea : %s (need >= %s)\n' "$(count_files 'data/raw/'${CPSC_EXPECTED_NAME} .hea)" "${CPSC_MIN}"
printf '  P17  .mat : %s (need >= %s)\n' "$(count_files 'data/raw/challenge_2017' .mat)" "${P17_MIN}"
printf '  MIT  .dat : %s (need >= %s)\n' "$(count_files 'data/raw/mit_afib' .dat)" "${MIT_MIN}"

echo
echo "Running preflight validator..."
"${PY}" -m src.experiments.validate_data --require-p17 --require-mit

echo
echo "Data staging complete. Launch the pipeline with:"
echo "  bash scripts/hpc/submit_pipeline.sh"
