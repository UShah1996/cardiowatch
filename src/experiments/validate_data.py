"""
validate_data.py — fail-fast dataset preflight for HPC runs.

This script intentionally checks only local filesystem state. It does not
download data. Counts can be overridden for dry runs or changed mirrors.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from src.experiments.provenance import get_run_id, results_dir, write_json, write_run_metadata


DEFAULT_CPSC_DIR = (
    "data/raw/classification-of-12-lead-ecgs-the-physionetcomputing"
    "-in-cardiology-challenge-2020-1.0.2/training/cpsc_2018"
)
P17_CANDIDATES = [
    "data/raw/challenge_2017/training2017",
    "data/raw/challenge_2017",
    "data/raw/training2017",
    (
        "data/raw/af-classification-from-a-short-single-lead-ecg-recording"
        "-the-physionetcomputing-in-cardiology-challenge-2017-1.0.0/training2017"
    ),
]
MITBIH_CANDIDATES = [
    "data/raw/mit_afib/files",
    "data/raw/mit_afib/mit-bih-atrial-fibrillation-database-1.0.0/files",
    "data/raw/mit_afib/mit-bih-atrial-fibrillation-database-1.0.0",
]


def count_files(root: str | Path, suffix: str) -> int:
    path = Path(root)
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob(f"*{suffix}") if p.is_file())


def first_existing(paths: list[str]) -> str | None:
    return next((p for p in paths if Path(p).exists()), None)


def validate(args: argparse.Namespace) -> dict[str, Any]:
    cpsc_count = count_files(args.cpsc_dir, ".hea")
    p17_dir = first_existing(P17_CANDIDATES)
    p17_count = count_files(p17_dir, ".mat") if p17_dir else 0
    mit_dir = first_existing(MITBIH_CANDIDATES)
    mit_count = count_files(mit_dir, ".dat") if mit_dir else 0
    heart_exists = Path(args.clinical_csv).exists()

    checks = {
        "clinical_csv": {"path": args.clinical_csv, "exists": heart_exists},
        "cpsc": {
            "path": args.cpsc_dir,
            "hea_count": cpsc_count,
            "expected_min": args.expected_cpsc_min,
            "ok": cpsc_count >= args.expected_cpsc_min,
        },
        "physionet_2017": {
            "path": p17_dir,
            "mat_count": p17_count,
            "expected_min": args.expected_p17_min,
            "ok": p17_count >= args.expected_p17_min,
        },
        "mit_bih_afib": {
            "path": mit_dir,
            "dat_count": mit_count,
            "expected_min": args.expected_mit_min,
            "ok": mit_count >= args.expected_mit_min,
        },
        "run_id": get_run_id(),
    }

    required_ok = heart_exists and checks["cpsc"]["ok"]
    if args.require_p17:
        required_ok = required_ok and checks["physionet_2017"]["ok"]
    if args.require_mit:
        required_ok = required_ok and checks["mit_bih_afib"]["ok"]
    checks["ok"] = required_ok
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CardioWatch dataset staging")
    parser.add_argument("--cpsc-dir", default=DEFAULT_CPSC_DIR)
    parser.add_argument("--clinical-csv", default="data/raw/heart.csv")
    parser.add_argument("--expected-cpsc-min", type=int, default=6800)
    parser.add_argument("--expected-p17-min", type=int, default=8200)
    parser.add_argument("--expected-mit-min", type=int, default=20)
    parser.add_argument("--require-p17", action="store_true")
    parser.add_argument("--require-mit", action="store_true")
    args = parser.parse_args()

    out_dir = write_run_metadata(extra={"stage": "validate_data"})
    checks = validate(args)
    write_json(out_dir / "data_validation.json", checks)
    print(f"Run ID: {get_run_id()}")
    print(f"Wrote {out_dir / 'data_validation.json'}")
    if not checks["ok"]:
        raise SystemExit(f"Data validation failed: {checks}")


if __name__ == "__main__":
    main()

