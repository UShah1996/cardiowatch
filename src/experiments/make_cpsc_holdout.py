"""
make_cpsc_holdout.py — pre-registered CPSC holdout manifest.

Creates a common holdout before training so RR+RF and CNN-LSTM can be
trained on the complement and compared on identical unseen records.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import wfdb
from sklearn.model_selection import train_test_split

from src.experiments.provenance import (
    get_run_id,
    results_dir,
    sha256_items,
    write_json,
    write_run_metadata,
)


DEFAULT_CPSC_DIR = (
    "data/raw/classification-of-12-lead-ecgs-the-physionetcomputing"
    "-in-cardiology-challenge-2020-1.0.2/training/cpsc_2018"
)
AFIB_CODE = "164889003"


def record_id(path: str) -> str:
    return Path(path).name


def infer_patient_id(path: str) -> str:
    rid = record_id(path)
    # CPSC files are generally one ECG per patient. If a mirror appends
    # multiple segment suffixes, collapse trailing segment tokens.
    return re.sub(r"[_-](seg|segment|window|part)\d+$", "", rid, flags=re.I)


def collect_records(data_dir: str) -> list[dict[str, Any]]:
    paths: list[str] = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith(".hea"):
                paths.append(os.path.join(root, fname.replace(".hea", "")))
    paths.sort()

    records = []
    skipped = 0
    for path in paths:
        try:
            header = wfdb.rdheader(path)
            dx_codes: list[str] = []
            for c in header.comments:
                if c.startswith("Dx:"):
                    dx_codes = [x.strip() for x in c.replace("Dx:", "").split(",")]
                    break
            label = int(AFIB_CODE in dx_codes)
            records.append(
                {
                    "path": path,
                    "record_id": record_id(path),
                    "patient_id": infer_patient_id(path),
                    "label": label,
                    "dx_codes": dx_codes,
                }
            )
        except Exception:
            skipped += 1
    if skipped:
        print(f"Skipped {skipped} unreadable headers")
    return records


def choose_split_unit(records: list[dict[str, Any]], force_patient: bool) -> str:
    counts = Counter(r["patient_id"] for r in records)
    has_multi = any(v > 1 for v in counts.values())
    if has_multi:
        return "patient"
    if force_patient:
        return "patient"
    return "record"


def patient_level_labels(records: list[dict[str, Any]]) -> tuple[list[str], list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for r in records:
        grouped[r["patient_id"]].append(int(r["label"]))
    ids, labels = [], []
    for pid in sorted(grouped):
        vals = grouped[pid]
        if len(set(vals)) > 1:
            raise ValueError(f"Mixed labels inside patient_id={pid}: {vals}")
        ids.append(pid)
        labels.append(vals[0])
    return ids, labels


def make_manifest(args: argparse.Namespace) -> dict[str, Any]:
    records = collect_records(args.cpsc_dir)
    if not records:
        raise FileNotFoundError(f"No readable CPSC headers found under {args.cpsc_dir}")

    split_unit = choose_split_unit(records, args.force_patient_split)
    if split_unit == "patient":
        units, unit_labels = patient_level_labels(records)
    else:
        units = [r["record_id"] for r in records]
        unit_labels = [int(r["label"]) for r in records]

    train_units, holdout_units = train_test_split(
        units,
        test_size=args.holdout_fraction,
        stratify=unit_labels,
        random_state=args.seed,
    )
    holdout_set = set(holdout_units)
    train_set = set(train_units)

    def in_holdout(r: dict[str, Any]) -> bool:
        return (r["patient_id"] if split_unit == "patient" else r["record_id"]) in holdout_set

    holdout_records = [r for r in records if in_holdout(r)]
    train_records = [r for r in records if not in_holdout(r)]

    overlap = holdout_set & train_set
    if overlap:
        raise AssertionError(f"Split overlap detected: {sorted(overlap)[:5]}")

    all_paths = [r["path"] for r in records]
    holdout_paths = [r["path"] for r in holdout_records]
    train_paths = [r["path"] for r in train_records]

    manifest = {
        "run_id": get_run_id(),
        "cpsc_dir": args.cpsc_dir,
        "seed": args.seed,
        "holdout_fraction": args.holdout_fraction,
        "split_unit": split_unit,
        "split_unit_note": (
            "split_unit='record' is patient-grouped here: CPSC 2018 has one ECG per "
            "patient after collapsing segment suffixes (n_records == n_units), so a "
            "record-level split does not leak a patient across train/holdout. "
            "split_unit switches to 'patient' automatically if multi-record patients exist."
        ),
        "patient_id_rule": "record basename, with trailing segment/window suffix collapsed",
        "n_records": len(records),
        "n_units": len(units),
        "n_train_records": len(train_records),
        "n_holdout_records": len(holdout_records),
        "label_counts_all": dict(Counter(int(r["label"]) for r in records)),
        "label_counts_train": dict(Counter(int(r["label"]) for r in train_records)),
        "label_counts_holdout": dict(Counter(int(r["label"]) for r in holdout_records)),
        "record_list_sha256": sha256_items(all_paths),
        "train_record_sha256": sha256_items(train_paths),
        "holdout_record_sha256": sha256_items(holdout_paths),
        "records": [
            {
                "path": r["path"],
                "record_id": r["record_id"],
                "patient_id": r["patient_id"],
                "label": int(r["label"]),
                "split": "holdout" if in_holdout(r) else "train",
                "dx_codes": r["dx_codes"],
            }
            for r in records
        ],
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create pre-registered CPSC holdout")
    parser.add_argument("--cpsc-dir", default=DEFAULT_CPSC_DIR)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-patient-split", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_dir = write_run_metadata(extra={"stage": "make_cpsc_holdout"})
    manifest = make_manifest(args)
    out_path = Path(args.out) if args.out else out_dir / "manifests" / "cpsc_holdout.json"
    write_json(out_path, manifest)
    print(f"Wrote holdout manifest -> {out_path}")
    print(f"Holdout hash: {manifest['holdout_record_sha256']}")


if __name__ == "__main__":
    main()
