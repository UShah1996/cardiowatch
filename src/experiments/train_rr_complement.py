"""
train_rr_complement.py — train RR+RF excluding the pre-registered CPSC holdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import wfdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.experiments.provenance import read_json, results_dir, write_json, write_run_metadata
from src.models.rr_afib_detector import extract_rr_features


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = read_json(path)
    if "records" not in manifest:
        raise ValueError(f"Invalid manifest: {path}")
    return manifest


def manifest_records(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return [r for r in manifest["records"] if r["split"] == split]


def load_rr_features(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    rows, labels, paths = [], [], []
    skipped = 0
    for r in records:
        path = r["path"]
        try:
            record = wfdb.rdrecord(path)
            leads = [n.strip().upper() for n in record.sig_name]
            if "I" not in leads:
                skipped += 1
                continue
            sig = record.p_signal[:, leads.index("I")].astype(np.float32)
            sig = np.nan_to_num(sig)
            sig = np.clip(sig, -2.0, 2.0)
            sig = (sig - sig.mean()) / (sig.std() + 1e-8)
            sig = np.clip(sig, -5.0, 5.0)
            feats = extract_rr_features(sig, fs=500)
            if feats is None:
                skipped += 1
                continue
            rows.append(feats)
            labels.append(int(r["label"]))
            paths.append(path)
        except Exception:
            skipped += 1
    print(f"RR features loaded: {len(rows)} | skipped={skipped}")
    return pd.DataFrame(rows), np.array(labels, dtype=int), paths


def threshold_for_specificity(y_true: np.ndarray, probs: np.ndarray, specificity: float = 0.90) -> float:
    negatives = probs[y_true == 0]
    if len(negatives) == 0:
        return 0.5
    # P(pred < t | negative) ~= specificity.
    return float(np.quantile(negatives, specificity))


def train(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.manifest)
    train_records = manifest_records(manifest, "train")
    holdout_records = manifest_records(manifest, "holdout")
    train_paths = {r["path"] for r in train_records}
    holdout_paths = {r["path"] for r in holdout_records}
    overlap = train_paths & holdout_paths
    if overlap:
        raise AssertionError(f"Manifest train/holdout overlap: {sorted(overlap)[:5]}")

    X, y, paths = load_rr_features(train_records)
    if len(np.unique(y)) != 2:
        raise ValueError("RR training complement must contain both labels")

    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=args.validation_fraction, stratify=y, random_state=args.seed
    )
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_results = cross_validate(model, X_fit, y_fit, cv=cv, scoring=["recall", "f1", "roc_auc"])
    model.fit(X_fit, y_fit)

    val_probs = model.predict_proba(X_val)[:, 1]
    fixed_threshold = args.threshold
    matched_threshold = threshold_for_specificity(y_val, val_probs, specificity=args.matched_specificity)
    val_auc = float(roc_auc_score(y_val, val_probs))

    out_dir = write_run_metadata(extra={"stage": "train_rr_complement"})
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_path = processed_dir / "rr_rf_cpsc_complement.pkl"
    joblib.dump({"model": model, "feature_names": list(X.columns)}, model_path)
    legacy_model_path = processed_dir / "rr_rf_model.pkl"
    joblib.dump({"model": model, "feature_names": list(X.columns)}, legacy_model_path)

    summary = {
        "model": "RR+RF CPSC complement",
        "manifest": str(args.manifest),
        "manifest_holdout_sha256": manifest["holdout_record_sha256"],
        "n_train_feature_rows": int(len(X_fit)),
        "n_validation_rows": int(len(X_val)),
        "validation_auc": val_auc,
        "fixed_threshold": fixed_threshold,
        "matched_specificity": args.matched_specificity,
        "matched_specificity_threshold": matched_threshold,
        "cv_recall": [float(x) for x in cv_results["test_recall"]],
        "cv_f1": [float(x) for x in cv_results["test_f1"]],
        "cv_auc": [float(x) for x in cv_results["test_roc_auc"]],
        "model_path": str(model_path),
        "legacy_model_path": str(legacy_model_path),
    }
    write_json(out_dir / "rr_complement_training.json", summary)
    print(f"Saved RR complement model -> {model_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RR+RF excluding CPSC holdout")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--matched-specificity", type=float, default=0.90)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
