"""
clinical_report.py — score the saved clinical models on the held-out clinical
test split and emit a result-table-compatible JSON. No retraining: it loads
rf_model.pkl / xgb_model.pkl produced by job 02 and re-derives the same
deterministic test split from clinical.full_pipeline().

Output: docs/results/<run_id>/clinical_eval.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.evaluation.result_tables import metrics_for
from src.experiments.provenance import write_json, write_run_metadata
from src.preprocessing.clinical import full_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Clinical model test-set report (no retrain)")
    parser.add_argument("--rf-model", default="data/processed/rf_model.pkl")
    parser.add_argument("--xgb-model", default="data/processed/xgb_model.pkl")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    (_, _, X_te, _, _, y_te), _ = full_pipeline()
    y = list(y_te)
    rows = []

    if Path(args.rf_model).exists():
        rf = joblib.load(args.rf_model)
        rf_p = rf.predict_proba(X_te)[:, 1]
        rows.append({
            "model": "random_forest",
            "dataset": "Kaggle clinical test split",
            "metrics": metrics_for(y, rf_p, 0.50),
        })
    else:
        print(f"WARNING: {args.rf_model} not found — skipping RF")

    if Path(args.xgb_model).exists():
        saved = joblib.load(args.xgb_model)
        if isinstance(saved, dict):
            xgb, thr = saved["model"], float(saved.get("threshold", 0.40))
        else:
            xgb, thr = saved, 0.40
        xgb_p = xgb.predict_proba(X_te)[:, 1]
        rows.append({
            "model": "xgboost",
            "dataset": "Kaggle clinical test split",
            "metrics": metrics_for(y, xgb_p, thr),
        })
    else:
        print(f"WARNING: {args.xgb_model} not found — skipping XGBoost")

    out_dir = write_run_metadata(extra={"stage": "clinical_report"})
    out_path = Path(args.out) if args.out else out_dir / "clinical_eval.json"
    write_json(out_path, {
        "rows": rows,
        "note": "Clinical (tabular) cohort is distinct from the ECG cohorts; "
                "do not read these rows as a cross-modal ablation.",
    })
    print(f"Wrote clinical eval JSON -> {out_path}")


if __name__ == "__main__":
    main()
