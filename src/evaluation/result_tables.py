"""
result_tables.py — paper-ready JSON/Markdown result summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluation.confidence_intervals import wilson_ci
from src.evaluation.stat_tests import delong_roc_test
from src.experiments.provenance import read_json, write_json, write_run_metadata


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else None


def metrics_for(y_true, probs, threshold: float) -> dict[str, Any]:
    y = np.array(y_true, dtype=int)
    p = np.array(probs, dtype=float)
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    acc_k = int(tp + tn)
    acc_n = int(len(y))
    acc, acc_lo, acc_hi = wilson_ci(acc_k, acc_n)
    recall = recall_score(y, pred, zero_division=0)
    spec = tn / max(tn + fp, 1)
    ppv = precision_score(y, pred, zero_division=0)
    npv = tn / max(tn + fn, 1)
    return {
        "n": acc_n,
        "threshold": threshold,
        "auc": safe_auc(y, p),
        "accuracy": acc,
        "accuracy_wilson_ci": [acc_lo, acc_hi],
        "recall": float(recall),
        "specificity": float(spec),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def md_table(title: str, rows: list[dict[str, Any]]) -> str:
    lines = [f"## {title}", "", "| Model | Dataset | n | AUC | Recall | Specificity | Balanced Acc | PPV | NPV | F1 | Threshold |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        m = r["metrics"]
        lines.append(
            f"| {r['model']} | {r['dataset']} | {m['n']} | "
            f"{m['auc']:.3f} | {m['recall']:.3f} | {m['specificity']:.3f} | "
            f"{m['balanced_accuracy']:.3f} | {m['ppv']:.3f} | {m['npv']:.3f} | "
            f"{m['f1']:.3f} | {m['threshold']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def build_from_paired(paired: dict[str, Any]) -> dict[str, Any]:
    y = paired["labels"]
    rows = []
    for name in paired["model_names"]:
        threshold = paired["thresholds_fixed"].get(name, 0.5)
        rows.append(
            {
                "model": name,
                "dataset": "CPSC pre-registered holdout",
                "metrics": metrics_for(y, paired["probabilities"][name], threshold),
            }
        )

    comparisons = {}
    for a, b in [("rr_rf", "cnn_cpsc"), ("rr_rf", "cnn_combined_deploy"), ("cnn_cpsc", "cnn_combined_deploy")]:
        if a in paired["probabilities"] and b in paired["probabilities"]:
            d = delong_roc_test(y, paired["probabilities"][a], paired["probabilities"][b])
            comparisons[f"{a}_vs_{b}"] = d.__dict__

    return {
        "manifest_holdout_sha256": paired.get("manifest_holdout_sha256"),
        "ecg_controlled_and_deployment": rows,
        "comparisons": comparisons,
        "notes": [
            "Clinical-only and ECG-only models are intentionally not mixed into one ablation table.",
            "cnn_combined_deploy is data-asymmetric and should be labeled as deployment comparison.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-ready result tables")
    parser.add_argument("--paired-cpsc-json", required=True)
    parser.add_argument("--latency-json", default=None)
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()

    out_dir = write_run_metadata(extra={"stage": "result_tables"})
    paired = read_json(args.paired_cpsc_json)
    tables = build_from_paired(paired)
    if args.latency_json:
        tables["latency_bootstrap"] = read_json(args.latency_json)

    prefix = Path(args.out_prefix) if args.out_prefix else out_dir / "paper_tables"
    write_json(prefix.with_suffix(".json"), tables)
    md = [
        "# CardioWatch Paper Result Tables",
        "",
        md_table("ECG Controlled And Deployment Results", tables["ecg_controlled_and_deployment"]),
        "## Validity Notes",
        "",
        "- Clinical-only and ECG-only models use different cohorts and are reported separately.",
        "- Apple Watch agreement is not a clinical gold standard unless independently adjudicated.",
        "- Fusion rows are exploratory unless based on truly paired clinical + ECG data.",
        "- Detection latency is measured after AFib onset.",
        "",
    ]
    prefix.with_suffix(".md").write_text("\n".join(md))
    print(f"Wrote {prefix.with_suffix('.json')} and {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
