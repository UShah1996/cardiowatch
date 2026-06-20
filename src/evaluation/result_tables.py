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


def _fmt(x: Any, spec: str = ".3g") -> str:
    return format(x, spec) if isinstance(x, (int, float)) else "—"


def md_table(title: str, rows: list[dict[str, Any]]) -> str:
    lines = [f"## {title}", "", "| Model | Dataset | n | AUC | Recall | Specificity | Balanced Acc | PPV | NPV | F1 | Threshold |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        m = r["metrics"]
        auc = f"{m['auc']:.3f}" if m.get("auc") is not None else "—"
        lines.append(
            f"| {r['model']} | {r['dataset']} | {m['n']} | "
            f"{auc} | {m['recall']:.3f} | {m['specificity']:.3f} | "
            f"{m['balanced_accuracy']:.3f} | {m['ppv']:.3f} | {m['npv']:.3f} | "
            f"{m['f1']:.3f} | {m['threshold']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def apple_watch_md(aw: dict[str, Any]) -> str:
    acc = aw.get("accuracy")
    lo, hi = (aw.get("accuracy_wilson_ci") or [None, None])
    ci = f"{lo:.3f}–{hi:.3f}" if isinstance(lo, (int, float)) else "—"
    lines = [
        "## Apple Watch (exploratory — device-classifier labels)",
        "",
        "| Model | Dataset | n | n AFib | Accuracy | 95% CI (Wilson) | Threshold |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| {aw.get('model', 'cnn_lstm')} | {aw.get('dataset', 'Apple Watch')} | "
        f"{aw.get('n', '?')} | {aw.get('n_afib', '?')} | "
        f"{_fmt(acc, '.3f')} | {ci} | {_fmt(aw.get('threshold', 0.5), '.2f')} |",
    ]
    if aw.get("ground_truth_note"):
        lines += ["", f"_{aw['ground_truth_note']}_"]
    return "\n".join(lines) + "\n"


def latency_md(L: dict[str, Any]) -> str:
    lines = [
        "## Detection Latency (bootstrap CPSC transitions)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Transitions | {L.get('n_transitions', '?')} |",
        f"| Detected | {L.get('n_detected', '?')} |",
        f"| Threshold | {L.get('threshold', '?')} |",
        f"| Stride (s) | {L.get('stride_sec', '?')} |",
        f"| Median latency (min) | {L.get('latency_median_min')} |",
        f"| Latency IQR (min) | {L.get('latency_iqr_min')} |",
        f"| Normal-phase FP rate (median) | {_fmt(L.get('fp_rate_median'), '.3f')} |",
    ]
    if L.get("independence_caveat"):
        lines += ["", f"_{L['independence_caveat']}_"]
    return "\n".join(lines) + "\n"


def onset_md(o: dict[str, Any]) -> str:
    """Exploratory AFib onset-prediction table (pooled + per-source rows)."""
    def row(name: str, n_rec, auc, sens, sens_ci, lt, lt_iqr, fa):
        ci = (f" ({sens_ci[0]:.2f}–{sens_ci[1]:.2f})"
              if isinstance(sens_ci, (list, tuple)) and sens_ci and sens_ci[0] is not None else "")
        iqr = (f" (IQR {lt_iqr[0]:.1f}–{lt_iqr[1]:.1f})"
               if isinstance(lt_iqr, (list, tuple)) and lt_iqr and lt_iqr[0] is not None else "")
        sens_s = f"{sens:.2f}{ci}" if isinstance(sens, (int, float)) else "—"
        lt_s = f"{lt:.1f}{iqr}" if isinstance(lt, (int, float)) else "—"
        return (f"| {name} | {n_rec} | {_fmt(auc, '.3f')} | {sens_s} | "
                f"{lt_s} | {_fmt(fa, '.2f')} |")

    lines = [
        "## AFib Onset Prediction (exploratory — pre-onset sinus)",
        "",
        f"Features: {o.get('feature_mode', '?')}"
        + (" + P-wave" if o.get('pwave_features') else "")
        + (", per-patient normalized" if o.get('per_patient_normalized') else "")
        + f"; horizon {o.get('horizon_min', '?')} min; patient-grouped CV.",
        "",
        "| Cohort | n patients | window AUC | onset sensitivity (95% CI) | median lead time, min | false alarms/h |",
        "|---|---:|---:|---:|---:|---:|",
        row("pooled", o.get("n_records", "?"), o.get("window_auc_grouped_cv"),
            o.get("onset_sensitivity"), o.get("onset_sensitivity_wilson_ci"),
            o.get("lead_time_median_min"), o.get("lead_time_iqr_min"),
            o.get("false_alarms_per_hour")),
    ]
    for src, v in (o.get("per_source") or {}).items():
        lines.append(row(src, v.get("n_records", "?"),
                         v.get("window_auc_grouped_cv_oof"),
                         v.get("onset_sensitivity"), None,
                         v.get("lead_time_median_min"), v.get("lead_time_iqr_min"),
                         v.get("false_alarms_per_hour")))
    lines += ["", "_Exploratory: threshold set at fixed control specificity on "
              "out-of-fold scores; modest, sub-clinical signal. Not a clinical "
              "early-warning claim._"]
    return "\n".join(lines) + "\n"


def stats_md(stats: dict[str, Any]) -> str:
    lines = [
        "## Statistical Comparisons (CPSC holdout)",
        "",
        f"Primary endpoint: `{stats.get('primary_endpoint', '?')}`. "
        f"Holm family size: {stats.get('holm_family_size', '?')} "
        "(DeLong + fixed-threshold McNemar over model-vs-model pairs; "
        "matched-specificity and random-baseline rows are raw sensitivity/sanity).",
        "",
        "| Comparison | ΔAUC | 95% CI | DeLong p | DeLong p(Holm) | McNemar b/c | McNemar p | McNemar p(Holm) | In family |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    for key, c in stats.get("comparisons", {}).items():
        d = c.get("delong", {})
        m = c.get("mcnemar_fixed", {})
        ci = (f"[{d['ci_low']:.3f}, {d['ci_high']:.3f}]"
              if isinstance(d.get("ci_low"), (int, float)) else "—")
        lines.append(
            f"| {key} | {_fmt(d.get('delta_auc'), '+.3f')} | {ci} | "
            f"{_fmt(d.get('p_value'))} | {_fmt(d.get('p_value_holm'))} | "
            f"{m.get('b', '?')}/{m.get('c', '?')} | {_fmt(m.get('p_value'))} | "
            f"{_fmt(m.get('p_value_holm'))} | {'yes' if c.get('in_holm_family') else 'no'} |"
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
    parser.add_argument("--clinical-json", default=None)
    parser.add_argument("--mitbih-json", default=None)
    parser.add_argument("--apple-watch-json", default=None)
    parser.add_argument("--stats-json", default=None)
    parser.add_argument("--onset-json", default=None)
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()

    out_dir = write_run_metadata(extra={"stage": "result_tables"})
    paired = read_json(args.paired_cpsc_json)
    tables = build_from_paired(paired)

    def _maybe(path):
        return read_json(path) if path and Path(path).exists() else None

    clinical = _maybe(args.clinical_json)
    mitbih = _maybe(args.mitbih_json)
    apple = _maybe(args.apple_watch_json)
    stats = _maybe(args.stats_json)
    latency = _maybe(args.latency_json)
    onset = _maybe(args.onset_json)

    if clinical:
        tables["clinical"] = clinical
    if mitbih:
        tables["mitbih"] = mitbih
    if apple:
        tables["apple_watch"] = apple
    if stats:
        tables["statistical_comparisons"] = stats
    if latency:
        tables["latency_bootstrap"] = latency
    if onset:
        tables["onset_prediction"] = onset

    prefix = Path(args.out_prefix) if args.out_prefix else out_dir / "paper_tables"
    write_json(prefix.with_suffix(".json"), tables)

    md = ["# CardioWatch Paper Result Tables", ""]
    if clinical and clinical.get("rows"):
        md.append(md_table("Clinical Models (Kaggle cohort, distinct from ECG cohorts)", clinical["rows"]))
    md.append(md_table("ECG Controlled And Deployment Results (CPSC holdout)",
                       tables["ecg_controlled_and_deployment"]))
    if stats:
        md.append(stats_md(stats))
    if mitbih and mitbih.get("rows"):
        md.append(md_table("Cross-Device: MIT-BIH AFib (zero-shot)", mitbih["rows"]))
    if apple:
        md.append(apple_watch_md(apple))
    if latency:
        md.append(latency_md(latency))
    if onset:
        md.append(onset_md(onset))
    md += [
        "## Validity Notes",
        "",
        "- Clinical-only and ECG-only models use different cohorts and are reported separately (not a cross-modal ablation).",
        "- Apple Watch agreement is not a clinical gold standard unless independently adjudicated.",
        "- Fusion rows are exploratory unless based on truly paired clinical + ECG data.",
        "- Detection latency is measured after AFib onset; the old 30-minute lead-time framing is invalid.",
        "- Corrected p-values come from stat_tests.json (pre-registered Holm family); matched-specificity and random-baseline tests are raw.",
        "",
    ]
    prefix.with_suffix(".md").write_text("\n".join(md))
    print(f"Wrote {prefix.with_suffix('.json')} and {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
