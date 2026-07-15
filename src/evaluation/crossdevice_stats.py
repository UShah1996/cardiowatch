"""
crossdevice_stats.py — provable cross-device generalization statistics.

Reads the per-cohort files written by crossdevice_eval.py
(docs/results/<run_id>/crossdevice/<cohort>.json) and produces the
pre-registered cross-device result (amendment 2026-07):

  1. Per cohort with both classes present and both models scored: a *paired*
     DeLong test of RR+RF vs CNN-LSTM(CPSC) AUC on the identical windows.
     These form a SINGLE Holm family (separate from the same-data family in
     stat_tests.json).
  2. Every external AUC (each model, each cohort) with a patient-clustered
     bootstrap 95% CI (cluster by record_id, not window).

This is the canonical source for the cross-device p-values and CIs; it converts
"RR holds, deep degrades" from two point estimates into a signed, CI-bounded,
Holm-corrected result.

Run:
    python -m src.evaluation.crossdevice_stats --run-dir docs/results/<run_id>
    python -m src.evaluation.crossdevice_stats --self-test
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.confidence_intervals import cluster_bootstrap_auc
from src.evaluation.stat_tests import delong_roc_test, holm_correction
from src.experiments.provenance import results_dir, write_json

PRIMARY_PAIR = ("rr_rf", "cnn_cpsc")


def _load_cohorts(crossdevice_dir: Path) -> list[dict[str, Any]]:
    files = sorted(crossdevice_dir.glob("*.json"))
    cohorts = []
    for f in files:
        with f.open() as fh:
            cohorts.append(json.load(fh))
    return cohorts


def compute(cohorts: list[dict[str, Any]], n_boot: int = 2000, seed: int = 42) -> dict[str, Any]:
    per_cohort: dict[str, Any] = {}
    family_keys: list[str] = []
    family_pvals: list[float] = []

    for c in cohorts:
        name = c["cohort"]
        y = np.array(c["labels"], dtype=int)
        groups = np.array(c["record_ids"])
        probs = c["probabilities"]
        both_classes = len(np.unique(y)) == 2

        # Clustered-CI AUC for every model scored in this cohort.
        model_cis: dict[str, Any] = {}
        for model, p in probs.items():
            point, lo, hi = cluster_bootstrap_auc(
                y, np.array(p, dtype=float), groups, n_boot=n_boot, seed=seed
            )
            model_cis[model] = {
                "auc": point,
                "cluster_bootstrap_ci": [lo, hi],
                "n_records": int(len(np.unique(groups))),
                "n_windows": int(len(y)),
            }

        entry: dict[str, Any] = {
            "device": c.get("device", name),
            "both_classes_present": bool(both_classes),
            "n_afib_windows": int(y.sum()),
            "model_auc_clustered_ci": model_cis,
        }

        # Paired DeLong RR+RF vs CNN-LSTM(CPSC) on identical windows.
        a, b = PRIMARY_PAIR
        if both_classes and a in probs and b in probs:
            d = asdict(delong_roc_test(y, probs[a], probs[b]))
            entry["paired_delong_rr_vs_cnn_cpsc"] = d
            key = f"{name}:delong_{a}_vs_{b}"
            family_keys.append(key)
            family_pvals.append(d["p_value"])
        else:
            entry["paired_delong_rr_vs_cnn_cpsc"] = None
            entry["delong_skipped_reason"] = (
                "single class in cohort" if not both_classes
                else "primary pair not both scored (e.g. plausibility-only cohort)"
            )

        per_cohort[name] = entry

    # Holm across the cross-device DeLong family (separate from same-data family).
    holm = holm_correction(family_pvals) if family_pvals else []
    for key, adj in zip(family_keys, holm):
        cohort_name = key.split(":", 1)[0]
        per_cohort[cohort_name]["paired_delong_rr_vs_cnn_cpsc"]["p_value_holm"] = adj

    return {
        "primary_pair": f"{PRIMARY_PAIR[0]}_vs_{PRIMARY_PAIR[1]}",
        "holm_family": family_keys,
        "holm_family_size": len(family_pvals),
        "n_boot": n_boot,
        "seed": seed,
        "cohorts": per_cohort,
        "notes": [
            "Cross-device DeLong tests form a SINGLE pre-registered Holm family, "
            "separate from the same-data family in stat_tests.json.",
            "AUC CIs are patient/record-clustered bootstrap (windows within a "
            "patient are correlated; naive per-window CIs understate uncertainty).",
            "Apple Watch has a single confirmed positive: treat as a plausibility "
            "check, not a discrimination claim; its DeLong may be skipped.",
        ],
    }


def _self_test() -> None:
    rng = np.random.default_rng(1)
    labels, groups, rr, cnn = [], [], [], []
    for patient in range(8):
        for _ in range(6):
            y = int(patient % 2 == 0)
            labels.append(y)
            groups.append(f"p{patient}")
            rr.append(0.85 * y + 0.08 + rng.normal(0, 0.04))
            cnn.append(0.5 + rng.normal(0, 0.04))
    cohort = {
        "cohort": "afdb",
        "device": "test Holter",
        "labels": labels,
        "record_ids": groups,
        "probabilities": {"rr_rf": rr, "cnn_cpsc": cnn},
    }
    out = compute([cohort], n_boot=200, seed=0)
    e = out["cohorts"]["afdb"]
    assert e["paired_delong_rr_vs_cnn_cpsc"] is not None
    assert "p_value_holm" in e["paired_delong_rr_vs_cnn_cpsc"]
    rr_auc = e["model_auc_clustered_ci"]["rr_rf"]["auc"]
    cnn_auc = e["model_auc_clustered_ci"]["cnn_cpsc"]["auc"]
    assert rr_auc > cnn_auc
    lo, hi = e["model_auc_clustered_ci"]["rr_rf"]["cluster_bootstrap_ci"]
    assert lo <= rr_auc <= hi
    print("crossdevice_stats self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-device paired stats + clustered CIs")
    parser.add_argument("--run-dir", help="docs/results/<run_id> (defaults to current run)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    run_dir = Path(args.run_dir) if args.run_dir else results_dir(create=False)
    crossdevice_dir = run_dir / "crossdevice"
    if not crossdevice_dir.is_dir():
        parser.error(f"No crossdevice/ dir under {run_dir} — run crossdevice_eval first")
    cohorts = _load_cohorts(crossdevice_dir)
    result = compute(cohorts, n_boot=args.n_boot, seed=args.seed)
    out_path = Path(args.out) if args.out else run_dir / "crossdevice_stats.json"
    write_json(out_path, result)
    print(f"Wrote {out_path}  ({result['holm_family_size']} DeLong tests in Holm family)")


if __name__ == "__main__":
    main()
