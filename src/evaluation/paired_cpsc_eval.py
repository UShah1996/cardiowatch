"""
paired_cpsc_eval.py — score ECG models on the same CPSC holdout manifest.

Outputs paired probabilities/predictions for RR+RF, controlled CPSC-only
CNN-LSTM, optional combined deployment CNN-LSTM, and a deterministic random
baseline. This file is the input to stat_tests.py and result_tables.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import wfdb

from src.evaluation.stat_tests import delong_roc_test, holm_correction, mcnemar_test
from src.experiments.provenance import read_json, results_dir, write_json, write_run_metadata
from src.models.cnn_lstm import build_model
from src.models.rr_afib_detector import extract_rr_features


def preprocess_cnn(sig: np.ndarray, target_len: int = 5000) -> np.ndarray:
    sig = np.nan_to_num(sig.astype(np.float32))
    sig = np.clip(sig, -2.0, 2.0)
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    sig = np.clip(sig, -5.0, 5.0)
    if len(sig) >= target_len:
        return sig[:target_len]
    return np.pad(sig, (0, target_len - len(sig)))


def load_lead_i(path: str) -> np.ndarray | None:
    try:
        rec = wfdb.rdrecord(path)
        leads = [n.strip().upper() for n in rec.sig_name]
        if "I" not in leads:
            return None
        return rec.p_signal[:, leads.index("I")].astype(np.float32)
    except Exception:
        return None


def load_cnn(path: str | None, device: torch.device):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"WARNING: CNN checkpoint missing: {path}")
        return None
    model = build_model(input_length=5000).to(device)
    model.load_state_dict(torch.load(p, map_location=device))
    model.eval()
    return model


def cnn_prob(model, sig: np.ndarray, device: torch.device) -> float:
    w = preprocess_cnn(sig)
    x = torch.tensor(w, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(torch.sigmoid(model(x).squeeze()).detach().cpu().item())


def threshold_predictions(probs: list[float], threshold: float) -> list[int]:
    return (np.array(probs) >= threshold).astype(int).tolist()


def training_threshold(summary_name: str, fallback: float | None) -> float:
    if fallback is not None:
        return fallback
    path = results_dir(create=False) / summary_name
    if not path.exists():
        return 0.40
    try:
        return float(read_json(path)["matched_specificity_threshold"])
    except Exception:
        return 0.40


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.manifest)
    holdout = [r for r in manifest["records"] if r["split"] == "holdout"]
    if not holdout:
        raise ValueError("Manifest has no holdout records")

    rr_saved = joblib.load(args.rr_model)
    rr_model = rr_saved["model"]
    rr_features = rr_saved["feature_names"]
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    cnn_cpsc = load_cnn(args.cnn_cpsc_checkpoint, device)
    cnn_combined = load_cnn(args.cnn_combined_checkpoint, device)

    labels: list[int] = []
    paths: list[str] = []
    probabilities: dict[str, list[float]] = {
        "rr_rf": [],
        "random_baseline": [],
    }
    if cnn_cpsc is not None:
        probabilities["cnn_cpsc"] = []
    if cnn_combined is not None:
        probabilities["cnn_combined_deploy"] = []

    rng = np.random.default_rng(args.seed)
    skipped = 0
    for r in holdout:
        sig = load_lead_i(r["path"])
        if sig is None:
            skipped += 1
            continue

        rr_feats = extract_rr_features(preprocess_cnn(sig), fs=500)
        if rr_feats is None:
            skipped += 1
            continue
        rr_vec = np.array([[rr_feats.get(k, 0) for k in rr_features]])
        rr_prob = float(rr_model.predict_proba(rr_vec)[0, 1])
        if rr_feats.get("rr_cv", 0) < 0.15:
            rr_prob = min(rr_prob, 0.25)

        labels.append(int(r["label"]))
        paths.append(r["path"])
        probabilities["rr_rf"].append(rr_prob)
        probabilities["random_baseline"].append(float(rng.random()))
        if cnn_cpsc is not None:
            probabilities["cnn_cpsc"].append(cnn_prob(cnn_cpsc, sig, device))
        if cnn_combined is not None:
            probabilities["cnn_combined_deploy"].append(cnn_prob(cnn_combined, sig, device))

    thresholds_fixed = {
        "rr_rf": args.rr_threshold,
        "cnn_cpsc": args.cnn_threshold,
        "cnn_combined_deploy": args.cnn_threshold,
        "random_baseline": 0.5,
    }
    thresholds_matched = {
        "rr_rf": training_threshold("rr_complement_training.json", args.rr_matched_threshold),
        "cnn_cpsc": training_threshold("cnn_cpsc_training.json", args.cnn_matched_threshold),
        "cnn_combined_deploy": training_threshold(
            "cnn_combined_training.json", args.cnn_combined_matched_threshold
        ),
        "random_baseline": 0.5,
    }

    model_names = list(probabilities.keys())
    predictions_fixed = {
        name: threshold_predictions(probabilities[name], thresholds_fixed.get(name, 0.5))
        for name in model_names
    }
    predictions_matched = {
        name: threshold_predictions(probabilities[name], thresholds_matched.get(name, thresholds_fixed.get(name, 0.5)))
        for name in model_names
    }

    stats_out: dict[str, Any] = {}
    p_values: list[float] = []
    p_keys: list[tuple[str, str]] = []
    y = np.array(labels, dtype=int)
    for a, b in [("rr_rf", "cnn_cpsc"), ("rr_rf", "cnn_combined_deploy"), ("cnn_cpsc", "cnn_combined_deploy")]:
        if a not in probabilities or b not in probabilities:
            continue
        key = f"{a}_vs_{b}"
        correct_a = np.array(predictions_fixed[a]) == y
        correct_b = np.array(predictions_fixed[b]) == y
        delong = delong_roc_test(y, probabilities[a], probabilities[b])
        mcnemar_fixed = mcnemar_test(correct_a, correct_b)
        correct_a_m = np.array(predictions_matched[a]) == y
        correct_b_m = np.array(predictions_matched[b]) == y
        mcnemar_matched = mcnemar_test(correct_a_m, correct_b_m)
        stats_out[key] = {
            "delong": delong.__dict__,
            "mcnemar_fixed": mcnemar_fixed.__dict__,
            "mcnemar_matched_specificity": mcnemar_matched.__dict__,
        }
        for test_name in ["delong", "mcnemar_fixed", "mcnemar_matched_specificity"]:
            p_keys.append((key, test_name))
            p_values.append(stats_out[key][test_name]["p_value"])
    for (key, test_name), adj in zip(p_keys, holm_correction(p_values)):
        stats_out[key][test_name]["p_value_holm"] = adj

    out_dir = write_run_metadata(extra={"stage": "paired_cpsc_eval"})
    payload = {
        "manifest": str(args.manifest),
        "manifest_holdout_sha256": manifest["holdout_record_sha256"],
        "n_holdout_manifest": len(holdout),
        "n_scored": len(labels),
        "n_skipped": skipped,
        "paths": paths,
        "labels": labels,
        "model_names": model_names,
        "probabilities": probabilities,
        "thresholds_fixed": thresholds_fixed,
        "thresholds_matched_specificity": thresholds_matched,
        "predictions_fixed": predictions_fixed,
        "predictions_matched_specificity": predictions_matched,
        "stats": stats_out,
        "notes": [
            "Primary controlled comparison is rr_rf vs cnn_cpsc.",
            "cnn_combined_deploy is a secondary data-asymmetric deployment row.",
            "random_baseline is a deterministic sanity floor, not a clinical model.",
        ],
    }
    out_path = Path(args.out) if args.out else out_dir / "paired_cpsc_predictions.json"
    write_json(out_path, payload)
    print(f"Wrote paired predictions -> {out_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired CPSC holdout evaluation")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rr-model", default="data/processed/rr_rf_cpsc_complement.pkl")
    parser.add_argument("--cnn-cpsc-checkpoint", default="data/processed/cnn_lstm_cpsc_complement.pt")
    parser.add_argument("--cnn-combined-checkpoint", default="data/processed/cnn_lstm_combined_deploy.pt")
    parser.add_argument("--rr-threshold", type=float, default=0.40)
    parser.add_argument("--cnn-threshold", type=float, default=0.40)
    parser.add_argument("--rr-matched-threshold", type=float, default=None)
    parser.add_argument("--cnn-matched-threshold", type=float, default=None)
    parser.add_argument("--cnn-combined-matched-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
