"""
representation_shift.py — the mechanism: why device-agnostic features transfer.

Core idea: a representation only transfers across devices if it does NOT encode
device identity. We test this directly. For the SAME recordings we build two
representations and ask "can a classifier tell which device this came from?":

  1. RR/HRV feature vector  — the device-agnostic model's input.
  2. CNN-LSTM penultimate embedding — the 128-d LSTM hidden state (hidden[-1])
     taken BEFORE the classifier head (src/models/cnn_lstm.py:forward).

For each device pair (CPSC hospital vs Apple Watch; CPSC vs MIT-BIH Holter) we
train a record-grouped logistic-regression DEVICE classifier on each
representation and report the out-of-fold device-separability AUC.

Hypothesis (the mechanism of the wearable domain gap): CNN embeddings are highly
device-separable (AUC -> ~1.0, they encode the device), while RR features are
near chance (AUC ~0.5-0.7, device-invariant). Supporting analyses:
  (B) AFib-vs-normal separability of top RR features WITHIN each device
      (the AFib timing signature is preserved across hardware).
  (C) cross-device calibration (ECE) of each model.

Usage:
    python -m src.evaluation.representation_shift \
        --manifest docs/results/<run_id>/manifests/cpsc_holdout.json \
        --out docs/results/<run_id>/representation_shift.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.rr_afib_detector import extract_rr_features  # noqa: E402

WINDOW = 5000  # 10 s at 500 Hz for the CNN


# ── signal preprocessing (matches training / paired_cpsc_eval) ────────
def _norm_window(sig: np.ndarray) -> np.ndarray:
    w = np.nan_to_num(sig.astype(np.float32))
    w = np.clip(w, -2.0, 2.0)
    w = (w - w.mean()) / (w.std() + 1e-8)
    w = np.clip(w, -5.0, 5.0)
    return w[:WINDOW] if len(w) >= WINDOW else np.pad(w, (0, WINDOW - len(w)))


# ── per-device Lead I loaders (full 500 Hz signal) ────────────────────
def load_cpsc(manifest_path: str, max_n: int, seed: int) -> list[dict]:
    import wfdb
    from src.experiments.provenance import read_json
    recs = [r for r in read_json(manifest_path)["records"] if r["split"] == "holdout"]
    rng = np.random.default_rng(seed)
    if len(recs) > max_n:
        recs = [recs[i] for i in rng.choice(len(recs), max_n, replace=False)]
    out = []
    for r in recs:
        try:
            rec = wfdb.rdrecord(r["path"])
            leads = [n.strip().upper() for n in rec.sig_name]
            if "I" not in leads:
                continue
            sig = rec.p_signal[:, leads.index("I")].astype(np.float32)
            out.append({"device": "cpsc", "record": os.path.basename(r["path"]),
                        "sig": np.nan_to_num(sig), "afib": int(r["label"])})
        except Exception:
            continue
    return out


def load_apple_watch(base_dir: str, max_n: int) -> list[dict]:
    import glob
    from scipy.signal import resample
    from src.models.build_fusion_apple_watch import load_apple_watch_csv, PEOPLE
    out = []
    for person in PEOPLE:
        for fp in sorted(glob.glob(os.path.join(base_dir, person, "electrocardiograms", "*.csv"))):
            try:
                sig_uv, cls, _ = load_apple_watch_csv(fp)
                if cls == "Poor Recording":
                    continue
                sig = sig_uv / 1000.0
                sig = resample(sig, int(len(sig) * 500 / 512)).astype(np.float32)
                sig = sig[int(5 * 500):]  # skip placement artifact
                out.append({"device": "apple_watch", "record": os.path.basename(fp),
                            "sig": np.nan_to_num(sig),
                            "afib": int(cls == "Atrial Fibrillation")})
            except Exception:
                continue
    return out[:max_n] if len(out) > max_n else out


def load_mitbih(mit_dir: str, max_n: int, seed: int) -> list[dict]:
    import wfdb
    from scipy.signal import resample
    paths = sorted({os.path.join(mit_dir, f.replace(".hea", ""))
                    for f in os.listdir(mit_dir) if f.endswith(".hea")}) if os.path.isdir(mit_dir) else []
    rng = np.random.default_rng(seed)
    out = []
    for p in paths:
        try:
            rec = wfdb.rdrecord(p, sampfrom=0, sampto=250 * 60)  # first minute
            sig = rec.p_signal[:, 0].astype(np.float32)
            sig = resample(sig, int(len(sig) * 500 / rec.fs)).astype(np.float32)
            # split into a few 30 s windows per record for more samples
            step = 30 * 500
            for s in range(0, len(sig) - step + 1, step):
                out.append({"device": "holter", "record": os.path.basename(p),
                            "sig": np.nan_to_num(sig[s:s + step]), "afib": -1})
        except Exception:
            continue
    if len(out) > max_n:
        out = [out[i] for i in rng.choice(len(out), max_n, replace=False)]
    return out


# ── representations ───────────────────────────────────────────────────
def cnn_embedding(model, sig: np.ndarray, torch_dev) -> np.ndarray:
    import torch
    x = torch.tensor(_norm_window(sig)).unsqueeze(0).unsqueeze(0).to(torch_dev)
    with torch.no_grad():
        c = model.cnn(x).permute(0, 2, 1)
        _, (h, _) = model.lstm(c)
        return h[-1].squeeze(0).detach().cpu().numpy().astype(np.float32)


def cnn_prob(model, sig: np.ndarray, torch_dev) -> float:
    import torch
    x = torch.tensor(_norm_window(sig)).unsqueeze(0).unsqueeze(0).to(torch_dev)
    with torch.no_grad():
        return float(torch.sigmoid(model(x).squeeze()).detach().cpu().item())


# ── device-separability (analysis A) — pure logic, self-tested ────────
def device_separability(X: np.ndarray, device_bin: np.ndarray,
                        groups: np.ndarray) -> float | None:
    """
    Out-of-fold AUC of a record-grouped logistic regression predicting DEVICE
    from a representation. High -> representation encodes device (won't transfer).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_groups = len(set(groups.tolist()))
    if len(set(device_bin.tolist())) < 2 or n_groups < 2:
        return None
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, class_weight="balanced"))
    oof = cross_val_predict(clf, X, device_bin,
                            cv=GroupKFold(n_splits=min(5, n_groups)),
                            groups=groups, method="predict_proba")[:, 1]
    return float(roc_auc_score(device_bin, oof))


def afib_feature_separability(rows: list[dict], device: str,
                              feats: list[str]) -> dict[str, Any]:
    """(B) Per-feature AFib-vs-normal AUC within one device."""
    from sklearn.metrics import roc_auc_score
    sub = [r for r in rows if r["device"] == device and r["afib"] in (0, 1)]
    y = np.array([r["afib"] for r in sub])
    if len(set(y.tolist())) < 2:
        return {"n": len(sub), "note": "single-class or unlabeled device"}
    out = {"n": len(sub)}
    for f in feats:
        v = np.array([r["rr"].get(f, np.nan) for r in sub], dtype=float)
        m = np.isfinite(v)
        if m.sum() < 4 or len(set(y[m].tolist())) < 2:
            continue
        auc = roc_auc_score(y[m], v[m])
        out[f] = round(float(max(auc, 1 - auc)), 3)  # direction-agnostic separation
    return out


def _ece(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float | None:
    from sklearn.calibration import calibration_curve
    if len(set(y_true.tolist())) < 2:
        return None
    frac, mean_pred = calibration_curve(y_true, y_prob, n_bins=bins, strategy="uniform")
    return float(np.mean(np.abs(frac - mean_pred)))


# ── main experiment ───────────────────────────────────────────────────
def run(args: argparse.Namespace) -> dict[str, Any]:
    import joblib
    import torch
    from src.models.cnn_lstm import build_model
    from src.experiments.provenance import write_json, write_run_metadata

    rows: list[dict] = []
    rows += load_cpsc(args.manifest, args.max_per_device, args.seed)
    rows += load_apple_watch(args.apple_dir, args.max_per_device)
    rows += load_mitbih(args.mit_dir, args.max_per_device, args.seed)
    if not rows:
        raise RuntimeError("No recordings loaded — check dataset paths")

    # RR feature vectors (device-agnostic representation)
    for r in rows:
        r["rr"] = extract_rr_features(_norm_window(r["sig"]) if len(r["sig"]) < WINDOW
                                      else r["sig"], fs=500) or {}
    feats = sorted({k for r in rows for k in r["rr"].keys()})

    # CNN embeddings (deep representation)
    tdev = torch.device("cpu")
    cnn_path = next((p for p in [
        "data/processed/cnn_lstm_cpsc_complement.pt",
        "data/processed/cnn_lstm_combined_deploy.pt",
        "data/processed/cnn_lstm_combined_best.pt",
    ] if os.path.exists(p)), None)
    model = None
    if cnn_path:
        model = build_model(input_length=WINDOW).to(tdev)
        model.load_state_dict(torch.load(cnn_path, map_location="cpu"))
        model.eval()
        for r in rows:
            r["emb"] = cnn_embedding(model, r["sig"], tdev)

    def _matrix(rs, key):
        if key == "rr":
            return np.array([[r["rr"].get(f, 0.0) for f in feats] for r in rs], float)
        return np.array([r["emb"] for r in rs], float)

    # ── A. device-separability per pair ──────────────────────────────
    pairs = [("cpsc", "apple_watch"), ("cpsc", "holter")]
    separability = {}
    for a, b in pairs:
        sub = [r for r in rows if r["device"] in (a, b) and r["rr"]]
        if len({r["device"] for r in sub}) < 2:
            continue
        dev_bin = np.array([1 if r["device"] == b else 0 for r in sub])
        groups = np.array([f'{r["device"]}:{r["record"]}' for r in sub])
        entry = {"n": len(sub),
                 "rr_features_device_auc": device_separability(_matrix(sub, "rr"), dev_bin, groups)}
        if model is not None:
            entry["cnn_embedding_device_auc"] = device_separability(_matrix(sub, "emb"), dev_bin, groups)
        separability[f"{a}_vs_{b}"] = entry

    # ── B. AFib-signature invariance (RR features, per device) ───────
    top_rr = [f for f in ["rr_cv", "rr_rmssd", "rr_pnn50", "rr_entropy", "rr_iqr"] if f in feats]
    invariance = {d: afib_feature_separability(rows, d, top_rr) for d in ("cpsc", "apple_watch")}

    # ── C. cross-device calibration (ECE) of each model ──────────────
    calibration = {}
    if model is not None:
        for d in ("cpsc", "apple_watch"):
            sub = [r for r in rows if r["device"] == d and r["afib"] in (0, 1)]
            if len(sub) < 10:
                continue
            y = np.array([r["afib"] for r in sub])
            cnn_p = np.array([cnn_prob(model, r["sig"], tdev) for r in sub])
            calibration[d] = {"n": len(sub), "cnn_ece": _ece(y, cnn_p)}

    summary = {
        "task": "representation device-separability (mechanism of cross-device transfer)",
        "hypothesis": "CNN embeddings encode device (high AUC); RR features do not (near chance).",
        "cnn_checkpoint": cnn_path,
        "n_features_rr": len(feats),
        "device_separability_auc": separability,
        "afib_feature_invariance_auc": invariance,
        "cross_device_calibration_ece": calibration,
        "notes": [
            "Device-separability AUC near 1.0 means the representation encodes the device "
            "and will not transfer; near 0.5 means device-invariant.",
            "Compare cnn_embedding_device_auc (expected high) vs rr_features_device_auc "
            "(expected low) — this is the mechanism of the wearable domain gap.",
        ],
    }
    if args.out:
        write_run_metadata(extra={"stage": "representation_shift"})
        write_json(Path(args.out), summary)
        print(f"Wrote {args.out}")
    for pair, e in separability.items():
        print(f"{pair}: RR-feature device-AUC={e.get('rr_features_device_auc')} | "
              f"CNN-embedding device-AUC={e.get('cnn_embedding_device_auc')}")
    return summary


def _self_test() -> None:
    # A device-separable representation -> AUC ~1; a device-invariant one -> ~0.5.
    rng = np.random.default_rng(0)
    n = 60
    groups = np.array([f"r{i}" for i in range(2 * n)])
    dev = np.array([0] * n + [1] * n)
    # device-encoding representation: shifted means per device
    X_dev = np.vstack([rng.normal(0, 1, (n, 4)), rng.normal(5, 1, (n, 4))])
    # device-invariant representation: same distribution regardless of device
    X_inv = rng.normal(0, 1, (2 * n, 4))
    auc_dev = device_separability(X_dev, dev, groups)
    auc_inv = device_separability(X_inv, dev, groups)
    assert auc_dev > 0.95, auc_dev
    assert auc_inv < 0.75, auc_inv
    print(f"self-test passed: device-encoding AUC={auc_dev:.2f}, invariant AUC={auc_inv:.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Representation device-separability (mechanism)")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--manifest", default=None)
    p.add_argument("--apple-dir", default="data/apple_health_export")
    p.add_argument("--mit-dir", default="data/raw/mit_afib/files")
    p.add_argument("--max-per-device", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.manifest:
        p.error("--manifest is required (CPSC hold-out manifest) unless --self-test")
    run(args)


if __name__ == "__main__":
    main()
