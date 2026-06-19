"""
latency_bootstrap.py — bootstrap detection-latency transitions.

Builds many fixed-length normal -> AFib CPSC transitions from verified
records. Samples are record-level draws with replacement from a shared pool,
so the resulting transitions are useful robustness checks but are not fully
independent clinical episodes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wfdb

from src.evaluation.lead_time import (
    AFIB_CODE,
    FS,
    NORMAL_CODE,
    compute_detection_latency,
    ecg_risk_over_time,
    normal_phase_false_positives,
)
from src.experiments.provenance import write_json, write_run_metadata
from src.models.cnn_lstm import build_model


def list_records(data_dir: str) -> tuple[list[str], list[str]]:
    normal, afib = [], []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if not fname.endswith(".hea"):
                continue
            path = os.path.join(root, fname.replace(".hea", ""))
            try:
                h = wfdb.rdheader(path)
                codes = []
                for c in h.comments:
                    if c.startswith("Dx:"):
                        codes = [x.strip() for x in c.replace("Dx:", "").split(",")]
                        break
                if NORMAL_CODE in codes:
                    normal.append(path)
                elif AFIB_CODE in codes:
                    afib.append(path)
            except Exception:
                continue
    return sorted(normal), sorted(afib)


def load_lead_i(path: str) -> np.ndarray | None:
    try:
        rec = wfdb.rdrecord(path)
        leads = [n.strip().upper() for n in rec.sig_name]
        if "I" not in leads:
            return None
        sig = rec.p_signal[:, leads.index("I")].astype(np.float32)
        sig = np.nan_to_num(sig)
        sig = np.clip(sig, -2.0, 2.0)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        sig = np.clip(sig, -5.0, 5.0)
        return sig
    except Exception:
        return None


def concatenate_to_minutes(paths: list[str], target_minutes: float, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    target_samples = int(target_minutes * 60 * FS)
    chunks, used = [], []
    total = 0
    attempts = 0
    while total < target_samples and attempts < max(1000, len(paths) * 5):
        attempts += 1
        path = str(rng.choice(paths))
        sig = load_lead_i(path)
        if sig is None:
            continue
        chunks.append(sig)
        used.append(path)
        total += len(sig)
    if total < target_samples:
        raise RuntimeError(f"Could not build {target_minutes} min signal from record pool")
    return np.concatenate(chunks)[:target_samples], used


def load_model(path: str):
    model = build_model(input_length=5000)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def run(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    normal_paths, afib_paths = list_records(args.cpsc_dir)
    if not normal_paths or not afib_paths:
        raise FileNotFoundError("Need both normal and AFib CPSC records")
    model = load_model(args.checkpoint)

    transitions = []
    for i in range(args.n_transitions):
        normal_sig, normal_used = concatenate_to_minutes(normal_paths, args.normal_minutes, rng)
        afib_sig, afib_used = concatenate_to_minutes(afib_paths, args.afib_minutes, rng)
        signal = np.concatenate([normal_sig, afib_sig])
        onset = len(normal_sig) / FS / 60.0
        times, probs = ecg_risk_over_time(signal, model, stride_sec=args.stride_sec)
        latency, alert_time = compute_detection_latency(times, probs, onset, threshold=args.threshold)
        n_fp, n_normal, fp_rate = normal_phase_false_positives(times, probs, onset, threshold=args.threshold)
        transitions.append(
            {
                "transition": i + 1,
                "latency_min": latency,
                "first_alert_min": alert_time,
                "onset_min": onset,
                "normal_phase_false_positives": n_fp,
                "normal_phase_windows": n_normal,
                "normal_phase_fp_rate": fp_rate,
                "normal_records": normal_used,
                "afib_records": afib_used,
            }
        )

    latencies = np.array([t["latency_min"] for t in transitions if t["latency_min"] is not None], dtype=float)
    fp_rates = np.array([t["normal_phase_fp_rate"] for t in transitions], dtype=float)
    summary = {
        "n_transitions": args.n_transitions,
        "n_detected": int(len(latencies)),
        "threshold": args.threshold,
        "stride_sec": args.stride_sec,
        "normal_minutes": args.normal_minutes,
        "afib_minutes": args.afib_minutes,
        "latency_median_min": float(np.median(latencies)) if len(latencies) else None,
        "latency_iqr_min": [
            float(np.percentile(latencies, 25)) if len(latencies) else None,
            float(np.percentile(latencies, 75)) if len(latencies) else None,
        ],
        "fp_rate_median": float(np.median(fp_rates)),
        "fp_rate_iqr": [float(np.percentile(fp_rates, 25)), float(np.percentile(fp_rates, 75))],
        "independence_caveat": (
            "Transitions are sampled with replacement from a shared record pool; "
            "treat as robustness analysis, not independent prospective episodes."
        ),
        "transitions": transitions,
    }
    out_dir = write_run_metadata(extra={"stage": "latency_bootstrap"})
    out_path = Path(args.out) if args.out else out_dir / "latency_bootstrap.json"
    write_json(out_path, summary)
    print(f"Wrote latency bootstrap -> {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CPSC detection latency")
    parser.add_argument("--cpsc-dir", default=(
        "data/raw/classification-of-12-lead-ecgs-the-physionetcomputing"
        "-in-cardiology-challenge-2020-1.0.2/training/cpsc_2018"
    ))
    parser.add_argument("--checkpoint", default="data/processed/cnn_lstm_cpsc_complement.pt")
    parser.add_argument("--n-transitions", type=int, default=100)
    parser.add_argument("--normal-minutes", type=float, default=35.0)
    parser.add_argument("--afib-minutes", type=float, default=31.0)
    parser.add_argument("--stride-sec", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

