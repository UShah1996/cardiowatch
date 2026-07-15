"""
crossdevice_eval.py — zero-shot cross-device scoring on identical windows.

Pre-registered by the 2026-07 amendment in docs/results/ANALYSIS_PLAN.md.

For one external cohort, this scores the device-agnostic RR+RF model and the
CPSC-only CNN-LSTM (and, where legitimate, the combined deployment CNN-LSTM)
on the SAME 10-second Lead-I windows (5000 samples at 500 Hz). Emitting the
per-window probabilities of every model on identical windows is what lets
crossdevice_stats.py run a *paired* DeLong test (RR+RF vs CNN-LSTM-CPSC) and
patient-clustered bootstrap CIs — turning the "deep model degrades off its
training device" assertion into a signed, CI-bounded, p-valued result.

Design choices (documented so reviewers can see there is no shortcut):
  * Identical windows for all models. RR features and the CNN both consume the
    same `preprocess_cnn(...)` 10 s window, exactly as in paired_cpsc_eval.py,
    so the comparison is truly paired.
  * Long Holter records (afdb 250 Hz, ltafdb 128 Hz) are resampled to 500 Hz
    and cut into consecutive, non-overlapping 10 s windows; the window label is
    the active rhythm at its midpoint (reusing evaluate_mitbih_afib helpers).
  * The combined CNN-LSTM was TRAINED on PhysioNet/CinC 2017, so scoring it on
    the CinC-2017 cohort would be train-on-test. It is therefore scored only on
    cohorts it never saw (afdb, ltafdb, apple_watch) and OMITTED for cinc2017.
    The primary comparison (RR+RF vs CNN-LSTM-CPSC) is clean zero-shot on every
    cohort.
  * `record_id` is the patient/record grouping key used later for clustered CIs.

One cohort per invocation → one file docs/results/<run_id>/crossdevice/<cohort>.json,
so a SLURM array over cohorts runs them in parallel with no cross-job merge.

Run:
    python -m src.evaluation.crossdevice_eval --cohort afdb
    python -m src.evaluation.crossdevice_eval --self-test
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from src.experiments.provenance import results_dir, write_run_metadata, write_json

# 10 s at 500 Hz — the CNN-LSTM's native input and the CPSC training window.
TARGET_FS = 500
WINDOW_SEC = 10
WINDOW_SAMPLES = WINDOW_SEC * TARGET_FS  # 5000

# Which models each cohort may legitimately score zero-shot. The combined
# deployment model saw CinC 2017 in training, so it is excluded there.
DEFAULT_MODELS = ("rr_rf", "cnn_cpsc", "cnn_combined_deploy")
COHORT_MODEL_BLOCKLIST = {
    "cinc2017": ("cnn_combined_deploy",),  # training overlap → would be leakage
}

COHORTS = ("afdb", "ltafdb", "cinc2017", "apple_watch")

# A single 10 s window ready to score: (record_id, label, preprocessed_signal).
Window = tuple[str, int, np.ndarray]


# ── Cohort loaders — each yields identical 10 s windows for all models ──────

def _load_holter_windows(data_dir: str, src_fs: int) -> Iterator[Window]:
    """afdb / ltafdb: rhythm-annotated WFDB Holter records, channel 0.

    Reuses the annotation + window-label logic already validated in
    evaluate_mitbih_afib.py, but windows at 10 s (not 30 s) to match the CNN.
    """
    import wfdb
    from scipy.signal import resample as scipy_resample

    from src.evaluation.evaluate_mitbih_afib import (
        get_afib_annotations,
        get_window_label,
    )

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Holter cohort dir not found: {data_dir}")

    records = sorted({
        f[:-4] for f in os.listdir(data_dir) if f.endswith(".hea")
    })
    scale = TARGET_FS / src_fs
    for rec in records:
        rec_path = os.path.join(data_dir, rec)
        try:
            record = wfdb.rdrecord(rec_path)
            signal = np.nan_to_num(record.p_signal[:, 0].astype(np.float32))
            rhythms = get_afib_annotations(rec_path)
            if not rhythms:
                continue
            n_target = int(len(signal) * scale)
            signal = scipy_resample(signal, n_target).astype(np.float32)
            rhythms = [(int(s * scale), lab) for s, lab in rhythms]
        except Exception as exc:  # pragma: no cover - depends on data on HPC
            print(f"  {rec}: load error — {exc}")
            continue

        n_windows = len(signal) // WINDOW_SAMPLES
        for i in range(n_windows):
            start = i * WINDOW_SAMPLES
            end = start + WINDOW_SAMPLES
            label = get_window_label(start, end, rhythms)
            if label is None:
                continue
            yield rec, int(label), signal[start:end]


def _load_cinc2017_windows(data_dir: str, src_fs: int = 300) -> Iterator[Window]:
    """PhysioNet/CinC 2017 single-lead AliveCor records (.mat/.hea).

    Labels from REFERENCE(-v3).csv: 'A' = AFib (positive), else negative.
    First 10 s window per record; record_id = record name (one per subject).
    """
    import wfdb
    from scipy.signal import resample as scipy_resample

    ref = None
    for name in ("REFERENCE-v3.csv", "REFERENCE.csv"):
        cand = os.path.join(data_dir, name)
        if os.path.exists(cand):
            ref = cand
            break
    if ref is None:
        raise FileNotFoundError(f"CinC-2017 REFERENCE csv not found under {data_dir}")

    labels: dict[str, int] = {}
    with open(ref) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[0]:
                labels[parts[0]] = 1 if parts[1].strip().upper() == "A" else 0

    scale = TARGET_FS / src_fs
    for rec, label in sorted(labels.items()):
        rec_path = os.path.join(data_dir, rec)
        if not os.path.exists(rec_path + ".hea"):
            continue
        try:
            record = wfdb.rdrecord(rec_path)
            signal = np.nan_to_num(record.p_signal[:, 0].astype(np.float32))
            signal = scipy_resample(signal, int(len(signal) * scale)).astype(np.float32)
        except Exception as exc:  # pragma: no cover
            print(f"  {rec}: load error — {exc}")
            continue
        yield rec, int(label), signal[:WINDOW_SAMPLES]


def _load_apple_watch_windows(base_dir: str) -> Iterator[Window]:
    """Apple Watch personal exports. Reuses the volunteer enumerator and CSV
    parser from build_fusion_apple_watch.py. record_id = volunteer (person),
    so clustered CIs group by person. Labels are the watch classifier output.
    """
    from src.models.build_fusion_apple_watch import (
        PEOPLE,
        load_apple_watch_csv,
        preprocess_for_cnn,
    )

    for person_key in PEOPLE:
        ecg_dir = os.path.join(base_dir, person_key, "electrocardiograms")
        if not os.path.isdir(ecg_dir):
            continue
        person = person_key.replace("apple_health_export_", "")
        for fpath in sorted(glob.glob(os.path.join(ecg_dir, "*.csv"))):
            try:
                sig_uv, cls, _ = load_apple_watch_csv(fpath)
                if cls == "Poor Recording":
                    continue
                label = 1 if cls == "Atrial Fibrillation" else 0
                # preprocess_for_cnn already resamples 512->500, skips the 5 s
                # placement artifact, normalizes, and returns exactly 5000.
                yield person, int(label), preprocess_for_cnn(sig_uv)
            except Exception as exc:  # pragma: no cover
                print(f"  {os.path.basename(fpath)}: load error — {exc}")
                continue


def cohort_windows(cohort: str) -> Iterator[Window]:
    if cohort == "afdb":
        return _load_holter_windows("data/raw/mit_afib/files", src_fs=250)
    if cohort == "ltafdb":
        return _load_holter_windows("data/raw/ltafdb/files", src_fs=128)
    if cohort == "cinc2017":
        return _load_cinc2017_windows(
            "data/raw/challenge_2017/training2017", src_fs=300
        )
    if cohort == "apple_watch":
        return _load_apple_watch_windows("data/apple_health_export")
    raise ValueError(f"Unknown cohort: {cohort}")


COHORT_DEVICE = {
    "afdb": "MIT-BIH AFib — ambulatory Holter, 250 Hz",
    "ltafdb": "Long-Term AF — ambulatory Holter, 128 Hz",
    "cinc2017": "PhysioNet/CinC 2017 — AliveCor single-lead, 300 Hz",
    "apple_watch": "Apple Watch — consumer wearable, 512 Hz",
}


# ── Scoring ─────────────────────────────────────────────────────────────────

def _score(cohort: str, args: argparse.Namespace) -> dict[str, Any]:
    import torch  # imported here so --self-test works without torch

    import joblib

    from src.evaluation.paired_cpsc_eval import load_cnn, preprocess_cnn
    from src.models.rr_afib_detector import extract_rr_features

    allowed = [m for m in DEFAULT_MODELS if m not in COHORT_MODEL_BLOCKLIST.get(cohort, ())]

    rr_saved = joblib.load(args.rr_model)
    rr_model = rr_saved["model"]
    rr_features = rr_saved["feature_names"]

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    cnn = {
        "cnn_cpsc": load_cnn(args.cnn_cpsc_checkpoint, device) if "cnn_cpsc" in allowed else None,
        "cnn_combined_deploy": (
            load_cnn(args.cnn_combined_checkpoint, device)
            if "cnn_combined_deploy" in allowed else None
        ),
    }

    record_ids: list[str] = []
    labels: list[int] = []
    probs: dict[str, list[float]] = {m: [] for m in allowed}
    skipped = 0

    cnn_models = [(name, cnn[name]) for name in ("cnn_cpsc", "cnn_combined_deploy")
                  if cnn.get(name) is not None]
    CNN_BATCH = 256
    win_buf: list[np.ndarray] = []

    def _flush_cnn() -> None:
        # Batched CNN inference: one GPU call per CNN_BATCH windows instead of
        # one (with a .item() sync) per window. Same windows, same order, same
        # scores — but minutes instead of hours on long Holter cohorts (ltafdb).
        if not win_buf or not cnn_models:
            win_buf.clear()
            return
        x = torch.tensor(np.stack(win_buf), dtype=torch.float32).unsqueeze(1).to(device)
        for name, model in cnn_models:
            with torch.no_grad():
                p = torch.sigmoid(model(x).squeeze(-1)).detach().cpu().numpy()
            probs[name].extend(float(v) for v in np.atleast_1d(p))
        win_buf.clear()

    for rec, label, sig in cohort_windows(cohort):
        window = preprocess_cnn(sig)  # identical window for every model
        rr_feats = extract_rr_features(window, fs=TARGET_FS)
        if rr_feats is None:
            skipped += 1
            continue
        rr_vec = np.array([[rr_feats.get(k, 0) for k in rr_features]])
        rr_prob = float(rr_model.predict_proba(rr_vec)[0, 1])
        if rr_feats.get("rr_cv", 0) < 0.15:  # same clinical cap as elsewhere
            rr_prob = min(rr_prob, 0.25)

        record_ids.append(rec)
        labels.append(int(label))
        probs["rr_rf"].append(rr_prob)
        win_buf.append(window)
        if len(win_buf) >= CNN_BATCH:
            _flush_cnn()
    _flush_cnn()

    return _summarize(cohort, allowed, record_ids, labels, probs, skipped)


def _summarize(
    cohort: str,
    models: list[str],
    record_ids: list[str],
    labels: list[int],
    probs: dict[str, list[float]],
    skipped: int,
) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    y = np.array(labels, dtype=int)
    n_pos = int(y.sum())
    per_model: dict[str, Any] = {}
    for name in models:
        p = np.array(probs[name], dtype=float)
        auc = None
        if len(np.unique(y)) == 2 and len(p) == len(y):
            auc = float(roc_auc_score(y, p))
        per_model[name] = {"auc": auc, "threshold": 0.40}

    return {
        "cohort": cohort,
        "device": COHORT_DEVICE.get(cohort, cohort),
        "models_scored": models,
        "models_omitted": list(COHORT_MODEL_BLOCKLIST.get(cohort, ())),
        "omission_reason": (
            "combined CNN-LSTM was trained on this cohort — omitted to avoid "
            "train-on-test leakage"
            if COHORT_MODEL_BLOCKLIST.get(cohort) else None
        ),
        "n_windows": int(len(y)),
        "n_afib_windows": n_pos,
        "n_records": int(len(set(record_ids))),
        "n_skipped": skipped,
        "record_ids": record_ids,
        "labels": labels,
        "probabilities": probs,
        "per_model_auc": per_model,
        "window_seconds": WINDOW_SEC,
        "note": (
            "Per-window probabilities are aligned across models (identical "
            "windows) so a paired DeLong test is valid. record_ids are the "
            "patient/record grouping key for clustered bootstrap CIs."
        ),
    }


def run(cohort: str, args: argparse.Namespace) -> Path:
    payload = _score(cohort, args)
    out_dir = write_run_metadata(extra={"stage": f"crossdevice_eval:{cohort}"})
    out_path = out_dir / "crossdevice" / f"{cohort}.json"
    write_json(out_path, payload)
    auc_str = ", ".join(
        f"{m}={payload['per_model_auc'][m]['auc']}" for m in payload["models_scored"]
    )
    print(
        f"[{cohort}] windows={payload['n_windows']} afib={payload['n_afib_windows']} "
        f"records={payload['n_records']} | AUC: {auc_str}\n  -> {out_path}"
    )
    return out_path


# ── Self-test — exercises aggregation without data/torch ────────────────────

def _self_test() -> None:
    rng = np.random.default_rng(0)
    record_ids, labels, rr, cnn = [], [], [], []
    for patient in range(6):
        for _ in range(5):
            y = int(patient % 2 == 0)
            record_ids.append(f"p{patient}")
            labels.append(y)
            # RR separates classes; CNN is near-chance (mimics device shift).
            rr.append(0.8 * y + 0.1 + rng.normal(0, 0.05))
            cnn.append(0.5 + rng.normal(0, 0.05))
    summary = _summarize(
        "afdb", ["rr_rf", "cnn_cpsc"], record_ids, labels,
        {"rr_rf": rr, "cnn_cpsc": cnn}, skipped=0,
    )
    assert summary["n_records"] == 6
    assert summary["n_windows"] == 30
    assert summary["per_model_auc"]["rr_rf"]["auc"] > summary["per_model_auc"]["cnn_cpsc"]["auc"]
    # cinc2017 must never score the combined model.
    assert "cnn_combined_deploy" in COHORT_MODEL_BLOCKLIST["cinc2017"]
    print("crossdevice_eval self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot cross-device evaluation")
    parser.add_argument("--cohort", choices=COHORTS)
    parser.add_argument("--rr-model", default="data/processed/rr_rf_cpsc_complement.pkl")
    parser.add_argument("--cnn-cpsc-checkpoint", default="data/processed/cnn_lstm_cpsc_complement.pt")
    parser.add_argument("--cnn-combined-checkpoint", default="data/processed/cnn_lstm_combined_deploy.pt")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.cohort:
        parser.error("--cohort or --self-test is required")
    run(args.cohort, args)


if __name__ == "__main__":
    main()
