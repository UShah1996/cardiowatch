"""
afib_onset_prediction.py — EXPLORATORY AFib onset (early-warning) prediction.

This is a *prediction* task, distinct from every detection result in the repo:
from a window of SINUS rhythm, predict whether AFib begins within H minutes.
A detector that keys on AFib morphology cannot do this — there is no AFib in
the input window — so we train a fresh RR/HRV model on pre-onset sinus only.

Data: MIT-BIH AFib (afdb) — continuous Holter with rhythm onset annotations.
  pre-onset positive : sinus window whose NEXT AFib onset is in (0, H] min
  control negative    : sinus window with no AFib onset within FAR min
  excluded            : windows already in AFib (that would be detection),
                        and windows in the ambiguous (H, FAR] buffer.

Leakage control: split by RECORD (patient) with GroupKFold — never by window.
Out-of-fold scores are used for every reported number.

Evaluation:
  1. Window-level discrimination (pre-onset vs control): record-grouped CV AUC.
  2. Early-warning operating point: threshold set to a fixed control specificity
     on out-of-fold control scores; require k sustained windows to raise an
     alarm; then per true onset, lead time = minutes from the first sustained
     pre-onset alarm to onset. Report sensitivity (onsets warned), control
     false-alarm rate per hour, and the lead-time distribution (median/IQR).

Caveats (state in the paper):
  - afdb are selected AFib patients, not a general screening population.
  - Few records (~23) → wide CIs; report bootstrap/Wilson where applicable.
  - Threshold is set at a fixed control specificity on out-of-fold scores
    (documented operating point), not tuned to maximise lead time.
  - Exploratory.

Usage:
    python -m src.evaluation.afib_onset_prediction \
        --out docs/results/<run_id>/onset_prediction.json [--fig ...png]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.evaluation.confidence_intervals import wilson_ci
from src.models.rr_afib_detector import extract_rr_features

AFIB_LABEL = "AFIB"


# ── afdb parsing ──────────────────────────────────────────────────────
def list_records(data_dir: str) -> list[str]:
    import os as _os
    recs = sorted({
        f.replace(".hea", "")
        for f in _os.listdir(data_dir) if f.endswith(".hea")
    })
    return [os.path.join(data_dir, r) for r in recs]


def rhythm_segments(record_path: str) -> list[tuple[int, str]]:
    """Return [(sample, rhythm_label), ...] in order from the .atr file."""
    import wfdb
    ann = wfdb.rdann(record_path, "atr")
    segs = []
    for i, aux in enumerate(ann.aux_note):
        aux = aux.strip().strip("\x00")
        if aux.startswith("("):
            segs.append((int(ann.sample[i]), aux[1:].strip()))
    return segs


def rhythm_at(segments: list[tuple[int, str]], sample: int) -> str | None:
    active = None
    for s, label in segments:
        if s <= sample:
            active = label
        else:
            break
    return active


def afib_onsets(segments: list[tuple[int, str]]) -> list[int]:
    """Samples where the rhythm transitions INTO AFib from a non-AFib rhythm."""
    onsets = []
    prev = None
    for s, label in segments:
        if label == AFIB_LABEL and prev != AFIB_LABEL:
            onsets.append(s)
        prev = label
    return onsets


# ── window construction ───────────────────────────────────────────────
def build_record_windows(
    record_path: str, horizon_min: float, far_min: float,
    window_sec: float, stride_sec: float, source: str = "afdb",
) -> list[dict[str, Any]]:
    """
    Slide windows over one record; label sinus windows by time to next onset.
    Returns dicts with record, t_start_sec, time_to_onset_min, label, features.
    Windows already in AFib, or in the (H, FAR] buffer, or with too few beats
    are dropped (label is only 0 or 1 in the returned list).
    """
    import wfdb
    try:
        rec = wfdb.rdrecord(record_path)
    except Exception:
        return []
    fs = int(rec.fs)
    sig = np.nan_to_num(rec.p_signal[:, 0].astype(np.float32))
    segs = rhythm_segments(record_path)
    if not segs:
        return []
    onsets = np.array(afib_onsets(segs), dtype=int)

    win = int(window_sec * fs)
    stride = int(stride_sec * fs)
    out = []
    name = os.path.basename(record_path)
    for start in range(0, len(sig) - win + 1, stride):
        end = start + win
        mid = (start + end) // 2
        if rhythm_at(segs, mid) == AFIB_LABEL:
            continue  # already in AFib — that's detection, not prediction
        future = onsets[onsets > end]
        ttom = float((future[0] - end) / fs / 60.0) if future.size else np.inf
        if 0 < ttom <= horizon_min:
            label = 1
        elif ttom > far_min:
            label = 0
        else:
            continue  # ambiguous buffer
        feats = extract_rr_features(sig[start:end], fs=fs)
        if feats is None:
            continue
        out.append({
            "source": source,
            "record": name,
            "group": f"{source}:{name}",
            "t_start_sec": start / fs,
            "time_to_onset_min": ttom,
            "label": label,
            "features": feats,
            "onset_id": int(future[0]) if future.size else -1,
        })
    return out


# ── early-warning operating-point logic (pure; unit-tested) ───────────
def sustained_mask(alert: np.ndarray, k: int) -> np.ndarray:
    """True where a window is part of a run of >= k consecutive alerts."""
    alert = np.asarray(alert, dtype=bool)
    out = np.zeros_like(alert)
    run = 0
    for i, a in enumerate(alert):
        run = run + 1 if a else 0
        if run >= k:
            out[i - k + 1: i + 1] = True
    return out


def lead_times_and_alarms(
    rows: list[dict[str, Any]], scores: np.ndarray, threshold: float, k: int,
    stride_sec: float,
) -> dict[str, Any]:
    """
    Per-record early-warning evaluation on out-of-fold scores.
    Returns sensitivity, lead-time list, and control false-alarm rate / hour.
    """
    by_group: dict[str, list[int]] = {}
    for idx, r in enumerate(rows):
        by_group.setdefault(r["group"], []).append(idx)

    lead_times: list[float] = []
    onsets_total = 0
    onsets_warned = 0
    control_alarm_runs = 0
    control_window_seconds = 0.0

    for rec, idxs in by_group.items():
        idxs.sort(key=lambda i: rows[i]["t_start_sec"])
        s = scores[idxs]
        sustained = sustained_mask(s >= threshold, k)
        labels = np.array([rows[i]["label"] for i in idxs])
        onset_ids = np.array([rows[i]["onset_id"] for i in idxs])

        # Pre-onset sensitivity + lead time, per distinct onset.
        for oid in sorted(set(onset_ids[labels == 1].tolist())):
            onsets_total += 1
            sel = (labels == 1) & (onset_ids == oid)
            # earliest sustained warning for this onset = largest time_to_onset
            ttoms = np.array([rows[i]["time_to_onset_min"] for i in idxs])
            warned = sel & sustained
            if warned.any():
                onsets_warned += 1
                lead_times.append(float(ttoms[warned].max()))

        # Control false alarms: sustained runs among control windows.
        control = labels == 0
        control_window_seconds += float(control.sum()) * stride_sec
        prev = False
        for i in range(len(idxs)):
            if control[i] and sustained[i] and not prev:
                control_alarm_runs += 1
            prev = control[i] and sustained[i]

    control_hours = control_window_seconds / 3600.0
    return {
        "sensitivity": onsets_warned / onsets_total if onsets_total else None,
        "onsets_total": onsets_total,
        "onsets_warned": onsets_warned,
        "lead_times_min": lead_times,
        "lead_time_median_min": float(np.median(lead_times)) if lead_times else None,
        "lead_time_iqr_min": [float(np.percentile(lead_times, 25)),
                              float(np.percentile(lead_times, 75))] if lead_times else None,
        "control_false_alarms": control_alarm_runs,
        "control_hours": control_hours,
        "false_alarms_per_hour": control_alarm_runs / control_hours if control_hours else None,
    }


# ── main experiment ───────────────────────────────────────────────────
def run(args: argparse.Namespace) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    # Pool one or more WFDB rhythm-annotated databases ("source:path" tokens).
    specs = []
    for tok in args.data_dirs:
        source, _, path = tok.partition(":")
        if not path:
            source, path = "afdb", tok
        specs.append((source, path))

    rows: list[dict[str, Any]] = []
    used_sources = []
    for source, path in specs:
        if not os.path.isdir(path):
            print(f"WARNING: {source} dir not found, skipping: {path}")
            continue
        recs = list_records(path)
        before = len(rows)
        for rp in recs:
            rows.extend(build_record_windows(
                rp, args.horizon_min, args.far_min,
                args.window_sec, args.stride_sec, source=source))
        print(f"  {source}: {len(recs)} records, {len(rows) - before} labeled windows")
        if len(rows) > before:
            used_sources.append(source)

    if not rows:
        raise RuntimeError("No labeled windows built — check data dirs / annotations")

    feat_names = sorted(rows[0]["features"].keys())
    X = np.array([[r["features"].get(k, 0.0) for k in feat_names] for r in rows], dtype=float)
    y = np.array([r["label"] for r in rows], dtype=int)
    groups = np.array([r["group"] for r in rows])
    n_groups = len(set(groups.tolist()))
    n_splits = min(args.folds, n_groups)

    if len(set(y.tolist())) < 2 or n_splits < 2:
        raise RuntimeError(
            f"Insufficient data for grouped CV: classes={set(y.tolist())}, groups={n_groups}")

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced",
        random_state=args.seed, n_jobs=-1)
    gkf = GroupKFold(n_splits=n_splits)
    oof = cross_val_predict(clf, X, y, cv=gkf, groups=groups,
                            method="predict_proba", n_jobs=-1)[:, 1]

    auc = float(roc_auc_score(y, oof))
    # Operating point: threshold at fixed control specificity on OOF scores.
    control_scores = oof[y == 0]
    threshold = float(np.quantile(control_scores, args.target_specificity))
    ew = lead_times_and_alarms(rows, oof, threshold, args.k_sustained, args.stride_sec)

    # Per-source breakdown at the same pooled operating threshold.
    per_source: dict[str, Any] = {}
    for src in sorted(set(used_sources)):
        idx = [i for i, r in enumerate(rows) if r["source"] == src]
        sub_rows = [rows[i] for i in idx]
        sub_scores = oof[idx]
        ys = y[idx]
        ew_s = lead_times_and_alarms(sub_rows, sub_scores, threshold,
                                     args.k_sustained, args.stride_sec)
        per_source[src] = {
            "n_records": len({r["group"] for r in sub_rows}),
            "n_windows": len(sub_rows),
            "n_preonset_windows": int((ys == 1).sum()),
            "n_control_windows": int((ys == 0).sum()),
            "window_auc_grouped_cv_oof": (float(roc_auc_score(ys, sub_scores))
                                          if len(set(ys.tolist())) > 1 else None),
            "onset_sensitivity": ew_s["sensitivity"],
            "onsets_total": ew_s["onsets_total"],
            "onsets_warned": ew_s["onsets_warned"],
            "lead_time_median_min": ew_s["lead_time_median_min"],
            "lead_time_iqr_min": ew_s["lead_time_iqr_min"],
            "false_alarms_per_hour": ew_s["false_alarms_per_hour"],
        }

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    sens_k = ew["onsets_warned"]
    sens_n = ew["onsets_total"]
    sens_ci = wilson_ci(sens_k, sens_n) if sens_n else (None, None, None)

    summary = {
        "task": "AFib onset prediction (pre-onset sinus -> AFib within horizon)",
        "exploratory": True,
        "dataset": f"pooled WFDB rhythm-annotated DBs: {', '.join(sorted(set(used_sources)))}; "
                   "patient(record)-grouped CV",
        "sources": sorted(set(used_sources)),
        "horizon_min": args.horizon_min,
        "far_min": args.far_min,
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "k_sustained": args.k_sustained,
        "target_control_specificity": args.target_specificity,
        "operating_threshold": threshold,
        "n_records": n_groups,
        "n_windows": len(rows),
        "n_preonset_windows": n_pos,
        "n_control_windows": n_neg,
        "window_auc_grouped_cv": auc,
        "onset_sensitivity": ew["sensitivity"],
        "onset_sensitivity_wilson_ci": [sens_ci[1], sens_ci[2]],
        "onsets_total": ew["onsets_total"],
        "onsets_warned": ew["onsets_warned"],
        "lead_time_median_min": ew["lead_time_median_min"],
        "lead_time_iqr_min": ew["lead_time_iqr_min"],
        "false_alarms_per_hour": ew["false_alarms_per_hour"],
        "lead_times_min": ew["lead_times_min"],
        "per_source": per_source,
        "caveats": [
            "afdb are selected AFib patients, not a general screening population.",
            "Few records -> wide CIs; treat as exploratory early-warning evidence.",
            "Threshold set at fixed control specificity on out-of-fold scores.",
            "Prediction uses RR/HRV features on SINUS windows only (no AFib morphology).",
        ],
    }

    if args.out:
        from src.experiments.provenance import write_json, write_run_metadata
        write_run_metadata(extra={"stage": "afib_onset_prediction"})
        write_json(Path(args.out), summary)
        print(f"Wrote {args.out}")

    if args.fig and ew["lead_times_min"]:
        _plot(summary, args.fig)

    print(f"Window AUC (grouped CV): {auc:.3f} | "
          f"onset sensitivity: {ew['sensitivity']} "
          f"({ew['onsets_warned']}/{ew['onsets_total']}) | "
          f"median lead time: {ew['lead_time_median_min']} min | "
          f"FA/h: {ew['false_alarms_per_hour']}")
    return summary


def _plot(summary: dict[str, Any], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lt = summary["lead_times_min"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(lt, bins=min(20, max(5, len(lt))), color="#185FA5", alpha=0.85)
    med = summary["lead_time_median_min"]
    if med is not None:
        ax.axvline(med, color="#A32D2D", ls="--", lw=1.5, label=f"median={med:.1f} min")
        ax.legend(fontsize=9)
    ax.set_xlabel("Lead time before AFib onset (min)")
    ax.set_ylabel("Onsets warned")
    ax.set_title(
        f"AFib onset prediction (afdb, exploratory)\n"
        f"sensitivity={summary['onset_sensitivity']:.2f}, "
        f"FA/h={summary['false_alarms_per_hour']:.2f}, "
        f"window AUC={summary['window_auc_grouped_cv']:.3f}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def _self_test() -> None:
    # sustained_mask
    assert sustained_mask([0, 1, 1, 0, 1], 2).tolist() == [False, True, True, False, False]
    assert sustained_mask([1, 1, 1], 2).tolist() == [True, True, True]
    assert sustained_mask([1, 0, 1, 0], 2).tolist() == [False, False, False, False]
    # lead time: one record, one onset, pre-onset windows at ttom 28..2,
    # scores cross threshold sustained from ttom=20.
    rows = [{"record": "r", "t_start_sec": t, "time_to_onset_min": 30 - t / 60,
             "label": 1, "onset_id": 1} for t in range(0, 1680, 60)]  # 28 windows
    scores = np.zeros(len(rows))
    scores[5:] = 0.9  # sustained high from the 6th window onward
    res = lead_times_and_alarms(rows, scores, 0.5, 2, 60)
    assert res["onsets_total"] == 1 and res["onsets_warned"] == 1
    # earliest warning is window index 5 -> ttom = 30 - 300/60 = 25 min
    assert abs(res["lead_times_min"][0] - 25.0) < 1e-6
    print("afib_onset_prediction self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory AFib onset prediction")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--data-dirs", nargs="*",
        default=["afdb:data/raw/mit_afib/files", "ltafdb:data/raw/ltafdb/files"],
        help="one or more 'source:path' WFDB rhythm-annotated dirs to pool "
             "(missing dirs are skipped with a warning)")
    parser.add_argument("--horizon-min", type=float, default=30.0)
    parser.add_argument("--far-min", type=float, default=60.0)
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--stride-sec", type=float, default=30.0)
    parser.add_argument("--k-sustained", type=int, default=2)
    parser.add_argument("--target-specificity", type=float, default=0.95)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    parser.add_argument("--fig", default=None)
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
