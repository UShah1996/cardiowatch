"""
make_figures.py — turn pipeline result JSONs into paper figures (PNG).

Consumes the JSON written by paired_cpsc_eval.py and latency_bootstrap.py and
produces publication figures. Each figure is independent: a missing or
single-class input is skipped with a message rather than failing the run, so a
partial pipeline still yields whatever figures it can.

Figures:
  1. roc_cpsc_holdout.png   — ROC of every model on the shared CPSC holdout
  2. auc_cpsc_holdout.png    — AUC bar chart + DeLong delta/Holm-p annotation
  3. detection_latency.png   — latency + normal-phase FP-rate distributions

Usage:
    python -m src.evaluation.make_figures \
        --paired-json docs/results/<run_id>/paired_cpsc_predictions.json \
        --latency-json docs/results/<run_id>/latency_bootstrap.json \
        --out-dir docs/results/<run_id>/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import auc as sk_auc, roc_curve  # noqa: E402

MODEL_LABELS = {
    "rr_rf": "RR + RF",
    "cnn_cpsc": "CNN-LSTM (CPSC complement)",
    "cnn_combined_deploy": "CNN-LSTM (combined, deploy)",
    "random_baseline": "Random baseline",
}
MODEL_COLORS = {
    "rr_rf": "#185FA5",
    "cnn_cpsc": "#993C1D",
    "cnn_combined_deploy": "#3B6D11",
    "random_baseline": "#888780",
}
MODEL_ORDER = ["rr_rf", "cnn_cpsc", "cnn_combined_deploy", "random_baseline"]


def load_json(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        print(f"SKIP: {p} not found")
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP: could not parse {p}: {exc}")
        return None


def _label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def _present_models(probs: dict[str, list[float]]) -> list[str]:
    ordered = [m for m in MODEL_ORDER if m in probs]
    return ordered + [m for m in probs if m not in ordered]


def fig_roc(paired: dict[str, Any], out_dir: Path) -> bool:
    labels = np.array(paired.get("labels", []), dtype=int)
    probs = paired.get("probabilities", {})
    if labels.size == 0 or len(set(labels.tolist())) < 2:
        print("SKIP roc: holdout has fewer than two classes")
        return False

    fig, ax = plt.subplots(figsize=(6, 6))
    for name in _present_models(probs):
        p = np.asarray(probs[name], dtype=float)
        if p.size != labels.size:
            continue
        fpr, tpr, _ = roc_curve(labels, p)
        ax.plot(
            fpr, tpr, lw=2, color=MODEL_COLORS.get(name, "#444441"),
            label=f"{_label(name)} (AUC={sk_auc(fpr, tpr):.3f})",
        )
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#B4B2A9")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC — CPSC holdout (n={paired.get('n_scored', '?')})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    out = out_dir / "roc_cpsc_holdout.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return True


def fig_auc_bar(paired: dict[str, Any], out_dir: Path) -> bool:
    labels = np.array(paired.get("labels", []), dtype=int)
    probs = paired.get("probabilities", {})
    if labels.size == 0 or len(set(labels.tolist())) < 2:
        print("SKIP auc_bar: holdout has fewer than two classes")
        return False

    names = [n for n in _present_models(probs) if np.asarray(probs[n]).size == labels.size]
    aucs = []
    for name in names:
        fpr, tpr, _ = roc_curve(labels, np.asarray(probs[name], dtype=float))
        aucs.append(sk_auc(fpr, tpr))

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names)), 5))
    bars = ax.bar(
        range(len(names)), aucs,
        color=[MODEL_COLORS.get(n, "#444441") for n in names], width=0.6,
    )
    ax.axhline(0.5, ls="--", lw=1, color="#B4B2A9", label="chance (0.5)")
    for rect, a in zip(bars, aucs):
        ax.text(rect.get_x() + rect.get_width() / 2, a + 0.01, f"{a:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([_label(n) for n in names], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0, 1.05)
    ax.set_title("AUC on CPSC holdout")

    # Annotate the primary controlled comparison (DeLong + Holm-adjusted p).
    delong = (paired.get("stats", {}).get("rr_rf_vs_cnn_cpsc", {}) or {}).get("delong")
    if delong:
        dauc = delong.get("delta_auc")
        p_holm = delong.get("p_value_holm", delong.get("p_value"))
        if dauc is not None and p_holm is not None:
            ax.text(
                0.5, 0.02,
                f"RR vs CPSC-CNN: ΔAUC={dauc:+.3f}  "
                f"(95% CI {delong.get('ci_low', float('nan')):.3f}–"
                f"{delong.get('ci_high', float('nan')):.3f}), "
                f"DeLong p(Holm)={p_holm:.3g}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
                color="#444441",
            )
    out = out_dir / "auc_cpsc_holdout.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return True


def fig_latency(latency: dict[str, Any], out_dir: Path) -> bool:
    transitions = latency.get("transitions", [])
    lat = [t["latency_min"] for t in transitions if t.get("latency_min") is not None]
    fp = [t.get("normal_phase_fp_rate") for t in transitions
          if t.get("normal_phase_fp_rate") is not None]
    if not transitions:
        print("SKIP latency: no transitions in JSON")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    if lat:
        axes[0].hist(lat, bins=min(20, max(5, len(lat))), color="#185FA5", alpha=0.85)
        med = latency.get("latency_median_min")
        iqr = latency.get("latency_iqr_min") or [None, None]
        if med is not None:
            axes[0].axvline(med, color="#A32D2D", ls="--", lw=1.5,
                            label=f"median={med:.1f} min")
            if iqr[0] is not None and iqr[1] is not None:
                axes[0].axvspan(iqr[0], iqr[1], color="#A32D2D", alpha=0.10,
                                label=f"IQR {iqr[0]:.1f}–{iqr[1]:.1f}")
            axes[0].legend(fontsize=9)
    else:
        axes[0].text(0.5, 0.5, "no detections", ha="center", va="center",
                     transform=axes[0].transAxes)
    n_det = latency.get("n_detected", len(lat))
    n_tot = latency.get("n_transitions", len(transitions))
    axes[0].set_xlabel("Detection latency (min after onset)")
    axes[0].set_ylabel("Transitions")
    axes[0].set_title(f"Detection latency ({n_det}/{n_tot} detected)")

    axes[1].hist(fp, bins=min(20, max(5, len(fp))), color="#854F0B", alpha=0.85)
    fpm = latency.get("fp_rate_median")
    if fpm is not None:
        axes[1].axvline(fpm, color="#A32D2D", ls="--", lw=1.5,
                        label=f"median={fpm:.1%}")
        axes[1].legend(fontsize=9)
    axes[1].set_xlabel("Normal-phase false-positive rate")
    axes[1].set_ylabel("Transitions")
    axes[1].set_title("Normal-phase false positives")

    fig.suptitle(
        f"Bootstrap detection latency (threshold={latency.get('threshold', '?')}, "
        f"{n_tot} transitions)", fontsize=12,
    )
    out = out_dir / "detection_latency.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return True


def fig_mechanism(mech: dict[str, Any], out_dir: Path) -> bool:
    """Device-separability bar chart: RR features vs CNN embedding, per pair."""
    sep = mech.get("device_separability_auc", {})
    pairs = [p for p, e in sep.items()
             if e.get("rr_features_device_auc") is not None]
    if not pairs:
        print("SKIP mechanism: no device-separability results")
        return False

    labels = [p.replace("_vs_", " vs\n") for p in pairs]
    rr = [sep[p].get("rr_features_device_auc") for p in pairs]
    cnn = [sep[p].get("cnn_embedding_device_auc") for p in pairs]
    x = np.arange(len(pairs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 2.4 * len(pairs)), 4.8))
    ax.bar(x - w / 2, rr, w, color="#185FA5", label="RR/HRV features (device-agnostic)")
    if any(c is not None for c in cnn):
        ax.bar(x + w / 2, [c if c is not None else 0 for c in cnn], w,
               color="#993C1D", label="CNN-LSTM embedding")
    ax.axhline(0.5, ls="--", lw=1, color="#B4B2A9", label="chance (device-invariant)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Device-separability AUC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Does the representation encode the device?\n"
                 "Higher = device-specific (won't transfer)")
    ax.legend(fontsize=9, loc="upper left")
    for xi, (r, c) in enumerate(zip(rr, cnn)):
        if r is not None:
            ax.text(xi - w / 2, r + 0.01, f"{r:.2f}", ha="center", va="bottom", fontsize=8)
        if c is not None:
            ax.text(xi + w / 2, c + 0.01, f"{c:.2f}", ha="center", va="bottom", fontsize=8)
    out = out_dir / "device_separability.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CardioWatch paper figures")
    parser.add_argument("--paired-json", default=None,
                        help="paired_cpsc_predictions.json")
    parser.add_argument("--latency-json", default=None,
                        help="latency_bootstrap.json")
    parser.add_argument("--mechanism-json", default=None,
                        help="representation_shift.json")
    parser.add_argument("--out-dir", default="docs/results/figures",
                        help="directory for output PNGs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    if args.paired_json:
        paired = load_json(args.paired_json)
        if paired:
            made += int(fig_roc(paired, out_dir))
            made += int(fig_auc_bar(paired, out_dir))
    if args.latency_json:
        latency = load_json(args.latency_json)
        if latency:
            made += int(fig_latency(latency, out_dir))
    if args.mechanism_json:
        mech = load_json(args.mechanism_json)
        if mech:
            made += int(fig_mechanism(mech, out_dir))

    print(f"Done — {made} figure(s) written to {out_dir}")


if __name__ == "__main__":
    main()
