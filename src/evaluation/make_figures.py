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


def fig_crossdevice(cd: dict[str, Any], out_dir: Path) -> bool:
    """Head-to-head zero-shot cross-device degradation.

    Per external cohort: RR+RF vs CNN-LSTM(CPSC) AUC with patient/record-clustered
    bootstrap 95% CI error bars, chance line, and the Holm-corrected paired-DeLong
    p annotated above each cohort. This is the empirical centerpiece figure.
    Cohorts without both models scored (e.g. Apple Watch plausibility-only) are
    skipped so the figure only shows valid head-to-head comparisons.
    """
    cohorts = cd.get("cohorts", {})

    def _ci_err(entry, auc):
        lo, hi = (entry.get("cluster_bootstrap_ci") or [None, None])
        if auc is None or lo is None or hi is None or lo != lo or hi != hi:
            return 0.0, 0.0
        return max(0.0, auc - lo), max(0.0, hi - auc)

    def _valid(v):
        return v is not None and v == v  # rejects None and NaN (single-class cohorts)

    names, rr_auc, cnn_auc, rr_err, cnn_err, pvals = [], [], [], [], [], []
    for name, c in cohorts.items():
        mc = c.get("model_auc_clustered_ci", {})
        rr = mc.get("rr_rf", {})
        cnn = mc.get("cnn_cpsc", {})
        if not (_valid(rr.get("auc")) and _valid(cnn.get("auc"))):
            continue  # e.g. single-class / plausibility-only cohort (Apple Watch)
        names.append(name)
        rr_auc.append(rr["auc"]); cnn_auc.append(cnn["auc"])
        rlo, rhi = _ci_err(rr, rr["auc"]); rr_err.append([rlo, rhi])
        clo, chi = _ci_err(cnn, cnn["auc"]); cnn_err.append([clo, chi])
        d = c.get("paired_delong_rr_vs_cnn_cpsc") or {}
        pvals.append(d.get("p_value_holm", d.get("p_value")))

    if not names:
        print("SKIP crossdevice: no cohort has both RR+RF and CNN-LSTM(CPSC) scored")
        return False

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 2.6 * len(names)), 5))
    ax.bar(x - w / 2, rr_auc, w, color=MODEL_COLORS["rr_rf"],
           yerr=np.array(rr_err).T, capsize=4, ecolor="#444441",
           label="RR + RF (device-agnostic)")
    ax.bar(x + w / 2, cnn_auc, w, color=MODEL_COLORS["cnn_cpsc"],
           yerr=np.array(cnn_err).T, capsize=4, ecolor="#444441",
           label="CNN-LSTM (CPSC), zero-shot")
    ax.axhline(0.5, ls="--", lw=1, color="#B4B2A9", label="chance (0.5)")

    for xi, (r, c, p) in enumerate(zip(rr_auc, cnn_auc, pvals)):
        ax.text(xi - w / 2, r + 0.015, f"{r:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w / 2, c + 0.015, f"{c:.3f}", ha="center", va="bottom", fontsize=8)
        if p is not None:
            top = max(r, c)
            ax.text(xi, min(top + 0.08, 1.04), f"DeLong p(Holm)\n{p:.2g}",
                    ha="center", va="bottom", fontsize=7.5, color="#444441")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n{cohorts[n].get('device', '')}" for n in names], fontsize=8)
    ax.set_ylabel("AUC-ROC (zero-shot)")
    ax.set_ylim(0, 1.18)
    ax.set_title("Cross-device generalization: timing features transfer,\n"
                 "deep waveform model degrades off its training device")
    ax.legend(loc="lower left", fontsize=9)
    out = out_dir / "crossdevice_degradation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return True


def fig_inversion(paired: dict[str, Any], cd: dict[str, Any], out_dir: Path) -> bool:
    """Cross-device variability — the paper's headline visual.

    Left panel: the in-domain hold-out, where the deep model holds a small,
    reproducible advantage. Right panel: every external cohort, showing that the
    deep model's AUC swings far more widely than the device-agnostic model's and
    that the ordering between them changes from cohort to cohort — including
    between two cohorts of the same device class. Spread (SD) is annotated
    because the spread, not the mean, is the paper's claim.

    External points carry patient/record-clustered bootstrap 95% CIs. Apple Watch
    is drawn with a hatched marker: one confirmed positive, plausibility only.
    """
    from sklearn.metrics import roc_auc_score

    labels = np.array(paired.get("labels", []), dtype=int)
    probs = paired.get("probabilities", {})
    if labels.size == 0 or "rr_rf" not in probs or "cnn_cpsc" not in probs:
        print("SKIP variability: in-domain rr_rf / cnn_cpsc scores unavailable")
        return False

    def _valid(v):
        return v is not None and v == v

    in_rr = float(roc_auc_score(labels, probs["rr_rf"]))
    in_cnn = float(roc_auc_score(labels, probs["cnn_cpsc"]))

    # External cohorts, ordered as in the paper's table.
    order = (("apple_watch", "Apple Watch\n(consumer)", True),
             ("afdb", "MIT-BIH\n(Holter)", False),
             ("cinc2017", "CinC 2017\n(AliveCor)", False),
             ("ltafdb", "Long-Term AF\n(Holter)", False))
    cohorts = cd.get("cohorts", {})
    xs, rr, cnn, weak = [], [], [], []
    for key, pretty, is_weak in order:
        c = cohorts.get(key)
        if not c:
            continue
        mc = c.get("model_auc_clustered_ci", {})
        a, b = mc.get("rr_rf", {}), mc.get("cnn_cpsc", {})
        if not (_valid(a.get("auc")) and _valid(b.get("auc"))):
            continue
        xs.append(pretty)
        weak.append(is_weak)
        for entry, bucket in ((a, rr), (b, cnn)):
            lo, hi = (entry.get("cluster_bootstrap_ci") or [None, None])
            bucket.append((entry["auc"], lo if _valid(lo) else None,
                           hi if _valid(hi) else None))

    if not xs:
        print("SKIP variability: no external cohort has both models scored")
        return False

    def _series(vals):
        y = np.array([v[0] for v in vals], dtype=float)
        lo = np.array([v[0] - v[1] if v[1] is not None else 0.0 for v in vals])
        hi = np.array([v[2] - v[0] if v[2] is not None else 0.0 for v in vals])
        return y, np.vstack([np.maximum(lo, 0), np.maximum(hi, 0)])

    rr_y, rr_e = _series(rr)
    cnn_y, cnn_e = _series(cnn)
    x = np.arange(len(xs))

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(max(9, 2.1 * len(xs) + 3), 5.2),
        gridspec_kw={"width_ratios": [1, max(2.4, len(xs))]}, sharey=True)

    # ── Left: in-domain ────────────────────────────────────────────────
    axl.bar([0], [in_rr], 0.5, color=MODEL_COLORS["rr_rf"])
    axl.bar([1], [in_cnn], 0.5, color=MODEL_COLORS["cnn_cpsc"])
    for xi, v in ((0, in_rr), (1, in_cnn)):
        axl.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    axl.set_xticks([0, 1])
    axl.set_xticklabels(["RR + RF", "CNN-LSTM"], fontsize=8.5)
    axl.set_ylabel("AUC-ROC")
    axl.set_title("In-domain hold-out\n(deep model modestly ahead)", fontsize=10)
    axl.set_xlim(-0.6, 1.6)

    # ── Right: external cohorts ────────────────────────────────────────
    axr.errorbar(x, rr_y, yerr=rr_e, marker="o", ms=9, lw=2.5, capsize=5,
                 color=MODEL_COLORS["rr_rf"], ecolor=MODEL_COLORS["rr_rf"],
                 label=f"RR + RF (SD {rr_y.std(ddof=1):.3f})")
    axr.errorbar(x, cnn_y, yerr=cnn_e, marker="s", ms=9, lw=2.5, capsize=5,
                 color=MODEL_COLORS["cnn_cpsc"], ecolor=MODEL_COLORS["cnn_cpsc"],
                 label=f"CNN-LSTM zero-shot (SD {cnn_y.std(ddof=1):.3f})")

    # Shade each model's full range to make the spread difference immediate.
    for y, colour in ((rr_y, MODEL_COLORS["rr_rf"]),
                      (cnn_y, MODEL_COLORS["cnn_cpsc"])):
        axr.axhspan(y.min(), y.max(), color=colour, alpha=0.07, zorder=0)

    for xi, (a, b, is_weak) in enumerate(zip(rr_y, cnn_y, weak)):
        a_up = a >= b
        axr.annotate(f"{a:.3f}", (xi, a), textcoords="offset points",
                     xytext=(0, 12 if a_up else -18), ha="center", fontsize=8.5,
                     color=MODEL_COLORS["rr_rf"])
        axr.annotate(f"{b:.3f}", (xi, b), textcoords="offset points",
                     xytext=(0, -18 if a_up else 12), ha="center", fontsize=8.5,
                     color=MODEL_COLORS["cnn_cpsc"])
        if is_weak:
            axr.annotate("1 positive\n(plausibility)", (xi, min(a, b)),
                         textcoords="offset points", xytext=(0, -42),
                         ha="center", fontsize=7, color="#888780", style="italic")

    axr.axhline(0.5, ls="--", lw=1, color="#B4B2A9", label="chance (0.5)")
    axr.set_xticks(x)
    axr.set_xticklabels(xs, fontsize=8.5)
    axr.set_xlim(-0.45, len(x) - 0.55)
    axr.set_title("Zero-shot external cohorts — the deep model's AUC swings\n"
                  "far more widely, and the ordering changes by cohort",
                  fontsize=10)
    axr.legend(loc="lower right", fontsize=8.5)

    axl.set_ylim(0.45, 1.05)
    fig.suptitle("In-domain accuracy does not predict cross-device variability",
                 fontsize=12)
    out = out_dir / "crossdevice_variability.png"
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
    parser.add_argument("--crossdevice-json", default=None,
                        help="crossdevice_stats.json")
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
    if args.crossdevice_json:
        cd = load_json(args.crossdevice_json)
        if cd:
            made += int(fig_crossdevice(cd, out_dir))
            # Headline figure needs both in-domain and external results.
            if args.paired_json:
                paired = load_json(args.paired_json)
                if paired:
                    made += int(fig_inversion(paired, cd, out_dir))

    print(f"Done — {made} figure(s) written to {out_dir}")


if __name__ == "__main__":
    main()
