"""
lead_time_sweep.py — Detection-Latency / False-Positive Tradeoff
================================================================
Replaces the old circular "lead time" sweep with an honest operating-
point analysis. For each alert threshold (and each assumed clinical
prior rf_prob) it reports:

    - detection latency : minutes from AFib onset to the first alert at
                          or after onset (lower is better)
    - normal-phase FPs  : false alerts during the verified normal-sinus
                          phase, before onset (fewer is better)

The resulting latency-vs-false-positive curve is a genuine clinical
decision-support tradeoff: lower thresholds (or a higher clinical prior)
detect AFib sooner but raise more false alerts in normal sinus. No
manually-defined "event" time and no >=30-min target — those made the
previous result true by construction.

Note: when the assumed clinical prior rf_prob is high, the fused
baseline alone can approach the threshold, producing alerts in the
normal phase. Those are reported honestly as false positives rather
than as "early" detections.

Usage:
    python3 src/evaluation/lead_time_sweep.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple, Optional

from src.models.cnn_lstm import build_model
from src.evaluation.lead_time import (
    build_real_signal,
    ecg_risk_over_time,
    preprocess_window,
    WEIGHTS_PATH,
    FS,
    WINDOW_SAMPLES,
)

# ── Config ────────────────────────────────────────────────────────────
# Sweep these RF probabilities — covers low-risk to high-risk patients
RF_PROBS       = [0.45, 0.55, 0.65, 0.75]
RF_WEIGHTS     = 0.6
ECG_WEIGHT     = 0.4

# Sweep these alert thresholds
THRESHOLDS     = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

# Signal structure (same as lead_time.py)
NORMAL_MINUTES = 35
AFIB_MINUTES   = 31


def fuse(rf_prob: float, ecg_prob: float) -> float:
    return RF_WEIGHTS * rf_prob + ECG_WEIGHT * ecg_prob


def false_positive_rate(
    times_min:    List[float],
    fused_probs:  List[float],
    onset_min:    float,
    threshold:    float,
) -> Tuple[int, int, float]:
    """
    Count false positives in the normal sinus phase (before AFib onset).

    Returns:
        (n_fp, n_windows_normal, fp_rate)
    """
    normal_mask = [t < onset_min for t in times_min]
    n_normal    = sum(normal_mask)
    n_fp        = sum(
        1 for t, p, is_normal in zip(times_min, fused_probs, normal_mask)
        if is_normal and p >= threshold
    )
    fp_rate = n_fp / max(n_normal, 1)
    return n_fp, n_normal, fp_rate


def first_detection_after_onset(
    times_min:    List[float],
    fused_probs:  List[float],
    onset_min:    float,
    threshold:    float,
) -> Tuple[Optional[float], Optional[float]]:
    """
    First alert at or after AFib onset. Returns (detection_latency,
    alert_time), where latency = alert_time - onset_min. Alerts before
    onset are false positives, not detections, and are excluded here.
    """
    for t, p in zip(times_min, fused_probs):
        if t >= onset_min and p >= threshold:
            return t - onset_min, t
    return None, None


def run_sweep(
    times_min:   List[float],
    ecg_probs:   List[float],
    onset_min:   float,
    rf_prob:     float,
) -> List[dict]:
    """
    Run the threshold sweep for a single rf_prob value.
    Returns list of result dicts, one per threshold.
    """
    fused_probs = [fuse(rf_prob, e) for e in ecg_probs]
    results     = []

    for thresh in THRESHOLDS:
        latency, alert_time = first_detection_after_onset(
            times_min, fused_probs, onset_min, thresh
        )
        n_fp, n_norm, fp_rate = false_positive_rate(
            times_min, fused_probs, onset_min, thresh
        )
        results.append({
            'threshold'  : thresh,
            'rf_prob'    : rf_prob,
            'latency'    : latency,
            'alert_time' : alert_time,
            'n_fp'       : n_fp,
            'n_normal'   : n_norm,
            'fp_rate'    : fp_rate,
        })

    return results


def print_sweep_table(all_results: List[List[dict]]) -> None:
    """Print a formatted table of all sweep results."""
    print("\n" + "="*80)
    print("DETECTION-LATENCY THRESHOLD SWEEP RESULTS")
    print("="*80)
    header = (f"{'rf_prob':>8} {'thresh':>8} {'latency':>12} "
              f"{'alert_t':>10} {'FP':>6} {'FP_rate':>10}")
    print(header)
    print("-"*80)

    for rf_results in all_results:
        for r in rf_results:
            lt_str = (f"{r['latency']:.1f} min" if r['latency'] is not None
                      else "NOT DET.")
            at_str = f"{r['alert_time']:.1f} min" if r['alert_time'] else "—"
            print(
                f"{r['rf_prob']:>8.2f} "
                f"{r['threshold']:>8.2f} "
                f"{lt_str:>12} "
                f"{at_str:>10} "
                f"{r['n_fp']:>6} "
                f"{r['fp_rate']:>10.1%}"
            )
        print()

    print("="*80)
    print("Interpretation:")
    print("  Lower threshold (or higher clinical prior) → shorter detection")
    print("  latency but more false positives in the normal-sinus phase.")
    print("  The latency-vs-FP curve shows the achievable operating points.")
    print("="*80 + "\n")


def plot_sweep(
    all_results:  List[List[dict]],
    times_min:    List[float],
    ecg_probs:    List[float],
    onset_min:    float,
    total_min:    float,
    save_path:    str = 'docs/lead_time_tradeoff.png',
) -> None:
    """
    Three-panel plot:
      Panel 1: Detection latency vs threshold (one line per rf_prob)
      Panel 2: False positive rate vs threshold
      Panel 3: Detection latency vs FP rate (operating point curve)
    Latency is plotted only where AFib was detected; NaN gaps mean the
    threshold never fired after onset.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    afib_min = total_min - onset_min          # max possible latency
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        'CardioWatch Detection-Latency Threshold Sweep\n'
        f'(Normal sinus: {NORMAL_MINUTES} min → AFib onset → '
        f'AFib phase: {AFIB_MINUTES} min)',
        fontsize=13, fontweight='bold'
    )

    def latency_or_nan(r):
        return r['latency'] if r['latency'] is not None else np.nan

    # ── Panel 1: Detection latency vs threshold ──────────────────────
    ax = axes[0]
    for i, (rf_results, rf_p) in enumerate(zip(all_results, RF_PROBS)):
        lats = [latency_or_nan(r) for r in rf_results]
        ax.plot(THRESHOLDS, lats,
                'o-', color=colors[i], linewidth=2, markersize=7,
                label=f'RF prob={rf_p}')
    ax.set_xlabel('Alert threshold', fontsize=11)
    ax.set_ylabel('Detection latency (minutes)', fontsize=11)
    ax.set_title('Detection Latency vs Threshold\n(lower is better)', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(min(THRESHOLDS) - 0.02, max(THRESHOLDS) + 0.02)
    ax.set_ylim(-0.5, max(afib_min, 1))
    ax.grid(alpha=0.3)

    # ── Panel 2: FP rate vs threshold ────────────────────────────────
    ax = axes[1]
    for i, (rf_results, rf_p) in enumerate(zip(all_results, RF_PROBS)):
        fps = [r['fp_rate'] for r in rf_results]
        ax.plot(THRESHOLDS, fps,
                's-', color=colors[i], linewidth=2, markersize=7,
                label=f'RF prob={rf_p}')

    ax.axhline(0.05, color='orange', linestyle=':', linewidth=1.5,
               label='5% FP reference')
    ax.set_xlabel('Alert threshold', fontsize=11)
    ax.set_ylabel('False positive rate (normal phase)', fontsize=11)
    ax.set_title('False Positive Rate vs Threshold\n(lower is better)', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(min(THRESHOLDS) - 0.02, max(THRESHOLDS) + 0.02)
    ax.set_ylim(-0.01, 0.6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(alpha=0.3)

    # ── Panel 3: Latency vs FP rate (operating point curve) ──────────
    ax = axes[2]
    for i, (rf_results, rf_p) in enumerate(zip(all_results, RF_PROBS)):
        fps  = [r['fp_rate'] for r in rf_results]
        lats = [latency_or_nan(r) for r in rf_results]
        ax.plot(fps, lats, 'o-', color=colors[i], linewidth=2, markersize=7,
                label=f'RF prob={rf_p}')
        for r, fp, lt in zip(rf_results, fps, lats):
            if r['threshold'] in [0.40, 0.50, 0.60] and not np.isnan(lt):
                ax.annotate(
                    f't={r["threshold"]}',
                    xy=(fp, lt), xytext=(fp + 0.01, lt + 0.3),
                    fontsize=7, color=colors[i], alpha=0.8,
                )

    ax.axvline(0.05, color='orange', linestyle=':', linewidth=1.5,
               label='5% FP reference')
    ax.set_xlabel('False positive rate (normal phase)', fontsize=11)
    ax.set_ylabel('Detection latency (minutes)', fontsize=11)
    ax.set_title('Latency vs FP Rate\n(operating point curve — lower-left is best)',
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.5, max(afib_min, 1))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.01, 0.55)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Tradeoff plot saved → {save_path}")


def run_full_sweep(plot: bool = True) -> List[List[dict]]:
    """
    Run the complete lead-time sweep.

    1. Builds the signal once (expensive — loads all recordings)
    2. Runs CNN-LSTM inference once (reused across all thresholds)
    3. Sweeps thresholds and rf_prob values (fast — just arithmetic)
    4. Prints table and saves plot
    """
    print("Loading CNN-LSTM model...")
    cnn_model = build_model(input_length=5000)
    cnn_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
    cnn_model.eval()
    print("CNN-LSTM loaded.\n")

    print("Building signal from real CPSC recordings...")
    signal, onset_min, total_min = build_real_signal(
        normal_minutes=NORMAL_MINUTES,
        afib_minutes=AFIB_MINUTES,
    )
    print()

    print("Running CNN-LSTM inference (stride=10s, one pass)...")
    times_min, ecg_probs = ecg_risk_over_time(signal, cnn_model, stride_sec=10)
    print(f"Evaluated {len(times_min)} windows.\n")

    # Sweep thresholds for each rf_prob
    all_results = []
    for rf_p in RF_PROBS:
        print(f"Sweeping thresholds for rf_prob={rf_p}...")
        results = run_sweep(times_min, ecg_probs, onset_min, rf_p)
        all_results.append(results)

    print_sweep_table(all_results)

    if plot:
        plot_sweep(all_results, times_min, ecg_probs,
                   onset_min, total_min)

    # Summary stats: best (lowest-latency) operating point that keeps the
    # normal-phase false-positive rate at or below 5%.
    print("\nKEY FINDING:")
    print("─"*50)
    for rf_results in all_results:
        rf_p   = rf_results[0]['rf_prob']
        usable = [r for r in rf_results
                  if r['latency'] is not None and r['fp_rate'] <= 0.05]
        if usable:
            best = min(usable, key=lambda r: r['latency'])
            print(f"  rf_prob={rf_p}: best latency {best['latency']:.1f} min "
                  f"at threshold {best['threshold']:.2f} "
                  f"(FP rate {best['fp_rate']:.1%})")
        else:
            print(f"  rf_prob={rf_p}: no threshold detects AFib with FP rate <= 5%")
    print("─"*50)
    print("Detection latency and false-positive rate are read off real")
    print("model behaviour — no manually-defined event, no fixed target.")

    return all_results


if __name__ == '__main__':
    run_full_sweep(plot=True)
