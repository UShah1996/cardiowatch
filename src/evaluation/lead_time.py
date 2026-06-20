"""
lead_time.py — Detection-Latency Evaluation for CardioWatch
Concatenates real verified Normal + AFib CPSC recordings into one long
signal with a known AFib onset point, then measures how quickly the
CNN-LSTM detects AFib after it begins, and how many false alerts it
raises during the preceding normal-sinus phase.

Why this replaces the old "lead time" metric:
  The previous version defined the cardiac event as exactly
  `onset + 30 min` and reported `event_time - first_alert` as "lead
  time", which made the 30-minute result true by construction. Worse,
  the "first alert" could be a false positive in the normal phase, so a
  spurious early alert inflated the reported lead time. A detector
  trained on AFib morphology cannot predict AFib before it starts on a
  normal→AFib concatenation; it can only detect it once present. The
  honest, measurable quantities are therefore:

    - detection latency : minutes from AFib onset to the first true
                          alert (at or after onset). Lower is better.
    - normal-phase FPs  : alerts raised during the verified normal-sinus
                          phase (before onset). Fewer is better.

Signal structure:
  - 35 min of verified Normal Sinus Rhythm recordings
  - 31 min of verified AFib recordings
  - AFib onset = boundary between the two phases (a real, known point)

Usage:
    python3 src/evaluation/lead_time.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import wfdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.models.cnn_lstm import build_model

# ── Config ────────────────────────────────────────────────────────────
WEIGHTS_PATH   = 'data/processed/cnn_lstm_combined_best.pt'
THRESHOLD      = 0.5
WINDOW_SAMPLES = 5000      # 10s at 500 Hz — must match training
FS             = 500
DATA_DIR       = ('data/raw/classification-of-12-lead-ecgs-the-physionetcomputing'
                  '-in-cardiology-challenge-2020-1.0.2/training/cpsc_2018')
NORMAL_CODE    = '426783006'
AFIB_CODE      = '164889003'


# ── Helpers ───────────────────────────────────────────────────────────
def preprocess_window(window):
    """Normalize a 10s window — identical to ECGDataset preprocessing."""
    w = window.astype(np.float32)
    w = np.clip(w, -2.0, 2.0)
    w = (w - w.mean()) / (w.std() + 1e-8)
    w = np.clip(w, -5.0, 5.0)
    if len(w) >= WINDOW_SAMPLES:
        w = w[:WINDOW_SAMPLES]
    else:
        w = np.pad(w, (0, WINDOW_SAMPLES - len(w)))
    return w


def load_real_recording(path):
    """Load Lead I from a CPSC recording, normalized."""
    record = wfdb.rdrecord(path)
    leads  = [n.strip().upper() for n in record.sig_name]
    sig    = record.p_signal[:, leads.index('I')].astype(np.float32)
    sig    = np.nan_to_num(sig)
    sig    = np.clip(sig, -2.0, 2.0)
    sig    = (sig - sig.mean()) / (sig.std() + 1e-8)
    sig    = np.clip(sig, -5.0, 5.0)
    return sig


# ── Build signal from real recordings ────────────────────────────────
def build_real_signal(normal_minutes=35, afib_minutes=5):
    """
    Scans the CPSC dataset for verified Normal and AFib recordings,
    concatenates enough of each to reach the requested durations,
    and returns the full signal with onset/event timestamps.

    Args:
        normal_minutes : minutes of Normal Sinus Rhythm at the start
        afib_minutes   : minutes of AFib after onset

    Returns:
        full_signal    : concatenated float32 array
        onset_minutes  : when AFib starts (= end of normal phase)
        total_minutes  : full signal duration
    """
    normal_paths, afib_paths = [], []

    print("  Scanning for verified Normal and AFib recordings...")
    for root, dirs, files in os.walk(DATA_DIR):
        for fname in files:
            if not fname.endswith('.hea'):
                continue
            path = os.path.join(root, fname.replace('.hea', ''))
            try:
                h = wfdb.rdheader(path)
                for c in h.comments:
                    if c.startswith('Dx:'):
                        codes = [x.strip() for x in c.replace('Dx:', '').split(',')]
                        if NORMAL_CODE in codes:
                            normal_paths.append(path)
                        elif AFIB_CODE in codes:
                            afib_paths.append(path)
                        break
            except:
                continue

    print(f"  Found {len(normal_paths)} Normal | {len(afib_paths)} AFib recordings")

    def load_until(paths, target_min):
        segs           = []
        total_samples  = 0
        target_samples = int(target_min * 60 * FS)
        for p in paths:
            if total_samples >= target_samples:
                break
            try:
                seg = load_real_recording(p)
                segs.append(seg)
                total_samples += len(seg)
            except:
                continue
        return np.concatenate(segs) if segs else np.array([], dtype=np.float32)

    print(f"  Loading {normal_minutes} min of Normal...")
    normal_signal = load_until(normal_paths, normal_minutes)

    print(f"  Loading {afib_minutes} min of AFib...")
    afib_signal = load_until(afib_paths, afib_minutes)

    full_signal   = np.concatenate([normal_signal, afib_signal])
    onset_minutes = len(normal_signal) / FS / 60.0
    total_minutes = len(full_signal) / FS / 60.0

    print(f"  Normal phase : {onset_minutes:.2f} min ({len(normal_signal):,} samples)")
    print(f"  AFib phase   : {len(afib_signal)/FS/60:.2f} min ({len(afib_signal):,} samples)")
    print(f"  Total signal : {total_minutes:.2f} min")
    print(f"  AFib onset   : {onset_minutes:.2f} min (boundary between phases)")

    return full_signal, onset_minutes, total_minutes


# ── CNN-LSTM inference across full signal ─────────────────────────────
def ecg_risk_over_time(signal, cnn_model, stride_sec=10):
    """Slides a 10s window every stride_sec seconds. Returns times + probs."""
    stride    = int(stride_sec * FS)
    times_min = []
    ecg_probs = []

    for start in range(0, len(signal) - WINDOW_SAMPLES + 1, stride):
        w = preprocess_window(signal[start: start + WINDOW_SAMPLES])
        x = torch.tensor(w).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(cnn_model(x).squeeze()).item()
        times_min.append(start / FS / 60.0)
        ecg_probs.append(prob)

    return times_min, ecg_probs


# ── Detection-latency calculation ─────────────────────────────────────
def compute_detection_latency(times_min, probs, onset_min,
                              threshold=THRESHOLD):
    """
    Minutes from AFib onset to the first alert at or after onset.

    Only alerts at or after the true onset count as detections — an alert
    in the normal phase is a false positive, not an early detection, and
    is reported separately by normal_phase_false_positives().

    Returns (latency_minutes, alert_time_minutes), or (None, None) if the
    model never alerts after onset.
    """
    for t, p in zip(times_min, probs):
        if t >= onset_min and p >= threshold:
            return t - onset_min, t
    return None, None


def normal_phase_false_positives(times_min, probs, onset_min,
                                 threshold=THRESHOLD):
    """
    Count alerts during the verified normal-sinus phase (before onset).

    Returns (n_false_positives, n_normal_windows, fp_rate).
    """
    normal = [(t, p) for t, p in zip(times_min, probs) if t < onset_min]
    n_normal = len(normal)
    n_fp     = sum(1 for _, p in normal if p >= threshold)
    return n_fp, n_normal, n_fp / max(n_normal, 1)


# ── Main ──────────────────────────────────────────────────────────────
def evaluate_detection_latency(threshold=THRESHOLD, plot=True):
    """
    Detection-latency evaluation using real CPSC recordings.

    Measures the CNN-LSTM's intrinsic ability to detect AFib once it
    begins — using the ECG probability directly, with no clinical-score
    inflation — and counts any false alerts in the preceding normal
    phase. (The threshold × clinical-prior tradeoff is explored
    separately in lead_time_sweep.py.)
    """
    print("Loading CNN-LSTM model...")
    cnn_model = build_model(input_length=5000)
    cnn_model.load_state_dict(
        torch.load(WEIGHTS_PATH, map_location='cpu'))
    cnn_model.eval()
    print("CNN-LSTM loaded.\n")

    print("Building signal from real CPSC recordings...")
    signal, afib_onset_t, total_min = build_real_signal(
        normal_minutes=35,
        afib_minutes=31
    )
    print()

    print("Running CNN-LSTM inference (stride=10s)...")
    times_min, ecg_probs = ecg_risk_over_time(signal, cnn_model, stride_sec=10)
    print(f"Evaluated {len(times_min)} windows.\n")

    latency, first_alert = compute_detection_latency(
        times_min, ecg_probs, afib_onset_t, threshold)
    n_fp, n_normal, fp_rate = normal_phase_false_positives(
        times_min, ecg_probs, afib_onset_t, threshold)

    # ── Results ───────────────────────────────────────────────────────
    print("=" * 55)
    print(f"  Threshold              : {threshold}")
    print(f"  AFib onset             : {afib_onset_t:.2f} min")
    if latency is not None:
        print(f"  First detection (>=onset): {first_alert:.2f} min")
        print(f"  Detection latency      : {latency:.2f} min after onset")
    else:
        print(f"  First detection        : never (no alert after onset)")
    print(f"  Normal-phase windows   : {n_normal}")
    print(f"  Normal-phase false pos : {n_fp} ({fp_rate:.1%})")
    print("=" * 55)

    # ── Plot ──────────────────────────────────────────────────────────
    if plot:
        os.makedirs('docs', exist_ok=True)
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

        # Panel 1: Signal morphology comparison
        preview     = WINDOW_SAMPLES
        afib_start  = int(afib_onset_t * 60 * FS)
        t_sec       = np.arange(preview) / FS

        axes[0].plot(t_sec, signal[0: preview],
                     color='steelblue', linewidth=0.8,
                     label='Normal sinus (real CPSC, Lead I)')
        axes[0].plot(t_sec, signal[afib_start: afib_start + preview],
                     color='crimson', linewidth=0.8, alpha=0.9,
                     label=f'AFib (real CPSC, Lead I, onset={afib_onset_t:.1f} min)')
        axes[0].set_ylabel('Amplitude (normalized)')
        axes[0].set_xlabel('Seconds (10s excerpt from each phase)')
        axes[0].set_title('Real ECG — Normal Sinus vs AFib Morphology (Lead I)')
        axes[0].legend(fontsize=9)

        # Panel 2: ECG risk over time with onset, detection, FP
        axes[1].plot(times_min, ecg_probs, color='steelblue',
                     linewidth=1.5, label='ECG AFib risk (CNN-LSTM)')
        axes[1].axvspan(0, afib_onset_t, alpha=0.06, color='green',
                        label='Normal-sinus phase')
        axes[1].axvspan(afib_onset_t, total_min,
                        alpha=0.08, color='red', label='AFib phase')
        axes[1].axvline(afib_onset_t, color='orange', linestyle=':',
                        linewidth=1.5,
                        label=f'AFib onset ({afib_onset_t:.1f} min)')
        axes[1].axhline(threshold, color='red', linestyle='--',
                        linewidth=1.5, label=f'Threshold ({threshold})')
        # Mark false positives in the normal phase
        fp_t = [t for t, p in zip(times_min, ecg_probs)
                if t < afib_onset_t and p >= threshold]
        if fp_t:
            axes[1].scatter(fp_t, [threshold] * len(fp_t), marker='x',
                            color='black', s=40, zorder=5,
                            label=f'Normal-phase false pos ({len(fp_t)})')
        if first_alert is not None:
            axes[1].axvline(first_alert, color='limegreen',
                            linestyle='-.', linewidth=2,
                            label=f'First detection ({first_alert:.1f} min)')
            axes[1].annotate(
                f'{latency:.1f} min\nlatency',
                xy=(first_alert, threshold + 0.04),
                xytext=(first_alert + 1.5, threshold + 0.20),
                fontsize=10, color='darkgreen', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5)
            )
        axes[1].set_ylabel('AFib probability')
        axes[1].set_xlabel('Time (minutes)')
        axes[1].set_title(
            f'CNN-LSTM ECG Risk Over Time  |  Detection latency: '
            + (f'{latency:.1f} min' if latency is not None else 'not detected')
            + f'  |  Normal-phase FP: {n_fp}/{n_normal}'
        )
        axes[1].legend(fontsize=8)
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_xlim(0, total_min)

        plt.tight_layout()
        out_path = 'docs/lead_time_evaluation.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved -> {out_path}")
        plt.close()

    return latency, times_min, ecg_probs


if __name__ == '__main__':
    evaluate_detection_latency(plot=True)