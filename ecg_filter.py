"""
src/preprocessing/ecg_filter.py

Band-pass filtering and 5-minute windowing of ECG signals.
Targets Lead I (Apple Watch equivalent) from PhysioNet recordings.
"""

import numpy as np
from scipy.signal import butter, filtfilt
import wfdb
from pathlib import Path
import yaml


def load_config(config_path="configs/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float,
                    fs: float, order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter to an ECG signal.

    Args:
        signal:  Raw ECG signal (1D array).
        lowcut:  Low cutoff frequency in Hz (default 0.5).
        highcut: High cutoff frequency in Hz (default 100.0).
        fs:      Sampling frequency in Hz.
        order:   Filter order.

    Returns:
        Filtered ECG signal.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def extract_lead_i(record_path: str) -> np.ndarray:
    """
    Load a PhysioNet record and return Lead I signal only.
    Lead I approximates the electrical path measured by Apple Watch.

    Args:
        record_path: Path to PhysioNet .hea file (without extension).

    Returns:
        Lead I signal as 1D numpy array.
    """
    record = wfdb.rdrecord(record_path)
    lead_names = [name.strip().upper() for name in record.sig_name]

    if "I" not in lead_names:
        raise ValueError(f"Lead I not found in record. Available: {lead_names}")

    lead_idx = lead_names.index("I")
    return record.p_signal[:, lead_idx].astype(np.float32)


def segment_into_windows(signal: np.ndarray, fs: float,
                          window_minutes: float = 5.0) -> np.ndarray:
    """
    Segment a continuous ECG signal into fixed-length windows.

    Args:
        signal:         Filtered ECG signal.
        fs:             Sampling frequency in Hz.
        window_minutes: Window length in minutes.

    Returns:
        2D array of shape (n_windows, window_samples).
    """
    window_samples = int(window_minutes * 60 * fs)
    n_windows = len(signal) // window_samples
    trimmed = signal[:n_windows * window_samples]
    return trimmed.reshape(n_windows, window_samples)


def process_record(record_path: str, config: dict) -> np.ndarray:
    """
    Full pipeline: load → filter → window for one ECG record.

    Args:
        record_path: Path to PhysioNet record (no extension).
        config:      Loaded YAML config dict.

    Returns:
        Windowed, filtered Lead I segments.
    """
    cfg = config["preprocessing"]["ecg"]

    raw_lead = extract_lead_i(record_path)
    filtered = bandpass_filter(
        signal=raw_lead,
        lowcut=cfg["bandpass_low_hz"],
        highcut=cfg["bandpass_high_hz"],
        fs=cfg["sampling_rate_hz"],
    )
    windows = segment_into_windows(
        signal=filtered,
        fs=cfg["sampling_rate_hz"],
        window_minutes=cfg["window_minutes"],
    )
    return windows


if __name__ == "__main__":
    cfg = load_config()
    # Quick smoke test on a dummy signal
    dummy = np.random.randn(cfg["preprocessing"]["ecg"]["sampling_rate_hz"] * 600)
    filtered = bandpass_filter(dummy, 0.5, 100.0, fs=500)
    windows = segment_into_windows(filtered, fs=500, window_minutes=5)
    print(f"✓ Filter OK | Windows shape: {windows.shape}")
