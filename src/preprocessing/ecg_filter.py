import numpy as np
from scipy.signal import butter, filtfilt
import wfdb
import yaml
import os

def load_config(path='configs/config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

def bandpass_filter(signal, lowcut=0.5, highcut=100.0, fs=500, order=4):
    """Apply Butterworth band-pass filter to remove noise."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal).astype(np.float32)

def extract_lead_i(record_path):
    """Load PhysioNet record and return Lead I only."""
    record = wfdb.rdrecord(record_path)
    lead_names = [n.strip().upper() for n in record.sig_name]
    
    if 'I' not in lead_names:
        raise ValueError(f'Lead I not found. Available: {lead_names}')
    
    idx = lead_names.index('I')
    return record.p_signal[:, idx].astype(np.float32), record.fs

def segment_into_windows(signal, fs, window_minutes=5.0):
    """Split signal into fixed-length windows."""
    window_samples = int(window_minutes * 60 * fs)
    n_windows = len(signal) // window_samples
    if n_windows == 0:
        return signal.reshape(1, -1)  # Return as-is if too short
    trimmed = signal[:n_windows * window_samples]
    return trimmed.reshape(n_windows, window_samples)

def process_record(record_path):
    """Full pipeline: load -> filter -> window one ECG record."""
    cfg = load_config()['preprocessing']['ecg']
    raw, fs = extract_lead_i(record_path)
    filtered = bandpass_filter(raw,
        lowcut=cfg['bandpass_low_hz'],
        highcut=cfg['bandpass_high_hz'],
        fs=cfg['sampling_rate_hz'])
    windows = segment_into_windows(filtered,
        fs=cfg['sampling_rate_hz'],
        window_minutes=cfg['window_minutes'])
    return windows

if __name__ == '__main__':
    # Smoke test with synthetic signal
    dummy = np.random.randn(500 * 600).astype(np.float32)
    filtered = bandpass_filter(dummy, 0.5, 100.0, fs=500)
    windows = segment_into_windows(filtered, fs=500, window_minutes=5)
    print(f'Filter OK | Windows shape: {windows.shape}')
    print(f'Expected: (2, 150000)')