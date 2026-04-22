"""
ecg_dataset_combined.py — Combined ECG Dataset for CardioWatch
Loads both:
  - CPSC 2018 (hospital 12-lead, 500 Hz, SNOMED labels in .hea files)
  - PhysioNet 2017 Challenge (AliveCor wearable, 300 Hz, labels in REFERENCE.csv)

Design rationale:
  Training on both datasets gives the CNN-LSTM exposure to both clinical-grade
  (CPSC 2018) and wearable-grade (PhysioNet 2017) AFib signals, improving
  generalization to Apple Watch recordings without losing CPSC performance.
  Both datasets use Lead I. 2017 signals are resampled from 300 Hz to 500 Hz.

Usage:
    from src.preprocessing.ecg_dataset_combined import CombinedECGDataset
    dataset = CombinedECGDataset(cpsc_dir, physionet_dir)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb
import scipy.io
from scipy.signal import resample as scipy_resample

# SNOMED code for AFib in CPSC 2018
AFIB_CODE_CPSC = '164889003'

# PhysioNet 2017 label mapping
# N = Normal, A = AFib, O = Other, ~ = Noisy
PHYSIONET_2017_LABELS = {
    'N': 0,  # Normal → non-AFib
    'A': 1,  # AFib   → AFib
    'O': 0,  # Other  → non-AFib
    '~': 0,  # Noisy  → non-AFib (excluded from training ideally)
}


class CombinedECGDataset(Dataset):
    """
    Combined dataset loading CPSC 2018 + PhysioNet 2017 ECG recordings.

    Both are treated as binary AFib classification:
      label = 1 → AFib
      label = 0 → non-AFib

    Preprocessing (same for both sources):
      1. Extract Lead I
      2. Resample to 500 Hz (2017 is 300 Hz → upsampled)
      3. NaN removal
      4. Clip to ±2.0 mV
      5. Z-score normalize per recording
      6. Clip to ±5.0 after normalize
      7. Truncate or pad to target_len samples
    """

    def __init__(
        self,
        cpsc_dir,
        physionet_dir=None,
        target_len=5000,
        exclude_noisy_2017=True,
        verbose=True
    ):
        """
        Args:
            cpsc_dir          : path to CPSC 2018 training directory
            physionet_dir     : path to PhysioNet 2017 training2017/ directory
                                (None to use CPSC only)
            target_len        : output signal length in samples (default 5000 = 10s at 500Hz)
            exclude_noisy_2017: skip 2017 recordings labeled '~' (noisy)
            verbose           : print loading stats
        """
        self.records    = []
        self.labels     = []
        self.sources    = []   # track which dataset each recording came from
        self.target_len = target_len

        # ── Load CPSC 2018 ────────────────────────────────────────────
        cpsc_loaded, cpsc_skipped = self._load_cpsc(cpsc_dir)

        # ── Load PhysioNet 2017 ───────────────────────────────────────
        p17_loaded = 0
        p17_skipped = 0
        if physionet_dir is not None:
            p17_loaded, p17_skipped = self._load_physionet_2017(
                physionet_dir, exclude_noisy_2017)

        if verbose:
            total    = len(self.records)
            n_afib   = sum(self.labels)
            n_normal = total - n_afib
            n_cpsc   = sum(1 for s in self.sources if s == 'cpsc2018')
            n_p17    = sum(1 for s in self.sources if s == 'physionet2017')

            print(f"{'='*55}")
            print(f"Combined ECG Dataset loaded:")
            print(f"  CPSC 2018:       {n_cpsc:>5} recordings "
                  f"(skipped {cpsc_skipped})")
            print(f"  PhysioNet 2017:  {n_p17:>5} recordings "
                  f"(skipped {p17_skipped})")
            print(f"  Total:           {total:>5}")
            print(f"  AFib:            {n_afib:>5} ({n_afib/total*100:.1f}%)")
            print(f"  Non-AFib:        {n_normal:>5} ({n_normal/total*100:.1f}%)")
            print(f"{'='*55}")

    # ── CPSC 2018 loader ──────────────────────────────────────────────
    def _load_cpsc(self, data_dir):
        loaded  = 0
        skipped = 0

        for root, dirs, files in os.walk(data_dir):
            for fname in files:
                if not fname.endswith('.hea'):
                    continue
                path = os.path.join(root, fname.replace('.hea', ''))
                try:
                    record = wfdb.rdrecord(path)
                    header = wfdb.rdheader(path)
                    leads  = [n.strip().upper() for n in record.sig_name]

                    if 'I' not in leads:
                        skipped += 1
                        continue

                    sig = record.p_signal[
                        :, leads.index('I')].astype(np.float32)
                    sig = self._preprocess(sig, src_fs=500)

                    # Parse AFib label from SNOMED Dx code
                    dx_codes = []
                    for c in header.comments:
                        if c.startswith('Dx:'):
                            dx_codes = [
                                x.strip()
                                for x in c.replace('Dx:', '').split(',')
                            ]

                    label = 1 if AFIB_CODE_CPSC in dx_codes else 0
                    self.records.append(sig)
                    self.labels.append(label)
                    self.sources.append('cpsc2018')
                    loaded += 1

                except Exception:
                    skipped += 1
                    continue

        return loaded, skipped

    # ── PhysioNet 2017 loader ─────────────────────────────────────────
    def _load_physionet_2017(self, data_dir, exclude_noisy):
        """
        PhysioNet 2017 uses REFERENCE.csv for labels and .mat for signals.

        REFERENCE.csv format:
            A0001,N
            A0002,A
            A0003,O
            A0004,~
        """
        loaded  = 0
        skipped = 0

        # Find REFERENCE.csv — may be in subdir
        ref_path = None
        for root, dirs, files in os.walk(data_dir):
            for fname in files:
                if fname == 'REFERENCE.csv':
                    ref_path = os.path.join(root, fname)
                    break
            if ref_path:
                break

        if ref_path is None:
            print(f"WARNING: REFERENCE.csv not found in {data_dir}")
            print("  Expected structure: training2017/REFERENCE.csv")
            return 0, 0

        # Load labels
        import csv
        label_map = {}
        with open(ref_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    record_name = row[0].strip()
                    label_str   = row[1].strip()
                    label_map[record_name] = label_str

        ref_dir = os.path.dirname(ref_path)

        for record_name, label_str in label_map.items():
            # Skip noisy recordings if requested
            if exclude_noisy and label_str == '~':
                skipped += 1
                continue

            # Try loading .mat file
            mat_path = os.path.join(ref_dir, f'{record_name}.mat')
            hea_path = os.path.join(ref_dir, f'{record_name}.hea')

            if not os.path.exists(mat_path):
                skipped += 1
                continue

            try:
                # Load signal from .mat file
                mat  = scipy.io.loadmat(mat_path)

                # 2017 .mat files store signal in 'val' key
                if 'val' in mat:
                    sig = mat['val'].flatten().astype(np.float32)
                else:
                    # Try first numeric array
                    for key, val in mat.items():
                        if not key.startswith('_') and isinstance(val, np.ndarray):
                            sig = val.flatten().astype(np.float32)
                            break
                    else:
                        skipped += 1
                        continue

                # Get sampling rate from header if available
                src_fs = 300  # 2017 default
                if os.path.exists(hea_path):
                    try:
                        with open(hea_path, 'r') as f:
                            first_line = f.readline().split()
                            if len(first_line) >= 3:
                                src_fs = int(first_line[2])
                    except Exception:
                        src_fs = 300

                # Convert ADC counts to mV if needed
                # 2017 signals are in ADC units, gain is typically 1000
                if np.abs(sig).max() > 10:
                    sig = sig / 1000.0

                sig   = self._preprocess(sig, src_fs=src_fs)
                label = PHYSIONET_2017_LABELS.get(label_str, 0)

                self.records.append(sig)
                self.labels.append(label)
                self.sources.append('physionet2017')
                loaded += 1

            except Exception:
                skipped += 1
                continue

        return loaded, skipped

    # ── Signal preprocessing ──────────────────────────────────────────
    def _preprocess(self, sig, src_fs=500, target_fs=500):
        """
        Preprocess a raw ECG signal:
        1. Resample to target_fs if needed
        2. Remove NaNs
        3. Clip, z-score normalize, clip again
        """
        sig = np.nan_to_num(sig)

        # Resample if source fs differs from target
        if src_fs != target_fs:
            n_target = int(len(sig) * target_fs / src_fs)
            sig      = scipy_resample(sig, n_target).astype(np.float32)

        sig = np.clip(sig, -2.0, 2.0)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        sig = np.clip(sig, -5.0, 5.0)

        return sig.astype(np.float32)

    # ── Dataset interface ─────────────────────────────────────────────
    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        sig = self.records[idx]

        # Truncate or pad to target_len
        if len(sig) >= self.target_len:
            sig = sig[:self.target_len]
        else:
            sig = np.pad(sig, (0, self.target_len - len(sig)))

        sig = torch.tensor(sig).unsqueeze(0)  # (1, target_len)
        return sig, torch.tensor(self.labels[idx], dtype=torch.float32)

    def get_source_stats(self):
        """Return breakdown of AFib by source dataset."""
        stats = {}
        for source in ['cpsc2018', 'physionet2017']:
            indices = [i for i, s in enumerate(self.sources) if s == source]
            n_total = len(indices)
            n_afib  = sum(self.labels[i] for i in indices)
            stats[source] = {
                'total':    n_total,
                'afib':     n_afib,
                'non_afib': n_total - n_afib,
                'afib_pct': n_afib / n_total * 100 if n_total > 0 else 0
            }
        return stats


# ── Quick test ────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    CPSC_DIR = ('data/raw/classification-of-12-lead-ecgs-the-physionetcomputing'
                '-in-cardiology-challenge-2020-1.0.2/training/cpsc_2018')

    # Find PhysioNet 2017 directory
    P17_DIR = None
    for candidate in [
        'data/raw/challenge_2017/training2017',
        'data/raw/challenge_2017',
        'data/raw/training2017',
        'data/raw/af-classification-from-a-short-single-lead-ecg-recording'
        '-the-physionetcomputing-in-cardiology-challenge-2017-1.0.0/training2017',
    ]:
        if os.path.exists(candidate):
            P17_DIR = candidate
            print(f'Found PhysioNet 2017 at: {candidate}')
            break

    if P17_DIR is None:
        print('PhysioNet 2017 not found — loading CPSC 2018 only')
        print('Run download command first:')
        print('  curl -L "https://physionet.org/files/challenge-2017/1.0.0/'
              'training2017.zip" -o data/raw/training2017.zip')
        print('  unzip data/raw/training2017.zip -d data/raw/challenge_2017/')

    dataset = CombinedECGDataset(CPSC_DIR, P17_DIR)

    # Show source breakdown
    stats = dataset.get_source_stats()
    print('\nSource breakdown:')
    for source, s in stats.items():
        print(f'  {source}: {s["total"]} total, '
              f'{s["afib"]} AFib ({s["afib_pct"]:.1f}%), '
              f'{s["non_afib"]} non-AFib')

    # Test one item
    X, y = dataset[0]
    print(f'\nSample shape: {X.shape}')
    print(f'Sample label: {y.item()}')
    print(f'Sample min/max: {X.min():.3f} / {X.max():.3f}')

    # Compute pos_weight for training
    n_afib    = sum(dataset.labels)
    n_normal  = len(dataset) - n_afib
    pos_weight = n_normal / n_afib
    print(f'\nRecommended pos_weight for BCEWithLogitsLoss: {pos_weight:.2f}')
    print(f'  ({n_normal} non-AFib / {n_afib} AFib)')