import torch
from torch.utils.data import Dataset
import numpy as np
import wfdb, os

class ECGDataset(Dataset):
    NORMAL_LABEL = 'Normal'

    def __init__(self, data_dir, target_len=5000):
        self.records = []
        self.labels  = []
        self.paths   = []   # kept record paths, in dataset order
        self.target_len = target_len

        AFIB_CODE = '164889003'

        # Collect and sort all .hea paths before loading. os.walk order is
        # filesystem-dependent, so without this the dataset order — and
        # therefore the seeded train/val random_split — differs across
        # machines, making results non-reproducible. Sorting the full paths
        # gives a stable, machine-independent order.
        hea_paths = []
        for root, dirs, files in os.walk(data_dir):
            for fname in files:
                if fname.endswith('.hea'):
                    hea_paths.append(os.path.join(root, fname.replace('.hea', '')))
        hea_paths.sort()

        for path in hea_paths:
            try:
                record = wfdb.rdrecord(path)
                header = wfdb.rdheader(path)
                leads  = [n.strip().upper() for n in record.sig_name]
                if 'I' not in leads:
                    continue

                sig = record.p_signal[:, leads.index('I')].astype(np.float32)
                sig = np.nan_to_num(sig)
                sig = np.clip(sig, -2.0, 2.0)
                sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
                sig = np.clip(sig, -5.0, 5.0)

                dx_codes = []
                for c in header.comments:
                    if c.startswith('Dx:'):
                        dx_codes = [x.strip() for x in c.replace('Dx:', '').split(',')]

                label = 1 if AFIB_CODE in dx_codes else 0
                self.records.append(sig)
                self.labels.append(label)
                self.paths.append(path)
            except Exception:
                continue

        print(f"Loaded {len(self.records)} recordings")
        print(f"Normal: {self.labels.count(0)} | Abnormal: {self.labels.count(1)}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        sig = self.records[idx]
        if len(sig) >= self.target_len:
            sig = sig[:self.target_len]
        else:
            sig = np.pad(sig, (0, self.target_len - len(sig)))
        sig = torch.tensor(sig).unsqueeze(0)  # (1, 5000)
        return sig, torch.tensor(self.labels[idx], dtype=torch.float32)