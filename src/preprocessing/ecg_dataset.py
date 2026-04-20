import torch
from torch.utils.data import Dataset
import numpy as np
import wfdb, os

class ECGDataset(Dataset):
    NORMAL_LABEL = 'Normal'

    def __init__(self, data_dir, target_len=5000):
        self.records = []
        self.labels  = []
        self.target_len = target_len

        # CPSC 2018 SNOMED code for Normal Sinus Rhythm
        NORMAL_CODE = '426783006'

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
                        continue
                    sig = record.p_signal[:, leads.index('I')].astype(np.float32)
                    sig = np.nan_to_num(sig)

                    # Parse Dx field — e.g. 'Dx: 426783006' or 'Dx: 270492004,164931005'
                    dx_codes = []
                    for c in header.comments:
                        if c.startswith('Dx:'):
                            dx_codes = [x.strip() for x in c.replace('Dx:', '').split(',')]

                    label = 0 if NORMAL_CODE in dx_codes else 1
                    self.records.append(sig)
                    self.labels.append(label)
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