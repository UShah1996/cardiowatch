"""
train_cnn_lstm_combined.py — CNN-LSTM Training on Combined CPSC 2018 + PhysioNet 2017

Ablation study:
  train_cnn_lstm_2018only.py  → CPSC 2018 only    → AUC=0.968, AW=~50%
  train_cnn_lstm_2017only.py  → PhysioNet 2017     → TBD
  train_cnn_lstm_combined.py  → Both combined      → TBD  ← THIS FILE

Usage:
    python3 src/models/train_cnn_lstm_combined.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.models.cnn_lstm import build_model
from src.preprocessing.ecg_dataset_combined import CombinedECGDataset
from sklearn.metrics import (recall_score, precision_score, f1_score,
                             roc_auc_score, confusion_matrix)
import mlflow

CPSC_DIR = ('data/raw/classification-of-12-lead-ecgs-the-physionetcomputing'
            '-in-cardiology-challenge-2020-1.0.2/training/cpsc_2018')

P17_CANDIDATES = [
    'data/raw/challenge_2017/af-classification-from-a-short-single-lead-ecg-recording-the-physionet-computing-in-cardiology-challenge-2017-1.0.0/training2017',
    'data/raw/challenge_2017/training2017',
    'data/raw/challenge_2017',
    'data/raw/training2017',
]
P17_DIR = next((p for p in P17_CANDIDATES if os.path.exists(p)), None)

SAVE_PATH = 'data/processed/cnn_lstm_combined_best.pt'


def add_noise(signal, std=0.05):
    return signal + torch.randn_like(signal) * std


def train():
    if P17_DIR:
        print(f'Using COMBINED dataset: CPSC 2018 + PhysioNet 2017')
        print(f'CPSC 2018 path:      {CPSC_DIR}')
        print(f'PhysioNet 2017 path: {P17_DIR}')
    else:
        print('WARNING: PhysioNet 2017 not found — training on CPSC 2018 only')

    # ── KEY FIX: both cpsc_dir AND physionet_dir ──────────────────────
    dataset = CombinedECGDataset(cpsc_dir=CPSC_DIR, physionet_dir=P17_DIR)

    n_train  = int(0.8 * len(dataset))
    n_val    = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)

    n_cpsc = sum(1 for s in dataset.sources if s == 'cpsc2018')
    n_p17  = sum(1 for s in dataset.sources if s == 'physionet2017')
    print(f'\nDataset breakdown:')
    print(f'  CPSC 2018:       {n_cpsc} recordings')
    print(f'  PhysioNet 2017:  {n_p17} recordings')
    print(f'  Total:           {len(dataset)}')
    print(f'  AFib:            {sum(dataset.labels)} ({sum(dataset.labels)/len(dataset)*100:.1f}%)')
    print(f'Train: {len(train_ds)} | Val: {len(val_ds)}')

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    model    = build_model(input_length=5000).to(device)
    n_afib   = sum(dataset.labels)
    n_normal = len(dataset) - n_afib
    pw       = n_normal / n_afib
    print(f'pos_weight: {pw:.2f} ({n_normal} non-AFib / {n_afib} AFib)')

    pos_weight = torch.tensor([pw]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=0.0003)

    mlflow.start_run(run_name='cnn_lstm_combined_2017_2018')
    mlflow.log_params({
        'dataset':          'cpsc2018_plus_physionet2017',
        'cpsc_recordings':  n_cpsc,
        'p17_recordings':   n_p17,
        'total_recordings': len(dataset),
        'n_afib':           n_afib,
        'n_normal':         n_normal,
        'pos_weight':       round(pw, 3),
        'lr':               0.0003,
        'batch_size':       64,
    })

    patience = 7; no_improve = 0; best_auc = 0

    for epoch in range(40):
        model.train()
        train_loss = 0; n_batches = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            X    = add_noise(X)
            pred = model(X).squeeze()
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item(); n_batches += 1

        avg_loss = train_loss / n_batches

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y   = X.to(device), y.to(device)
                logits = model(X).squeeze()
                probs  = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().tolist())
                all_labels.extend(y.cpu().int().tolist())

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = 0.0

        for thresh in [0.3, 0.4, 0.5]:
            preds = [1 if p >= thresh else 0 for p in all_probs]
            recall    = recall_score(all_labels, preds, zero_division=0)
            precision = precision_score(all_labels, preds, zero_division=0)
            f1        = f1_score(all_labels, preds, zero_division=0)
            print(f'Epoch {epoch+1:>2} | loss={avg_loss:.4f} | '
                  f'thresh={thresh} | Recall={recall:.3f} | '
                  f'Precision={precision:.3f} | F1={f1:.3f}')

        cm = confusion_matrix(all_labels, [1 if p >= 0.4 else 0 for p in all_probs])
        print(f'  AUC-ROC={auc:.3f}')
        print(f'  Confusion matrix (thresh=0.4): TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}')

        mlflow.log_metrics({
            'train_loss': avg_loss, 'auc_roc': auc,
            'recall_t04': recall_score(all_labels, [1 if p>=0.4 else 0 for p in all_probs], zero_division=0),
            'f1_t04':     f1_score(all_labels, [1 if p>=0.4 else 0 for p in all_probs], zero_division=0),
            'epoch': epoch
        }, step=epoch)

        if auc > best_auc:
            best_auc = auc; no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f'  ✅ Saved best combined model (AUC={best_auc:.3f})')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    mlflow.end_run()
    print(f'\nTraining complete. Best AUC: {best_auc:.3f}')
    print(f'Model saved to {SAVE_PATH}')
    return best_auc


if __name__ == '__main__':
    train()