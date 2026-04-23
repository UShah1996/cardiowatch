"""
train_cnn_lstm_2018.py — CNN-LSTM Training on CPSC 2018 Only
Part of ablation study: 2018 vs 2017 vs Combined

Design rationale:
  CPSC 2018 is a hospital-grade 12-lead ECG dataset at 500 Hz.
  Training on this alone achieves high CPSC AUC (0.968) but fails
  on Apple Watch (~50%) due to domain gap between clinical and
  consumer wearable ECG signals.

Ablation study:
  train_cnn_lstm_2018.py     → CPSC 2018 only   → AUC=0.968, AW=~50%  ← THIS FILE
  train_cnn_lstm_2017.py     → PhysioNet 2017    → TBD
  train_cnn_lstm_combined.py → Both combined     → TBD

Usage:
    python3 src/models/train_cnn_lstm_2018.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from src.models.cnn_lstm import build_model
from src.preprocessing.ecg_dataset import ECGDataset
from sklearn.metrics import (recall_score, precision_score, f1_score,
                             roc_auc_score, confusion_matrix)
import mlflow
import random, numpy as np

DATA_DIR  = ('data/raw/classification-of-12-lead-ecgs-the-physionetcomputing'
             '-in-cardiology-challenge-2020-1.0.2/training/cpsc_2018')
SAVE_PATH = 'data/processed/cnn_lstm_2018_best.pt'

torch.manual_seed(42); random.seed(42); np.random.seed(42)


def add_noise(signal, std=0.05):
    return signal + torch.randn_like(signal) * std


def train():
    print('Training on CPSC 2018 ONLY (ablation)')
    print(f'Data: {DATA_DIR} | Output: {SAVE_PATH}')

    dataset  = ECGDataset(DATA_DIR)
    n_train  = int(0.8 * len(dataset))
    n_val    = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    # WeightedRandomSampler — balances AFib/non-AFib in each batch
    # Applied consistently across all three ablation models
    labels_array = [int(dataset[i][1].item()) for i in train_ds.indices]
    class_counts = [labels_array.count(0), labels_array.count(1)]
    weights      = [1.0 / class_counts[l] for l in labels_array]
    sampler      = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)
    print(f'Train: {len(train_ds)} | Val: {len(val_ds)}')

    device     = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')
    model      = build_model(input_length=5000).to(device)
    pos_weight = torch.tensor([5656 / 1221]).to(device)  # 4.63 — CPSC class ratio
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=0.0003)

    mlflow.start_run(run_name='cnn_lstm_2018_ablation')
    mlflow.log_params({'dataset': 'cpsc_2018_only', 'pos_weight': round(5656/1221, 3),
                       'lr': 0.0003, 'batch_size': 64, 'sampler': 'WeightedRandomSampler', 'ablation': 'True'})

    patience = 7; no_improve = 0; best_auc = 0

    for epoch in range(40):
        model.train()
        train_loss = 0; n_batches = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            X    = add_noise(X)
            pred = model(X).squeeze()
            loss = criterion(pred, y)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item(); n_batches += 1
        avg_loss = train_loss / n_batches

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                probs = torch.sigmoid(model(X).squeeze())
                all_probs.extend(probs.cpu().tolist())
                all_labels.extend(y.cpu().int().tolist())

        try: auc = roc_auc_score(all_labels, all_probs)
        except: auc = 0.0

        for thresh in [0.3, 0.4, 0.5]:
            preds = [1 if p >= thresh else 0 for p in all_probs]
            print(f'Epoch {epoch+1:>2} | loss={avg_loss:.4f} | thresh={thresh} | '
                  f'Recall={recall_score(all_labels,preds,zero_division=0):.3f} | '
                  f'Precision={precision_score(all_labels,preds,zero_division=0):.3f} | '
                  f'F1={f1_score(all_labels,preds,zero_division=0):.3f}')

        cm = confusion_matrix(all_labels, [1 if p>=0.4 else 0 for p in all_probs])
        print(f'  AUC-ROC={auc:.3f} | TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}')

        mlflow.log_metrics({'train_loss': avg_loss, 'auc_roc': auc,
                            'recall_t04': recall_score(all_labels,[1 if p>=0.4 else 0 for p in all_probs],zero_division=0),
                            'f1_t04': f1_score(all_labels,[1 if p>=0.4 else 0 for p in all_probs],zero_division=0),
                            'epoch': epoch}, step=epoch)

        if auc > best_auc:
            best_auc = auc; no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f'  ✅ Saved best 2018 model (AUC={best_auc:.3f})')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}'); break

    mlflow.end_run()
    print(f'\nDone. Best AUC: {best_auc:.3f} | Saved: {SAVE_PATH}')
    return best_auc

if __name__ == '__main__':
    train()