"""
train_cnn_lstm_combined.py — CNN-LSTM Training on Combined CPSC 2018 + PhysioNet 2017
Trains on both datasets to improve generalization to Apple Watch ECGs.

Design rationale:
  - CPSC 2018: hospital 12-lead (500 Hz) — strong AFib signal quality
  - PhysioNet 2017: AliveCor wearable (300 Hz → 500 Hz) — similar to Apple Watch
  - Combined training exposes model to both clinical and wearable signal characteristics
  - Consistent with domain randomization in transfer learning literature

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

# ── Paths ─────────────────────────────────────────────────────────────
CPSC_DIR = ('data/raw/classification-of-12-lead-ecgs-the-physionetcomputing'
            '-in-cardiology-challenge-2020-1.0.2/training/cpsc_2018')

# Auto-detect PhysioNet 2017 directory
P17_CANDIDATES = [
    'data/raw/challenge_2017/training2017',
    'data/raw/challenge_2017',
    'data/raw/training2017',
    ('data/raw/af-classification-from-a-short-single-lead-ecg-recording'
     '-the-physionetcomputing-in-cardiology-challenge-2017-1.0.0/training2017'),
]
P17_DIR = next((p for p in P17_CANDIDATES if os.path.exists(p)), None)


def add_noise(signal, std=0.05):
    """Gaussian noise augmentation for wearable robustness."""
    return signal + torch.randn_like(signal) * std


def train():
    # ── Load combined dataset ─────────────────────────────────────────
    if P17_DIR:
        print(f'Using combined dataset: CPSC 2018 + PhysioNet 2017')
        print(f'PhysioNet 2017 path: {P17_DIR}')
    else:
        print('WARNING: PhysioNet 2017 not found — training on CPSC 2018 only')
        print('Expected at one of:')
        for p in P17_CANDIDATES:
            print(f'  {p}')

    dataset = CombinedECGDataset(CPSC_DIR, P17_DIR)

    # ── Train/val split ───────────────────────────────────────────────
    n_train  = int(0.8 * len(dataset))
    n_val    = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)

    print(f'\nTrain: {len(train_ds)} | Val: {len(val_ds)}')

    # ── Device ────────────────────────────────────────────────────────
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(input_length=5000).to(device)

    # ── Loss — pos_weight from actual dataset distribution ────────────
    n_afib   = sum(dataset.labels)
    n_normal = len(dataset) - n_afib
    pw       = n_normal / n_afib
    print(f'pos_weight: {pw:.2f} ({n_normal} non-AFib / {n_afib} AFib)')

    pos_weight = torch.tensor([pw]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=0.0003)

    # ── Training ──────────────────────────────────────────────────────
    mlflow.start_run(run_name='cnn_lstm_combined_2017_2018')
    mlflow.log_params({
        'cpsc_recordings':    sum(1 for s in dataset.sources if s == 'cpsc2018'),
        'p17_recordings':     sum(1 for s in dataset.sources if s == 'physionet2017'),
        'total_recordings':   len(dataset),
        'n_afib':             n_afib,
        'n_normal':           n_normal,
        'pos_weight':         round(pw, 3),
        'lr':                 0.0003,
        'batch_size':         64,
    })

    patience   = 7      # slightly more patience for larger dataset
    no_improve = 0
    best_auc   = 0

    for epoch in range(40):  # more epochs for combined dataset
        # ── Train ─────────────────────────────────────────────────────
        model.train()
        train_loss = 0
        n_batches  = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            X    = add_noise(X)
            pred = model(X).squeeze()
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches  += 1

        avg_loss = train_loss / n_batches

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y   = X.to(device), y.to(device)
                logits = model(X).squeeze()
                probs  = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().tolist())
                all_labels.extend(y.cpu().int().tolist())

        # ── Metrics ───────────────────────────────────────────────────
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = 0.0

        for thresh in [0.3, 0.4, 0.5]:
            preds     = [1 if p >= thresh else 0 for p in all_probs]
            recall    = recall_score(all_labels, preds, zero_division=0)
            precision = precision_score(all_labels, preds, zero_division=0)
            f1        = f1_score(all_labels, preds, zero_division=0)
            print(f'Epoch {epoch+1:>2} | loss={avg_loss:.4f} | '
                  f'thresh={thresh} | Recall={recall:.3f} | '
                  f'Precision={precision:.3f} | F1={f1:.3f}')

        cm = confusion_matrix(
            all_labels,
            [1 if p >= 0.4 else 0 for p in all_probs]
        )
        print(f'  AUC-ROC={auc:.3f}')
        print(f'  Confusion matrix (thresh=0.4):')
        print(f'    TN={cm[0][0]}  FP={cm[0][1]}')
        print(f'    FN={cm[1][0]}  TP={cm[1][1]}')

        mlflow.log_metrics({
            'train_loss':   avg_loss,
            'auc_roc':      auc,
            'recall_t04':   recall_score(
                all_labels,
                [1 if p >= 0.4 else 0 for p in all_probs],
                zero_division=0
            ),
            'f1_t04':       f1_score(
                all_labels,
                [1 if p >= 0.4 else 0 for p in all_probs],
                zero_division=0
            ),
            'epoch':        epoch
        }, step=epoch)

        # ── Checkpoint ────────────────────────────────────────────────
        if auc > best_auc:
            best_auc   = auc
            no_improve = 0
            torch.save(
                model.state_dict(),
                'data/processed/cnn_lstm_combined_best.pt'
            )
            print(f'  ✅ Saved best combined model (AUC={best_auc:.3f})')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1} — '
                      f'no improvement for {patience} epochs')
                break

    mlflow.end_run()
    print(f'\nTraining complete. Best AUC: {best_auc:.3f}')
    print(f'Model saved to data/processed/cnn_lstm_combined_best.pt')

    return best_auc


if __name__ == '__main__':
    train()