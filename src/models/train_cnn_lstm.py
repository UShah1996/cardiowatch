import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.models.cnn_lstm import build_model
from src.preprocessing.ecg_dataset import ECGDataset
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import mlflow

DATA_DIR = 'data/raw/classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2/training/cpsc_2018'
# Reproducible training
torch.manual_seed(42)
import random, numpy as np
random.seed(42)
np.random.seed(42)


def add_noise(signal, std=0.05):
    """Gaussian noise augmentation for wearable robustness."""
    return signal + torch.randn_like(signal) * std

def train():
    dataset = ECGDataset(DATA_DIR)
    n_train = int(0.8 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)  # increased
    val_loader   = DataLoader(val_ds,   batch_size=64)

    # M1 GPU acceleration
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = build_model(input_length=5000).to(device)  # move model to M1 GPU

    optimizer  = torch.optim.Adam(model.parameters(), lr=0.0003)
    pos_weight = torch.tensor([5656 / 1221]).to(device)  # must be on same device
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    mlflow.start_run()
    patience   = 5
    no_improve = 0
    best_auc   = 0

    for epoch in range(30):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)  # move batch to M1 GPU
            X    = add_noise(X)
            pred = model(X).squeeze()
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y  = X.to(device), y.to(device)  # move batch to M1 GPU
                logits = model(X).squeeze()
                probs  = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().tolist())   # back to CPU for sklearn
                all_labels.extend(y.cpu().int().tolist())

        for thresh in [0.3, 0.4, 0.5]:
            preds     = [1 if p >= thresh else 0 for p in all_probs]
            recall    = recall_score(all_labels, preds, zero_division=0)
            precision = precision_score(all_labels, preds, zero_division=0)
            f1        = f1_score(all_labels, preds, zero_division=0)
            print(f'Epoch {epoch+1} | thresh={thresh} | Recall={recall:.3f} | Precision={precision:.3f} | F1={f1:.3f}')

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0

        cm = confusion_matrix(all_labels, [1 if p >= 0.4 else 0 for p in all_probs])
        print(f'  AUC-ROC={auc:.3f}')
        print(f'  Confusion matrix (thresh=0.4):')
        print(f'    TN={cm[0][0]}  FP={cm[0][1]}')
        print(f'    FN={cm[1][0]}  TP={cm[1][1]}')

        mlflow.log_metrics({
            'recall_t04': recall_score(all_labels, [1 if p >= 0.4 else 0 for p in all_probs], zero_division=0),
            'f1_t04':     f1_score(all_labels, [1 if p >= 0.4 else 0 for p in all_probs], zero_division=0),
            'auc_roc':    auc,
            'epoch':      epoch
        }, step=epoch)

        if auc > best_auc:
            best_auc   = auc
            no_improve = 0
            torch.save(model.state_dict(), 'data/processed/cnn_lstm_best.pt')
            print(f'  Saved best model (AUC={best_auc:.3f})')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1} — no improvement for {patience} epochs')
                break

if __name__ == '__main__':
    train()