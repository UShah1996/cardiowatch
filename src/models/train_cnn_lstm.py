import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.models.cnn_lstm import build_model
from src.preprocessing.ecg_dataset import ECGDataset
import mlflow

DATA_DIR = 'data/raw/classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2/training/cpsc_2018'

def add_noise(signal, std=0.05):
    """Gaussian noise augmentation for wearable robustness."""
    return signal + torch.randn_like(signal) * std

def train():
    dataset = ECGDataset(DATA_DIR)
    n_train = int(0.8 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32)

    # Use 5000-sample input (10 seconds, not 5 minutes — matches CPSC duration)
    model     = build_model(input_length=5000)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    pos_weight = torch.tensor([5959 / 918])  # abnormal/normal ratio
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    mlflow.start_run()
    best_recall = 0

    for epoch in range(30):
        model.train()
        for X, y in train_loader:
            X = add_noise(X)          # augment
            pred = model(X).squeeze()
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                pred = model(X).squeeze()
                all_preds.extend((torch.sigmoid(pred) >= 0.4).int().tolist())
                all_labels.extend(y.int().tolist())

        from sklearn.metrics import recall_score, roc_auc_score
        recall = recall_score(all_labels, all_preds, zero_division=0)
        mlflow.log_metrics({'recall': recall, 'epoch': epoch})
        print(f'Epoch {epoch+1} | Recall: {recall:.3f}')

        if recall > best_recall:
            best_recall = recall
            torch.save(model.state_dict(), 'data/processed/cnn_lstm_best.pt')

    mlflow.end_run()
    print(f'Best ECG recall: {best_recall:.3f}')

if __name__ == '__main__':
    train()