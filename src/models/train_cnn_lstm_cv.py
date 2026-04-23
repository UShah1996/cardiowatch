"""
train_cnn_lstm_cv.py — 3-Fold Stratified Cross-Validation for CNN-LSTM
=======================================================================
Addresses the "no CV on CNN-LSTM" weakness.

RF and XGBoost report 5-fold CV with mean ± std. CNN-LSTM previously
reported a single train/val split. This script runs 3-fold stratified
CV and reports mean ± std with 95% CIs, making the comparison fair.

Why 3-fold (not 5):
  - Training takes ~20 min per fold on M1 MPS → 3 folds = ~1 hour total
  - 5 folds would take ~1.7 hours
  - 3-fold CV with n≈6877 still provides ~4585 train / ~2292 val per fold
  - Standard in ECG deep learning literature for this dataset size

Design:
  - Stratified split preserves AFib/normal ratio across folds
  - Each fold uses the same hyperparameters (no tuning per fold)
  - Best model per fold saved; overall best saved as cnn_lstm_cv_best.pt
  - Bootstrap CIs computed on each fold's validation set
  - Final report: mean ± std AUC/Recall/F1 with 95% CI (t-distribution)

Usage:
    python3 src/models/train_cnn_lstm_cv.py

    # Optional: resume from a specific fold
    python3 src/models/train_cnn_lstm_cv.py --start-fold 2

    # Use combined dataset instead of CPSC-only
    python3 src/models/train_cnn_lstm_cv.py --combined
"""

import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
)
import mlflow
import json
from datetime import datetime

from src.models.cnn_lstm import build_model
from src.preprocessing.ecg_dataset import ECGDataset
from src.evaluation.confidence_intervals import bootstrap_all_metrics, cv_ci_report

# ── Config ─────────────────────────────────────────────────────────────
CPSC_DIR   = ('data/raw/classification-of-12-lead-ecgs-the-physionetcomputing'
              '-in-cardiology-challenge-2020-1.0.2/training/cpsc_2018')
SAVE_DIR   = 'data/processed'
N_FOLDS    = 3
EPOCHS     = 40
PATIENCE   = 7
BATCH_SIZE = 64
LR         = 0.0003
SEED       = 42

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ── Data augmentation ─────────────────────────────────────────────────

def add_noise(signal: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    """Gaussian noise augmentation for wearable robustness."""
    return signal + torch.randn_like(signal) * std


# ── Single fold training ──────────────────────────────────────────────

def train_one_fold(
    dataset:    ECGDataset,
    train_idx:  np.ndarray,
    val_idx:    np.ndarray,
    fold:       int,
    device:     torch.device,
    pos_weight: float,
) -> dict:
    """
    Train and evaluate one CV fold.

    Args:
        dataset    : full ECGDataset
        train_idx  : indices for training split
        val_idx    : indices for validation split
        fold       : fold number (1-indexed, for logging)
        device     : torch device
        pos_weight : BCEWithLogitsLoss positive class weight

    Returns:
        dict with fold metrics: auc, recall, f1, precision, confusion_matrix
    """
    print(f"\n{'─'*55}")
    print(f"FOLD {fold}/{N_FOLDS}  |  train={len(train_idx)}  val={len(val_idx)}")
    print(f"{'─'*55}")

    # ── Dataloaders — shuffle only, pos_weight handles class imbalance ──
    # WeightedRandomSampler + pos_weight together causes the model to
    # collapse to predicting AFib for everything. Use shuffle=True only,
    # matching train_cnn_lstm_combined.py which achieved AUC=0.974.
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model     = build_model(input_length=5000).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    fold_save_path = os.path.join(SAVE_DIR, f'cnn_lstm_cv_fold{fold}.pt')
    best_auc       = 0.0
    no_improve     = 0
    best_metrics   = {}

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
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

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y  = X.to(device), y.to(device)
                probs = torch.sigmoid(model(X).squeeze())
                all_probs.extend(probs.cpu().tolist())
                all_labels.extend(y.cpu().int().tolist())

        all_probs  = np.array(all_probs)
        all_labels = np.array(all_labels)

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = 0.0

        preds_04 = (all_probs >= 0.4).astype(int)
        recall   = recall_score(all_labels, preds_04, zero_division=0)
        f1       = f1_score(all_labels, preds_04, zero_division=0)

        print(f"  Ep {epoch+1:>2} | loss={avg_loss:.4f} | "
              f"AUC={auc:.3f} | Recall={recall:.3f} | F1={f1:.3f}")

        mlflow.log_metrics({
            f'fold{fold}_loss'  : avg_loss,
            f'fold{fold}_auc'   : auc,
            f'fold{fold}_recall': recall,
            f'fold{fold}_f1'    : f1,
        }, step=epoch)

        # ── Checkpoint ────────────────────────────────────────────────
        if auc > best_auc:
            best_auc   = auc
            no_improve = 0
            torch.save(model.state_dict(), fold_save_path)
            best_metrics = {
                'auc'    : float(auc),
                'recall' : float(recall),
                'f1'     : float(f1),
                'probs'  : all_probs.tolist(),
                'labels' : all_labels.tolist(),
            }
            print(f"  ✅ Saved fold {fold} best (AUC={best_auc:.3f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # ── Full metrics on best fold checkpoint ──────────────────────────
    model.load_state_dict(torch.load(fold_save_path, map_location=device))
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y  = X.to(device), y.to(device)
            probs = torch.sigmoid(model(X).squeeze())
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(y.cpu().int().tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc       = roc_auc_score(all_labels, all_probs)
    preds_04  = (all_probs >= 0.4).astype(int)
    recall    = recall_score(all_labels, preds_04, zero_division=0)
    precision = precision_score(all_labels, preds_04, zero_division=0)
    f1        = f1_score(all_labels, preds_04, zero_division=0)
    cm        = confusion_matrix(all_labels, preds_04)

    print(f"\n  Fold {fold} final best checkpoint:")
    print(f"    AUC-ROC   : {auc:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    Precision : {precision:.4f}")
    print(f"    F1        : {f1:.4f}")
    print(f"    CM        : TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    # Bootstrap CI on this fold's val set
    bootstrap_all_metrics(
        all_labels, all_probs,
        threshold=0.4,
        label=f'CNN-LSTM Fold {fold} (n={len(all_labels)})',
        n_boot=1000,   # 1000 per fold (faster), 2000 at final
    )

    return {
        'fold'       : fold,
        'auc'        : float(auc),
        'recall'     : float(recall),
        'precision'  : float(precision),
        'f1'         : float(f1),
        'cm'         : cm.tolist(),
        'probs'      : all_probs.tolist(),
        'labels'     : all_labels.tolist(),
        'best_path'  : fold_save_path,
    }


# ── Full CV loop ──────────────────────────────────────────────────────

def train_cv(
    data_dir:    str  = CPSC_DIR,
    start_fold:  int  = 1,
    use_combined: bool = False,
) -> dict:
    """
    Run N_FOLDS-fold stratified cross-validation on CNN-LSTM.

    Args:
        data_dir    : path to ECG dataset
        start_fold  : resume from this fold (1-indexed)
        use_combined: if True, use CombinedECGDataset instead of ECGDataset

    Returns:
        dict with per-fold results and aggregate statistics
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("="*60)
    print(f"CNN-LSTM {N_FOLDS}-Fold Stratified Cross-Validation")
    print(f"  Dataset   : {'Combined (CPSC+P17)' if use_combined else 'CPSC 2018 only'}")
    print(f"  Epochs    : {EPOCHS}  Patience: {PATIENCE}")
    print(f"  Batch     : {BATCH_SIZE}  LR: {LR}")
    print(f"  Start fold: {start_fold}")
    print("="*60)

    # ── Load dataset ──────────────────────────────────────────────────
    if use_combined:
        from src.preprocessing.ecg_dataset_combined import CombinedECGDataset
        P17_CANDIDATES = [
            'data/raw/challenge_2017/training2017',
            'data/raw/challenge_2017',
            'data/raw/training2017',
        ]
        p17_dir = next((p for p in P17_CANDIDATES if os.path.exists(p)), None)
        dataset = CombinedECGDataset(data_dir, p17_dir)
    else:
        dataset = ECGDataset(data_dir)

    all_labels = np.array([int(dataset[i][1].item()) for i in range(len(dataset))])
    n_afib     = int(all_labels.sum())
    n_normal   = len(all_labels) - n_afib
    pos_weight = n_normal / n_afib

    print(f"Dataset: {len(dataset)} recordings | "
          f"AFib: {n_afib} ({100*n_afib/len(dataset):.1f}%) | "
          f"pos_weight: {pos_weight:.2f}")

    # ── Device ────────────────────────────────────────────────────────
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ── Stratified K-Fold ─────────────────────────────────────────────
    skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_splits = list(skf.split(np.arange(len(dataset)), all_labels))

    # ── MLflow run ────────────────────────────────────────────────────
    run_name = f'cnn_lstm_{N_FOLDS}fold_cv_{datetime.now().strftime("%m%d_%H%M")}'
    mlflow.start_run(run_name=run_name)
    mlflow.log_params({
        'n_folds'   : N_FOLDS,
        'n_total'   : len(dataset),
        'n_afib'    : n_afib,
        'n_normal'  : n_normal,
        'pos_weight': round(pos_weight, 3),
        'epochs'    : EPOCHS,
        'patience'  : PATIENCE,
        'lr'        : LR,
        'batch_size': BATCH_SIZE,
        'combined'  : use_combined,
    })

    # ── Run each fold ─────────────────────────────────────────────────
    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        fold = fold_idx + 1
        if fold < start_fold:
            print(f"Skipping fold {fold} (--start-fold={start_fold})")
            continue
        result = train_one_fold(
            dataset, train_idx, val_idx,
            fold=fold, device=device, pos_weight=pos_weight,
        )
        fold_results.append(result)

    mlflow.end_run()

    if not fold_results:
        print("No folds were run.")
        return {}

    # ── Aggregate results ─────────────────────────────────────────────
    auc_scores    = np.array([r['auc']    for r in fold_results])
    recall_scores = np.array([r['recall'] for r in fold_results])
    f1_scores     = np.array([r['f1']     for r in fold_results])

    print("\n" + "="*60)
    print(f"CNN-LSTM {N_FOLDS}-FOLD CV — AGGREGATE RESULTS")
    print("="*60)
    print(f"  AUC-ROC  : {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
    print(f"  Recall   : {recall_scores.mean():.4f} ± {recall_scores.std():.4f}")
    print(f"  F1       : {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

    # ── t-distribution 95% CI across folds ───────────────────────────
    cv_results_fmt = {
        'test_roc_auc': auc_scores,
        'test_recall' : recall_scores,
        'test_f1'     : f1_scores,
    }
    cv_ci_report(cv_results_fmt, model_name=f'CNN-LSTM ({N_FOLDS}-fold CV)')

    # ── Identify and copy best fold model ─────────────────────────────
    best_fold   = fold_results[int(np.argmax(auc_scores))]
    best_path   = best_fold['best_path']
    final_path  = os.path.join(SAVE_DIR, 'cnn_lstm_cv_best.pt')
    import shutil
    shutil.copy2(best_path, final_path)
    print(f"\nBest fold: {best_fold['fold']} (AUC={best_fold['auc']:.4f})")
    print(f"Best model saved → {final_path}")

    # ── Bootstrap CI on pooled predictions ───────────────────────────
    all_probs  = np.concatenate([r['probs']  for r in fold_results])
    all_labels_pooled = np.concatenate([r['labels'] for r in fold_results])
    bootstrap_all_metrics(
        all_labels_pooled, all_probs,
        threshold=0.4,
        label=f'CNN-LSTM Pooled CV (n={len(all_labels_pooled)})',
        n_boot=2000,
    )

    # ── Save results JSON ─────────────────────────────────────────────
    summary = {
        'n_folds'       : N_FOLDS,
        'auc_mean'      : float(auc_scores.mean()),
        'auc_std'       : float(auc_scores.std()),
        'recall_mean'   : float(recall_scores.mean()),
        'recall_std'    : float(recall_scores.std()),
        'f1_mean'       : float(f1_scores.mean()),
        'f1_std'        : float(f1_scores.std()),
        'fold_results'  : [
            {k: v for k, v in r.items() if k not in ('probs', 'labels')}
            for r in fold_results
        ],
    }
    json_path = os.path.join(SAVE_DIR, 'cnn_lstm_cv_results.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"CV results saved → {json_path}")

    print("\n" + "="*60)
    print("WHAT TO SAY IN YOUR PRESENTATION:")
    print(f"  'CNN-LSTM: AUC {auc_scores.mean():.3f} ± {auc_scores.std():.3f} "
          f"({N_FOLDS}-fold CV)'")
    print(f"  'Recall {recall_scores.mean():.3f} ± {recall_scores.std():.3f}'")
    print("  This is directly comparable to RF and XGBoost CV results.")
    print("="*60 + "\n")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='3-fold CV for CNN-LSTM ECG model'
    )
    parser.add_argument(
        '--start-fold', type=int, default=1,
        help='Resume training from this fold number (default: 1)'
    )
    parser.add_argument(
        '--combined', action='store_true',
        help='Use combined CPSC+PhysioNet dataset instead of CPSC only'
    )
    parser.add_argument(
        '--data-dir', type=str, default=CPSC_DIR,
        help='Path to ECG dataset directory'
    )
    args = parser.parse_args()

    train_cv(
        data_dir     = args.data_dir,
        start_fold   = args.start_fold,
        use_combined = args.combined,
    )
