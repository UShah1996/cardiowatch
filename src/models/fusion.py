"""
fusion.py — Late Fusion Layer for CardioWatch
Combines RF clinical risk score + CNN-LSTM ECG risk score.
Weighted average fusion — RF gets higher weight (0.6) since it has
stronger validated performance on clinical data.
"""

import numpy as np
from sklearn.metrics import recall_score, roc_auc_score, f1_score


def fuse_predictions(rf_probs, cnn_probs, weight_rf=0.6, weight_cnn=0.4):
    """
    Weighted average of RF (clinical) and CNN-LSTM (ECG) probabilities.

    Args:
        rf_probs  : array of RF probability scores [0, 1]
        cnn_probs : array of CNN-LSTM probability scores [0, 1]
        weight_rf : weight for clinical model (default 0.6)
        weight_cnn: weight for ECG model (default 0.4)

    Returns:
        fused probabilities as numpy array
    """
    rf_probs  = np.array(rf_probs)
    cnn_probs = np.array(cnn_probs)
    return weight_rf * rf_probs + weight_cnn * cnn_probs


def evaluate_fusion(fused_probs, y_true, threshold=0.5):
    """
    Evaluate fused predictions against ground truth labels.

    Args:
        fused_probs : array of fused probability scores
        y_true      : ground truth binary labels
        threshold   : classification threshold (default 0.5)

    Returns:
        recall, auc_roc
    """
    fused_probs = np.array(fused_probs)
    preds       = (fused_probs >= threshold).astype(int)
    recall      = recall_score(y_true, preds, zero_division=0)
    f1          = f1_score(y_true, preds, zero_division=0)
    auc_roc     = roc_auc_score(y_true, fused_probs)
    print(f'Fused Recall: {recall:.3f} | F1: {f1:.3f} | AUC-ROC: {auc_roc:.3f}')
    return recall, auc_roc


def find_best_weights(rf_probs, cnn_probs, y_true):
    """
    Grid search over fusion weights to maximize recall.
    Useful for tuning after both models are fully trained.

    Returns best (weight_rf, weight_cnn, recall, threshold)
    """
    best = {'recall': 0, 'weight_rf': 0.6, 'weight_cnn': 0.4, 'threshold': 0.5}

    for w_rf in np.arange(0.3, 0.9, 0.1):
        w_cnn  = round(1.0 - w_rf, 1)
        fused  = fuse_predictions(rf_probs, cnn_probs, w_rf, w_cnn)
        for t in np.arange(0.3, 0.7, 0.05):
            preds  = (fused >= t).astype(int)
            recall = recall_score(y_true, preds, zero_division=0)
            if recall > best['recall']:
                best = {'recall': recall, 'weight_rf': w_rf,
                        'weight_cnn': w_cnn, 'threshold': round(t, 2)}

    print(f"Best weights — RF: {best['weight_rf']:.1f} | "
          f"ECG: {best['weight_cnn']:.1f} | "
          f"Threshold: {best['threshold']} | "
          f"Recall: {best['recall']:.3f}")
    return best


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    # Demonstrate fusion on synthetic scores
    np.random.seed(42)
    n = 200
    y_true    = np.array([1]*40 + [0]*160)          # 20% AFib
    rf_probs  = np.where(y_true, np.random.beta(5,2,n), np.random.beta(2,5,n))
    cnn_probs = np.where(y_true, np.random.beta(6,2,n), np.random.beta(2,6,n))

    print('=== Individual model performance ===')
    print(f'RF alone  — AUC: {roc_auc_score(y_true, rf_probs):.3f}')
    print(f'CNN alone — AUC: {roc_auc_score(y_true, cnn_probs):.3f}')

    print('\n=== Default fusion (RF=0.6, ECG=0.4) ===')
    fused = fuse_predictions(rf_probs, cnn_probs)
    evaluate_fusion(fused, y_true)

    print('\n=== Weight grid search ===')
    find_best_weights(rf_probs, cnn_probs, y_true)