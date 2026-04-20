"""
Fusion layer: combines RF clinical risk score + CNN-LSTM ECG risk score.
Simple late fusion — average weighted by validation performance.
"""
import numpy as np
from sklearn.metrics import recall_score, roc_auc_score

def fuse_predictions(rf_probs, cnn_probs, weight_rf=0.6, weight_cnn=0.4):
    """
    Weighted average of RF (clinical) and CNN-LSTM (ECG) probabilities.
    RF gets higher weight initially since it has real validation results.
    Adjust weights after CNN-LSTM training based on val recall.
    """
    return weight_rf * rf_probs + weight_cnn * cnn_probs

def evaluate_fusion(fused_probs, y_true, threshold=0.5):
    preds = (fused_probs >= threshold).astype(int)
    recall  = recall_score(y_true, preds)
    auc_roc = roc_auc_score(y_true, fused_probs)
    print(f'Fused Recall: {recall:.3f} | AUC-ROC: {auc_roc:.3f}')
    return recall, auc_roc