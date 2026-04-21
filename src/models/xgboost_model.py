"""
xgboost_model.py — XGBoost baseline for clinical risk prediction.
Saves trained model to data/processed/xgb_model.pkl for dashboard use.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import joblib
import numpy as np
import yaml
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import recall_score, f1_score, roc_auc_score

from src.preprocessing.clinical import full_pipeline
from src.preprocessing.smote_balance import apply_smote

PROCESSED_DIR = 'data/processed'


def build_xgb(config_path='configs/config.yaml'):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)['models']['xgboost']
    return XGBClassifier(
        n_estimators     = cfg['n_estimators'],
        max_depth        = cfg['max_depth'],
        learning_rate    = cfg['learning_rate'],
        scale_pos_weight = cfg['scale_pos_weight'],
        eval_metric      = 'aucpr',
        random_state     = cfg['random_state'],
        n_jobs           = -1
    )


def tune_threshold(model, X_val, y_val):
    """Find threshold that maximizes recall on validation set."""
    probs = model.predict_proba(X_val)[:, 1]
    best_t, best_recall = 0.5, 0
    for t in np.arange(0.3, 0.7, 0.01):
        recall = recall_score(y_val, (probs >= t).astype(int), zero_division=0)
        if recall > best_recall:
            best_recall, best_t = recall, t
    print(f'Best threshold: {best_t:.2f} | Recall: {best_recall:.3f}')
    return best_t


def train_and_evaluate():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    (X_tr, X_val, X_te, y_tr, y_val, y_te), _ = full_pipeline()
    X_res, y_res = apply_smote(X_tr, y_tr)

    model = build_xgb()

    # ── 5-fold cross-validation ───────────────────────────────────────
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(
        model, X_res, y_res, cv=cv,
        scoring=['recall', 'f1', 'roc_auc'],
        return_train_score=False
    )
    print('XGBoost 5-fold CV:')
    print(f'  Recall:  {results["test_recall"].mean():.3f} ± {results["test_recall"].std():.3f}')
    print(f'  F1:      {results["test_f1"].mean():.3f} ± {results["test_f1"].std():.3f}')
    print(f'  AUC-ROC: {results["test_roc_auc"].mean():.3f} ± {results["test_roc_auc"].std():.3f}')

    # ── Train final model ─────────────────────────────────────────────
    model.fit(X_res, y_res)

    # ── Threshold tuning on validation set ───────────────────────────
    print('\nThreshold tuning on validation set:')
    best_threshold = tune_threshold(model, X_val, y_val)

    # ── Test set evaluation ───────────────────────────────────────────
    y_proba = model.predict_proba(X_te)[:, 1]
    y_pred  = (y_proba >= best_threshold).astype(int)
    print(f'\nTest set results (threshold={best_threshold:.2f}):')
    print(f'  Recall:  {recall_score(y_te, y_pred):.3f}')
    print(f'  F1:      {f1_score(y_te, y_pred):.3f}')
    print(f'  AUC-ROC: {roc_auc_score(y_te, y_proba):.3f}')

    # ── Save model ────────────────────────────────────────────────────
    out_path = os.path.join(PROCESSED_DIR, 'xgb_model.pkl')
    joblib.dump(model, out_path)
    print(f'\nXGBoost model saved to {out_path}')

    return model, results, best_threshold


if __name__ == '__main__':
    train_and_evaluate()