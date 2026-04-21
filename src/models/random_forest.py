"""
random_forest.py — Random Forest baseline with 5-fold CV.
Saves trained model to data/processed/rf_model.pkl for dashboard use.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import recall_score, f1_score, roc_auc_score

from src.preprocessing.clinical import full_pipeline
from src.preprocessing.smote_balance import apply_smote

PROCESSED_DIR = 'data/processed'

def build_rf():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

def train_and_evaluate():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    (X_tr, X_val, X_te, y_tr, y_val, y_te), scaler = full_pipeline()
    X_res, y_res = apply_smote(X_tr, y_tr)

    # ── 5-fold cross-validation ───────────────────────────────────────
    model = build_rf()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(
        model, X_res, y_res, cv=cv,
        scoring=['recall', 'f1', 'roc_auc'],
        return_train_score=False
    )

    print(f"Recall:   {results['test_recall'].mean():.3f} ± {results['test_recall'].std():.3f}")
    print(f"F1:       {results['test_f1'].mean():.3f} ± {results['test_f1'].std():.3f}")
    print(f"AUC-ROC:  {results['test_roc_auc'].mean():.3f} ± {results['test_roc_auc'].std():.3f}")

    # ── Train final model on full SMOTE training set ──────────────────
    model.fit(X_res, y_res)

    # ── Save model for dashboard ──────────────────────────────────────
    joblib.dump(model, os.path.join(PROCESSED_DIR, 'rf_model.pkl'))
    print(f"RF model saved to {PROCESSED_DIR}/rf_model.pkl")

    # ── Test set evaluation ───────────────────────────────────────────
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    print(f"\nTest set results:")
    print(f"  Recall:  {recall_score(y_te, y_pred):.3f}")
    print(f"  F1:      {f1_score(y_te, y_pred):.3f}")
    print(f"  AUC-ROC: {roc_auc_score(y_te, y_proba):.3f}")

    return model, results


if __name__ == '__main__':
    train_and_evaluate()