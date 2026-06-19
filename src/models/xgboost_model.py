"""
xgboost_model.py — XGBoost baseline for clinical risk prediction + 95% CI.
Saves trained model to data/processed/xgb_model.pkl for dashboard use.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import joblib
import numpy as np
import yaml
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import recall_score, f1_score, roc_auc_score, fbeta_score, precision_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.preprocessing.clinical import full_pipeline
from src.preprocessing.smote_balance import apply_smote
from src.evaluation.confidence_intervals import cv_ci_report, bootstrap_all_metrics

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
        n_jobs           = -1,
    )


def tune_threshold(model, X_val, y_val, beta=2.0):
    """
    Find the threshold that maximises F-beta on the validation set.

    Maximising recall alone is not a valid selection criterion: recall
    decreases monotonically with the threshold, so the search always
    collapses to the lowest value in the grid (and, in the limit, to a
    trivial all-positive classifier). F-beta balances recall against
    precision; beta=2 weights recall twice as heavily as precision,
    which is appropriate for a screening tool where missing a true case
    is costlier than a false alarm — without the degenerate solution.
    """
    probs = model.predict_proba(X_val)[:, 1]
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= t).astype(int)
        f = fbeta_score(y_val, preds, beta=beta, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    preds_best = (probs >= best_t).astype(int)
    r = recall_score(y_val, preds_best, zero_division=0)
    p = precision_score(y_val, preds_best, zero_division=0)
    print(f'Best threshold (F{beta:.0f}): {best_t:.2f} | '
          f'F{beta:.0f}: {best_f:.3f} | Recall: {r:.3f} | Precision: {p:.3f}')
    return best_t


def train_and_evaluate():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    (X_tr, X_val, X_te, y_tr, y_val, y_te), _ = full_pipeline()

    # ── 5-fold CV with SMOTE applied INSIDE each fold ─────────────────
    # See random_forest.py: SMOTE-before-CV leaks synthetic samples from
    # validation rows into training rows. The imblearn Pipeline resamples
    # only each fold's training portion.
    cv_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=5)),
        ('xgb',   build_xgb()),
    ])
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(
        cv_pipeline, X_tr, y_tr, cv=cv,
        scoring=['recall', 'f1', 'roc_auc'],
        return_train_score=False
    )

    print("XGBoost — 5-fold CV (mean ± std):")
    print(f"  Recall:  {results['test_recall'].mean():.3f} ± {results['test_recall'].std():.3f}")
    print(f"  F1:      {results['test_f1'].mean():.3f} ± {results['test_f1'].std():.3f}")
    print(f"  AUC-ROC: {results['test_roc_auc'].mean():.3f} ± {results['test_roc_auc'].std():.3f}")

    # ── 95% CI on CV results ──────────────────────────────────────────
    cv_ci_report(results, model_name='XGBoost (5-fold CV)')

    # ── Train final model on full SMOTE training set ──────────────────
    X_res, y_res = apply_smote(X_tr, y_tr)
    model = build_xgb()
    model.fit(X_res, y_res)

    # ── Threshold tuning on validation set ───────────────────────────
    print('\nThreshold tuning on validation set:')
    best_threshold = tune_threshold(model, X_val, y_val)

    # ── Test set evaluation + bootstrap CI ───────────────────────────
    y_proba = model.predict_proba(X_te)[:, 1]
    y_pred  = (y_proba >= best_threshold).astype(int)

    print(f"\nXGBoost — Test set (n={len(y_te)}, threshold={best_threshold:.2f}):")
    print(f"  Recall:  {recall_score(y_te, y_pred):.3f}")
    print(f"  F1:      {f1_score(y_te, y_pred):.3f}")
    print(f"  AUC-ROC: {roc_auc_score(y_te, y_proba):.3f}")

    # Bootstrap CI — this is the key fix: shows how unreliable 0.980 recall
    # is on n=92 patients after threshold tuning
    bootstrap_all_metrics(
        np.array(y_te), y_proba,
        threshold=best_threshold,
        label=f'XGBoost Test Set (n={len(y_te)}, threshold={best_threshold:.2f})',
        n_boot=2000,
    )

    # ── Save model ────────────────────────────────────────────────────
    out_path = os.path.join(PROCESSED_DIR, 'xgb_model.pkl')
    joblib.dump({'model': model, 'threshold': best_threshold}, out_path)
    print(f'\nXGBoost model saved → {out_path}')

    return model, results, best_threshold


if __name__ == '__main__':
    train_and_evaluate()