# src/models/xgboost_model.py
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import recall_score
import numpy as np
import yaml

def build_xgb(config_path='configs/config.yaml'):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)['models']['xgboost']
    return XGBClassifier(
        n_estimators=cfg['n_estimators'],
        max_depth=cfg['max_depth'],
        learning_rate=cfg['learning_rate'],
        scale_pos_weight=cfg['scale_pos_weight'],
        eval_metric='aucpr',
        random_state=cfg['random_state']
    )
    
def tune_threshold(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    best_t, best_recall = 0.5, 0
    for t in np.arange(0.3, 0.7, 0.01):
        recall = recall_score(y_val, (probs >= t).astype(int))
        if recall > best_recall:
            best_recall, best_t = recall, t
    print(f'Best threshold: {best_t:.2f} | Recall: {best_recall:.3f}')
    return best_t

if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    from src.preprocessing.clinical import full_pipeline
    from src.preprocessing.smote_balance import apply_smote
    (X_tr,X_val,X_te,y_tr,y_val,y_te), _ = full_pipeline()
    X_res, y_res = apply_smote(X_tr, y_tr)
    model = build_xgb()
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    recall = cross_val_score(model, X_res, y_res, cv=cv, scoring='recall')
    auc    = cross_val_score(model, X_res, y_res, cv=cv, scoring='roc_auc')
    print(f'XGBoost Recall: {recall.mean():.3f} | AUC: {auc.mean():.3f}')