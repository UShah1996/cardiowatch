from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
import yaml
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def load_config(path='configs/config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

def build_rf(config_path='configs/config.yaml'):
    cfg = load_config(config_path)['models']['random_forest']
    return RandomForestClassifier(
        n_estimators=cfg['n_estimators'],
        max_depth=cfg['max_depth'],
        class_weight=cfg['class_weight'],
        random_state=cfg['random_state'],
    )

def train_and_evaluate(X_train, y_train, config_path='configs/config.yaml'):
    cfg = load_config(config_path)['evaluation']
    model = build_rf(config_path)
    cv = StratifiedKFold(n_splits=cfg['cv_folds'], shuffle=True, random_state=42)
    recall  = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall')
    f1      = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    roc_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f'Recall:  {recall.mean():.3f} +/- {recall.std():.3f}')
    print(f'F1:      {f1.mean():.3f} +/- {f1.std():.3f}')
    print(f'AUC-ROC: {roc_auc.mean():.3f} +/- {roc_auc.std():.3f}')
    model.fit(X_train, y_train)
    return model

if __name__ == '__main__':
    import sys; sys.path.insert(0,'..')
    from src.preprocessing.clinical import full_pipeline
    from src.preprocessing.smote_balance import apply_smote
    (X_tr,X_val,X_te,y_tr,y_val,y_te), _ = full_pipeline()
    X_res, y_res = apply_smote(X_tr, y_tr)
    print('\n=== Random Forest CV Results ===')
    model = train_and_evaluate(X_res, y_res)
    print('Training done!')