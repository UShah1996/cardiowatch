import numpy as np
from imblearn.over_sampling import SMOTE

def apply_smote(X_train, y_train, random_state=42, k_neighbors=5):
    print(f'Before SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}')
    sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f'After SMOTE:  {dict(zip(*np.unique(y_res, return_counts=True)))}')
    return X_res, y_res

if __name__ == '__main__':
    from clinical import full_pipeline
    (X_tr, X_val, X_te, y_tr, y_val, y_te), _ = full_pipeline()
    X_res, y_res = apply_smote(X_tr, y_tr)
    print('SMOTE applied. New training size:', len(X_res))