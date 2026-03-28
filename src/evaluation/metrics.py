from sklearn.metrics import (
    recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import numpy as np

def evaluate_model(model, X_test, y_test, threshold=0.4):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    recall  = recall_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred)
    
    print(f'Recall:  {recall:.3f}  (target >= 0.90)')
    print(f'F1:      {f1:.3f}')
    print(f'AUC-ROC: {auc_roc:.3f}')
    print('\nConfusion matrix:')
    print(cm)
    print('\nClassification report:')
    print(classification_report(y_test, y_pred,
        target_names=['No Disease', 'Heart Disease']))
    
    return {'recall': recall, 'f1': f1, 'auc_roc': auc_roc, 'cm': cm}