import shap
import numpy as np
import pandas as pd

def build_explainer(model, X_train):
    """Build a TreeExplainer for Random Forest."""
    explainer = shap.TreeExplainer(model)
    return explainer

def get_shap_values(explainer, X_input):
    shap_vals = explainer.shap_values(X_input)
    # For RandomForest classifiers, shap_values returns [class0_array, class1_array]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # class=1 (heart disease)
    # shap_vals is shape (1, n_features) — flatten to 1D
    shap_vals = np.array(shap_vals).flatten()
    return dict(zip(X_input.columns, shap_vals))

def top_features(shap_dict, n=6):
    """Return top-n features sorted by absolute SHAP value."""
    return sorted(shap_dict.items(),
                  key=lambda x: abs(x[1]), reverse=True)[:n]
