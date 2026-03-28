import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from collections import deque

from src.preprocessing.clinical import full_pipeline
from src.models.random_forest import build_rf, train_and_evaluate
from src.preprocessing.smote_balance import apply_smote
from src.evaluation.shap_explainer import (
    build_explainer, get_shap_values, top_features)

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioWatch",
    page_icon="U0001fac0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load & train model (cached so it only runs once) ─────────────────
@st.cache_resource
def load_model():
    (X_tr,X_val,X_te,y_tr,y_val,y_te), _ = full_pipeline()
    X_res, y_res = apply_smote(X_tr, y_tr)
    model = build_rf()
    model.fit(X_res, y_res)
    explainer = build_explainer(model, X_res)
    feature_names = X_tr.columns.tolist()
    return model, explainer, feature_names, X_te, y_te

model, explainer, feature_names, X_te, y_te = load_model()

# ── Sidebar: patient profile sliders ─────────────────────────────────
st.sidebar.title("U0001fac0 Patient Profile")
st.sidebar.markdown("Adjust patient parameters to simulate risk.")

age        = st.sidebar.slider("Age", 20, 80, 54)
resting_bp = st.sidebar.slider("Resting BP (mmHg)", 80, 200, 130)
cholesterol= st.sidebar.slider("Cholesterol (mg/dL)", 100, 400, 220)
max_hr     = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
oldpeak    = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1)
sex        = st.sidebar.selectbox("Sex", ["Male", "Female"])
chest_pain = st.sidebar.selectbox(
    "Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
ex_angina  = st.sidebar.selectbox("Exercise Angina", ["No", "Yes"])
resting_ecg= st.sidebar.selectbox(
    "Resting ECG", ["Normal", "LVH", "ST"])
st_slope   = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ── Build feature vector matching training columns ────────────────────
def build_patient_vector(feature_names):
    """Construct a normalised feature row from sidebar values."""
    from sklearn.preprocessing import MinMaxScaler
    # Raw values before normalisation
    raw = {
        "Age": age, "RestingBP": resting_bp,
        "Cholesterol": cholesterol, "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex": 1 if sex == "Male" else 0,
        "ExerciseAngina": 1 if ex_angina == "Yes" else 0,
        # One-hot for ChestPainType
        "ChestPainType_ASY": int(chest_pain == "ASY"),
        "ChestPainType_ATA": int(chest_pain == "ATA"),
        "ChestPainType_NAP": int(chest_pain == "NAP"),
        "ChestPainType_TA":  int(chest_pain == "TA"),
        # One-hot for RestingECG
        "RestingECG_LVH":    int(resting_ecg == "LVH"),
        "RestingECG_Normal": int(resting_ecg == "Normal"),
        "RestingECG_ST":     int(resting_ecg == "ST"),
        # One-hot for ST_Slope
        "ST_Slope_Down": int(st_slope == "Down"),
        "ST_Slope_Flat": int(st_slope == "Flat"),
        "ST_Slope_Up":   int(st_slope == "Up"),
    }
    # Normalise continuous features (same ranges as MinMaxScaler)
    raw["Age"]         = (age - 20) / 60
    raw["RestingBP"]   = (resting_bp - 80) / 120
    raw["Cholesterol"] = (cholesterol - 100) / 300
    raw["MaxHR"]       = (max_hr - 60) / 160
    raw["Oldpeak"]     = oldpeak / 6.0
    # Return as DataFrame with correct column order
    return pd.DataFrame([{k: raw.get(k, 0) for k in feature_names}])

patient_df = build_patient_vector(feature_names)
risk_prob  = model.predict_proba(patient_df)[0, 1]
shap_dict  = get_shap_values(explainer, patient_df)
top_feats  = top_features(shap_dict, n=6)

# ── Main panel ───────────────────────────────────────────────────────
st.title("U0001fac0 CardioWatch — Live Cardiac Risk Monitor")
st.markdown("Simulated Apple Watch-style continuous cardiac risk feed.")

col1, col2, col3 = st.columns([1, 1, 1])

THRESHOLD = 0.5
alert = risk_prob >= THRESHOLD

with col1:
    st.metric(
        label="U0001fac0 Cardiac Risk Score",
        value=f"{risk_prob:.1%}",
        delta="⚠️ HIGH RISK" if alert else "✅ Normal"
    )
    if alert:
        st.error("U0001f6a8 ALERT: Risk score above threshold! Consult a physician.")
    else:
        st.success("✅ Risk within normal range.")

with col2:
    st.subheader("Risk Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_prob * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red" if alert else "steelblue"},
            "steps": [
                {"range": [0, 40],  "color": "#d4edda"},
                {"range": [40, 65], "color": "#fff3cd"},
                {"range": [65, 100],"color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75, "value": 50
            }
        }
    ))
    fig_gauge.update_layout(height=250, margin=dict(t=20,b=0,l=20,r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col3:
    st.subheader("Top Risk Factors (SHAP)")
    feat_names = [f[0] for f in top_feats]
    feat_vals  = [f[1] for f in top_feats]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in feat_vals]
    fig_shap = go.Figure(go.Bar(
        x=feat_vals, y=feat_names,
        orientation="h",
        marker_color=colors
    ))
    fig_shap.update_layout(
        xaxis_title="SHAP value (impact on risk)",
        height=250,
        margin=dict(t=20, b=0, l=20, r=20)
    )
    st.plotly_chart(fig_shap, use_container_width=True)

# ── Rolling risk chart ───────────────────────────────────────────────
st.subheader("U0001f4c8 Rolling Risk History (last 30 readings)")
if "risk_history" not in st.session_state:
    st.session_state.risk_history = deque(maxlen=30)
st.session_state.risk_history.append(round(risk_prob, 4))

fig_roll = go.Figure()
fig_roll.add_trace(go.Scatter(
    y=list(st.session_state.risk_history),
    mode="lines+markers",
    line=dict(color="steelblue", width=2),
    name="Risk score"
))
fig_roll.add_hline(y=THRESHOLD, line_dash="dash",
                   line_color="red", annotation_text="Alert threshold")
fig_roll.update_layout(
    yaxis=dict(range=[0, 1], title="Risk probability"),
    xaxis_title="Reading #",
    height=300,
    margin=dict(t=20, b=40, l=40, r=20)
)
st.plotly_chart(fig_roll, use_container_width=True)

# ── Raw patient vector (for debugging) ───────────────────────────────
with st.expander("Show raw feature vector"):
    st.dataframe(patient_df)