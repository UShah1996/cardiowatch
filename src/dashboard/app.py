"""
app.py — CardioWatch Streamlit Dashboard
Combines Random Forest clinical risk + CNN-LSTM AFib ECG risk into a fused score.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import torch
from collections import deque

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioWatch",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load saved models (instant — no retraining) ──────────────────────
@st.cache_resource
def load_models():
    """Load pre-trained RF model, scaler, and CNN-LSTM from disk."""
    rf_model = joblib.load('data/processed/rf_model.pkl')
    scaler   = joblib.load('data/processed/scaler.pkl')

    # CNN-LSTM — load architecture then weights
    from src.models.cnn_lstm import build_model
    cnn_model = build_model(input_length=5000)
    weights_path = 'data/processed/cnn_lstm_best.pt'
    if os.path.exists(weights_path):
        cnn_model.load_state_dict(
            torch.load(weights_path, map_location='cpu'))
        cnn_model.eval()
        cnn_ready = True
    else:
        cnn_ready = False

    # Feature names from RF model
    feature_names = rf_model.feature_names_in_.tolist()

    return rf_model, scaler, cnn_model, cnn_ready, feature_names


rf_model, scaler, cnn_model, cnn_ready, feature_names = load_models()

# ── SHAP explainer ───────────────────────────────────────────────────
@st.cache_resource
def load_explainer(_rf_model):
    from src.evaluation.shap_explainer import build_explainer
    # Need training data for explainer — load a sample
    from src.preprocessing.clinical import full_pipeline
    from src.preprocessing.smote_balance import apply_smote
    (X_tr, _, _, y_tr, _, _), _ = full_pipeline()
    X_res, y_res = apply_smote(X_tr, y_tr)
    return build_explainer(_rf_model, X_res)

explainer = load_explainer(rf_model)

# ── Sidebar: patient profile sliders ─────────────────────────────────
st.sidebar.title("🫀 Patient Profile")
st.sidebar.markdown("Adjust parameters to simulate cardiac risk.")

age         = st.sidebar.slider("Age", 20, 80, 54)
resting_bp  = st.sidebar.slider("Resting BP (mmHg)", 80, 200, 130)
cholesterol = st.sidebar.slider("Cholesterol (mg/dL)", 100, 400, 220)
max_hr      = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
oldpeak     = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1)
sex         = st.sidebar.selectbox("Sex", ["Male", "Female"])
chest_pain  = st.sidebar.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
ex_angina   = st.sidebar.selectbox("Exercise Angina", ["No", "Yes"])
resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "LVH", "ST"])
st_slope    = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ── Build feature vector using saved scaler ───────────────────────────
def build_patient_vector(feature_names, scaler):
    """Construct a normalized feature row from sidebar values."""
    raw = {
        "Age":         age,
        "RestingBP":   resting_bp,
        "Cholesterol": cholesterol,
        "MaxHR":       max_hr,
        "Oldpeak":     oldpeak,
        "Sex":                    1 if sex == "Male" else 0,
        "ExerciseAngina":         1 if ex_angina == "Yes" else 0,
        "ChestPainType_ASY":      int(chest_pain == "ASY"),
        "ChestPainType_ATA":      int(chest_pain == "ATA"),
        "ChestPainType_NAP":      int(chest_pain == "NAP"),
        "ChestPainType_TA":       int(chest_pain == "TA"),
        "RestingECG_LVH":         int(resting_ecg == "LVH"),
        "RestingECG_Normal":      int(resting_ecg == "Normal"),
        "RestingECG_ST":          int(resting_ecg == "ST"),
        "ST_Slope_Down":          int(st_slope == "Down"),
        "ST_Slope_Flat":          int(st_slope == "Flat"),
        "ST_Slope_Up":            int(st_slope == "Up"),
    }

    df = pd.DataFrame([{k: raw.get(k, 0) for k in feature_names}])

    # Apply saved scaler to continuous features
    continuous = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    continuous_present = [c for c in continuous if c in df.columns]
    df[continuous_present] = scaler.transform(df[continuous_present])

    return df


# ── Fusion function ───────────────────────────────────────────────────
def fuse_scores(rf_prob, ecg_prob, rf_weight=0.6, ecg_weight=0.4):
    """Late fusion: weighted average of RF clinical + CNN-LSTM ECG scores."""
    return rf_weight * rf_prob + ecg_weight * ecg_prob


# ── Compute RF risk ───────────────────────────────────────────────────
patient_df = build_patient_vector(feature_names, scaler)
rf_prob    = rf_model.predict_proba(patient_df)[0, 1]

from src.evaluation.shap_explainer import get_shap_values, top_features
shap_dict = get_shap_values(explainer, patient_df)
top_feats = top_features(shap_dict, n=6)

THRESHOLD = 0.5
alert = rf_prob >= THRESHOLD

# ── Main panel ───────────────────────────────────────────────────────
st.title("🫀 CardioWatch — Live Cardiac Risk Monitor")
st.markdown("Multi-modal cardiac risk: **Random Forest** (clinical) + **CNN-LSTM** (ECG AFib detection)")

# ── Row 1: RF clinical risk ───────────────────────────────────────────
st.subheader("📊 Clinical Risk (Random Forest)")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.metric(
        label="Clinical Risk Score",
        value=f"{rf_prob:.1%}",
        delta="⚠️ HIGH RISK" if alert else "✅ Normal"
    )
    if alert:
        st.error("🚨 ALERT: Clinical risk above threshold — consult a physician.")
    else:
        st.success("✅ Clinical risk within normal range.")

with col2:
    st.subheader("Risk Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rf_prob * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red" if alert else "steelblue"},
            "steps": [
                {"range": [0, 40],   "color": "#d4edda"},
                {"range": [40, 65],  "color": "#fff3cd"},
                {"range": [65, 100], "color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    fig_gauge.update_layout(height=250, margin=dict(t=20, b=0, l=20, r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col3:
    st.subheader("Top Risk Factors (SHAP)")
    feat_names = [f[0] for f in top_feats]
    feat_vals  = [f[1] for f in top_feats]
    colors     = ["#e74c3c" if v > 0 else "#2ecc71" for v in feat_vals]
    fig_shap   = go.Figure(go.Bar(
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

st.divider()

# ── Row 2: ECG upload + CNN-LSTM AFib risk ────────────────────────────
st.subheader("📈 ECG Risk — Apple Watch AFib Detection (CNN-LSTM)")

if not cnn_ready:
    st.warning("⚠️ CNN-LSTM model not found at `data/processed/cnn_lstm_best.pt`. "
               "Run `python src/models/train_cnn_lstm.py` first.")

uploaded = st.file_uploader(
    "Upload Apple Watch ECG export (Health app → Browse → Heart → "
    "Electrocardiograms → Export as CSV)",
    type="csv"
)

ecg_prob   = None
fused_prob = None

if uploaded is not None:
    try:
        # Robust Apple Watch CSV parsing — skips comment/header rows
        raw_df = pd.read_csv(uploaded, comment='#', header=None)
        signal = pd.to_numeric(
            raw_df.iloc[:, 0], errors='coerce').dropna().values.astype(np.float32)

        st.info(f"Loaded {len(signal)} samples from Apple Watch ECG "
                f"({len(signal)/512:.1f}s at 512 Hz)")
        
        # ADD THIS — convert µV to mV to match training data scale
        signal = signal / 1000.0

        from src.preprocessing.ecg_filter import bandpass_filter, segment_into_windows
        filtered = bandpass_filter(signal, 0.5, 100.0, fs=512)
        windows  = segment_into_windows(filtered, fs=512, window_minutes=5)
        st.success(f"Preprocessed into {len(windows)} window(s)")

        if cnn_ready and len(windows) > 0:
            # Run CNN-LSTM inference on each window
            ecg_probs = []
            for window in windows:
                # Normalize window (same as training)
                w = window.astype(np.float32)
                w = np.clip(w, -2.0, 2.0)
                w = (w - w.mean()) / (w.std() + 1e-8)
                w = np.clip(w, -5.0, 5.0)

                # Truncate or pad to 5000 samples
                if len(w) >= 5000:
                    w = w[:5000]
                else:
                    w = np.pad(w, (0, 5000 - len(w)))

                x = torch.tensor(w).unsqueeze(0).unsqueeze(0)  # (1, 1, 5000)
                with torch.no_grad():
                    logit = cnn_model(x).squeeze()
                    prob  = torch.sigmoid(logit).item()
                ecg_probs.append(prob)

            ecg_prob = float(np.mean(ecg_probs))

            # Fused score
            fused_prob = fuse_scores(rf_prob, ecg_prob)

            # Display ECG + fused results
            ecg_col1, ecg_col2, ecg_col3 = st.columns(3)
            with ecg_col1:
                st.metric("🫀 AFib Probability (CNN-LSTM)",
                          f"{ecg_prob:.1%}",
                          delta="⚠️ AFib detected" if ecg_prob >= 0.5 else "✅ No AFib")
            with ecg_col2:
                st.metric("⚡ Fused Risk Score",
                          f"{fused_prob:.1%}",
                          delta="⚠️ HIGH" if fused_prob >= THRESHOLD else "✅ Normal")
            with ecg_col3:
                st.caption("Fusion weights")
                st.caption("Clinical (RF): 60%")
                st.caption("ECG (CNN-LSTM): 40%")

            if ecg_prob >= 0.5:
                st.error("🚨 AFib pattern detected in ECG — please consult a cardiologist.")
            else:
                st.success("✅ No AFib pattern detected in ECG.")

    except Exception as e:
        st.error(f"Error processing ECG file: {e}")
        st.caption("Make sure the file is a valid Apple Watch ECG CSV export.")

st.divider()

# ── Row 3: Rolling risk history ───────────────────────────────────────
st.subheader("📈 Rolling Risk History (last 30 readings)")

if "risk_history" not in st.session_state:
    st.session_state.risk_history        = deque(maxlen=30)
    st.session_state.fused_risk_history  = deque(maxlen=30)

st.session_state.risk_history.append(round(rf_prob, 4))
if fused_prob is not None:
    st.session_state.fused_risk_history.append(round(fused_prob, 4))

fig_roll = go.Figure()
fig_roll.add_trace(go.Scatter(
    y=list(st.session_state.risk_history),
    mode="lines+markers",
    line=dict(color="steelblue", width=2),
    name="Clinical risk (RF)"
))
if len(st.session_state.fused_risk_history) > 0:
    fig_roll.add_trace(go.Scatter(
        y=list(st.session_state.fused_risk_history),
        mode="lines+markers",
        line=dict(color="crimson", width=2, dash="dot"),
        name="Fused risk (RF + ECG)"
    ))
fig_roll.add_hline(
    y=THRESHOLD, line_dash="dash",
    line_color="red", annotation_text="Alert threshold"
)
fig_roll.update_layout(
    yaxis=dict(range=[0, 1], title="Risk probability"),
    xaxis_title="Reading #",
    height=300,
    margin=dict(t=20, b=40, l=40, r=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
st.plotly_chart(fig_roll, use_container_width=True)

# ── Debug expander ────────────────────────────────────────────────────
with st.expander("🔍 Debug — raw feature vector"):
    st.dataframe(patient_df)
    st.caption(f"RF probability: {rf_prob:.4f}")
    if ecg_prob is not None:
        st.caption(f"ECG probability: {ecg_prob:.4f}")
        st.caption(f"Fused probability: {fused_prob:.4f}")