"""
app.py — CardioWatch Streamlit Dashboard
Combines Random Forest / XGBoost clinical risk + CNN-LSTM AFib ECG risk
into a calibrated fused score with SHAP explainability, alert logging,
and model performance display.
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


# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioWatch",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Weight download (Streamlit Cloud) ────────────────────────────────
import os

def _download_weights_on_startup():
    PROCESSED_DIR = 'data/processed'
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Already downloaded — check all critical files
    critical = ['rf_model.pkl', 'scaler.pkl', 'xgb_model.pkl']
    if all(
        os.path.exists(f'{PROCESSED_DIR}/{f}') and
        os.path.getsize(f'{PROCESSED_DIR}/{f}') > 500
        for f in critical
    ):
        return True

    status_container = st.empty()
    status_container.info("⏳ Downloading model weights from Hugging Face (first load only — ~2 min)...")

    try:
        from huggingface_hub import hf_hub_download
        from src.dashboard.download_weights import WEIGHTS, OPTIONAL_WEIGHTS, REPO_ID

        progress = st.progress(0)
        results  = {}

        for i, name in enumerate(WEIGHTS):
            dest = os.path.join(PROCESSED_DIR, name)
            if os.path.exists(dest) and os.path.getsize(dest) > 500:
                results[name] = True
                progress.progress((i + 1) / len(WEIGHTS))
                continue

            status_container.info(f"⏳ Downloading {name} ({i+1}/{len(WEIGHTS)})...")
            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=name,
                    local_dir=PROCESSED_DIR,
                    local_dir_use_symlinks=False,
                )
                size = os.path.getsize(dest) if os.path.exists(dest) else 0
                if size > 500:
                    results[name] = True
                else:
                    st.error(f"❌ {name} downloaded but too small ({size} bytes)")
                    if os.path.exists(dest):
                        os.remove(dest)
                    results[name] = False
            except Exception as e:
                st.error(f"❌ {name} failed: {e}")
                results[name] = False

            progress.progress((i + 1) / len(WEIGHTS))

        # Optional pre-registered CNN checkpoints — fetch if on the Hub, but
        # never error if absent (dashboard falls back to legacy checkpoints).
        for name in OPTIONAL_WEIGHTS:
            dest = os.path.join(PROCESSED_DIR, name)
            if os.path.exists(dest) and os.path.getsize(dest) > 500:
                continue
            try:
                hf_hub_download(repo_id=REPO_ID, filename=name,
                                local_dir=PROCESSED_DIR, local_dir_use_symlinks=False)
            except Exception:
                pass

        progress.empty()
        n_ok = sum(results.values())
        if n_ok == len(WEIGHTS):
            status_container.success(f"✅ All {n_ok} weights downloaded.")
            return True
        else:
            failed = [k for k, v in results.items() if not v]
            status_container.warning(
                f"⚠️ {n_ok}/{len(WEIGHTS)} weights ready. "
                f"Failed: {', '.join(failed)}"
            )
            return n_ok > 0

    except ImportError:
        st.error("❌ huggingface_hub not installed — add it to requirements.txt")
        return False
    except Exception as e:
        st.error(f"❌ Download failed: {e}")
        return False

_weights_ready = _download_weights_on_startup()


# ── Load saved models ─────────────────────────────────────────────────
# ── Demo mode: synthetic model for Streamlit Cloud (no weights) ──────
class _DemoModel:
    """Stub model that returns plausible synthetic predictions."""
    feature_names_in_ = None

    def __init__(self, positive_rate=0.35):
        self._rate = positive_rate
        import numpy as _np
        self.feature_names_in_ = _np.array([
            'Age','RestingBP','Cholesterol','MaxHR','Oldpeak','Sex',
            'ExerciseAngina','ChestPainType_ASY','ChestPainType_ATA',
            'ChestPainType_NAP','ChestPainType_TA','RestingECG_LVH',
            'RestingECG_Normal','RestingECG_ST','ST_Slope_Down',
            'ST_Slope_Flat','ST_Slope_Up',
        ])

    def predict_proba(self, X):
        import numpy as _np
        # Make score loosely sensitive to feature values so sliders feel live
        try:
            age_idx = list(self.feature_names_in_).index('Age')
            raw_age = float(X.iloc[0, age_idx])  # already scaled 0-1
            score   = float(_np.clip(self._rate + raw_age * 0.3, 0.05, 0.95))
        except Exception:
            score = self._rate
        return _np.array([[1 - score, score]])


@st.cache_resource
def load_models():
    """
    Load all models from data/processed/.
    Prefers the pre-registered hold-out CNN checkpoints
    (cnn_lstm_combined_deploy.pt / cnn_lstm_cpsc_complement.pt), falling back to
    the legacy checkpoints if absent.
    Loads CalibratedFusion if fusion_model.pkl exists.
    Falls back to demo mode if weights are not present (Streamlit Cloud).
    """
    CRITICAL = [
        'data/processed/rf_model.pkl',
        'data/processed/scaler.pkl',
        'data/processed/xgb_model.pkl',
    ]
    DEMO = not all(
        os.path.exists(p) and os.path.getsize(p) > 500
        for p in CRITICAL
    )

    if DEMO:
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        demo_rf  = _DemoModel(positive_rate=0.35)
        demo_xgb = _DemoModel(positive_rate=0.30)
        demo_scaler = MinMaxScaler()
        # Fit on dummy data so transform() works
        dummy = np.array([[20,80,100,60,0],[80,200,400,220,6]], dtype=float)
        demo_scaler.fit(dummy)
        from src.models.cnn_lstm import build_model
        cnn_model = build_model(input_length=5000)
        return (demo_rf, demo_xgb, 0.30, demo_scaler,
                cnn_model, False, 'Demo mode (no weights)',
                None, [], False,
                None, 'Demo (0.60/0.40)',
                list(demo_rf.feature_names_in_))

    rf_model  = joblib.load('data/processed/rf_model.pkl')
    scaler    = joblib.load('data/processed/scaler.pkl')

    # XGBoost — stored as dict {model, threshold} from updated xgboost_model.py
    xgb_path = 'data/processed/xgb_model.pkl'
    if os.path.exists(xgb_path):
        xgb_saved  = joblib.load(xgb_path)
        # Handle both old format (bare model) and new format (dict)
        if isinstance(xgb_saved, dict):
            xgb_model      = xgb_saved['model']
            xgb_threshold  = xgb_saved.get('threshold', 0.30)
        else:
            xgb_model      = xgb_saved
            xgb_threshold  = 0.30
    else:
        xgb_model     = None
        xgb_threshold = 0.30

    # CNN-LSTM — prefer the pre-registered hold-out checkpoints; fall back to
    # the legacy ones so the app still runs if the new weights aren't present.
    cnn_candidates = [
        ('data/processed/cnn_lstm_combined_deploy.pt', 'CNN-LSTM Combined (deployment, hold-out AUC 0.975)'),
        ('data/processed/cnn_lstm_cpsc_complement.pt', 'CNN-LSTM CPSC (hold-out AUC 0.949)'),
        ('data/processed/cnn_lstm_combined_best.pt',   'CNN-LSTM Combined (legacy)'),
        ('data/processed/cnn_lstm_cv_best.pt',         'CNN-LSTM (legacy CV)'),
        ('data/processed/cnn_lstm_best.pt',            'CNN-LSTM CPSC-only (legacy)'),
    ]
    from src.models.cnn_lstm import build_model
    cnn_model   = build_model(input_length=5000)
    cnn_ready   = False
    cnn_label   = 'CNN-LSTM'
    for weights_path, label in cnn_candidates:
        if os.path.exists(weights_path):
            cnn_model.load_state_dict(
                torch.load(weights_path, map_location='cpu'))
            cnn_model.eval()
            cnn_ready = True
            cnn_label = label
            break

    # RR Traditional ML
    rr_path = 'data/processed/rr_rf_model.pkl'
    if os.path.exists(rr_path):
        rr_saved      = joblib.load(rr_path)
        rr_model      = rr_saved['model']
        rr_feat_names = rr_saved['feature_names']
        rr_ready      = True
    else:
        rr_model      = None
        rr_feat_names = []
        rr_ready      = False

    # Calibrated fusion model
    fusion_path = 'data/processed/fusion_model.pkl'
    if os.path.exists(fusion_path):
        from src.models.fusion_calibrated import CalibratedFusion
        fusion_model = CalibratedFusion.load(fusion_path)
        fusion_label = (
            f'Learned (RF={fusion_model.weight_rf:.2f}, '
            f'ECG={fusion_model.weight_ecg:.2f})'
            if fusion_model.fitted else 'Fallback (0.60/0.40)'
        )
    else:
        fusion_model = None
        fusion_label = 'Heuristic (0.60/0.40)'

    feature_names = rf_model.feature_names_in_.tolist()

    return (rf_model, xgb_model, xgb_threshold, scaler,
            cnn_model, cnn_ready, cnn_label,
            rr_model, rr_feat_names, rr_ready,
            fusion_model, fusion_label,
            feature_names)


(rf_model, xgb_model, xgb_threshold, scaler,
 cnn_model, cnn_ready, cnn_label,
 rr_model, rr_feat_names, rr_ready,
 fusion_model, fusion_label,
 feature_names) = load_models()


# ── SHAP explainers ───────────────────────────────────────────────────
@st.cache_resource
def load_explainers(_rf_model, _xgb_model):
    # Skip SHAP in demo mode
    if isinstance(_rf_model, _DemoModel):
        return {'Random Forest': None, 'XGBoost': None}
    from src.evaluation.shap_explainer import build_explainer
    from src.preprocessing.clinical import full_pipeline
    from src.preprocessing.smote_balance import apply_smote
    (X_tr, _, _, y_tr, _, _), _ = full_pipeline()
    X_res, _ = apply_smote(X_tr, y_tr)
    explainers = {'Random Forest': build_explainer(_rf_model, X_res)}
    if _xgb_model is not None and not isinstance(_xgb_model, _DemoModel):
        explainers['XGBoost'] = build_explainer(_xgb_model, X_res)
    return explainers


explainers = load_explainers(rf_model, xgb_model)

# ── Session state init ────────────────────────────────────────────────
for key in ['risk_history', 'fused_risk_history', 'alert_log']:
    if key not in st.session_state:
        st.session_state[key] = deque(maxlen=30) if 'history' in key else []

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("🫀 Patient Profile")
st.sidebar.markdown("Adjust parameters to explore the clinical AFib/stroke risk model.")

clinical_model_choice = st.sidebar.radio(
    "Clinical Model",
    ["Random Forest", "XGBoost"] if xgb_model is not None else ["Random Forest"],
    horizontal=True
)
st.sidebar.divider()

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


# ── Feature vector ────────────────────────────────────────────────────
def build_patient_vector(feature_names, scaler):
    raw = {
        "Age":               age,           "RestingBP":         resting_bp,
        "Cholesterol":       cholesterol,   "MaxHR":             max_hr,
        "Oldpeak":           oldpeak,       "Sex":               1 if sex == "Male" else 0,
        "ExerciseAngina":    1 if ex_angina == "Yes" else 0,
        "ChestPainType_ASY": int(chest_pain == "ASY"),
        "ChestPainType_ATA": int(chest_pain == "ATA"),
        "ChestPainType_NAP": int(chest_pain == "NAP"),
        "ChestPainType_TA":  int(chest_pain == "TA"),
        "RestingECG_LVH":    int(resting_ecg == "LVH"),
        "RestingECG_Normal": int(resting_ecg == "Normal"),
        "RestingECG_ST":     int(resting_ecg == "ST"),
        "ST_Slope_Down":     int(st_slope == "Down"),
        "ST_Slope_Flat":     int(st_slope == "Flat"),
        "ST_Slope_Up":       int(st_slope == "Up"),
    }
    df         = pd.DataFrame([{k: raw.get(k, 0) for k in feature_names}])
    continuous = [c for c in ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
                  if c in df.columns]
    df[continuous] = scaler.transform(df[continuous])
    return df


def compute_fused_score(clinical_prob: float, ecg_prob: float) -> tuple[float, str]:
    """
    Compute fused score using CalibratedFusion if available,
    falling back to hardcoded 0.6/0.4.

    Returns (fused_prob, method_description)
    """
    if fusion_model is not None and fusion_model.fitted:
        fused = fusion_model.predict_proba(clinical_prob, ecg_prob)
        return float(fused), fusion_label
    # Hardcoded fallback
    return float(0.6 * clinical_prob + 0.4 * ecg_prob), 'Heuristic (0.60/0.40)'


# ── Active clinical model ─────────────────────────────────────────────
active_model  = xgb_model if clinical_model_choice == "XGBoost" else rf_model
THRESHOLD     = xgb_threshold if clinical_model_choice == "XGBoost" else 0.50
model_color   = "#e67e22" if clinical_model_choice == "XGBoost" else "steelblue"

patient_df    = build_patient_vector(feature_names, scaler)
clinical_prob = active_model.predict_proba(patient_df)[0, 1]
alert         = clinical_prob >= THRESHOLD

from src.evaluation.shap_explainer import get_shap_values, top_features
explainer = explainers.get(clinical_model_choice)
if explainer is not None:
    shap_dict = get_shap_values(explainer, patient_df)
    top_feats = top_features(shap_dict, n=6)
else:
    # Demo mode — synthetic SHAP values
    top_feats = [
        ('ST_Slope_Flat', 0.42), ('ChestPainType_ASY', 0.31),
        ('MaxHR', -0.28), ('Age', 0.19),
        ('Oldpeak', 0.15), ('Sex', -0.08),
    ]

ecg_prob   = None
fused_prob = None

# ═══════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ═══════════════════════════════════════════════════════════════════════
st.title("🫀 CardioWatch — AFib Detection for Stroke Prevention")
st.markdown(
    "Device-agnostic **single-lead atrial-fibrillation detection** on "
    "consumer wearables. Catching AFib early enables anticoagulation that "
    "prevents strokes. _Detects atrial fibrillation — **not** heart attacks._ "
    "A clinical risk model and clinical+ECG fusion are shown as separate, "
    "exploratory components."
)

# Demo mode banner
if not os.path.exists('data/processed/rf_model.pkl'):
    st.warning(
        "⚠️ **Demo mode** — model weights not found. "
        "Scores are illustrative only. "
        "Clone the repo and run the training pipeline to use real models. "
        "See [github.com/UShah1996/cardiowatch](https://github.com/UShah1996/cardiowatch)."
    )

# ── Section 0: Architecture explainer (fusion = exploratory) ─────────
with st.expander("ℹ️ How CardioWatch works — architecture (fusion is exploratory)", expanded=False):
    weight_rf  = fusion_model.weight_rf  if (fusion_model and fusion_model.fitted) else 0.60
    weight_ecg = fusion_model.weight_ecg if (fusion_model and fusion_model.fitted) else 0.40
    st.markdown(f"""
The **validated** product is single-lead **AFib detection** from the ECG. A
clinical risk model and a clinical+ECG **late fusion** are shown as separate,
**exploratory** components — not a validated joint-patient score.

```
Patient's clinical data          Patient's Apple Watch ECG
(Age, BP, Cholesterol...)        (Lead I, 30s at 512 Hz)
        │                                   │
        ▼                                   ▼
Random Forest / XGBoost            CNN-LSTM  (AFib detection)
(separate clinical cohort)         (CPSC hold-out AUC 0.949)
        │                                   │
        └──────────────┬────────────────────┘
                       ▼
         Fused = {weight_rf:.2f} × clinical + {weight_ecg:.2f} × ECG   (EXPLORATORY)
                       ▼
              Alert if fused > threshold
```

**⚠️ Why fusion is exploratory:** the clinical and ECG models are trained on
**different patient populations** (Kaggle clinical data vs CPSC ECG), so their
scores are not combined per patient — the weights were learned from CPSC
validation ECG scores paired with *sampled* clinical scores. A deployable fused
score would require one cohort where each patient has both an ECG and clinical
data. Treat the fused number as illustrative only.""")

# ── Section 1: Model comparison banner ───────────────────────────────
if xgb_model is not None:
    m1, m2, m3 = st.columns(3)
    rf_score  = rf_model.predict_proba(patient_df)[0, 1]
    xgb_score = xgb_model.predict_proba(patient_df)[0, 1]
    with m1:
        st.metric("Random Forest Score", f"{rf_score:.1%}",
                  delta="threshold 50%", delta_color="off")
    with m2:
        st.metric("XGBoost Score", f"{xgb_score:.1%}",
                  delta=f"threshold {xgb_threshold:.0%}", delta_color="off")
    with m3:
        st.info(f"🔵 Active model: **{clinical_model_choice}**")
    st.divider()

# ── Section 2: Clinical risk ──────────────────────────────────────────
st.subheader(f"📊 Clinical Risk ({clinical_model_choice})")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.metric(
        label=f"Clinical Risk Score ({clinical_model_choice})",
        value=f"{clinical_prob:.1%}",
        delta="⚠️ HIGH RISK" if alert else "✅ Normal"
    )
    if alert:
        st.error("🚨 ALERT: Clinical risk above threshold — consult a physician.")
    else:
        st.success("✅ Clinical risk within normal range.")
    st.caption(
        f"Alert threshold: {THRESHOLD:.0%} "
        f"({'tuned on validation set' if clinical_model_choice == 'XGBoost' else 'default'})"
    )

with col2:
    st.subheader("Risk Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=clinical_prob * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "red" if alert else model_color},
            "steps": [
                {"range": [0,  40],  "color": "#d4edda"},
                {"range": [40, 65],  "color": "#fff3cd"},
                {"range": [65, 100], "color": "#f8d7da"},
            ],
            "threshold": {
                "line":      {"color": "black", "width": 3},
                "thickness": 0.75,
                "value":     THRESHOLD * 100,
            }
        }
    ))
    fig_gauge.update_layout(height=250, margin=dict(t=20, b=0, l=20, r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col3:
    st.subheader(f"Top Risk Factors (SHAP — {clinical_model_choice})")
    feat_names = [f[0] for f in top_feats]
    feat_vals  = [f[1] for f in top_feats]
    colors     = ["#e74c3c" if v > 0 else "#2ecc71" for v in feat_vals]
    fig_shap   = go.Figure(go.Bar(
        x=feat_vals, y=feat_names, orientation="h",
        marker_color=colors
    ))
    fig_shap.update_layout(
        xaxis_title="SHAP value (impact on risk)",
        height=250, margin=dict(t=20, b=0, l=20, r=20)
    )
    st.plotly_chart(fig_shap, use_container_width=True)

st.divider()

# ── Section 3: CNN-LSTM performance panel ─────────────────────────────
st.subheader("🧠 CNN-LSTM — AFib Detection (pre-registered hold-out)")

# Numbers below come from the pre-registered, leakage-safe, patient-grouped
# CPSC hold-out (n=1,376, Lead I). See docs/results/<run_id>/paper_tables.md.
if 'Combined' in cnn_label or 'CV' in cnn_label:
    auc_display    = "0.975"
    recall_display = "91.8%"
    f1_display     = "0.837"
    auc_delta      = "combined-data deployment model"
    caption_text   = (
        f"Loaded: {cnn_label}. Combined-data *deployment* model "
        "(CPSC 2018 + PhysioNet 2017) — higher because it trains on more, "
        "heterogeneous data, not a same-data comparison. On the same hold-out the "
        "device-agnostic RR+RF model scores AUC 0.935 — statistically "
        "indistinguishable from the CPSC-only CNN-LSTM (0.949; DeLong p=0.33)."
    )
    aw_note = "exploratory: 0.91 (CI 0.80–0.96)"
else:
    auc_display    = "0.949"
    recall_display = "95.9%"
    f1_display     = "0.701"
    auc_delta      = "leakage-safe CPSC hold-out"
    caption_text   = (
        f"Loaded: {cnn_label}. CPSC-only model on the pre-registered hold-out. "
        "⚠️ Domain gap: ~0.50 (chance) on Apple Watch — recovered only by adding "
        "wearable training data."
    )
    aw_note = "~0.50 (domain gap)"

p1, p2, p3, p4 = st.columns(4)
with p1:
    st.metric("AUC-ROC",  auc_display, delta=auc_delta, delta_color="off")
with p2:
    st.metric("Recall",   recall_display, delta="hold-out, thr 0.40", delta_color="off")
with p3:
    st.metric("Apple Watch", aw_note,
              delta="exploratory, 1 AFib case", delta_color="off")
with p4:
    st.metric("F1 Score",    f1_display)

st.caption(caption_text)
st.info(
    "⏱️ Detection-latency analysis is now reported as time from verified "
    "AFib onset to first alert, plus normal-phase false positives. "
    "The older 30-minute lead-time framing is intentionally retired."
)

st.divider()

# ── Section 4: ECG upload + inference ────────────────────────────────
st.subheader("📈 ECG Risk — Apple Watch AFib Detection")

ecg_model_choice = st.radio(
    "ECG Detection Method",
    ["CNN-LSTM (Deep Learning)", "RR Intervals (Traditional ML)"],
    horizontal=True,
    help=(
        "CNN-LSTM: learned features, high clinical accuracy. "
        "RR Intervals: timing-based, device-agnostic, validated on Apple Watch + MIT-BIH."
    )
)

if ecg_model_choice == "RR Intervals (Traditional ML)":
    st.info(
        "**RR Interval method** — device-agnostic heartbeat-timing features. "
        "Pre-registered CPSC hold-out AUC 0.935 (statistically indistinguishable "
        "from the deep model, DeLong p=0.33); zero-shot MIT-BIH Holter AUC 0.907; "
        "Apple Watch exploratory (49/54), no cross-device retraining."
    )
else:
    st.info(
        f"**CNN-LSTM method** ({cnn_label}). Pre-registered CPSC hold-out AUC "
        "0.949 (CPSC-only) / 0.975 (combined-data deployment). Apple Watch is "
        "exploratory: accuracy 0.91 (95% CI 0.80–0.96) on 54 recordings with one "
        "confirmed AFib case (device-classifier labels, not clinical adjudication)."
    )

if ecg_model_choice == "CNN-LSTM (Deep Learning)" and not cnn_ready:
    st.warning(
        "⚠️ No CNN-LSTM weights found. "
        "Run `python src/models/train_cnn_lstm_combined.py` first."
    )

if ecg_model_choice == "RR Intervals (Traditional ML)" and not rr_ready:
    st.warning(
        "⚠️ RR model not found. "
        "Run `python src/models/rr_afib_detector.py` first."
    )

uploaded = st.file_uploader(
    "Upload Apple Watch ECG export "
    "(Health app → Browse → Heart → Electrocardiograms → Export as CSV)",
    type="csv"
)

if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded, comment="#", header=None)
        signal = (pd.to_numeric(raw_df.iloc[:, 0], errors="coerce")
                    .dropna().values.astype(np.float32))

        st.info(f"Loaded {len(signal)} samples ({len(signal)/512:.1f}s at 512 Hz)")
        signal_mv = signal / 1000.0  # µV → mV

        if ecg_model_choice == "CNN-LSTM (Deep Learning)" and cnn_ready:
            from src.preprocessing.ecg_filter import bandpass_filter
            from scipy.signal import resample as scipy_resample

            # Match training preprocessing: the CNN-LSTM was trained on
            # 10-second (5000-sample) windows at 500 Hz. Apple Watch records
            # at 512 Hz, so resample first, then filter at 500 Hz.
            WINDOW_SAMPLES = 5000           # 10 s at 500 Hz
            n_500   = int(len(signal_mv) * 500 / 512)
            sig_500 = scipy_resample(signal_mv, n_500).astype(np.float32)
            filtered = bandpass_filter(sig_500, 0.5, 100.0, fs=500)

            # Segment into non-overlapping 10 s windows and score every one,
            # then average — the old code scored only the first 10 s and
            # discarded the rest of the recording.
            n_windows = max(1, len(filtered) // WINDOW_SAMPLES)
            st.success(
                f"Preprocessed into {n_windows} window(s) of 10 s "
                f"({len(sig_500)/500:.1f}s @ 500 Hz, resampled from 512 Hz)"
            )

            ecg_probs_list = []
            for i in range(n_windows):
                w = filtered[i * WINDOW_SAMPLES:(i + 1) * WINDOW_SAMPLES]
                w = w.astype(np.float32)
                w = np.clip(w, -2.0, 2.0)
                w = (w - w.mean()) / (w.std() + 1e-8)
                w = np.clip(w, -5.0, 5.0)
                if len(w) < WINDOW_SAMPLES:
                    w = np.pad(w, (0, WINDOW_SAMPLES - len(w)))
                x = torch.tensor(w).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    prob = torch.sigmoid(cnn_model(x).squeeze()).item()
                ecg_probs_list.append(prob)

            ecg_prob = float(np.mean(ecg_probs_list))

            if 'Combined' not in cnn_label and 'CV' not in cnn_label:
                st.caption(
                    "⚠️ Domain gap warning: CPSC-only CNN-LSTM may show "
                    "~0.50 uncertainty on Apple Watch data. "
                    "Use cnn_lstm_combined_best.pt for better results."
                )

        elif ecg_model_choice == "RR Intervals (Traditional ML)" and rr_ready:
            from scipy.signal import resample as scipy_resample
            from src.models.rr_afib_detector import (
                APPLE_WATCH_SKIP_SEC,
                extract_rr_features,
            )

            n_500   = int(len(signal_mv) * 500 / 512)
            sig_500 = scipy_resample(signal_mv, n_500).astype(np.float32)
            sig_500 = sig_500[int(APPLE_WATCH_SKIP_SEC * 500):]
            feats   = extract_rr_features(sig_500, fs=500)

            if feats is None:
                st.warning(
                    "Could not detect enough R peaks for RR analysis. "
                    "Try a longer recording (minimum ~15 seconds)."
                )
            else:
                feat_vec = pd.DataFrame(
                    [{k: feats.get(k, 0) for k in rr_feat_names}]
                )
                ecg_prob = float(rr_model.predict_proba(feat_vec)[0, 1])
                if feats.get("rr_cv", 0) < 0.15:
                    ecg_prob = min(ecg_prob, 0.25)

                rr_col1, rr_col2, rr_col3, rr_col4 = st.columns(4)
                with rr_col1:
                    st.metric("Heart Rate", f"{feats['heart_rate']:.0f} bpm")
                with rr_col2:
                    cv = feats["rr_cv"]
                    st.metric(
                        "RR Coeff. of Variation", f"{cv:.3f}",
                        delta="AFib likely" if cv > 0.15 else "Normal",
                        delta_color="inverse" if cv > 0.15 else "normal"
                    )
                with rr_col3:
                    st.metric("RMSSD", f"{feats['rr_rmssd']:.1f} ms")
                with rr_col4:
                    st.metric("pNN50", f"{feats['rr_pnn50']:.1%}")

                st.caption(
                    "RR CV > 0.15 = AFib criterion (clinical threshold). "
                    "Device-agnostic — validated across Apple Watch (512 Hz), "
                    "CPSC (500 Hz), MIT-BIH (250 Hz)."
                )

        # ── Fused score ───────────────────────────────────────────────
        if ecg_prob is not None:
            fused_prob, fusion_method = compute_fused_score(clinical_prob, ecg_prob)

            ecg_col1, ecg_col2, ecg_col3 = st.columns(3)
            method_label = "CNN-LSTM" if "CNN" in ecg_model_choice else "RR+RF"

            with ecg_col1:
                st.metric(
                    f"🫀 AFib Probability ({method_label})",
                    f"{ecg_prob:.1%}",
                    delta="⚠️ AFib detected" if ecg_prob >= 0.5 else "✅ No AFib"
                )
            with ecg_col2:
                st.metric(
                    "⚡ Fused Score (exploratory)", f"{fused_prob:.1%}",
                    delta="⚠️ HIGH" if fused_prob >= THRESHOLD else "✅ Normal"
                )
            with ecg_col3:
                st.caption("Fusion method")
                st.caption(fusion_method)

            if ecg_prob >= 0.5:
                st.error(
                    "🚨 AFib pattern detected in ECG — please consult a cardiologist."
                )
            else:
                st.success("✅ No AFib pattern detected in ECG.")

    except Exception as e:
        st.error(f"Error processing ECG file: {e}")
        st.caption("Make sure the file is a valid Apple Watch ECG CSV export.")

st.divider()

# ── Section 5: Rolling risk history ──────────────────────────────────
st.subheader("📈 Rolling Risk History (last 30 readings)")

st.session_state.risk_history.append(round(clinical_prob, 4))
if fused_prob is not None:
    st.session_state.fused_risk_history.append(round(fused_prob, 4))

fig_roll = go.Figure()
fig_roll.add_trace(go.Scatter(
    y=list(st.session_state.risk_history),
    mode="lines+markers",
    line=dict(color=model_color, width=2),
    name=f"Clinical risk ({clinical_model_choice})"
))
if len(st.session_state.fused_risk_history) > 0:
    fig_roll.add_trace(go.Scatter(
        y=list(st.session_state.fused_risk_history),
        mode="lines+markers",
        line=dict(color="crimson", width=2, dash="dot"),
        name=f"Fused risk (Clinical + ECG | {fusion_label})"
    ))
fig_roll.add_hline(
    y=THRESHOLD, line_dash="dash", line_color="red",
    annotation_text=f"Alert threshold ({THRESHOLD:.0%})"
)
fig_roll.update_layout(
    yaxis=dict(range=[0, 1], title="Risk probability"),
    xaxis_title="Reading #",
    height=300,
    margin=dict(t=20, b=40, l=40, r=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
st.plotly_chart(fig_roll, use_container_width=True)

st.divider()

# ── Section 6: Alert history log ─────────────────────────────────────
st.subheader("🚨 Alert History")

if alert or (fused_prob is not None and fused_prob >= THRESHOLD):
    st.session_state.alert_log.append({
        "Time":           pd.Timestamp.now().strftime("%H:%M:%S"),
        "Model":          clinical_model_choice,
        "Clinical Score": f"{clinical_prob:.1%}",
        "ECG Score":      f"{ecg_prob:.1%}" if ecg_prob is not None else "—",
        "Fused Score":    f"{fused_prob:.1%}" if fused_prob is not None else "—",
        "Fusion Method":  fusion_label,
        "Trigger":        "🚨 Clinical" if alert else "⚡ Fused",
    })

if st.session_state.alert_log:
    alert_df = pd.DataFrame(st.session_state.alert_log)
    st.dataframe(alert_df, use_container_width=True)
    if st.button("Clear alert log"):
        st.session_state.alert_log = []
        st.rerun()
else:
    st.success("✅ No alerts triggered this session.")

st.divider()

# ── Section 7: Debug expander ─────────────────────────────────────────
with st.expander("🔍 Debug — raw feature vector"):
    st.dataframe(patient_df)
    st.caption(f"Clinical probability ({clinical_model_choice}): {clinical_prob:.4f}")
    st.caption(f"Alert threshold: {THRESHOLD}")
    st.caption(f"CNN-LSTM weights loaded: {cnn_label}")
    st.caption(f"Fusion method: {fusion_label}")
    if ecg_prob is not None:
        st.caption(f"ECG probability: {ecg_prob:.4f}")
        st.caption(f"Fused probability: {fused_prob:.4f}")
