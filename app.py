"""
src/dashboard/app.py

CardioWatch Streamlit Dashboard
Simulates an Apple Watch-style real-time cardiac risk feed.
Run with: streamlit run src/dashboard/app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="CardioWatch",
    page_icon="🫀",
    layout="wide",
)

# ── Header ─────────────────────────────────────────────────
st.markdown("# 🫀 CardioWatch")
st.markdown("**Real-Time Cardiac Risk Monitoring** — Apple Watch Simulator")
st.divider()

# ── Sidebar: Patient Profile ────────────────────────────────
with st.sidebar:
    st.header("Patient Profile")
    age = st.slider("Age", 30, 80, 55)
    cholesterol = st.slider("Cholesterol (mg/dL)", 150, 400, 220)
    bp = st.slider("Resting BP (mmHg)", 90, 200, 130)
    st.divider()
    st.header("Monitoring")
    refresh = st.slider("Refresh interval (s)", 2, 10, 5)
    risk_threshold = st.slider("Alert threshold", 0.5, 0.95, 0.70)
    running = st.toggle("Live Monitoring", value=True)
    st.divider()
    st.caption("CardioWatch v0.1 | Research prototype")

# ── Layout: Metrics + Chart ─────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

# Simulate a risk score (replace with real model inference)
def simulate_risk(age, cholesterol, bp):
    base = (age / 80 * 0.4) + (cholesterol / 400 * 0.3) + (bp / 200 * 0.3)
    noise = np.random.normal(0, 0.05)
    return float(np.clip(base + noise, 0, 1))

def risk_color(score):
    if score >= 0.70:
        return "🔴", "High"
    elif score >= 0.45:
        return "🟡", "Moderate"
    return "🟢", "Low"

# ── Persistent state for rolling chart ─────────────────────
if "history" not in st.session_state:
    st.session_state.history = {
        "times": [],
        "scores": [],
    }

placeholder = st.empty()

# ── Main Loop ───────────────────────────────────────────────
for _ in range(200 if running else 1):
    score = simulate_risk(age, cholesterol, bp)
    icon, label = risk_color(score)
    now = datetime.now()

    st.session_state.history["times"].append(now)
    st.session_state.history["scores"].append(score)

    # Keep rolling window of last 60 readings
    if len(st.session_state.history["times"]) > 60:
        st.session_state.history["times"] = st.session_state.history["times"][-60:]
        st.session_state.history["scores"] = st.session_state.history["scores"][-60:]

    with placeholder.container():
        # ── KPI Cards ──────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk Score", f"{score:.2f}", label_visibility="visible")
        c2.metric("Risk Level", f"{icon} {label}")
        c3.metric("Age", age)
        c4.metric("Cholesterol", f"{cholesterol} mg/dL")

        # ── Alert Banner ───────────────────────────────────
        if score >= risk_threshold:
            st.error(f"⚠️ **High Risk Detected** — Score {score:.2f} exceeds threshold {risk_threshold:.2f}. Consult a cardiologist.")
        else:
            st.success(f"✅ Risk within acceptable range ({score:.2f})")

        # ── Rolling Risk Chart ─────────────────────────────
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=st.session_state.history["times"],
            y=st.session_state.history["scores"],
            mode="lines+markers",
            line=dict(color="#e74c3c", width=2),
            marker=dict(size=4),
            name="Risk Score",
        ))

        fig.add_hline(
            y=risk_threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Alert threshold ({risk_threshold:.2f})",
        )

        fig.update_layout(
            title="Cardiac Risk Trend (Live)",
            xaxis_title="Time",
            yaxis_title="Risk Score",
            yaxis=dict(range=[0, 1]),
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── SHAP Placeholder ───────────────────────────────
        st.subheader("🔍 Feature Contributions (SHAP)")
        shap_vals = {
            "Age": round(age / 80 * 0.4, 3),
            "Cholesterol": round(cholesterol / 400 * 0.3, 3),
            "Blood Pressure": round(bp / 200 * 0.3, 3),
            "ECG Signal (CNN-LSTM)": round(np.random.uniform(0.05, 0.15), 3),
        }
        shap_fig = go.Figure(go.Bar(
            x=list(shap_vals.values()),
            y=list(shap_vals.keys()),
            orientation="h",
            marker_color=["#e74c3c" if v > 0.1 else "#3498db" for v in shap_vals.values()],
        ))
        shap_fig.update_layout(
            xaxis_title="SHAP Value (contribution to risk)",
            height=220,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(shap_fig, use_container_width=True)

        st.caption(f"Last updated: {now.strftime('%H:%M:%S')} | Refresh: {refresh}s")

    if running:
        time.sleep(refresh)
    else:
        break
