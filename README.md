# 🫀 CardioWatch
### Early Detection & Short-Term Risk Prediction of Heart Attacks Using ECG and Clinical Data

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## Overview

CardioWatch is an ML research project that explores whether **temporal patterns in cardiovascular data** can be used for the early detection of heart disease. Rather than predicting *whether* a patient has heart disease, this system aims to estimate **when** a cardiac event might be approaching — providing an early warning window to improve treatment outcomes.

The system combines structured clinical data with ECG time-series signals, processed through a multi-modal pipeline (Random Forest + CNN-LSTM), and surfaced through an Apple Watch-style Streamlit risk dashboard with SHAP explainability. The project is built around the Lead I ECG signal that Apple Watch already records — making real-world wearable deployment a realistic future step.

---

## Project Structure

```
cardiowatch/
├── data/
│   ├── raw/               # Downloaded datasets (gitignored — see Data Setup below)
│   ├── processed/         # Cleaned, windowed data (gitignored)
│   └── simulated/         # Synthetic HealthKit streams (gitignored)
├── src/
│   ├── preprocessing/
│   │   ├── clinical.py           # Imputation, encoding, normalization, train/val/test split
│   │   ├── ecg_filter.py         # Band-pass filtering (0.5–100 Hz), Lead I extraction, windowing
│   │   └── smote_balance.py      # SMOTE class imbalance handling
│   ├── models/
│   │   ├── random_forest.py      # RF baseline with 5-fold CV
│   │   ├── xgboost_model.py      # XGBoost baseline (stub)
│   │   └── cnn_lstm.py           # CNN-LSTM temporal ECG model
│   ├── evaluation/
│   │   ├── metrics.py            # Recall, AUC-ROC, F1, confusion matrix
│   │   └── shap_explainer.py     # SHAP TreeExplainer for RF
│   └── dashboard/
│       └── app.py                # Streamlit Apple Watch risk simulator
├── notebooks/
│   ├── 01_eda_clinical.ipynb     # Clinical dataset EDA
│   └── 02_eda_ecg_signals.ipynb  # ECG signal EDA, raw vs filtered plots
├── configs/
│   └── config.yaml               # Hyperparameters & paths
├── docs/
│   └── ecg_raw_vs_filtered.png   # ECG filter output visualization
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Source | Size | Purpose |
|---|---|---|---|
| Heart Failure Prediction | [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) | 918 samples, 11 features | Baseline risk modeling |
| PhysioNet Challenge 2020 (CPSC subset) | [Kaggle mirror](https://www.kaggle.com/datasets/gamalasran/physionet-challenge-2020) | 6,877 ECG recordings | CNN-LSTM temporal modeling |

> **Note:** Raw data files are excluded from this repository (gitignored). See the [Data Setup](#data-setup) section below for download instructions.

---

## Models

| Model | Input | Status | Key Results |
|---|---|---|---|
| Random Forest | Clinical features (19) | ✅ Complete | Recall 0.887, AUC-ROC 0.938 |
| XGBoost | Clinical features (19) | 🔲 Planned — Week 7 | — |
| CNN-LSTM | ECG time-series (150,000 samples/window) | ✅ Architecture complete | Output: torch.Size([4,1]), 254k params |

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- A [Kaggle account](https://www.kaggle.com) with API credentials configured (required for data download)

### 1. Clone and set up environment

```bash
git clone https://github.com/UShah1996/cardiowatch.git
cd cardiowatch

python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

pip3 install -r requirements.txt
```

### 2. Data Setup

The datasets are not included in the repo. Download them using the Kaggle API.

**Step 1 — Configure Kaggle credentials** (skip if already done):
```bash
# Place your kaggle.json from kaggle.com/settings/api into:
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

**Step 2 — Download both datasets:**
```bash
# Clinical dataset (small, ~50 KB)
kaggle datasets download fedesoriano/heart-failure-prediction -p data/raw --unzip

# ECG dataset — CPSC subset via Kaggle mirror (~3 GB, takes several minutes)
kaggle datasets download gamalasran/physionet-challenge-2020 -p data/raw --unzip
```

### 3. Run the pipelines

```bash
# Clinical preprocessing pipeline
python3 src/preprocessing/clinical.py
# Expected: Train: 734 | Val: 92 | Test: 92

# Random Forest baseline (5-fold CV)
python3 src/models/random_forest.py
# Expected: Recall ~0.887, AUC-ROC ~0.938

# Validate CNN-LSTM architecture
python3 src/models/cnn_lstm.py
# Expected: Output shape: torch.Size([4, 1])

# Launch Streamlit dashboard
streamlit run src/dashboard/app.py
# Opens at http://localhost:8501
```

---

## Validation Checklist

Run these to confirm everything is working after setup:

| Component | Command | Expected Result |
|---|---|---|
| Packages | `python3 -c "import torch, pandas, sklearn, wfdb, shap, streamlit"` | No errors |
| Dataset | `python3 -c "import pandas as pd; df=pd.read_csv('data/raw/heart.csv'); print(df.shape)"` | `(918, 12)` |
| Clinical pipeline | `python3 src/preprocessing/clinical.py` | Train: 734, Val: 92, Test: 92 |
| SMOTE | `python3 src/preprocessing/smote_balance.py` | After SMOTE: {0: 406, 1: 406} |
| ECG filter | `python3 src/preprocessing/ecg_filter.py` | Windows shape: (2, 150000) |
| CNN-LSTM | `python3 src/models/cnn_lstm.py` | Output shape: torch.Size([4, 1]) |
| RF baseline | `python3 src/models/random_forest.py` | Recall ≥ 0.85, AUC ≥ 0.88 |
| Dashboard | `streamlit run src/dashboard/app.py` | Opens at localhost:8501 |

---

## What Is Built (Weeks 1–6b, as of Mar 30, 2026)

### Weeks 1–2: Environment & Setup
- Python 3.9 virtual environment with all dependencies installed
- GitHub repo initialized, Kaggle API configured, PhysioNet account approved
- Literature review completed: Jin et al. (2022), Chadaga (2025), AHA (2025), Salet et al. (2024)

### Week 3: Clinical EDA
- Kaggle Heart Failure dataset downloaded (`data/raw/heart.csv`, 918 × 12)
- EDA notebook `01_eda_clinical.ipynb`: missing values, zero-cholesterol detection (172 affected rows), class balance (508 vs 410), feature distributions, correlation heatmap, outlier analysis

### Week 4: Clinical Preprocessing Pipeline
- `src/preprocessing/clinical.py` — median imputation, binary + one-hot encoding, MinMaxScaler, stratified 80/10/10 split → **Train: 734 | Val: 92 | Test: 92**
- `src/preprocessing/smote_balance.py` — SMOTE on training set → **{0: 406, 1: 406}**
- `configs/config.yaml` — all hyperparameters centralized

### Week 5: ECG Signal Pipeline
- `src/preprocessing/ecg_filter.py` — Butterworth band-pass filter (0.5–100 Hz), Lead I extraction, 5-min windowing → **Windows shape: (2, 150000)**
- CPSC subset downloaded (6,877 recordings)
- `02_eda_ecg_signals.ipynb` — raw vs filtered ECG plot, signal quality stats, `docs/ecg_raw_vs_filtered.png`

### Week 6a: Model Architecture
- `src/models/cnn_lstm.py` — CNN-LSTM: Conv1d blocks [32, 64] + BatchNorm + MaxPool, 2-layer LSTM (hidden=128), Sigmoid output. **Output: torch.Size([4, 1]) | Parameters: 254,593**
- `src/models/random_forest.py` — 5-fold CV. **Recall: 0.887 ± 0.041 | F1: 0.871 ± 0.013 | AUC-ROC: 0.938 ± 0.010**
- `src/evaluation/metrics.py` — `evaluate_model()` with configurable threshold, confusion matrix + classification report

### Week 6b: Streamlit Dashboard
- `src/evaluation/shap_explainer.py` — SHAP TreeExplainer: `build_explainer()`, `get_shap_values()`, `top_features()`
- `src/dashboard/app.py` — Apple Watch-style dashboard:
  - 10 patient profile sliders (Age, BP, Cholesterol, MaxHR, Oldpeak, Sex, ExerciseAngina, ChestPainType, RestingECG, ST_Slope)
  - Live risk gauge (green/yellow/red zones, alert banner above threshold)
  - SHAP bar chart — top 6 features, red = risk-increasing, green = risk-decreasing
  - Rolling 30-reading risk history chart with threshold line
  - Model cached with `@st.cache_resource`

---

### Week 7: XGBoost Baseline + Threshold Tuning
- Implement `src/models/xgboost_model.py` with `scale_pos_weight` for class imbalance
- Compare XGBoost vs Random Forest on Recall, F1, AUC-ROC on held-out test set
- Tune decision threshold on both models to push Recall to **≥ 0.93**
- Log results to `docs/baseline_results.md`

### Week 8: CNN-LSTM Training on ECG Data
- Build `src/preprocessing/ecg_dataset.py` — PyTorch `Dataset` for windowed ECG files
- Train CNN-LSTM on CPSC subset, track with MLflow
- Add Gaussian noise augmentation for wearable-quality robustness
- Save best checkpoint to `data/processed/cnn_lstm_best.pt`

### Week 9: Multi-Modal Fusion + Lead-Time Evaluation
- Combine RF + CNN-LSTM scores into a fusion layer
- Implement `src/evaluation/lead_time.py` — measure advance warning before simulated events
- Target: **≥ 30-minute lead time**
- Simulate Apple Watch-style data streams in `data/simulated/`

### Week 10: Apple Watch Integration Path + Final Report
- Build pipeline to read Apple Watch ECG exports directly (Health app → Export as CSV, 512 Hz)
- The same band-pass filter and windowing code handles Apple Watch CSVs with no modification, since Apple Watch records Lead I — the same signal the model was trained on
- Update dashboard to accept an uploaded Apple Watch CSV and show a real risk score
- Full real-time WatchOS streaming is out of scope for this semester (requires Apple Developer account + Swift app), but the export-based path is testable with a real device
- SHAP summary plots, robustness tests, final cross-validation pass, Streamlit Cloud deployment

---

## Evaluation Targets

| Metric | Target | Current Status |
|---|---|---|
| Recall (Sensitivity) | ≥ 93% | 88.7% (RF baseline — threshold tuning planned) |
| Lead-Time Warning | ≥ 30 minutes | Planned — Week 9 |
| AUC-ROC | Maximize | 0.938 (RF baseline) ✅ |
| F1-Score | Maximize | 0.871 (RF baseline) |
| Cross-Validation | 5-fold stratified | ✅ Implemented |

---

## References

1. World Health Organization — Cardiovascular Diseases Fact Sheet (2021)
2. Soriano — Heart Failure Prediction Dataset, Kaggle (2021)
3. Asran — PhysioNet Challenge 2020: Classification of 12-lead ECGs, Kaggle
4. American Heart Association — AI tool detected structural heart disease using a smartwatch (2025)
5. Michigan Medicine — Daylight Saving Time and Heart Attack Risk (2017)
6. Apple Inc. — HealthKit Framework Documentation (2024)
7. Salet et al. — Predicting Myocardial Infarction in Primary Care, PLoS ONE (2024)
8. Jin et al. — Transfer learning for single-lead ECG myocardial injury prediction, JAMIA (2022)
9. Chadaga — Predicting heart attack using time series data, Mendeley Data (2025)
