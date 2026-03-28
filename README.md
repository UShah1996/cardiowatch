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

The system combines structured clinical data with ECG time-series signals, processed through a multi-modal pipeline (Random Forest + CNN-LSTM), and surfaced through an Apple Watch-style Streamlit risk dashboard with SHAP explainability.

---

## Project Structure

```
cardiowatch/
├── data/
│   ├── raw/               # Downloaded datasets (gitignored)
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
| PhysioNet Challenge 2020 (CPSC subset) | [PhysioNet](https://physionet.org/content/challenge-2020/) | 6,877 ECG recordings | CNN-LSTM temporal modeling |

---

## Models

| Model | Input | Status | Key Results |
|---|---|---|---|
| Random Forest | Clinical features (19) | ✅ Complete | Recall 0.887, AUC-ROC 0.938 |
| XGBoost | Clinical features (19) | 🔲 Planned | — |
| CNN-LSTM | ECG time-series (150,000 samples/window) | ✅ Architecture complete | Output: torch.Size([4,1]), 254k params |

---

## ✅ What Is Built (Weeks 1–6b, as of Mar 30 2026)

### Weeks 1–2: Environment & Setup
- Python 3.9 virtual environment with all dependencies installed (`torch`, `sklearn`, `wfdb`, `shap`, `streamlit`, `plotly`, etc.)
- GitHub repo initialized and connected: [github.com/UShah1996/cardiowatch](https://github.com/UShah1996/cardiowatch)
- Kaggle API configured, PhysioNet account approved
- Literature review completed: Jin et al. (2022), Chadaga (2025), AHA (2025), Salet et al. (2024)

### Week 3: Clinical EDA
- Kaggle Heart Failure dataset downloaded (`data/raw/heart.csv`, 918 × 12)
- EDA notebook `01_eda_clinical.ipynb` with 6 cells covering: missing values, zero-cholesterol detection (172 affected rows), class balance, feature distributions, correlation heatmap, and outlier analysis

### Week 4: Clinical Preprocessing Pipeline
- `src/preprocessing/clinical.py` — median imputation for zero-cholesterol, binary encoding (Sex, ExerciseAngina), one-hot encoding (ChestPainType, RestingECG, ST_Slope), MinMaxScaler normalization, stratified 80/10/10 train/val/test split → **Train: 734 | Val: 92 | Test: 92**
- `src/preprocessing/smote_balance.py` — SMOTE applied to training set → **{0: 406, 1: 406}** balanced classes
- `configs/config.yaml` — all hyperparameters, paths, and model settings in one place

### Week 5: ECG Signal Pipeline
- `src/preprocessing/ecg_filter.py` — Butterworth band-pass filter (0.5–100 Hz), Lead I extraction by name lookup, 5-minute window segmentation at 500 Hz → **Windows shape: (2, 150000)**
- PhysioNet Challenge 2020 CPSC subset downloaded (6,877 recordings via Kaggle mirror)
- `02_eda_ecg_signals.ipynb` — raw vs filtered Lead I signal plot, signal quality stats across 20 recordings, `docs/ecg_raw_vs_filtered.png` saved

### Week 6a: Model Architecture
- `src/models/cnn_lstm.py` — CNN-LSTM in PyTorch: two Conv1d blocks `[32, 64]` with BatchNorm + MaxPool, 2-layer LSTM (hidden=128), classification head with Sigmoid output. **Output: torch.Size([4, 1]) ✅ | Parameters: 254,593**
- `src/models/random_forest.py` — RandomForestClassifier with 5-fold stratified CV. **Recall: 0.887 ± 0.041 | F1: 0.871 ± 0.013 | AUC-ROC: 0.938 ± 0.010**
- `src/evaluation/metrics.py` — `evaluate_model()` with configurable threshold, prints confusion matrix + classification report

### Week 6b: Streamlit Dashboard
- `src/evaluation/shap_explainer.py` — `build_explainer()`, `get_shap_values()`, `top_features()` using SHAP TreeExplainer for Random Forest
- `src/dashboard/app.py` — full Apple Watch-style Streamlit dashboard:
  - **10 patient profile sliders** (Age, BP, Cholesterol, MaxHR, Oldpeak, Sex, ExerciseAngina, ChestPainType, RestingECG, ST_Slope)
  - **Live risk gauge** (green/yellow/red zones, alert banner above 50% threshold)
  - **SHAP bar chart** — top 6 features, red = risk-increasing, green = risk-decreasing
  - **Rolling 30-reading risk history chart** with alert threshold line
  - Model cached with `@st.cache_resource` — loads once, updates on every slider change

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/UShah1996/cardiowatch.git
cd cardiowatch

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run clinical pipeline
python3 src/preprocessing/clinical.py

# 5. Run Random Forest baseline
python3 src/models/random_forest.py

# 6. Validate CNN-LSTM architecture
python3 src/models/cnn_lstm.py

# 7. Launch dashboard
streamlit run src/dashboard/app.py
```

---

## Validation Checklist

| Week | Command | Expected Result |
|---|---|---|
| Wk 1–2: Packages | `python3 -c "import torch, pandas, sklearn, wfdb"` | No errors |
| Wk 1–2: Git | `git log --oneline` | Shows all commits |
| Wk 3: Dataset | `python3 -c "import pandas as pd; df=pd.read_csv('data/raw/heart.csv'); print(df.shape)"` | `(918, 12)` |
| Wk 4: Pipeline | `python3 src/preprocessing/clinical.py` | Train: 734, Val: 92, Test: 92 |
| Wk 4: SMOTE | `python3 src/preprocessing/smote_balance.py` | After SMOTE: {0: 406, 1: 406} |
| Wk 5: ECG filter | `python3 src/preprocessing/ecg_filter.py` | Windows shape: (2, 150000) |
| Wk 6a: CNN-LSTM | `python3 src/models/cnn_lstm.py` | Output shape: torch.Size([4, 1]) |
| Wk 6a: RF baseline | `python3 src/models/random_forest.py` | Recall ≥ 0.85, AUC ≥ 0.88 |
| Wk 6b: Dashboard | `streamlit run src/dashboard/app.py` | Opens at localhost:8501 |
| Wk 6b: SHAP | `python3 -c "import shap; print(shap.__version__)"` | 0.45.x |

---

## 🗓️ Next 4 Weeks — Upcoming Plan (Apr–May 2026)

### Week 7: XGBoost Baseline + Threshold Tuning
- Implement `src/models/xgboost_model.py` with `scale_pos_weight` for class imbalance
- Compare XGBoost vs Random Forest on Recall, F1, AUC-ROC on held-out test set
- Tune decision threshold on both models to push Recall to **≥ 0.90** (target from proposal)
- Add `evaluate_model()` calls with threshold=0.40 sweep in `src/evaluation/metrics.py`
- Run final test-set evaluation and log results to `docs/baseline_results.md`

### Week 8: CNN-LSTM Training on ECG Data
- Build `src/preprocessing/ecg_dataset.py` — PyTorch `Dataset` class for windowed ECG `.npz` files
- Train CNN-LSTM on CPSC subset with labels from PhysioNet header files
- Track train/val loss and recall per epoch using MLflow (`mlflow ui`)
- Target: val recall ≥ 0.85 on ECG-only classification
- Save best checkpoint to `data/processed/cnn_lstm_best.pt`
- Add noise augmentation (Gaussian noise injection) to improve robustness on wearable-quality signals

### Week 9: Multi-Modal Fusion + Lead-Time Evaluation
- Combine Random Forest risk score + CNN-LSTM ECG risk score into a fusion layer
- Implement `src/evaluation/lead_time.py` — measure how many minutes before a simulated event the fused score crosses the alert threshold
- Target: **≥ 30-minute lead time** (core proposal requirement)
- Simulate Apple Watch-style streaming in `data/simulated/` using HealthKit reference patterns

### Week 10: Final Evaluation, SHAP Audit & Report
- Full evaluation on held-out test set across all three models
- SHAP summary plots for Random Forest — beeswarm and bar charts saved to `docs/`
- Robustness testing: evaluate on noisy ECG inputs (motion artifact simulation)
- 5-fold stratified cross-validation final pass with confusion matrices
- Write final project report and update dashboard with all three model comparisons
- Deploy dashboard to Streamlit Cloud for live demo link in README

---

## Evaluation Targets

| Metric | Target | Current Status |
|---|---|---|
| Recall (Sensitivity) | ≥ 90% | 88.7% (RF baseline, close) |
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
