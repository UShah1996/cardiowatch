# 🫀 CardioWatch
### Early Detection & Short-Term Risk Prediction of Atrial Fibrillation Using ECG and Clinical Data

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## Overview

CardioWatch is an ML research project that explores whether **temporal patterns in cardiovascular data** can be used for the early detection of heart disease. Rather than predicting *whether* a patient has heart disease, this system aims to estimate **when** a cardiac event might be approaching — providing an early warning window to improve treatment outcomes.

The system combines structured clinical data with ECG time-series signals, processed through a multi-modal pipeline (Random Forest + CNN-LSTM), and surfaced through an Apple Watch-style Streamlit risk dashboard with SHAP explainability.

The ECG component is specifically designed around **Atrial Fibrillation (AFib) detection using Lead I only** — the same single-lead signal that Apple Watch Series 4+ already records. This makes real-world wearable deployment a realistic next step rather than an afterthought.

---

## Project Structure

```
cardiowatch/
├── data/
│   ├── raw/               # Downloaded datasets (gitignored — see Data Setup below)
│   ├── processed/         # Cleaned, windowed data, best model checkpoint (gitignored)
│   └── simulated/         # Synthetic HealthKit streams (gitignored)
├── src/
│   ├── preprocessing/
│   │   ├── clinical.py           # Imputation, encoding, normalization, train/val/test split
│   │   ├── ecg_dataset.py        # PyTorch Dataset for CPSC ECG recordings — AFib labels
│   │   ├── ecg_filter.py         # Band-pass filtering (0.5–100 Hz), Lead I extraction, windowing
│   │   └── smote_balance.py      # SMOTE class imbalance handling
│   ├── models/
│   │   ├── random_forest.py      # RF baseline with 5-fold CV
│   │   ├── xgboost_model.py      # XGBoost baseline (planned)
│   │   ├── cnn_lstm.py           # CNN-LSTM temporal ECG model
│   │   └── train_cnn_lstm.py     # CNN-LSTM training loop with MLflow tracking
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
| Heart Failure Prediction | [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) | 918 samples, 11 features | Baseline clinical risk modeling |
| PhysioNet Challenge 2020 (CPSC subset) | [Kaggle mirror](https://www.kaggle.com/datasets/gamalasran/physionet-challenge-2020) | 6,877 ECG recordings | CNN-LSTM AFib detection |

> **Note:** Raw data files are excluded from this repository (gitignored). See [Data Setup](#2-data-setup) below.

---

## Models

| Model | Input | Status | Key Results |
|---|---|---|---|
| Random Forest | Clinical features (19) | ✅ Complete | Recall 0.887, AUC-ROC 0.938 |
| XGBoost | Clinical features (19) | 🔲 Planned — Week 7 | — |
| CNN-LSTM | Lead I ECG, 5000 samples (10s @ 500Hz) | ✅ Trained | **AUC-ROC 0.968, Recall 0.931, F1 0.844** |

### CNN-LSTM Architecture

| Layer | Detail |
|---|---|
| Input | `(batch, 1, 5000)` — single Lead I channel, 10 seconds at 500 Hz |
| Conv Block 1 | Conv1d(1→32, kernel=7) + BatchNorm + ReLU + MaxPool(2) → length 2500 |
| Conv Block 2 | Conv1d(32→64, kernel=7) + BatchNorm + ReLU + MaxPool(2) → length 1250 |
| Conv Block 3 | Conv1d(64→128, kernel=7) + BatchNorm + ReLU + MaxPool(2) → length 625 |
| LSTM | 2-layer, hidden=128, batch_first=True |
| Classifier | Dropout(0.3) → Linear(128→64) → ReLU → Linear(64→1) |
| Parameters | 254,593 trainable |

---

## Why AFib and Why Lead I?

Atrial Fibrillation is the world's most common serious arrhythmia, affecting ~37 million people and being a leading cause of stroke. Unlike most arrhythmias, AFib has two unmistakable electrical signatures visible in any single lead — absent P waves and irregular R-R intervals — making it reliably detectable from Lead I alone.

Apple Watch received FDA clearance for AFib detection in 2018 using its single-lead ECG sensor. CardioWatch is trained on the same signal type (Lead I, 500 Hz), making the path from research model to wearable deployment direct and realistic. No architectural changes are needed to accept real Apple Watch ECG exports.

---

## Quick Start

### Prerequisites

- Python 3.9+
- Apple M1/M2/M3 recommended (MPS GPU acceleration enabled automatically)
- [Kaggle account](https://www.kaggle.com) with API credentials configured

### 1. Clone and Set Up Environment

```bash
git clone https://github.com/UShah1996/cardiowatch.git
cd cardiowatch

python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

pip3 install -r requirements.txt
```

### 2. Data Setup

```bash
# Configure Kaggle credentials (skip if already done)
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Clinical dataset (~50 KB)
kaggle datasets download fedesoriano/heart-failure-prediction -p data/raw --unzip

# ECG dataset — CPSC subset (~3 GB, takes several minutes)
kaggle datasets download gamalasran/physionet-challenge-2020 -p data/raw --unzip
```

### 3. Run the Pipelines

```bash
# Clinical preprocessing
python3 src/preprocessing/clinical.py
# Expected: Train: 734 | Val: 92 | Test: 92

# Random Forest baseline (5-fold CV)
python3 src/models/random_forest.py
# Expected: Recall ~0.887, AUC-ROC ~0.938

# Validate CNN-LSTM architecture
python3 src/models/cnn_lstm.py
# Expected: Output shape: torch.Size([4, 1]), 254,593 params

# Train CNN-LSTM AFib detector (M1 GPU accelerated)
python3 src/models/train_cnn_lstm.py
# Expected: AUC-ROC ~0.968 by epoch 28 | Using device: mps

# Launch Streamlit dashboard
streamlit run src/dashboard/app.py
# Opens at http://localhost:8501

# View MLflow training curves
mlflow ui
# Opens at http://localhost:5000
```

---

## Validation Checklist

| Component | Command | Expected Result |
|---|---|---|
| Packages | `python3 -c "import torch, pandas, sklearn, wfdb, shap, streamlit"` | No errors |
| Dataset | `python3 -c "import pandas as pd; df=pd.read_csv('data/raw/heart.csv'); print(df.shape)"` | `(918, 12)` |
| Clinical pipeline | `python3 src/preprocessing/clinical.py` | Train: 734, Val: 92, Test: 92 |
| SMOTE | `python3 src/preprocessing/smote_balance.py` | After SMOTE: {0: 406, 1: 406} |
| ECG dataset | `python3 src/preprocessing/ecg_dataset.py` | Loaded 6877 recordings, AFib: 1221 |
| CNN-LSTM arch | `python3 src/models/cnn_lstm.py` | Output shape: torch.Size([4, 1]) |
| CNN-LSTM train | `python3 src/models/train_cnn_lstm.py` | AUC-ROC ≥ 0.95 by epoch 20 |
| RF baseline | `python3 src/models/random_forest.py` | Recall ≥ 0.85, AUC ≥ 0.88 |
| Dashboard | `streamlit run src/dashboard/app.py` | Opens at localhost:8501 |

---

## What Is Built

### Weeks 1–2: Environment & Setup
- Python 3.9 virtual environment with all dependencies installed (`torch`, `sklearn`, `wfdb`, `shap`, `streamlit`, `mlflow`, etc.)
- GitHub repo initialized, Kaggle API configured, PhysioNet account approved
- Literature review: Jin et al. (2022), Chadaga (2025), AHA (2025), Salet et al. (2024)

### Week 3: Clinical EDA
- Kaggle Heart Failure dataset downloaded (`data/raw/heart.csv`, 918 × 12)
- EDA notebook `01_eda_clinical.ipynb`: missing values, zero-cholesterol detection (172 affected rows), class balance (508 vs 410), feature distributions, correlation heatmap, outlier analysis

### Week 4: Clinical Preprocessing Pipeline
- `src/preprocessing/clinical.py` — median imputation for zero-cholesterol, binary encoding (Sex, ExerciseAngina), one-hot encoding (ChestPainType, RestingECG, ST_Slope), MinMaxScaler normalization, stratified 80/10/10 split → **Train: 734 | Val: 92 | Test: 92**
- `src/preprocessing/smote_balance.py` — SMOTE on training set only → **{0: 406, 1: 406}**
- `configs/config.yaml` — all hyperparameters centralized

### Week 5: ECG Signal Pipeline
- `src/preprocessing/ecg_filter.py` — Butterworth band-pass filter (0.5–100 Hz), Lead I extraction by name lookup, 5-min windowing at 500 Hz
- CPSC subset downloaded (6,877 recordings)
- `02_eda_ecg_signals.ipynb` — raw vs filtered Lead I plot, signal quality stats, `docs/ecg_raw_vs_filtered.png`

### Week 6a: Model Architecture & RF Baseline
- `src/models/cnn_lstm.py` — CNN-LSTM: 3 Conv1d blocks [32, 64, 128] + BatchNorm + MaxPool, 2-layer LSTM (hidden=128), classifier head. **Output: torch.Size([4, 1]) | Parameters: 254,593**
- `src/models/random_forest.py` — 5-fold stratified CV. **Recall: 0.887 ± 0.041 | F1: 0.871 ± 0.013 | AUC-ROC: 0.938 ± 0.010**
- `src/evaluation/metrics.py` — `evaluate_model()` with configurable threshold, confusion matrix + classification report

### Week 6b: Streamlit Dashboard
- `src/evaluation/shap_explainer.py` — SHAP TreeExplainer: `build_explainer()`, `get_shap_values()`, `top_features()`
- `src/dashboard/app.py` — Apple Watch-style dashboard:
  - 10 patient profile sliders (Age, BP, Cholesterol, MaxHR, Oldpeak, Sex, ExerciseAngina, ChestPainType, RestingECG, ST_Slope)
  - Live risk gauge (green/yellow/red zones, alert banner above threshold)
  - SHAP bar chart — top 6 features, red = risk-increasing, green = risk-decreasing
  - Rolling 30-reading risk history chart with alert threshold line
  - Model cached with `@st.cache_resource`

### Weeks 7–8: CNN-LSTM Training — AFib Detection
- `src/preprocessing/ecg_dataset.py` — PyTorch Dataset with SNOMED AFib code parsing (`164889003`), Lead I extraction, z-score normalization, outlier clipping (±2.0 pre-normalize, ±5.0 post-normalize)
- `src/models/train_cnn_lstm.py` — full training loop:
  - Gaussian noise augmentation (std=0.05) for wearable robustness
  - `BCEWithLogitsLoss` with `pos_weight=4.63` (5656/1221) for class imbalance
  - Gradient clipping (`max_norm=1.0`) for training stability
  - Early stopping (patience=5 epochs on AUC-ROC)
  - MLflow metric tracking per epoch
  - Apple M1 MPS GPU acceleration (`torch.device('mps')`)
- **Best checkpoint saved at epoch 28:**

| Metric | Value |
|---|---|
| AUC-ROC | **0.968** |
| Recall | **0.931** |
| Precision | 0.773 |
| F1 | 0.844 |
| TN | 1064 |
| FP | 67 |
| FN | 17 |
| TP | 228 |

- Best model saved to `data/processed/cnn_lstm_best.pt`

---

## Upcoming

### Week 9: Multi-Modal Fusion + Lead-Time Evaluation
- Combine RF clinical score + CNN-LSTM AFib score into a late fusion layer
- `src/evaluation/lead_time.py` — measure how many minutes before a simulated event the fused score crosses the alert threshold
- Target: **≥ 30-minute lead time**
- Simulate Apple Watch-style streaming in `data/simulated/`

### Week 10: Apple Watch Integration + Final Report
- Dashboard upload path for Apple Watch ECG CSV exports (Health app → Export, 512 Hz)
- The same band-pass filter and Lead I pipeline handles Apple Watch input with no code changes — Apple Watch records Lead I at 512 Hz, same as CPSC training data at 500 Hz
- SHAP summary plots, robustness tests on noisy inputs, final 5-fold CV pass
- Streamlit Cloud deployment for live demo link in README
- Full WatchOS streaming (requires Apple Developer account + Swift) is out of scope for this semester; the export-based path is fully testable with a real device

---

## Evaluation Targets

| Metric | Target | CNN-LSTM | Random Forest |
|---|---|---|---|
| Recall (Sensitivity) | ≥ 93% | **93.1% ✅** | 88.7% |
| AUC-ROC | Maximize | **0.968 ✅** | 0.938 ✅ |
| F1-Score | Maximize | 0.844 | 0.871 |
| Lead-Time Warning | ≥ 30 minutes | Planned — Week 9 | — |
| Cross-Validation | 5-fold stratified | Planned — Week 10 | ✅ Implemented |

---

## References

1. World Health Organization — Cardiovascular Diseases Fact Sheet (2021)
2. Soriano — Heart Failure Prediction Dataset, Kaggle (2021)
3. Asran — PhysioNet Challenge 2020: Classification of 12-lead ECGs, Kaggle
4. American Heart Association — AI tool detected structural heart disease using a smartwatch (2025)
5. Apple Inc. — HealthKit Framework Documentation (2024)
6. Salet et al. — Predicting Myocardial Infarction in Primary Care, PLoS ONE (2024)
7. Jin et al. — Transfer learning for single-lead ECG myocardial injury prediction, JAMIA (2022)
8. Chadaga — Predicting heart attack using time series data, Mendeley Data (2025)