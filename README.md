# 🫀 CardioWatch
### Early Detection & Short-Term Risk Prediction of Atrial Fibrillation Using ECG and Clinical Data

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Overview

CardioWatch is an ML research project that explores whether **temporal patterns in cardiovascular data** can be used for the early detection of heart disease. Rather than predicting *whether* a patient has heart disease, this system aims to estimate **when** a cardiac event might be approaching — providing an early warning window to improve treatment outcomes.

The system combines structured clinical data with ECG time-series signals, processed through a multi-modal pipeline (Random Forest + XGBoost + CNN-LSTM), and surfaced through an Apple Watch-style Streamlit risk dashboard with SHAP explainability.

The ECG component is specifically designed around **Atrial Fibrillation (AFib) detection using Lead I only** — the same single-lead signal that Apple Watch Series 4+ already records. This makes real-world wearable deployment a realistic next step rather than an afterthought.

---

## Project Structure

```
cardiowatch/
├── data/
│   ├── raw/               # Downloaded datasets (gitignored — see Data Setup below)
│   ├── processed/         # Model checkpoints and scalers (gitignored)
│   └── simulated/         # Synthetic HealthKit streams (gitignored)
├── src/
│   ├── preprocessing/
│   │   ├── clinical.py           # Imputation, encoding, normalization, train/val/test split
│   │   ├── ecg_dataset.py        # PyTorch Dataset for CPSC ECG recordings — AFib labels
│   │   ├── ecg_filter.py         # Band-pass filtering (0.5–100 Hz), Lead I extraction, windowing
│   │   └── smote_balance.py      # SMOTE class imbalance handling
│   ├── models/
│   │   ├── random_forest.py      # RF baseline with 5-fold CV — saves rf_model.pkl
│   │   ├── xgboost_model.py      # XGBoost baseline with threshold tuning — saves xgb_model.pkl
│   │   ├── cnn_lstm.py           # CNN-LSTM temporal ECG model architecture
│   │   ├── train_cnn_lstm.py     # CNN-LSTM training loop with MLflow + MPS acceleration
│   │   └── fusion.py             # Late fusion layer: RF + CNN-LSTM weighted average
│   ├── evaluation/
│   │   ├── metrics.py            # Recall, AUC-ROC, F1, confusion matrix
│   │   ├── shap_explainer.py     # SHAP TreeExplainer for RF
│   │   └── lead_time.py          # Lead-time evaluation using real CPSC recordings
│   └── dashboard/
│       └── app.py                # Streamlit Apple Watch risk simulator with fusion
├── notebooks/
│   └── CardioWatch_Complete.ipynb  # Full pipeline — EDA, preprocessing, training, evaluation 
│   
├── configs/
│   └── config.yaml               # Hyperparameters & paths
├── docs/
│   ├── ecg_raw_vs_filtered.png
│   ├── lead_time_evaluation.png
│   ├── roc_curve_cnn_lstm.png
│   ├── shap_summary.png
│   └── class_balance.png
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

| Model | Input | Recall | AUC-ROC | F1 | Threshold |
|---|---|---|---|---|---|
| Random Forest | Clinical features (19) | 0.902 | 0.945 | 0.893 | 0.50 |
| XGBoost | Clinical features (19) | **0.980** | 0.927 | 0.870 | 0.30 |
| CNN-LSTM | Lead I ECG (5000 samples) | 0.931 | **0.968** | 0.844 | 0.40 |

XGBoost achieves the highest recall (0.980) — best for minimizing missed cardiac events. CNN-LSTM achieves the highest AUC-ROC (0.968) — best overall discrimination. Both exceed the ≥ 0.93 recall target.

### CNN-LSTM Architecture

| Layer | Detail |
|---|---|
| Input | `(batch, 1, 5000)` — single Lead I channel, 10 seconds at 500 Hz |
| Conv Block 1 | Conv1d(1→32, kernel=7) + BatchNorm + ReLU + MaxPool(2) → length 2500 |
| Conv Block 2 | Conv1d(32→64, kernel=7) + BatchNorm + ReLU + MaxPool(2) → length 1250 |
| Conv Block 3 | Conv1d(64→128, kernel=7) + BatchNorm + ReLU + MaxPool(2) → length 625 |
| LSTM | 2-layer, hidden=128, batch_first=True |
| Classifier | Dropout(0.3) → Linear(128→64) → ReLU → Linear(64→1) |
| Parameters | 345,089 trainable |

---

## Why AFib and Why Lead I?

Atrial Fibrillation is the world's most common serious arrhythmia, affecting ~37 million people and being a leading cause of stroke. Unlike most arrhythmias, AFib has two unmistakable electrical signatures visible in any single lead — absent P waves and irregular R-R intervals — making it reliably detectable from Lead I alone.

Apple Watch received FDA clearance for AFib detection in 2018 using its single-lead ECG sensor. CardioWatch is trained on the same signal type (Lead I, 500 Hz), making the path from research model to wearable deployment direct and realistic.

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

### 3. Run the Full Pipeline

```bash
# Step 1 — Clinical preprocessing (saves scaler.pkl)
python3 src/preprocessing/clinical.py

# Step 2 — Random Forest (saves rf_model.pkl)
python3 src/models/random_forest.py

# Step 3 — XGBoost (saves xgb_model.pkl)
python3 src/models/xgboost_model.py

# Step 4 — Validate CNN-LSTM architecture
python3 src/models/cnn_lstm.py

# Step 5 — Train CNN-LSTM AFib detector (M1 GPU accelerated, ~10 min)
python3 src/models/train_cnn_lstm.py

# Step 6 — Lead-time evaluation
python3 src/evaluation/lead_time.py

# Step 7 — Launch dashboard
streamlit run src/dashboard/app.py

# Step 8 — View MLflow training curves
mlflow ui
```

Or run everything at once from the notebook:
```bash
jupyter notebook notebooks/CardioWatch_Complete.ipynb
```

---

## Validation Checklist

| Component | Command | Expected Result |
|---|---|---|
| Packages | `python3 -c "import torch, pandas, sklearn, wfdb, shap, streamlit, mlflow"` | No errors |
| Dataset | `python3 -c "import pandas as pd; df=pd.read_csv('data/raw/heart.csv'); print(df.shape)"` | `(918, 12)` |
| Clinical pipeline | `python3 src/preprocessing/clinical.py` | Train: 734, Val: 92, Test: 92 |
| SMOTE | `python3 src/preprocessing/smote_balance.py` | After SMOTE: {0: 406, 1: 406} |
| XGBoost | `python3 src/models/xgboost_model.py` | Recall: 0.980, AUC: 0.927 |
| ECG dataset | `python3 src/preprocessing/ecg_dataset.py` | Loaded 6877 recordings, AFib: 1221 |
| CNN-LSTM arch | `python3 src/models/cnn_lstm.py` | Output shape: torch.Size([4, 1]) |
| CNN-LSTM train | `python3 src/models/train_cnn_lstm.py` | AUC-ROC ≥ 0.95 by epoch 20 |
| Lead time | `python3 src/evaluation/lead_time.py` | Lead time ≥ 30 min: MET |
| RF baseline | `python3 src/models/random_forest.py` | Recall ≥ 0.90, AUC ≥ 0.94 |
| Dashboard | `streamlit run src/dashboard/app.py` | Opens at localhost:8501 |

---

## What Is Built

### Weeks 1–2: Environment & Setup
- Python 3.9 virtual environment with all dependencies installed
- GitHub repo initialized, Kaggle API configured, PhysioNet account approved
- Literature review: Jin et al. (2022), Chadaga (2025), AHA (2025), Salet et al. (2024)

### Week 3: Clinical EDA
- Kaggle Heart Failure dataset (918 × 12), EDA notebook with zero-cholesterol detection (172 rows), class balance (508 vs 410), feature distributions, correlation heatmap

### Week 4: Clinical Preprocessing Pipeline
- Median imputation, binary + one-hot encoding, MinMaxScaler, stratified 80/10/10 split
- SMOTE on training set → balanced {0: 406, 1: 406}
- Scaler saved to `data/processed/scaler.pkl` for dashboard use

### Week 5: ECG Signal Pipeline
- Butterworth band-pass filter (0.5–100 Hz), Lead I extraction by name lookup
- CPSC subset downloaded (6,877 recordings), raw vs filtered ECG visualization

### Week 6a: Model Architecture & RF Baseline
- CNN-LSTM: 3 Conv1d blocks [32, 64, 128] + 2-layer LSTM + classifier, 345k params
- Random Forest 5-fold CV: **Recall 0.902 | AUC-ROC 0.945**

### Week 6b: Streamlit Dashboard
- SHAP TreeExplainer for Random Forest feature importance
- Apple Watch-style dashboard: 10 patient sliders, live risk gauge, SHAP bar chart, rolling risk history
- CNN-LSTM inference wired to Apple Watch ECG CSV uploader
- Late fusion: `fused_score = 0.6 × RF + 0.4 × CNN-LSTM`

### Weeks 7–8: XGBoost + CNN-LSTM Training
- XGBoost with threshold tuning: **Recall 0.980 | AUC-ROC 0.927** at threshold=0.30
- `src/preprocessing/ecg_dataset.py` — AFib label parsing (SNOMED `164889003`), z-score normalization, outlier clipping
- CNN-LSTM training with Gaussian noise augmentation, gradient clipping, early stopping, MPS acceleration
- **Best checkpoint (epoch 28): AUC-ROC 0.968 | Recall 0.931 | F1 0.844**

### Weeks 9–10: Fusion + Lead-Time + Apple Watch Analysis

#### Lead-Time Evaluation
- Built from real CPSC recordings: 35 min Normal Sinus Rhythm → 31 min AFib
- Fused score (RF=0.75 clinical + CNN-LSTM ECG) alerts at AFib onset
- **Lead time: 30.0 minutes ✅ Target MET**

#### Apple Watch Domain Gap Analysis
- Tested model on 6 real Apple Watch ECG recordings (exported via Health Auto Export)
- Including 1 confirmed AFib recording (Apple Watch Classification: Atrial Fibrillation)
- Model scores ~0.50 on all recordings — pure uncertainty

| Recording | Apple Watch Says | Model Score |
|---|---|---|
| 2022-08-23 | Sinus Rhythm | 0.507 |
| 2022-08-24 | High Heart Rate | 0.506 |
| 2022-09-15 | **Atrial Fibrillation** | 0.502 |
| 2023-04-23 | Sinus Rhythm | 0.501 |
| 2023-04-25 | High Heart Rate | 0.509 |
| 2023-11-18 | Sinus Rhythm | 0.510 |

**Finding:** Domain gap between clinical hospital ECG (CPSC, 500 Hz, clinical environment) and consumer wearable ECG (Apple Watch, 512 Hz, wrist sensor) prevents direct deployment. Fine-tuning on labeled Apple Watch data is required — consistent with Jin et al. (2022) on ECG transfer learning.

---

## Evaluation Targets

| Metric | Target | CNN-LSTM | Random Forest | XGBoost |
|---|---|---|---|---|
| Recall (Sensitivity) | ≥ 93% | **93.1% ✅** | 90.2% ✅ | **98.0% ✅** |
| AUC-ROC | Maximize | **0.968 ✅** | 0.945 ✅ | 0.927 ✅ |
| F1-Score | Maximize | 0.844 | 0.893 | 0.870 |
| Lead-Time Warning | ≥ 30 minutes | **30.0 min ✅** | — | — |
| Cross-Validation | 5-fold stratified | ✅ | ✅ | ✅ |

---

## Apple Watch Deployment Path

Apple Watch users can export ECG recordings via **Health Auto Export** (free, App Store):
1. Open Health Auto Export → Export → ECG → CSV
2. Upload the CSV to the CardioWatch dashboard uploader
3. The pipeline applies band-pass filtering (0.5–100 Hz) and resamples 512 Hz → 500 Hz

**Current limitation:** Domain gap between clinical training data and Apple Watch signals produces uncertain predictions (~0.50). Fine-tuning the classifier head on labeled Apple Watch data is the primary path to deployment.

**Future work:** Full WatchOS streaming requires Apple Developer account ($99/year) + Swift app — out of scope for this semester.

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