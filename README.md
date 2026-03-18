# 🫀 CardioWatch
### Early Detection & Short-Term Risk Prediction of Heart Attacks Using ECG and Clinical Data

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

CardioWatch is an ML research project that explores whether **temporal patterns in cardiovascular data** can be used for the early detection of heart disease. Rather than predicting *whether* a patient has heart disease, this system aims to estimate **when** a cardiac event might be approaching — providing an early warning window to improve treatment outcomes.

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
│   │   ├── download_data.py      # Kaggle + PhysioNet download
│   │   ├── ecg_filter.py         # Band-pass filtering (0.5–100 Hz)
│   │   ├── windowing.py          # 5-min sliding windows
│   │   └── smote_balance.py      # Class imbalance handling
│   ├── models/
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   └── cnn_lstm.py           # Temporal ECG model
│   ├── evaluation/
│   │   ├── metrics.py            # Recall, AUC-ROC, F1, lead-time
│   │   └── shap_explain.py       # SHAP interpretability
│   └── dashboard/
│       └── app.py                # Streamlit Apple Watch simulator
├── notebooks/
│   ├── 01_eda_clinical.ipynb
│   ├── 02_eda_ecg_signals.ipynb
│   └── 03_model_exploration.ipynb
├── configs/
│   └── config.yaml               # Hyperparameters & paths
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
├── setup_github.sh
└── README.md
```

---

## Datasets

| Dataset | Source | Size | Purpose |
|---|---|---|---|
| Heart Failure Prediction | [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) | 918 samples, 11 features | Baseline risk modeling |
| PhysioNet Challenge 2020 | [PhysioNet](https://physionet.org/content/challenge-2020/) | 40,000+ ECG recordings | CNN-LSTM temporal modeling |

---

## Models

| Model | Input | Goal |
|---|---|---|
| Random Forest | Clinical features | Interpretable baseline risk score |
| XGBoost | Clinical features | Strong structured-data baseline |
| CNN-LSTM | ECG time-series | Temporal risk trend forecasting |

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/cardiowatch.git
cd cardiowatch

# 2. Run setup (creates venv, installs deps, pushes to GitHub)
bash setup_github.sh <your-github-username>

# 3. Activate environment
source venv/bin/activate

# 4. Download datasets
python src/preprocessing/download_data.py

# 5. Launch dashboard
streamlit run src/dashboard/app.py
```

---

## Evaluation Targets

- **Recall ≥ 90%** — minimize missed cardiac events
- **Lead-time ≥ 30 minutes** — meaningful early warning window
- **AUC-ROC** — balance sensitivity vs. false alarm rate
- **5-fold stratified cross-validation** — ensure generalization

---

## Timeline

| Phase | Weeks | Focus |
|---|---|---|
| Data Strategy | 1–4 | Signal processing, ECG windowing, Lead-I extraction |
| Model Architecture | 5–8 | Baseline → CNN-LSTM training & tuning |
| Deployment & Audit | 9–12 | Streamlit dashboard, SHAP interpretability |

---

## References

1. WHO — Cardiovascular Diseases Fact Sheet (2021)
2. Soriano — Heart Failure Prediction Dataset, Kaggle (2021)
3. PhysioNet Challenge 2020 — 12-lead ECG Classification
4. Jin et al. — Transfer learning for single-lead ECG, JAMIA (2022)
5. Chadaga — Predicting heart attack using time series data (2025)
