"""
clinical.py — Clinical preprocessing pipeline
Saves scaler.pkl and rf_model.pkl to data/processed/ for dashboard use.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

DATA_PATH      = 'data/raw/heart.csv'
PROCESSED_DIR  = 'data/processed'

def full_pipeline(data_path=DATA_PATH):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = pd.read_csv(data_path)

    # ── Step 1: Zero-cholesterol imputation ──────────────────────────
    median_chol = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].median()
    df['Cholesterol'] = df['Cholesterol'].replace(0, median_chol)

    # ── Step 2: Binary encoding ──────────────────────────────────────
    df['Sex']            = (df['Sex'] == 'M').astype(int)
    df['ExerciseAngina'] = (df['ExerciseAngina'] == 'Y').astype(int)

    # ── Step 3: One-hot encoding ─────────────────────────────────────
    df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ST_Slope'])

    # ── Step 4: Train/val/test split (stratified 80/10/10) ───────────
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # ── Step 5: MinMaxScaler (fit on train only) ─────────────────────
    continuous = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    scaler = MinMaxScaler()
    X_train[continuous] = scaler.fit_transform(X_train[continuous])
    X_val[continuous]   = scaler.transform(X_val[continuous])
    X_test[continuous]  = scaler.transform(X_test[continuous])

    # ── Save scaler for dashboard use ────────────────────────────────
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, 'scaler.pkl'))
    print(f"Scaler saved to {PROCESSED_DIR}/scaler.pkl")

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return (X_train, X_val, X_test, y_train, y_val, y_test), scaler


if __name__ == '__main__':
    full_pipeline()