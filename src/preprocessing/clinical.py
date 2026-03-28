import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))

def load_and_clean(path=None):
    if path is None:
        path = os.path.join(_HERE, '..', '..', 'data', 'raw', 'heart.csv')
    df = pd.read_csv(path)
    
    # Fix zero-cholesterol (physiologically impossible)
    median_chol = df[df['Cholesterol'] != 0]['Cholesterol'].median()
    df['Cholesterol'] = df['Cholesterol'].replace(0, median_chol)
    print(f'Imputed {(df["Cholesterol"]==median_chol).sum()} cholesterol zeros')
    return df

def encode_features(df):
    # Binary features
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
    
    # One-hot encode multi-class features
    df = pd.get_dummies(df,
        columns=['ChestPainType', 'RestingECG', 'ST_Slope'],
        drop_first=False)
    
    print(f'Features after encoding: {df.shape[1]}')
    return df

def normalize_features(df):
    num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print('Normalized:', num_cols)
    return df, scaler

def split_data(df, target='HeartDisease', random_state=42):
    X = df.drop(target, axis=1)
    y = df[target]
    
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=random_state)
    
    print(f'Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}')
    print(f'Train class ratio: {y_tr.mean():.2f}')
    return X_tr, X_val, X_te, y_tr, y_val, y_te

def full_pipeline(path=None):
    df = load_and_clean(path)
    df = encode_features(df)
    df, scaler = normalize_features(df)
    splits = split_data(df)
    return splits, scaler

if __name__ == '__main__':
    splits, scaler = full_pipeline()
    X_tr, X_val, X_te, y_tr, y_val, y_te = splits
    print('Pipeline complete!')
    print('Train X shape:', X_tr.shape)