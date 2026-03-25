import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


def preprocess(dataset):
    """
    Loads, cleans, normalizes and splits a dataset.
    dataset: 'diabetes' or 'heart'
    """

    # ── 1. Load ──────────────────────────────────────────────
    if dataset == 'diabetes':
        df = pd.read_csv('data/raw/diabetes.csv')
        target_col = 'Outcome'
    elif dataset == 'heart':
        df = pd.read_csv('data/raw/heart.csv')
        target_col = 'target'
    else:
        raise ValueError("dataset must be 'diabetes' or 'heart'")

    print(f"\n[{dataset}] Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # ── 2. Handle missing/zero values ────────────────────────
    # Diabetes dataset uses 0 as a placeholder for missing values
    if dataset == 'diabetes':
        zero_cols = ['Glucose', 'BloodPressure',
                     'SkinThickness', 'Insulin', 'BMI']
        df[zero_cols] = df[zero_cols].replace(0, np.nan)
        df.fillna(df.median(), inplace=True)
        print(f"[{dataset}] Zero values replaced with median")

    # ── 3. Split features and target ─────────────────────────
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ── 4. Train / val / test split (70 / 15 / 15) ───────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"[{dataset}] Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # ── 5. Normalize with StandardScaler ─────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ── 6. Save processed data ───────────────────────────────
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    np.save(f'data/processed/{dataset}_X_train.npy', X_train)
    np.save(f'data/processed/{dataset}_X_val.npy',   X_val)
    np.save(f'data/processed/{dataset}_X_test.npy',  X_test)
    np.save(f'data/processed/{dataset}_y_train.npy', y_train)
    np.save(f'data/processed/{dataset}_y_val.npy',   y_val)
    np.save(f'data/processed/{dataset}_y_test.npy',  y_test)

    # Save scaler for use during prediction
    joblib.dump(scaler, f'models/{dataset}_scaler.pkl')

    print(f"[{dataset}] Saved processed files to data/processed/")
    print(f"[{dataset}] Saved scaler to models/{dataset}_scaler.pkl")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    preprocess('diabetes')
    preprocess('heart')
