# Copyright (c) DI-PRED contributors.
# Licensed under the MIT License. See LICENSE.

"""Prepare raw datasets into normalized train/validation/test splits for model training."""

from typing import Literal, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


def preprocess(
    dataset: Literal['diabetes', 'heart']
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series, pd.Series]:
    """
    Load, clean, scale, and split a dataset into train/validation/test sets.

    `dataset` must be either `diabetes` or `heart`.
    """

    # load the right csv depending on which dataset we're working with
    if dataset == 'diabetes':
        df = pd.read_csv('data/raw/diabetes.csv')
        target_col = 'Outcome'
    elif dataset == 'heart':
        df = pd.read_csv('data/raw/heart.csv')
        target_col = 'target'
    else:
        raise ValueError("dataset must be 'diabetes' or 'heart'")

    print(f"\n[{dataset}] Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # the diabetes dataset uses 0 as a stand-in for missing values
    # so we replace those zeros with the median of each column
    if dataset == 'diabetes':
        zero_cols = ['Glucose', 'BloodPressure',
                     'SkinThickness', 'Insulin', 'BMI']
        df[zero_cols] = df[zero_cols].replace(0, np.nan)
        df.fillna(df.median(), inplace=True)
        print(f"[{dataset}] Zero values replaced with median")

    # separate the features from what we're trying to predict
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # split into train / val / test (70% / 15% / 15%)
    # stratify makes sure both splits have a similar ratio of positives to negatives
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    # `val_size=0.50` is applied to the remaining 30% (X_temp), not the full dataset.
    # That means each of val and test receives half of 30%, resulting in 15% each overall.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"[{dataset}] Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # normalize so all features are on the same scale
    # we fit only on training data so the test set doesn't leak into the scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # save everything so train.py can load it without rerunning this file
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    np.save(f'data/processed/{dataset}_X_train.npy', X_train)
    np.save(f'data/processed/{dataset}_X_val.npy',   X_val)
    np.save(f'data/processed/{dataset}_X_test.npy',  X_test)
    np.save(f'data/processed/{dataset}_y_train.npy', y_train)
    np.save(f'data/processed/{dataset}_y_val.npy',   y_val)
    np.save(f'data/processed/{dataset}_y_test.npy',  y_test)

    # save the scaler too so predict.py can use the same normalization later
    joblib.dump(scaler, f'models/{dataset}_scaler.pkl')

    print(f"[{dataset}] Saved processed files to data/processed/")
    print(f"[{dataset}] Saved scaler to models/{dataset}_scaler.pkl")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    preprocess('diabetes')
    preprocess('heart')
