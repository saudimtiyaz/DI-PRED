# Copyright (c) DI-PRED contributors.
# Licensed under the MIT License. See LICENSE.

"""Centralized configuration for data splitting, model hyperparameters, and feature ordering."""

# ── Data ─────────────────────────────────────────────────
DATA_CONFIG = {
    'random_state': 42,
    'test_size':    0.30,  # Fraction held out from full data before val/test split.
    'val_size':     0.50,  # Fraction of temp split sent to validation (0.5 -> equal val/test).
}

# ── Logistic Regression ──────────────────────────────────
LR_CONFIG = {
    'C':          1.0,   # Inverse regularization strength; lower values regularize more.
    'max_iter':   1000,  # Iteration cap to avoid premature solver stopping.
    'random_state': 42,
}

# ── Random Forest ────────────────────────────────────────
RF_CONFIG = {
    'n_estimators': 100,  # Number of trees in the ensemble.
    'max_depth':    10,   # Max tree depth to limit overfitting.
    'random_state': 42,
    'n_jobs': -1,         # Use all CPU cores during fitting.
}

# ── Datasets ─────────────────────────────────────────────
DATASETS = ['diabetes', 'heart']

FEATURE_NAMES = {
    'diabetes': [
        'Pregnancies', 'Glucose', 'BloodPressure',
        'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age'
    ],
    'heart': [
        'age', 'sex', 'cp', 'trestbps', 'chol',
        'fbs', 'restecg', 'thalach', 'exang',
        'oldpeak', 'slope', 'ca', 'thal'
    ]
}
