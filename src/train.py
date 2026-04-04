# Copyright (c) DI-PRED contributors.
# Licensed under the MIT License. See LICENSE.

"""Train baseline and ensemble classifiers on processed disease datasets."""

from typing import Literal, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


def load_data(dataset: Literal['diabetes', 'heart']) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed train/validation arrays for a dataset."""
    # load the processed train and val splits we saved in preprocess.py
    X_train = np.load(f'data/processed/{dataset}_X_train.npy')
    X_val = np.load(f'data/processed/{dataset}_X_val.npy')
    y_train = np.load(f'data/processed/{dataset}_y_train.npy')
    y_val = np.load(f'data/processed/{dataset}_y_val.npy')
    return X_train, X_val, y_train, y_val


def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray, dataset: Literal['diabetes', 'heart']
) -> LogisticRegression:
    """Train and persist the logistic regression baseline model."""
    print(f"\n[{dataset}] Training Logistic Regression...")

    # simple baseline model, max_iter bumped up so it always converges
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0
    )
    model.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{dataset}_logistic_regression.pkl')
    print(f"[{dataset}] Logistic Regression saved to models/")

    return model


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray, dataset: Literal['diabetes', 'heart']
) -> RandomForestClassifier:
    """Train and persist the random forest model."""
    print(f"\n[{dataset}] Training Random Forest...")

    # 100 trees with a max depth to avoid overfitting too badly
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1  # use all cpu cores to speed things up
    )
    model.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{dataset}_random_forest.pkl')
    print(f"[{dataset}] Random Forest saved to models/")

    return model


def evaluate_val(
    model: LogisticRegression | RandomForestClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    dataset: Literal['diabetes', 'heart']
) -> float:
    """Return validation accuracy for a trained model."""
    # quick sanity check on val set right after training
    val_acc = model.score(X_val, y_val)
    print(f"[{dataset}] {model_name} Validation Accuracy: {val_acc:.4f}")
    return val_acc


def train_all(dataset: Literal['diabetes', 'heart']) -> None:
    """Train both model families and save the better validation performer as the default model."""
    print(f"\n{'='*40}")
    print(f" Training models for: {dataset.upper()}")
    print(f"{'='*40}")

    X_train, X_val, y_train, y_val = load_data(dataset)

    # train both models back to back
    lr_model = train_logistic_regression(X_train, y_train, dataset)
    rf_model = train_random_forest(X_train, y_train, dataset)

    # print val accuracy for both so we can spot any obvious issues
    lr_acc = evaluate_val(lr_model, X_val, y_val, 'Logistic Regression', dataset)
    rf_acc = evaluate_val(rf_model, X_val, y_val, 'Random Forest', dataset)

    # SAVE BEST MODEL LOGIC (Explicitly used by predict.py)
    # This keeps it simple: if LR is better, save as 'best', else RF
    best_model = lr_model if lr_acc >= rf_acc else rf_model
    best_name = 'logistic_regression' if lr_acc >= rf_acc else 'random_forest'
    
    joblib.dump(best_model, f'models/{dataset}_best_model.pkl')
    print(f"[{dataset}] Best model ({best_name}) saved to models/{dataset}_best_model.pkl")


if __name__ == '__main__':
    # Ensure processed data exists
    if not os.path.exists('data/processed/diabetes_X_train.npy'):
        print("[!] Warning: Processed data not found. Run preprocess.py first.")
    else:
        train_all('diabetes')
        train_all('heart')
