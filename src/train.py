import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


def load_data(dataset):
    # load the processed train and val splits we saved in preprocess.py
    X_train = np.load(f'data/processed/{dataset}_X_train.npy')
    X_val = np.load(f'data/processed/{dataset}_X_val.npy')
    y_train = np.load(f'data/processed/{dataset}_y_train.npy')
    y_val = np.load(f'data/processed/{dataset}_y_val.npy')
    return X_train, X_val, y_train, y_val


def train_logistic_regression(X_train, y_train, dataset):
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


def train_random_forest(X_train, y_train, dataset):
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


def evaluate_val(model, X_val, y_val, model_name, dataset):
    # quick sanity check on val set right after training
    val_acc = model.score(X_val, y_val)
    print(f"[{dataset}] {model_name} Validation Accuracy: {val_acc:.4f}")


def train_all(dataset):
    print(f"\n{'='*40}")
    print(f" Training models for: {dataset.upper()}")
    print(f"{'='*40}")

    X_train, X_val, y_train, y_val = load_data(dataset)

    # train both models back to back
    lr_model = train_logistic_regression(X_train, y_train, dataset)
    rf_model = train_random_forest(X_train, y_train, dataset)

    # print val accuracy for both so we can spot any obvious issues
    evaluate_val(lr_model, X_val, y_val, 'Logistic Regression', dataset)
    evaluate_val(rf_model, X_val, y_val, 'Random Forest', dataset)


if __name__ == '__main__':
    train_all('diabetes')
    train_all('heart')
