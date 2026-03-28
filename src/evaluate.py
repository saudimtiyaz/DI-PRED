import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from src.visualize import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance
)


def load_test_data(dataset):
    # load only the test split — this data was never touched during training
    X_test = np.load(f'data/processed/{dataset}_X_test.npy')
    y_test = np.load(f'data/processed/{dataset}_y_test.npy')
    return X_test, y_test


def load_model(dataset, model_name):
    path = f'models/{dataset}_{model_name}.pkl'
    return joblib.load(path)


def get_metrics(model, X_test, y_test):
    # get both hard predictions and probabilities
    # probabilities are needed for auc
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall':    round(recall_score(y_test, y_pred), 4),
        'F1':        round(f1_score(y_test, y_pred), 4),
        'AUC':       round(roc_auc_score(y_test, y_proba), 4),
    }, y_pred, y_proba


def evaluate_dataset(dataset, feature_names):
    print(f"\n{'='*40}")
    print(f" Evaluating: {dataset.upper()}")
    print(f"{'='*40}")

    os.makedirs('results/figures', exist_ok=True)

    X_test, y_test = load_test_data(dataset)
    model_names = ['logistic_regression', 'random_forest']
    all_metrics = []
    roc_data = []

    for model_name in model_names:
        print(f"\n  Model: {model_name.replace('_', ' ').title()}")
        model = load_model(dataset, model_name)
        metrics, y_pred, y_proba = get_metrics(model, X_test, y_test)

        for k, v in metrics.items():
            print(f"    {k}: {v}")

        # Plot confusion matrix using refactored tool
        cm_path = f'results/figures/{dataset}_{model_name}_confusion_matrix.png'
        plot_confusion_matrix(y_test, y_pred, dataset, model_name, cm_path)

        # collect roc data so we can plot all curves together
        roc_data.append((model_name, y_test, y_proba))

        all_metrics.append({
            'Dataset': dataset,
            'Model': model_name.replace('_', ' ').title(),
            **metrics
        })

    # Plot ROC curves using refactored tool
    roc_path = f'results/figures/{dataset}_roc_curves.png'
    plot_roc_curves(dataset, roc_data, roc_path)

    # Plot feature importance using refactored tool
    rf = load_model(dataset, 'random_forest')
    fi_path = f'results/figures/{dataset}_feature_importance.png'
    plot_feature_importance(dataset, feature_names, rf.feature_importances_, fi_path)

    return all_metrics


if __name__ == '__main__':
    from src.config import FEATURE_NAMES

    all_metrics = []
    for dataset in ['diabetes', 'heart']:
        all_metrics += evaluate_dataset(dataset, FEATURE_NAMES[dataset])

    # save everything to a csv so it's easy to reference in the report
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(all_metrics)
    df.to_csv('results/metrics.csv', index=False)
    print(f"\n  Saved: results/metrics.csv")
    print(f"\n{'='*40}")
    print(" All evaluation complete!")
    print(f"{'='*40}")
