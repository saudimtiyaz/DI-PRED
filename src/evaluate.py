import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import joblib
import os


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


def plot_confusion_matrix(y_test, y_pred, dataset, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No', 'Yes'],
                yticklabels=['No', 'Yes'])
    plt.title(
        f'{dataset.capitalize()} — {model_name.replace("_", " ").title()}\nConfusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    path = f'results/figures/{dataset}_{model_name}_confusion_matrix.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curves(dataset, models_data):
    # plot all models on the same chart so they're easy to compare
    plt.figure(figsize=(7, 5))

    for model_name, y_test, y_proba in models_data:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(
            fpr, tpr, label=f'{model_name.replace("_", " ").title()} (AUC={auc:.3f})')

    # diagonal line represents a random guess — anything above this is useful
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset.capitalize()} — ROC Curves')
    plt.legend()
    plt.tight_layout()
    path = f'results/figures/{dataset}_roc_curves.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(dataset, feature_names):
    # random forest gives us importance scores for free after training
    rf = load_model(dataset, 'random_forest')
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]  # sort highest to lowest

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(importances)),
            importances[indices],
            color='steelblue', edgecolor='black')
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in indices],
               rotation=45, ha='right')
    plt.title(f'{dataset.capitalize()} — Random Forest Feature Importance')
    plt.tight_layout()
    path = f'results/figures/{dataset}_feature_importance.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


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

        plot_confusion_matrix(y_test, y_pred, dataset, model_name)

        # collect roc data so we can plot all curves together
        roc_data.append((model_name, y_test, y_proba))

        all_metrics.append({
            'Dataset': dataset,
            'Model': model_name.replace('_', ' ').title(),
            **metrics
        })

    plot_roc_curves(dataset, roc_data)
    plot_feature_importance(dataset, feature_names)

    return all_metrics


if __name__ == '__main__':
    diabetes_features = [
        'Pregnancies', 'Glucose', 'BloodPressure',
        'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age'
    ]

    heart_features = [
        'age', 'sex', 'cp', 'trestbps', 'chol',
        'fbs', 'restecg', 'thalach', 'exang',
        'oldpeak', 'slope', 'ca', 'thal'
    ]

    all_metrics = []
    all_metrics += evaluate_dataset('diabetes', diabetes_features)
    all_metrics += evaluate_dataset('heart', heart_features)

    # save everything to a csv so it's easy to reference in the report
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(all_metrics)
    df.to_csv('results/metrics.csv', index=False)
    print(f"\n  Saved: results/metrics.csv")
    print(f"\n{'='*40}")
    print(" All evaluation complete!")
    print(f"{'='*40}")
