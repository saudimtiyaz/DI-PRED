import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

def plot_confusion_matrix(y_true, y_pred, dataset_name, model_name, save_path):
    """Plots and saves a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No', 'Yes'],
                yticklabels=['No', 'Yes'])
    plt.title(f'{dataset_name.capitalize()} — {model_name.replace("_", " ").title()}\nConfusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved Confusion Matrix: {save_path}")

def plot_roc_curves(dataset_name, models_data, save_path):
    """Plots multiple ROC curves on a single chart."""
    plt.figure(figsize=(7, 5))
    for model_name, y_true, y_proba in models_data:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, label=f'{model_name.replace("_", " ").title()} (AUC={auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_name.capitalize()} — ROC Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved ROC Curves: {save_path}")

def plot_feature_importance(dataset_name, feature_names, importances, save_path):
    """Plots and saves feature importance bar chart."""
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(importances)),
            importances[indices],
            color='steelblue', edgecolor='black')
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in indices],
               rotation=45, ha='right')
    plt.title(f'{dataset_name.capitalize()} — Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved Feature Importance: {save_path}")
