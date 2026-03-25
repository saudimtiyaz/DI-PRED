import pandas as pd
from src.config import DATASETS, FEATURE_NAMES
from src.evaluate import evaluate_dataset
from src.train import train_all
from src.preprocess import preprocess
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def main():
    print("\n" + "="*50)
    print("  DISEASE PREDICTION — FULL PIPELINE")
    print("="*50)

    # step 1 — clean and split the raw data
    print("\n[STEP 1] Preprocessing datasets...")
    for dataset in DATASETS:
        preprocess(dataset)

    # step 2 — train logistic regression and random forest on both datasets
    print("\n[STEP 2] Training models...")
    for dataset in DATASETS:
        train_all(dataset)

    # step 3 — evaluate on the test set and save all figures and metrics
    print("\n[STEP 3] Evaluating models...")
    all_metrics = []
    for dataset in DATASETS:
        all_metrics += evaluate_dataset(dataset, FEATURE_NAMES[dataset])

    # combine metrics from both datasets into one csv
    df = pd.DataFrame(all_metrics)
    df.to_csv('results/metrics.csv', index=False)

    print("\n" + "="*50)
    print("  PIPELINE COMPLETE!")
    print("  Results saved to results/metrics.csv")
    print("  Figures saved to results/figures/")
    print("="*50)


if __name__ == '__main__':
    main()
