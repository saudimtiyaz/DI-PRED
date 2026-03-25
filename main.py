import pandas as pd
from src.config import DATASETS, FEATURE_NAMES
from src.evaluate import evaluate_dataset
from src.train import train_all
from src.preprocess import preprocess
import sys
import os

# Make sure src/ is on the path
sys.path.insert(0, os.path.dirname(__file__))


def main():
    print("\n" + "="*50)
    print("  DISEASE PREDICTION — FULL PIPELINE")
    print("="*50)

    # Step 1 — Preprocess
    print("\n[STEP 1] Preprocessing datasets...")
    for dataset in DATASETS:
        preprocess(dataset)

    # Step 2 — Train
    print("\n[STEP 2] Training models...")
    for dataset in DATASETS:
        train_all(dataset)

    # Step 3 — Evaluate
    print("\n[STEP 3] Evaluating models...")
    all_metrics = []
    for dataset in DATASETS:
        all_metrics += evaluate_dataset(dataset, FEATURE_NAMES[dataset])

    # Save combined metrics
    df = pd.DataFrame(all_metrics)
    df.to_csv('results/metrics.csv', index=False)

    print("\n" + "="*50)
    print("  PIPELINE COMPLETE!")
    print("  Results saved to results/metrics.csv")
    print("  Figures saved to results/figures/")
    print("="*50)


if __name__ == '__main__':
    main()
