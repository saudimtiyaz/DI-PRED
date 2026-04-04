"""Download and normalize raw datasets used by the DI-PRED pipeline."""

import os
import pandas as pd
import requests

def download_file(url: str, filename: str) -> bool:
    """Downloads a file from a URL if it doesn't already exist. Returns True if downloaded."""
    if os.path.exists(filename):
        print(f"  File already exists: {filename}")
        return False

    print(f"  Downloading: {url} ...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"  Saved: {filename}")
        return True
    else:
        print(f"  Failed to download: {url} (Status: {response.status_code})")
        return False


def remove_duplicate_header_rows(csv_path: str, expected_header: str) -> None:
    """
    Remove repeated in-file header rows caused by interrupted/manual re-runs.

    This guards against malformed files where header lines reappear in the body,
    which would otherwise be parsed as invalid samples.
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines:
        return

    deduped_lines = [lines[0]]
    for line in lines[1:]:
        # Keep only data rows after the first header to prevent accidental
        # "header as data" corruption during repeated processing attempts.
        if line.strip() != expected_header:
            deduped_lines.append(line)

    if len(deduped_lines) != len(lines):
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.writelines(deduped_lines)
        print(f"  Removed duplicate headers in: {csv_path}")


def main() -> None:
    print("\n--- Data Acquisition ---")
    os.makedirs('data/raw', exist_ok=True)

    # Dataset URLs
    diabetes_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    diabetes_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    heart_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    heart_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

    # Download and process Diabetes ONLY if it doesn't exist.
    if download_file(diabetes_url, 'data/raw/diabetes.csv'):
        df_diabetes = pd.read_csv('data/raw/diabetes.csv', header=None, names=diabetes_cols)
        df_diabetes.to_csv('data/raw/diabetes.csv', index=False)
        print("  Processed Diabetes dataset.")
    remove_duplicate_header_rows('data/raw/diabetes.csv', ','.join(diabetes_cols))

    # Download and process Heart Disease ONLY if it doesn't exist
    if download_file(heart_url, 'data/raw/heart.csv'):
        df_heart = pd.read_csv('data/raw/heart.csv', header=None, names=heart_cols, na_values='?')
        df_heart['target'] = pd.to_numeric(df_heart['target'], errors='coerce')
        df_heart.dropna(subset=['target'], inplace=True)
        df_heart['target'] = (df_heart['target'] > 0).astype(int)
        df_heart.dropna(inplace=True)
        df_heart.to_csv('data/raw/heart.csv', index=False)
        print("  Processed Heart dataset.")
    remove_duplicate_header_rows('data/raw/heart.csv', ','.join(heart_cols))

    print("\nData download and initialization complete.")

if __name__ == '__main__':
    main()
