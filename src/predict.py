from src.config import FEATURE_NAMES
import numpy as np
import joblib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def predict(dataset, input_values):
    scaler = joblib.load(f'models/{dataset}_scaler.pkl')
    model = joblib.load(f'models/{dataset}_random_forest.pkl')

    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    label = 'POSITIVE' if prediction == 1 else 'NEGATIVE'
    print(f"\n  Dataset:    {dataset.capitalize()}")
    print(f"  Prediction: {label}")
    print(f"  Confidence: {probability:.2%}")

    return prediction, probability


def interactive_predict(dataset):
    features = FEATURE_NAMES[dataset]
    print(f"\nEnter patient data for {dataset.capitalize()} prediction:")
    print("-" * 40)

    values = []
    for feature in features:
        val = float(input(f"  {feature}: "))
        values.append(val)

    predict(dataset, values)


if __name__ == '__main__':
    print("Which dataset?")
    print("  1. Diabetes")
    print("  2. Heart Disease")
    choice = input("Enter 1 or 2: ").strip()

    if choice == '1':
        interactive_predict('diabetes')
    elif choice == '2':
        interactive_predict('heart')
    else:
        print("Invalid choice.")
