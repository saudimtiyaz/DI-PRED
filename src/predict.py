import numpy as np
import joblib
import sys
import os

# add the project root to the path so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# feature names defined here directly to avoid import order issues
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


def predict(dataset, input_values):
    # load the scaler and best model for this dataset
    scaler = joblib.load(f'models/{dataset}_scaler.pkl')
    model = joblib.load(f'models/{dataset}_random_forest.pkl')

    # normalize the input the same way we normalized training data
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

    # collect each feature value one at a time from the user
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
