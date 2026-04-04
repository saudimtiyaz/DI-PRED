# Copyright (c) DI-PRED contributors.
# Licensed under the MIT License. See LICENSE.

"""Run interactive CLI predictions using persisted scalers and trained models."""

from typing import Literal, Sequence

import numpy as np
import joblib
import sys
import os

# add the project root to the path so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Realistic ranges and clear descriptions for features
FEATURE_INFO = {
    'diabetes': {
        'Pregnancies': {'range': (0, 17), 'desc': 'Number of times pregnant'},
        'Glucose': {'range': (0, 199), 'desc': 'Plasma glucose concentration (2 hours in an oral glucose tolerance test)'},
        'BloodPressure': {'range': (0, 122), 'desc': 'Diastolic blood pressure (mm Hg)'},
        'SkinThickness': {'range': (0, 99), 'desc': 'Triceps skin fold thickness (mm)'},
        'Insulin': {'range': (0, 846), 'desc': '2-Hour serum insulin (mu U/ml)'},
        'BMI': {'range': (0, 67.1), 'desc': 'Body mass index (weight in kg/(height in m)^2)'},
        'DiabetesPedigreeFunction': {'range': (0.08, 2.42), 'desc': 'Diabetes pedigree function (hereditary risk)'},
        'Age': {'range': (21, 81), 'desc': 'Age in years'}
    },
    'heart': {
        'age': {'range': (29, 77), 'desc': 'Age in years'},
        'sex': {'range': (0, 1), 'desc': 'Sex (1 = male; 0 = female)'},
        'cp': {'range': (0, 3), 'desc': 'Chest pain type (0: typical, 1: atypical, 2: non-anginal, 3: asymptomatic)'},
        'trestbps': {'range': (94, 200), 'desc': 'Resting blood pressure (mm Hg)'},
        'chol': {'range': (126, 564), 'desc': 'Serum cholestoral in mg/dl'},
        'fbs': {'range': (0, 1), 'desc': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)'},
        'restecg': {'range': (0, 2), 'desc': 'Resting electrocardiographic results (0, 1, or 2)'},
        'thalach': {'range': (71, 202), 'desc': 'Maximum heart rate achieved'},
        'exang': {'range': (0, 1), 'desc': 'Exercise induced angina (1 = yes; 0 = no)'},
        'oldpeak': {'range': (0, 6.2), 'desc': 'ST depression induced by exercise relative to rest'},
        'slope': {'range': (0, 2), 'desc': 'Slope of the peak exercise ST segment'},
        'ca': {'range': (0, 4), 'desc': 'Number of major vessels (0-4) colored by flourosopy'},
        'thal': {'range': (0, 3), 'desc': 'Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)'}
    }
}

def predict(dataset: Literal['diabetes', 'heart'], input_values: Sequence[float]) -> None:
    """Generate and print a binary prediction with confidence for a single patient record."""
    scaler = joblib.load(f'models/{dataset}_scaler.pkl')
    
    # Load the best model saved during training
    model_path = f'models/{dataset}_best_model.pkl'
    if not os.path.exists(model_path):
        # Fallback to specific models if best_model isn't found
        if dataset == 'diabetes':
            model_path = f'models/{dataset}_logistic_regression.pkl'
        else:
            model_path = f'models/{dataset}_random_forest.pkl'
            
    model = joblib.load(model_path)

    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    label = 'POSITIVE' if prediction == 1 else 'NEGATIVE'
    print(f"\n" + "="*40)
    print(f"  RESULT FOR {dataset.upper()}")
    print(f"  Prediction: {label}")
    print(f"  Confidence: {probability:.2%}")
    print("="*40)

def interactive_predict(dataset: Literal['diabetes', 'heart']) -> None:
    """Collect feature inputs from the CLI and run prediction for the selected dataset."""
    info = FEATURE_INFO[dataset]
    print(f"\nEnter patient data for {dataset.capitalize()} prediction:")
    print("-" * 60)

    values = []
    for feature, data in info.items():
        fmin, fmax = data['range']
        desc = data['desc']
        
        while True:
            try:
                print(f"\n> {feature.upper()}: {desc}")
                val = float(input(f"  Enter value ({fmin} to {fmax}): "))
                
                if fmin <= val <= fmax:
                    values.append(val)
                    break
                else:
                    print(f"    [!] Error: Value must be between {fmin} and {fmax}.")
            except ValueError:
                print("    [!] Error: Please enter a numeric value.")

    predict(dataset, values)

if __name__ == '__main__':
    print("\n" + "="*40)
    print("  DI-PRED: ENHANCED INTERACTIVE CLI")
    print("="*40)
    print("Which dataset?")
    print("  1. Diabetes")
    print("  2. Heart Disease")
    
    choice = input("\nEnter 1 or 2: ").strip()

    if choice == '1':
        interactive_predict('diabetes')
    elif choice == '2':
        interactive_predict('heart')
    else:
        print("Invalid choice.")
