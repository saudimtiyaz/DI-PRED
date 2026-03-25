# ── Data ─────────────────────────────────────────────────
DATA_CONFIG = {
    'random_state': 42,
    'test_size':    0.30,
    'val_size':     0.50,
}

# ── Logistic Regression ──────────────────────────────────
LR_CONFIG = {
    'C':          1.0,
    'max_iter':   1000,
    'random_state': 42,
}

# ── Random Forest ────────────────────────────────────────
RF_CONFIG = {
    'n_estimators': 100,
    'max_depth':    10,
    'random_state': 42,
    'n_jobs': -1,
}

# ── Datasets ─────────────────────────────────────────────
DATASETS = ['diabetes', 'heart']

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
