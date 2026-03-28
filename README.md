# DI-PRED: Advanced Disease Prediction System

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-orange.svg" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <p align="center">
    <b>An end-to-end clinical diagnostic tool for Diabetes and Heart Disease risk assessment.</b>
  </p>
</div>

---

## 🌟 Overview

**DI-PRED** (Diagnostic Predictor) is a high-performance machine learning framework designed to assist healthcare professionals in early-stage risk detection. By analyzing patient clinical data, the system provides real-time, probabilistic risk assessments for two of the most prevalent chronic conditions: **Diabetes** and **Heart Disease**.

> [!IMPORTANT]
> This tool is intended for research and educational purposes only. Always consult a medical professional for actual diagnosis.

---

## ✨ Key Features

- **🚀 Dual-Engine Pipeline:** Specialized models optimized for different pathologies.
- **📊 Automated EDA:** Generates comprehensive visual insights (ROC, Confusion Matrices, Feature Importance).
- **🖥️ Enhanced CLI:** Interactive diagnostic tool with real-time input validation and medical context.
- **🔄 Reproducible Research:** One-command data acquisition and processing script.
- **📈 Proven Accuracy:** High-AUC performance (up to 0.99 for Heart Disease).

---

## 🛠️ Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/sfu-cmpt340/2026_1_project_15.git
cd 2026_1_project_15

# Install dependencies
pip install -r requirements.txt
```

### 2. Quick Start (The "Demo")

For a fully automated demonstration of the data acquisition, model training, and diagnostic evaluation, run:

```bash
# Make the script executable
chmod +x demo_walkthrough.sh

# Run the automated demo
./demo_walkthrough.sh
```

Alternatively, you can run the steps manually:

### 3. Interactive Diagnosis

Launch the diagnostic CLI to test a patient profile:

```bash
python src/predict.py
```

---

## 🏗️ Architecture & How It Works

DI-PRED uses a decoupled architecture to ensure modularity and scalability:

1.  **Data Layer:** Idempotent scripts fetch raw data from UCI and Pima repositories, performing initial binarization.
2.  **Preprocessing Layer:** Handles median-imputation for missing values, feature scaling (StandardScaler), and stratified splitting.
3.  **Model Layer:**
    *   **Diabetes:** Uses **Logistic Regression** (Optimized for linear separability).
    *   **Heart Disease:** Uses **Random Forest** (Optimized for non-linear feature interactions).
4.  **Evaluation Layer:** Generates a `metrics.csv` leaderboard and high-resolution PNG visualizations in `results/figures/`.

---

## 📊 Comparison: Model Performance

| Dataset | Best Model | Accuracy | AUC Score |
| :--- | :--- | :--- | :--- |
| **Diabetes** | Logistic Regression | 77.6% | 0.86 |
| **Heart Disease** | Random Forest | 98.1% | 0.99 |

---

## 📂 Project Structure

```bash
repository
├── data/
│   ├── raw/             ## Original medical CSVs
│   ├── processed/       ## Scaled .npy train/test splits
│   └── download_data.py ## Data acquisition & cleaning
├── models/              ## Serialized .pkl models & scalers
├── notebooks/           ## EDA & Result Analysis
├── results/             ## Metrics & PNG visualizations
└── src/                 ## Core modular source code
```

---

## 🤝 Contributing

Contributions are welcome! Please check the `CONTRIBUTORS.md` for our team details. 

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
