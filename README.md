# DI-PRED: Advanced Disease Prediction System
**DI-PRED** (Diagnostic Predictor) is a high-performance machine learning framework for early-stage risk detection of Diabetes and Heart Disease from patient clinical data.

## Important Links
| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/IQDgUIFRK-A9QIVzN8vSNjCfAcLem4gWM2vQJVOOsGWfFOg) | [Slack channel](https://cmpt340spring2026.slack.com/archives/C0A9M1WBFPW) | [Project report](https://www.overleaf.com/project/696dc0102d552e14795dafb5) |
| -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------- |

- Timesheet: Link your timesheet (pinned in your project's Slack channel) where you track per student the time and tasks completed/participated for this project.
- Slack channel: Link your private Slack project channel.
- Project report: Link your Overleaf project report document.

## Video/demo/GIF
Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.

## Table of Contents
1. [Demo](#demo)
2. [Installation](#installation)
3. [Reproducing this project](#repro)
4. [Guidance](#guide)

<a name="demo"></a>
## 1. Example Demo

Launch the interactive diagnostic CLI to get a real-time risk assessment for a patient profile:

```bash
python src/predict.py
```

The CLI will prompt you to enter patient clinical values (e.g. glucose, blood pressure, cholesterol), validate your inputs, and return a probabilistic risk score for Diabetes and Heart Disease.

### What to find where

```bash
repository
├── data/
│   ├── raw/                  ## Original medical CSVs (diabetes.csv, heart.csv)
│   ├── processed/            ## Scaled .npy train/test splits
│   ├── samples/              ## Sample input files for testing
│   └── download_data.py      ## Data acquisition & cleaning script
├── models/                   ## Serialized .pkl models & scalers
│   ├── diabetes_best_...pkl
│   ├── diabetes_logistic...pkl
│   ├── diabetes_random...pkl
│   ├── diabetes_scaler...pkl
│   ├── heart_best_mo...pkl
│   ├── heart_logistic_r...pkl
│   ├── heart_random_...pkl
│   └── heart_scaler.pkl
├── notebooks/
│   ├── EDA.ipynb             ## Exploratory data analysis
│   └── results.ipynb         ## Result analysis
├── results/
│   ├── figures/              ## ROC curves, confusion matrices, feature importance PNGs
│   └── metrics.csv           ## Model performance leaderboard
├── src/                      ## Core modular source code
│   ├── config.py             ## Project configuration & constants
│   ├── evaluate.py           ## Model evaluation & figure generation
│   ├── predict.py            ## Interactive diagnostic CLI
│   ├── preprocess.py         ## Data cleaning & feature scaling
│   ├── train.py              ## Model training pipeline
│   ├── utils.py              ## Shared utility functions
│   └── visualize.py          ## Plotting helpers
├── main.py                   ## Entry point
├── requirements.txt          ## Pip dependencies
├── requirements.yml          ## Conda dependencies
├── CONTRIBUTORS.md           ## Team details
├── LICENSE
└── README.md                 ## You are here
```

<a name="installation"></a>
## 2. Installation

Tested on Python 3.9+. Clone the repository and install dependencies:

```bash
git clone https://github.com/sfu-cmpt340/2026_1_project_15.git
cd 2026_1_project_15
pip install -r requirements.txt
```

> **Note:** This tool is intended for research and educational purposes only. Always consult a medical professional for actual diagnosis.

<a name="repro"></a>
## 3. Reproduction

To reproduce the results reported in our paper, run the following steps:

```bash
# Step 1: Download and preprocess data from UCI and Pima repositories
python data/download_data.py

# Step 2: Run model training and evaluation
python src/train.py

# Step 3: View the results
cat results/metrics.csv
```

Data is fetched automatically from the UCI Heart Disease repository and the Pima Indians Diabetes dataset. No manual download is required.

Output will be saved in:
- `results/metrics.csv` — accuracy and AUC scores for all models
- `results/figures/` — ROC curves, confusion matrices, and feature importance plots (high-resolution PNGs)
- `models/` — serialized `.pkl` model and scaler files

### Model Performance

| Dataset        | Best Model          | Accuracy | AUC Score |
| :------------- | :------------------ | :------- | :-------- |
| Diabetes       | Logistic Regression | 77.6%    | 0.86      |
| Heart Disease  | Random Forest       | 98.1%    | 0.99      |

<a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
  - Do NOT use history re-editing (rebase)
  - Commit messages should be informative:
    - No: `'this should fix it'`, `'bump'`
    - Yes: `'Resolve invalid API call in updating X'`
  - Do NOT include IDE folders (`.idea`) or hidden files. Update your `.gitignore` where needed.
  - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/)
