#!/bin/bash

# CMPT 340 Project: DI-PRED Demo Script
# This script walks through the entire pipeline for a demo presentation.

echo "=========================================================="
echo "          DI-PRED: DISEASE PREDICTION SYSTEM              "
echo "=========================================================="
echo ""

# 1. DOWNLOAD DATA
echo "[PHASE 1] Fetching raw medical datasets..."
python3 data/download_data.py
echo ""
sleep 2

# 2. RUN PIPELINE
echo "[PHASE 2] Executing ML Pipeline (Preprocessing -> Training -> Evaluation)..."
python3 main.py
echo ""
sleep 2

# 3. SHOW RESULTS
echo "[PHASE 3] Pipeline complete! Viewing final metrics..."
head -n 5 results/metrics.csv | column -t -s,
echo ""
echo "Visualizations generated in results/figures/ (e.g., ROC curves, Heatmaps)"
ls results/figures/ | head -n 5
echo ""
sleep 2

# 4. INTERACTIVE DEMO
echo "[PHASE 4] Interactive Prediction Simulation"
echo "Running a sample patient prediction for Heart Disease..."
echo ""
# Inputs for Heart: 1 (Diabetes), 5 pregnancies, 120 glucose, 70 BP, 20 skin, 100 insulin, 28.5 BMI, 0.5 DP, 45 age
printf "1\n5\n120\n70\n20\n100\n28.5\n0.5\n45\n" | python3 src/predict.py
echo ""

echo "=========================================================="
echo "          DEMO COMPLETE! READY FOR PRESENTATION           "
echo "=========================================================="
