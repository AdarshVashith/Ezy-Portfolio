#!/usr/bin/env python3
"""
STEP 7: Time-Series Cross-Validation — Robust Accuracy Estimate
--------------------------------------------------------------------
Ek single train/test split pe bharosa mat karo — accuracy sirf luck se
upar-neeche ho sakti hai. Isliye multiple splits pe test karke
AVERAGE + VARIABILITY dekhte hain — ye zyada honest estimate deta hai.

Chalane ka tarika:
    python3 train_model_v4_cv.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# Ensure models directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Support both running from `models/` directory or root workspace
if os.path.exists("ml_dataset.csv"):
    CSV_FILE = "ml_dataset.csv"
elif os.path.exists("../ml_dataset.csv"):
    CSV_FILE = "../ml_dataset.csv"
else:
    CSV_FILE = "/Users/adarshvashistha/Desktop/EZ/ml_dataset.csv"

from train_model_v2 import engineer_features


def run_cross_validation(df, feature_columns, n_splits=5):
    """
    TimeSeriesSplit: har fold mein training set thoda bada hota jaata hai,
    aur test set hamesha training ke 'baad' ke time period se hota hai
    (normal K-Fold ki tarah random nahi, taaki future leak na ho).
    """
    df = df.sort_values("Date").reset_index(drop=True)
    X = df[feature_columns].values
    y = df["NextDayDirection"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_accuracies = []
    fold_baselines = []

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Baseline for this fold
        most_common = pd.Series(y_train).mode()[0]
        baseline_acc = accuracy_score(y_test, [most_common] * len(y_test))

        # Model for this fold
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        model_acc = accuracy_score(y_test, predictions)

        fold_accuracies.append(model_acc)
        fold_baselines.append(baseline_acc)

        print(f"Fold {fold_num}: Train={len(train_idx):4d} Test={len(test_idx):4d} | "
              f"Baseline={baseline_acc*100:.1f}% | Model={model_acc*100:.1f}% | "
              f"Diff={((model_acc-baseline_acc)*100):+.1f}%")

    return fold_accuracies, fold_baselines


def main():
    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=["NextDayDirection"])
    df = engineer_features(df)

    feature_columns = [
        "Daily_Return_Pct", "Intraday_Spread_Pct", "Volume_Change_Pct",
        "Sentiment_Lag_1", "Return_MA_3", "NewsCount"
    ]

    print("Running 5-fold Time-Series Cross-Validation...\n")
    accuracies, baselines = run_cross_validation(df, feature_columns, n_splits=5)

    print("\n" + "="*60)
    print(f"Average Model Accuracy:    {np.mean(accuracies)*100:.1f}% (+/- {np.std(accuracies)*100:.1f}%)")
    print(f"Average Baseline Accuracy: {np.mean(baselines)*100:.1f}% (+/- {np.std(baselines)*100:.1f}%)")
    print(f"Average Improvement:       {(np.mean(accuracies)-np.mean(baselines))*100:+.1f} points")
    print("="*60)
    print("\nAgar average improvement chhota hai (< 2-3 points) aur variability")
    print("(the +/- number) bada hai, to model reliably baseline beat NAHI kar raha.")


if __name__ == "__main__":
    main()
