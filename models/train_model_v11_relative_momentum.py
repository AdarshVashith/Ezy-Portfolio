#!/usr/bin/env python3
"""
STEP 13: Cross-Sectional Relative Momentum — Naya Angle
--------------------------------------------------------------------
Ab hum "absolute direction" nahi, "RELATIVE performance vs peers" predict
karte hain. Market-wide common noise cancel ho jaata hai is approach mein,
aur company-specific momentum clearer dikhta hai.

TARGET: Kal ye stock apne 11-stock basket ke average return se BETTER
        perform karega ya WORSE (relative outperformance).

Chalane ka tarika:
    python3 train_model_v11_relative_momentum.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from scipy import stats

# Support both running from `models/` directory or root workspace
if os.path.exists("ml_dataset.csv"):
    CSV_FILE = "ml_dataset.csv"
elif os.path.exists("../ml_dataset.csv"):
    CSV_FILE = "../ml_dataset.csv"
elif os.path.exists("ml_dataset_bigdata.csv"):
    CSV_FILE = "ml_dataset_bigdata.csv"
else:
    CSV_FILE = "/Users/adarshvashistha/Desktop/EZ/ml_dataset.csv"


def engineer_relative_momentum_features(df):
    """
    1. Har din, har stock ka Daily_Return_Pct calculate karo
    2. Har din ka CROSS-SECTIONAL AVERAGE return nikaalo (sab 11 stocks ka average)
    3. Har stock ka 'Relative_Return' = uska return - us din ka average return
    4. Rolling relative momentum banao (5-day, 10-day, 20-day average relative performance)
    5. TARGET: kal ka relative outperformance (1 = peers se better, 0 = worse)
    """
    df = df.sort_values(["Symbol", "Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Step 1: Daily return per stock
    df["Daily_Return_Pct"] = ((df["Close"] - df["Open"]) / df["Open"]) * 100

    # Step 2: Cross-sectional average per DATE (across all symbols that day)
    df["Cross_Sectional_Avg"] = df.groupby("Date")["Daily_Return_Pct"].transform("mean")

    # Step 3: Relative return (positive = better than peers today)
    df["Relative_Return"] = df["Daily_Return_Pct"] - df["Cross_Sectional_Avg"]

    engineered_rows = []
    for symbol, group in df.groupby("Symbol"):
        group = group.sort_values("Date").copy()

        # Step 4: Rolling relative momentum (trailing relative strength)
        group["Relative_Momentum_5"] = group["Relative_Return"].rolling(window=5).mean()
        group["Relative_Momentum_10"] = group["Relative_Return"].rolling(window=10).mean()
        group["Relative_Momentum_20"] = group["Relative_Return"].rolling(window=20).mean()

        # Extra context features
        group["Volume_Change_Pct"] = group["Volume"].pct_change() * 100
        group["Own_Volatility_10"] = group["Daily_Return_Pct"].rolling(window=10).std()

        # Step 5: TARGET -- kal relative outperform karega ya nahi
        group["Today_Outperform"] = (group["Relative_Return"] > 0).astype(int)
        group["Next_Day_Outperform"] = group["Today_Outperform"].shift(-1)

        engineered_rows.append(group)

    result = pd.concat(engineered_rows, ignore_index=True)

    check_cols = [
        "Relative_Momentum_5", "Relative_Momentum_10", "Relative_Momentum_20",
        "Volume_Change_Pct", "Own_Volatility_10", "Today_Outperform", "Next_Day_Outperform"
    ]
    before = len(result)
    result = result.dropna(subset=check_cols)
    result["Next_Day_Outperform"] = result["Next_Day_Outperform"].astype(int)
    print(f"Rows before cleanup: {before}, after: {len(result)}")

    return result


def filter_recent_years(df, years=5):
    df["Date"] = pd.to_datetime(df["Date"])
    cutoff = df["Date"].max() - pd.DateOffset(years=years)
    return df[df["Date"] >= cutoff].copy()


def run_relative_momentum_cv(df, feature_columns, n_splits=5):
    df = df.sort_values("Date").reset_index(drop=True)
    X = df[feature_columns].values
    y = df["Next_Day_Outperform"].values
    today_state = df["Today_Outperform"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    model_accs, majority_accs, persistence_accs = [], [], []

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        today_test = today_state[test_idx]

        majority_class = pd.Series(y_train).mode()[0]
        majority_acc = accuracy_score(y_test, [majority_class] * len(y_test))

        # Persistence: "aaj outperform kiya to kal bhi karega" (momentum continuation baseline)
        persistence_acc = accuracy_score(y_test, today_test)

        model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        model_acc = accuracy_score(y_test, model.predict(X_test))

        model_accs.append(model_acc)
        majority_accs.append(majority_acc)
        persistence_accs.append(persistence_acc)

        print(f"Fold {fold_num}: Majority={majority_acc*100:.1f}% | Persistence={persistence_acc*100:.1f}% | "
              f"Model={model_acc*100:.1f}% | Diff vs Maj={((model_acc-majority_acc)*100):+.2f}%")

    return model_accs, majority_accs, persistence_accs, model


def main():
    print("Loading dataset...")
    df = pd.read_csv(CSV_FILE)

    print("Engineering cross-sectional relative momentum features across dataset...")
    df = engineer_relative_momentum_features(df)

    print("Filtering to recent 5 years (2021-2026)...")
    df = filter_recent_years(df, years=5)

    # Fair features list with Today_Outperform included
    feature_columns = [
        "Today_Outperform", "Relative_Momentum_5", "Relative_Momentum_10", "Relative_Momentum_20",
        "Volume_Change_Pct", "Own_Volatility_10"
    ]

    print(f"\nRunning 5-fold Time-Series CV for RELATIVE MOMENTUM prediction...\n")
    model_accs, majority_accs, persistence_accs, model = run_relative_momentum_cv(df, feature_columns)

    print("\n" + "="*60)
    avg_model = np.mean(model_accs)
    avg_majority = np.mean(majority_accs)
    avg_persistence = np.mean(persistence_accs)

    print(f"Average Majority Baseline:    {avg_majority*100:.2f}%")
    print(f"Average Persistence Baseline: {avg_persistence*100:.2f}%")
    print(f"Average Model Accuracy:       {avg_model*100:.2f}%")

    diffs_vs_majority = [m - maj for m, maj in zip(model_accs, majority_accs)]
    t_stat, p_value = stats.ttest_1samp(diffs_vs_majority, 0)
    print(f"Model improvement vs Majority: {np.mean(diffs_vs_majority)*100:+.2f} points, p={p_value:.4f}")
    print("="*60)

    if np.mean(diffs_vs_majority) > 2 and p_value < 0.05:
        print("\n✅ Cross-sectional relative momentum se genuine signal mila!")
    else:
        print("\n⚠️  Is angle se bhi strong signal nahi mila.")

    print("\nFeature Importance:")
    importances = sorted(zip(feature_columns, model.feature_importances_), key=lambda x: -x[1])
    for feature, imp in importances:
        print(f"  {feature:<22} {imp:.3f}")


if __name__ == "__main__":
    main()
