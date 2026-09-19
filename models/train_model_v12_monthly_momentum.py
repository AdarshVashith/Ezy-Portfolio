#!/usr/bin/env python3
"""
STEP 14: Classic Academic Momentum Test (Jegadeesh-Titman Style)
--------------------------------------------------------------------
Literature claim: "Winners" (high past 12-month return) tend to keep
outperforming over the NEXT month. Ye daily-frequency se bilkul alag
timescale hai -- monthly resampling karke test karte hain.

FORMATION: Pichle 12 mahine ka cumulative return (winner/loser signal)
TARGET: Agle mahine peers (cross-sectional average) se better ya worse

Chalane ka tarika:
    python3 train_model_v12_monthly_momentum.py
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


def resample_to_monthly(df):
    """Daily data ko month-end closing price mein convert karta hai, per symbol"""
    df["Date"] = pd.to_datetime(df["Date"])
    monthly_rows = []

    for symbol, group in df.groupby("Symbol"):
        group = group.set_index("Date").sort_index()
        # Support both 'ME' (pandas >= 2.2) and 'M' (older pandas)
        try:
            monthly_close = group["Close"].resample("ME").last().dropna()
        except ValueError:
            monthly_close = group["Close"].resample("M").last().dropna()
            
        monthly_df = monthly_close.reset_index()
        monthly_df["Symbol"] = symbol
        monthly_rows.append(monthly_df)

    result = pd.concat(monthly_rows, ignore_index=True)
    return result


def engineer_momentum_features(monthly_df):
    """
    Formation_Return_12M: pichle 12 mahine ka cumulative return
    Target: agle mahine cross-sectional median se better ya worse
    """
    monthly_df = monthly_df.sort_values(["Symbol", "Date"]).copy()
    monthly_df["Monthly_Return_Pct"] = monthly_df.groupby("Symbol")["Close"].pct_change() * 100

    engineered_rows = []
    for symbol, group in monthly_df.groupby("Symbol"):
        group = group.sort_values("Date").copy()

        # Formation period: trailing 12-month cumulative return (as of month t)
        growth_factor = (1 + group["Monthly_Return_Pct"] / 100)
        group["Formation_Return_12M"] = (
            growth_factor.rolling(window=12).apply(lambda x: x.prod(), raw=True) - 1
        ) * 100

        # Shorter formation window bhi try karte hain comparison ke liye
        group["Formation_Return_6M"] = (
            growth_factor.rolling(window=6).apply(lambda x: x.prod(), raw=True) - 1
        ) * 100

        # Volatility of monthly returns (risk context)
        group["Monthly_Volatility_12M"] = group["Monthly_Return_Pct"].rolling(window=12).std()

        engineered_rows.append(group)

    result = pd.concat(engineered_rows, ignore_index=True)

    # Cross-sectional median return PER MONTH (across all symbols) -- for the target
    result["Cross_Sectional_Median"] = result.groupby("Date")["Monthly_Return_Pct"].transform("median")
    result["Outperformed_Peers"] = (result["Monthly_Return_Pct"] > result["Cross_Sectional_Median"]).astype(int)

    # TARGET: NEXT month's outperformance (shift -1 within each symbol)
    result = result.sort_values(["Symbol", "Date"])
    result["Next_Month_Outperform"] = result.groupby("Symbol")["Outperformed_Peers"].shift(-1)

    check_cols = ["Formation_Return_12M", "Formation_Return_6M", "Monthly_Volatility_12M",
                  "Outperformed_Peers", "Next_Month_Outperform"]
    before = len(result)
    result = result.dropna(subset=check_cols)
    result["Next_Month_Outperform"] = result["Next_Month_Outperform"].astype(int)
    print(f"Monthly rows before cleanup: {before}, after: {len(result)}")
    print(f"(12-month formation window needs warmup, isliye shuru ke 12+ months har symbol ke drop honge)")

    return result


def run_momentum_cv(df, feature_columns, n_splits=3):
    """n_splits chhota rakha hai (3) kyunki monthly data mein samples kam hote hain"""
    df = df.sort_values("Date").reset_index(drop=True)
    X = df[feature_columns].values
    y = df["Next_Month_Outperform"].values
    today_state = df["Outperformed_Peers"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    model_accs, majority_accs, persistence_accs = [], [], []

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        today_test = today_state[test_idx]

        majority_class = pd.Series(y_train).mode()[0]
        majority_acc = accuracy_score(y_test, [majority_class] * len(y_test))
        persistence_acc = accuracy_score(y_test, today_test)

        model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        model_acc = accuracy_score(y_test, model.predict(X_test))

        model_accs.append(model_acc)
        majority_accs.append(majority_acc)
        persistence_accs.append(persistence_acc)

        print(f"Fold {fold_num}: Train={len(train_idx):4d} Test={len(test_idx):4d} | "
              f"Majority={majority_acc*100:.1f}% | Persistence={persistence_acc*100:.1f}% | "
              f"Model={model_acc*100:.1f}%")

    return model_accs, majority_accs, persistence_accs, model


def main():
    print("Loading dataset...")
    df = pd.read_csv(CSV_FILE)

    print("Resampling daily data to MONTHLY (this is the key change)...")
    monthly_df = resample_to_monthly(df)
    print(f"Monthly observations created: {len(monthly_df)}\n")

    print("Engineering 12-month formation momentum features...")
    df_final = engineer_momentum_features(monthly_df)

    feature_columns = [
        "Outperformed_Peers", "Formation_Return_12M", "Formation_Return_6M", "Monthly_Volatility_12M"
    ]

    print(f"\nTotal usable monthly observations: {len(df_final)}")
    print("Running 3-fold Time-Series CV for MONTHLY MOMENTUM...\n")

    model_accs, majority_accs, persistence_accs, model = run_momentum_cv(df_final, feature_columns)

    print("\n" + "="*60)
    avg_model = np.mean(model_accs)
    avg_majority = np.mean(majority_accs)
    avg_persistence = np.mean(persistence_accs)

    print(f"Average Majority Baseline:    {avg_majority*100:.2f}%")
    print(f"Average Persistence Baseline: {avg_persistence*100:.2f}%")
    print(f"Average Model Accuracy:       {avg_model*100:.2f}%")

    diffs = [m - maj for m, maj in zip(model_accs, majority_accs)]
    t_stat, p_value = stats.ttest_1samp(diffs, 0)
    print(f"Model vs Majority: {np.mean(diffs)*100:+.2f} points, p={p_value:.4f}")
    print("="*60)
    print("\n⚠️  IMPORTANT CAVEAT: Sirf 3 folds hain (monthly data se sample chhota hai).")
    print("   P-value yahan kam reliable hai bade daily-data experiments ke comparison mein.")
    print("   Isse 'directional hint' lo, 'proof' nahi.")

    print("\nFeature Importance:")
    importances = sorted(zip(feature_columns, model.feature_importances_), key=lambda x: -x[1])
    for feature, imp in importances:
        print(f"  {feature:<25} {imp:.3f}")


if __name__ == "__main__":
    main()
