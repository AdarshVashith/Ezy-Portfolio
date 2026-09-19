#!/usr/bin/env python3
"""
STEP 9: Big Historical Dataset Test (76,000+ Multi-Decade Candles)
-------------------------------------------------------------------
Testing Pure Technical Indicators (RSI, Bollinger %B, Returns, Momentum)
on 30 years of historical data across 11 top NSE stocks.

Chalane ka tarika:
    python3 train_model_v6_bigdata.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# Support both running from `models/` directory or root workspace
if os.path.exists("ml_dataset.csv"):
    CSV_FILE = "ml_dataset.csv"
elif os.path.exists("../ml_dataset.csv"):
    CSV_FILE = "../ml_dataset.csv"
else:
    CSV_FILE = "/Users/adarshvashistha/Desktop/EZ/ml_dataset.csv"


def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_percent_b(prices, window=20, num_std=2):
    rolling_mean = prices.rolling(window=window, min_periods=window).mean()
    rolling_std = prices.rolling(window=window, min_periods=window).std()

    upper_band = rolling_mean + (num_std * rolling_std)
    lower_band = rolling_mean - (num_std * rolling_std)

    percent_b = (prices - lower_band) / (upper_band - lower_band)
    return percent_b


def engineer_pure_price_features(df):
    """Calculates stock-agnostic technical features per symbol on 76k+ rows"""
    df = df.sort_values(["Symbol", "Date"]).copy()
    engineered_rows = []

    for symbol, group in df.groupby("Symbol"):
        group = group.sort_values("Date").copy()

        group["Daily_Return_Pct"] = ((group["Close"] - group["Open"]) / group["Open"]) * 100
        group["Intraday_Spread_Pct"] = ((group["High"] - group["Low"]) / group["Open"]) * 100
        group["Volume_Change_Pct"] = group["Volume"].pct_change() * 100
        group["Return_MA_3"] = group["Daily_Return_Pct"].rolling(window=3).mean()
        group["Return_MA_7"] = group["Daily_Return_Pct"].rolling(window=7).mean()

        group["RSI_14"] = calculate_rsi(group["Close"], period=14)
        group["Bollinger_PercentB"] = calculate_bollinger_percent_b(group["Close"], window=20)

        engineered_rows.append(group)

    result = pd.concat(engineered_rows, ignore_index=True)

    feature_cols = [
        "Daily_Return_Pct", "Intraday_Spread_Pct", "Volume_Change_Pct",
        "Return_MA_3", "Return_MA_7", "RSI_14", "Bollinger_PercentB", "NextDayDirection"
    ]
    before = len(result)
    result = result.dropna(subset=feature_cols)
    result["NextDayDirection"] = result["NextDayDirection"].astype(int)
    print(f"Total Rows: {before} -> Usable Clean Rows: {len(result)}")
    return result


def run_cross_validation(df, feature_columns, n_splits=5):
    df = df.sort_values("Date").reset_index(drop=True)
    X = df[feature_columns].values
    y = df["NextDayDirection"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_accuracies, fold_baselines = [], []

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        most_common = pd.Series(y_train).mode()[0]
        baseline_acc = accuracy_score(y_test, [most_common] * len(y_test))

        model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        model_acc = accuracy_score(y_test, model.predict(X_test))

        fold_accuracies.append(model_acc)
        fold_baselines.append(baseline_acc)

        print(f"Fold {fold_num}: Train={len(train_idx):5d} | Test={len(test_idx):5d} | "
              f"Baseline={baseline_acc*100:.1f}% | Model={model_acc*100:.1f}% | "
              f"Diff={((model_acc-baseline_acc)*100):+.2f}%")

    return fold_accuracies, fold_baselines, model


def main():
    print("Loading 76k+ multi-decade dataset...")
    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=["NextDayDirection"])

    print("Engineering technical features (RSI, Bollinger, Momentum)...")
    df = engineer_pure_price_features(df)

    feature_columns = [
        "Daily_Return_Pct", "Intraday_Spread_Pct", "Volume_Change_Pct",
        "Return_MA_3", "Return_MA_7", "RSI_14", "Bollinger_PercentB"
    ]

    print(f"\nFeatures ({len(feature_columns)}): {feature_columns}")
    print("\nRunning 5-fold Time-Series Cross-Validation on ~75,000 samples...\n")

    accuracies, baselines, last_model = run_cross_validation(df, feature_columns, n_splits=5)

    print("\n" + "="*65)
    print(f"Average Model Accuracy:    {np.mean(accuracies)*100:.2f}% (+/- {np.std(accuracies)*100:.2f}%)")
    print(f"Average Baseline Accuracy: {np.mean(baselines)*100:.2f}% (+/- {np.std(baselines)*100:.2f}%)")
    print(f"Average Improvement:       {(np.mean(accuracies)-np.mean(baselines))*100:+.2f} points")
    print("="*65)

    print("\nFeature Importance (on 75k+ samples):")
    importances = sorted(zip(feature_columns, last_model.feature_importances_), key=lambda x: -x[1])
    for feat, imp in importances:
        print(f"  {feat:<22} {imp:.3f}")


if __name__ == "__main__":
    main()
