#!/usr/bin/env python3
"""
STEP 9: Recent-Regime Test — Kya Signal AAJ Bhi Kaam Karta Hai?
--------------------------------------------------------------------
Poore 30-saal ke dataset mein signal mila, lekin fold 4-5 (recent years)
mein weak tha. Ye script SIRF last 5 saal ka data leke test karta hai —
taaki pata chale ki signal 'aaj ke market' mein bhi exist karta hai ya
sirf purane (1996-2010) data mein tha.

Chalane ka tarika:
    python3 train_model_v7_recent_only.py
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
    """Calculates stock-agnostic technical features per symbol on full dataset"""
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
    result = result.dropna(subset=feature_cols)
    result["NextDayDirection"] = result["NextDayDirection"].astype(int)
    return result


def filter_recent_years(df, years=5):
    """Sirf last N saal ka data rakhta hai"""
    df["Date"] = pd.to_datetime(df["Date"])
    cutoff_date = df["Date"].max() - pd.DateOffset(years=years)
    recent_df = df[df["Date"] >= cutoff_date].copy()
    print(f"Filtered to last {years} years: {len(recent_df)} rows "
          f"(from {recent_df['Date'].min().date()} to {recent_df['Date'].max().date()})")
    return recent_df


def run_cross_validation(df, feature_columns, n_splits=5):
    df = df.sort_values("Date").reset_index(drop=True)
    X = df[feature_columns].values
    y = df["NextDayDirection"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_accuracies, fold_baselines, fold_diffs = [], [], []

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        most_common = pd.Series(y_train).mode()[0]
        baseline_acc = accuracy_score(y_test, [most_common] * len(y_test))

        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        model_acc = accuracy_score(y_test, model.predict(X_test))

        diff = (model_acc - baseline_acc) * 100
        fold_accuracies.append(model_acc)
        fold_baselines.append(baseline_acc)
        fold_diffs.append(diff)

        print(f"Fold {fold_num}: Train={len(train_idx):5d} Test={len(test_idx):5d} | "
              f"Baseline={baseline_acc*100:.1f}% | Model={model_acc*100:.1f}% | Diff={diff:+.2f}%")

    return fold_accuracies, fold_baselines, fold_diffs, model


def main():
    print("Loading dataset...")
    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=["NextDayDirection"])

    print("Engineering technical features (RSI, Bollinger, Momentum)...")
    df = engineer_pure_price_features(df)

    print("="*60)
    print("TEST: Sirf Recent 5 Years (2021-2026) ka data")
    print("="*60)
    recent_df = filter_recent_years(df, years=5)

    feature_columns = [
        "Daily_Return_Pct", "Intraday_Spread_Pct", "Volume_Change_Pct",
        "Return_MA_3", "Return_MA_7", "RSI_14", "Bollinger_PercentB"
    ]

    accuracies, baselines, diffs, last_model = run_cross_validation(recent_df, feature_columns, n_splits=5)

    print("\n" + "="*60)
    print(f"Average Model Accuracy:    {np.mean(accuracies)*100:.2f}%")
    print(f"Average Baseline Accuracy: {np.mean(baselines)*100:.2f}%")
    print(f"Average Improvement:       {np.mean(diffs):+.2f} points")

    t_stat, p_value = stats.ttest_1samp(diffs, 0)
    print(f"P-value (is this signal reliable?): {p_value:.4f}")
    print("="*60)

    if np.mean(diffs) > 0.5 and p_value < 0.05:
        print("\n✅ Signal AAJ BHI exist karta hai recent market mein — genuinely useful!")
    elif np.mean(diffs) > 0:
        print("\n⚠️  Weak positive signal hai, but statistically confirm nahi ho raha pura.")
        print("   -> Signal 'alpha decay' ho raha hai — purane data mein strong tha, ab weak hai.")
    else:
        print("\n❌ Recent data mein signal GAYAB ho gaya hai.")
        print("   -> Purana pattern (1996-2010) ab kaam nahi karta — market ne 'adapt' kar liya.")
        print("   -> Ye bhi valuable finding hai: 'alpha decay' ka real example dekha tumne.")

    print("\nFeature Importance (Recent 5-Year Regime):")
    importances = sorted(zip(feature_columns, last_model.feature_importances_), key=lambda x: -x[1])
    for feat, imp in importances:
        print(f"  {feat:<22} {imp:.3f}")


if __name__ == "__main__":
    main()
