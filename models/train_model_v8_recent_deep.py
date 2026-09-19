#!/usr/bin/env python3
"""
STEP 10: Deep Feature Exploration — Recent Regime Only (2021-2026)
------------------------------------------------------------------------
Naye feature categories jo pehle try nahi kiye:
1. Gap_Pct - overnight gap (aaj Open vs kal Close)
2. MACD - trend + momentum combined indicator
3. Volatility_10 - rolling volatility (kitna 'jhatakedar' hai market)
4. Momentum_5/10/20 - multi-window trend strength
5. Prev_Direction - kal ka actual outcome (autocorrelation, NOT leakage)
6. DayOfWeek - Monday/Friday jaisa seasonal effect

Chalane ka tarika:
    python3 train_model_v8_recent_deep.py
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
    return 100 - (100 / (1 + rs))


def calculate_bollinger_percent_b(prices, window=20, num_std=2):
    rolling_mean = prices.rolling(window=window, min_periods=window).mean()
    rolling_std = prices.rolling(window=window, min_periods=window).std()
    upper = rolling_mean + (num_std * rolling_std)
    lower = rolling_mean - (num_std * rolling_std)
    return (prices - lower) / (upper - lower)


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    MACD Line = 12-day EMA - 26-day EMA
    Signal Line = 9-day EMA of MACD Line
    Histogram = MACD Line - Signal Line (positive = bullish momentum)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return histogram


def engineer_deep_features(df):
    """Purane + naye saare features, per-symbol group karke"""
    df = df.sort_values(["Symbol", "Date"]).copy()
    engineered_rows = []

    for symbol, group in df.groupby("Symbol"):
        group = group.sort_values("Date").copy()
        group["Date"] = pd.to_datetime(group["Date"])

        # Purane features
        group["Daily_Return_Pct"] = ((group["Close"] - group["Open"]) / group["Open"]) * 100
        group["Intraday_Spread_Pct"] = ((group["High"] - group["Low"]) / group["Open"]) * 100
        group["Volume_Change_Pct"] = group["Volume"].pct_change() * 100
        group["Return_MA_3"] = group["Daily_Return_Pct"].rolling(window=3).mean()
        group["RSI_14"] = calculate_rsi(group["Close"], period=14)
        group["Bollinger_PercentB"] = calculate_bollinger_percent_b(group["Close"], window=20)

        # NAYE features
        group["Gap_Pct"] = ((group["Open"] - group["Close"].shift(1)) / group["Close"].shift(1)) * 100
        group["MACD_Histogram"] = calculate_macd(group["Close"])
        group["Volatility_10"] = group["Daily_Return_Pct"].rolling(window=10).std()
        group["Momentum_5"] = group["Close"].pct_change(periods=5) * 100
        group["Momentum_10"] = group["Close"].pct_change(periods=10) * 100
        group["Momentum_20"] = group["Close"].pct_change(periods=20) * 100
        group["Prev_Direction"] = group["NextDayDirection"].shift(1)  # kal ka ACTUAL outcome
        group["DayOfWeek"] = group["Date"].dt.dayofweek  # 0=Monday, 4=Friday

        engineered_rows.append(group)

    result = pd.concat(engineered_rows, ignore_index=True)

    feature_check_cols = [
        "Daily_Return_Pct", "Intraday_Spread_Pct", "Volume_Change_Pct", "Return_MA_3",
        "RSI_14", "Bollinger_PercentB", "Gap_Pct", "MACD_Histogram", "Volatility_10",
        "Momentum_5", "Momentum_10", "Momentum_20", "Prev_Direction", "NextDayDirection"
    ]
    before = len(result)
    result = result.dropna(subset=feature_check_cols)
    result["NextDayDirection"] = result["NextDayDirection"].astype(int)
    print(f"Rows before cleanup: {before}, after: {len(result)}")

    return result


def filter_recent_years(df, years=5):
    df["Date"] = pd.to_datetime(df["Date"])
    cutoff = df["Date"].max() - pd.DateOffset(years=years)
    return df[df["Date"] >= cutoff].copy()


def run_cross_validation(df, feature_columns, n_splits=5):
    df = df.sort_values("Date").reset_index(drop=True)
    X = df[feature_columns].values
    y = df["NextDayDirection"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    accuracies, baselines, diffs = [], [], []

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        most_common = pd.Series(y_train).mode()[0]
        baseline_acc = accuracy_score(y_test, [most_common] * len(y_test))

        model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        model_acc = accuracy_score(y_test, model.predict(X_test))

        diff = (model_acc - baseline_acc) * 100
        accuracies.append(model_acc)
        baselines.append(baseline_acc)
        diffs.append(diff)

        print(f"Fold {fold_num}: Train={len(train_idx):5d} Test={len(test_idx):5d} | "
              f"Baseline={baseline_acc*100:.1f}% | Model={model_acc*100:.1f}% | Diff={diff:+.2f}%")

    return accuracies, baselines, diffs, model


def main():
    print("Loading dataset...")
    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=["NextDayDirection"])

    print("Engineering deep feature set on dataset...")
    df = engineer_deep_features(df)

    print("Filtering to recent 5 years (2021-2026)...")
    df = filter_recent_years(df, years=5)

    feature_columns = [
        "Daily_Return_Pct", "Intraday_Spread_Pct", "Volume_Change_Pct", "Return_MA_3",
        "RSI_14", "Bollinger_PercentB", "Gap_Pct", "MACD_Histogram", "Volatility_10",
        "Momentum_5", "Momentum_10", "Momentum_20", "Prev_Direction", "DayOfWeek"
    ]

    print(f"\nTotal features: {len(feature_columns)}")
    print("Running 5-fold Time-Series Cross-Validation on RECENT DATA ONLY...\n")

    accuracies, baselines, diffs, model = run_cross_validation(df, feature_columns, n_splits=5)

    print("\n" + "="*60)
    print(f"Average Model Accuracy:    {np.mean(accuracies)*100:.2f}%")
    print(f"Average Baseline Accuracy: {np.mean(baselines)*100:.2f}%")
    print(f"Average Improvement:       {np.mean(diffs):+.2f} points")

    t_stat, p_value = stats.ttest_1samp(diffs, 0)
    print(f"P-value: {p_value:.4f}")
    print("="*60)

    if np.mean(diffs) > 0.5 and p_value < 0.05:
        print("\n✅ Naye features ne genuinely modern-regime signal diya!")
    else:
        print("\n⚠️  Abhi bhi reliable signal nahi mila modern data mein.")

    print("\nFeature Importance (Deep Recent Model):")
    importances = sorted(zip(feature_columns, model.feature_importances_), key=lambda x: -x[1])
    for feature, imp in importances:
        print(f"  {feature:<20} {imp:.3f}")


if __name__ == "__main__":
    main()
