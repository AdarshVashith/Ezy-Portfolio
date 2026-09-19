#!/usr/bin/env python3
"""
STEP 8: Advanced Features — Multi-day Sentiment + RSI + Bollinger Bands
---------------------------------------------------------------------------
Naye features:
1. Sentiment_MA_3 / Sentiment_MA_7 — single-day sentiment ki jagah rolling average
   (noise kam karta hai, single din ki random news ka bura effect nahi padega)
2. RSI (Relative Strength Index) — momentum indicator, overbought/oversold batata hai
3. Bollinger Bands (%B) — price kahan hai apne "normal range" ke comparison mein

Chalane ka tarika:
    python3 train_model_v5_advanced.py
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
    """
    RSI formula: 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss (over 'period' days)

    RSI > 70 = "overbought" (shayad neeche aayega)
    RSI < 30 = "oversold" (shayad upar jayega)
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)  # divide by zero se bachne ke liye
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_percent_b(prices, window=20, num_std=2):
    """
    %B batata hai price kahan hai Bollinger Band ke andar:
    %B = 0   -> price lower band pe hai (oversold zone)
    %B = 1   -> price upper band pe hai (overbought zone)
    %B = 0.5 -> price bilkul middle mein hai (average)
    """
    rolling_mean = prices.rolling(window=window, min_periods=window).mean()
    rolling_std = prices.rolling(window=window, min_periods=window).std()

    upper_band = rolling_mean + (num_std * rolling_std)
    lower_band = rolling_mean - (num_std * rolling_std)

    percent_b = (prices - lower_band) / (upper_band - lower_band)
    return percent_b


def engineer_advanced_features(df):
    """Saare features ek saath — pichhle wale + naye advanced wale, per-symbol"""
    df = df.sort_values(["Symbol", "Date"]).copy()
    engineered_rows = []

    for symbol, group in df.groupby("Symbol"):
        group = group.sort_values("Date").copy()

        # Purane features (v2 se)
        group["Daily_Return_Pct"] = ((group["Close"] - group["Open"]) / group["Open"]) * 100
        group["Intraday_Spread_Pct"] = ((group["High"] - group["Low"]) / group["Open"]) * 100
        group["Volume_Change_Pct"] = group["Volume"].pct_change() * 100
        group["Return_MA_3"] = group["Daily_Return_Pct"].rolling(window=3).mean()

        # NAYE: Multi-day sentiment (single-day ki jagah rolling average)
        group["Sentiment_MA_3"] = group["AvgSentiment"].shift(1).rolling(window=3).mean()
        group["Sentiment_MA_7"] = group["AvgSentiment"].shift(1).rolling(window=7).mean()

        # NAYE: RSI aur Bollinger Bands
        group["RSI_14"] = calculate_rsi(group["Close"], period=14)
        group["Bollinger_PercentB"] = calculate_bollinger_percent_b(group["Close"], window=20)

        engineered_rows.append(group)

    result = pd.concat(engineered_rows, ignore_index=True)

    feature_cols_to_check = [
        "Daily_Return_Pct", "Intraday_Spread_Pct", "Volume_Change_Pct", "Return_MA_3",
        "Sentiment_MA_3", "Sentiment_MA_7", "RSI_14", "Bollinger_PercentB", "NextDayDirection"
    ]
    before = len(result)
    result = result.dropna(subset=feature_cols_to_check)
    result["NextDayDirection"] = result["NextDayDirection"].astype(int)
    print(f"Rows before cleanup: {before}, after: {len(result)}")
    print("(RSI/Bollinger ko 14-20 din ka warmup chahiye, isliye shuru ke kuch din har symbol ke drop honge)")

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

        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        model_acc = accuracy_score(y_test, model.predict(X_test))

        fold_accuracies.append(model_acc)
        fold_baselines.append(baseline_acc)

        print(f"Fold {fold_num}: Baseline={baseline_acc*100:.1f}% | Model={model_acc*100:.1f}% | "
              f"Diff={((model_acc-baseline_acc)*100):+.1f}%")

    return fold_accuracies, fold_baselines


def main():
    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=["NextDayDirection"])

    print("Engineering advanced features (RSI, Bollinger, multi-day sentiment)...")
    df = engineer_advanced_features(df)

    feature_columns = [
        "Daily_Return_Pct", "Intraday_Spread_Pct", "Volume_Change_Pct", "Return_MA_3",
        "Sentiment_MA_3", "Sentiment_MA_7", "RSI_14", "Bollinger_PercentB", "NewsCount"
    ]

    print(f"\n--- TEST A: Advanced features (RSI, Bollinger, multi-day sentiment) ---")
    accuracies_adv, baselines_adv = run_cross_validation(df, feature_columns, n_splits=5)

    basic_columns = [
        "Daily_Return_Pct", "Intraday_Spread_Pct", "Volume_Change_Pct", "Return_MA_3", "NewsCount"
    ]
    print(f"\n--- TEST B: Basic features ONLY, SAME rows (fair comparison) ---")
    accuracies_basic, baselines_basic = run_cross_validation(df, basic_columns, n_splits=5)

    print("\n" + "="*60)
    print("FAIR COMPARISON (both tested on identical 1368 rows):")
    print(f"Basic features    -> Avg Accuracy: {np.mean(accuracies_basic)*100:.1f}% | vs baseline: {(np.mean(accuracies_basic)-np.mean(baselines_basic))*100:+.1f}%")
    print(f"Advanced features -> Avg Accuracy: {np.mean(accuracies_adv)*100:.1f}% | vs baseline: {(np.mean(accuracies_adv)-np.mean(baselines_adv))*100:+.1f}%")
    diff_real = (np.mean(accuracies_adv) - np.mean(accuracies_basic)) * 100
    print(f"Real difference from adding RSI/Bollinger/multi-day sentiment: {diff_real:+.1f} points")
    print("="*60)
    print("\nAgar ye 'real difference' bhi 2-3 points se kam hai, to advanced features ne")
    print("genuinely kuch nahi add kiya — jo dikha wo dataset-change ka side-effect tha.")


if __name__ == "__main__":
    main()
