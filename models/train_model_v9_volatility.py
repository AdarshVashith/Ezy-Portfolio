#!/usr/bin/env python3
"""
STEP 11: Volatility Prediction — Naya Target, Naya Problem Framing (3-Way Baseline Test)
----------------------------------------------------------------------------------------
Direction (UP/DOWN) predict karne ki jagah, ab hum "kal kitna BADA
movement hoga" predict karenge (regression). Ye "volatility clustering"
phenomenon exploit karta hai, jo direction se zyada well-established hai.

3-Way Comparison:
1. Naive persistence baseline ("kal ka volatility aaj jaisa hoga")
2. Mean baseline ("hamesha training set ka mean predict karo")
3. Real Random Forest Regressor

Chalane ka tarika:
    python3 train_model_v9_volatility.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
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


def engineer_volatility_features(df):
    """
    TARGET: Next_Day_Volatility (agle din ka Intraday_Spread_Pct)
    FEATURES: aaj tak ka jo bhi pata hai (volatility history, volume, momentum)
    """
    df = df.sort_values(["Symbol", "Date"]).copy()
    engineered_rows = []

    for symbol, group in df.groupby("Symbol"):
        group = group.sort_values("Date").copy()
        group["Date"] = pd.to_datetime(group["Date"])

        # Aaj ka volatility measure (High-Low spread)
        group["Intraday_Spread_Pct"] = ((group["High"] - group["Low"]) / group["Open"]) * 100
        group["Abs_Return_Pct"] = ((group["Close"] - group["Open"]) / group["Open"]).abs() * 100
        group["Daily_Return_Pct"] = ((group["Close"] - group["Open"]) / group["Open"]) * 100

        # Rolling volatility features (past ka pattern)
        group["Volatility_5"] = group["Daily_Return_Pct"].rolling(window=5).std()
        group["Volatility_10"] = group["Daily_Return_Pct"].rolling(window=10).std()
        group["Spread_MA_5"] = group["Intraday_Spread_Pct"].rolling(window=5).mean()
        group["Volume_Change_Pct"] = group["Volume"].pct_change() * 100
        group["RSI_14"] = calculate_rsi(group["Close"], period=14)

        # TARGET: agle din ka volatility (isliye .shift(-1) — future value laate hain bas target ke liye)
        group["Next_Day_Volatility"] = group["Intraday_Spread_Pct"].shift(-1)

        engineered_rows.append(group)

    result = pd.concat(engineered_rows, ignore_index=True)

    feature_check_cols = [
        "Intraday_Spread_Pct", "Abs_Return_Pct", "Volatility_5", "Volatility_10",
        "Spread_MA_5", "Volume_Change_Pct", "RSI_14", "Next_Day_Volatility"
    ]
    before = len(result)
    result = result.dropna(subset=feature_check_cols)
    print(f"Rows before cleanup: {before}, after: {len(result)}")

    return result


def filter_recent_years(df, years=5):
    df["Date"] = pd.to_datetime(df["Date"])
    cutoff = df["Date"].max() - pd.DateOffset(years=years)
    return df[df["Date"] >= cutoff].copy()


def run_volatility_cv(df, feature_columns, n_splits=5):
    """
    Har fold mein 3 cheezein compare karenge:
    1. NAIVE BASELINE: "kal ka volatility aaj jaisa hoga" (persistence)
    2. MEAN BASELINE: "hamesha training set ka mean predict karo"
    3. MODEL: Random Forest Regressor jo features use karta hai
    """
    df = df.sort_values("Date").reset_index(drop=True)
    X = df[feature_columns].values
    y = df["Next_Day_Volatility"].values
    naive_predictions_source = df["Intraday_Spread_Pct"].values  # "aaj ka volatility" as naive guess

    tscv = TimeSeriesSplit(n_splits=n_splits)
    model_rmses, naive_rmses, mean_baseline_rmses, r2_scores = [], [], [], []

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        naive_test = naive_predictions_source[test_idx]

        # Baseline 1: Naive persistence ("kal jaisa aaj")
        naive_rmse = np.sqrt(mean_squared_error(y_test, naive_test))

        # Baseline 2: Sirf TRAINING data ka average volatility predict karo (constant)
        train_mean = y_train.mean()
        mean_baseline_predictions = np.full_like(y_test, train_mean)
        mean_baseline_rmse = np.sqrt(mean_squared_error(y_test, mean_baseline_predictions))

        # Real model
        model = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        model_rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        model_rmses.append(model_rmse)
        naive_rmses.append(naive_rmse)
        mean_baseline_rmses.append(mean_baseline_rmse)
        r2_scores.append(r2)

        vs_naive = ((naive_rmse - model_rmse) / naive_rmse) * 100
        vs_mean = ((mean_baseline_rmse - model_rmse) / mean_baseline_rmse) * 100
        print(f"Fold {fold_num}: Naive={naive_rmse:.3f} | MeanBaseline={mean_baseline_rmse:.3f} | "
              f"Model={model_rmse:.3f} | R²={r2:.3f} | vs Naive={vs_naive:+.1f}% | vs Mean={vs_mean:+.1f}%")

    return model_rmses, naive_rmses, mean_baseline_rmses, r2_scores, model


def main():
    print("Loading dataset...")
    df = pd.read_csv(CSV_FILE)

    print("Engineering volatility features across full dataset...")
    df = engineer_volatility_features(df)

    print("Filtering to recent 5 years (2021-2026)...")
    recent_df = filter_recent_years(df, years=5)

    feature_columns = [
        "Intraday_Spread_Pct", "Abs_Return_Pct", "Volatility_5", "Volatility_10",
        "Spread_MA_5", "Volume_Change_Pct", "RSI_14"
    ]

    print(f"\nRunning 5-fold Time-Series CV for VOLATILITY prediction...\n")
    model_rmses, naive_rmses, mean_baseline_rmses, r2_scores, model = run_volatility_cv(recent_df, feature_columns)

    print("\n" + "="*60)
    avg_model_rmse = np.mean(model_rmses)
    avg_naive_rmse = np.mean(naive_rmses)
    avg_mean_baseline_rmse = np.mean(mean_baseline_rmses)
    avg_r2 = np.mean(r2_scores)

    print(f"Average Naive (persistence) RMSE: {avg_naive_rmse:.3f}")
    print(f"Average Mean-Baseline RMSE:       {avg_mean_baseline_rmse:.3f}  <- 'bas average predict karo'")
    print(f"Average Model RMSE:               {avg_model_rmse:.3f}")
    print(f"Average R² score:                 {avg_r2:.3f}")
    print(f"Improvement vs Naive:             {((avg_naive_rmse-avg_model_rmse)/avg_naive_rmse)*100:+.1f}%")
    print(f"Improvement vs Mean-Baseline:      {((avg_mean_baseline_rmse-avg_model_rmse)/avg_mean_baseline_rmse)*100:+.1f}%")

    rmse_diffs_vs_naive = [n - m for n, m in zip(naive_rmses, model_rmses)]
    rmse_diffs_vs_mean = [mb - m for mb, m in zip(mean_baseline_rmses, model_rmses)]
    _, p_vs_naive = stats.ttest_1samp(rmse_diffs_vs_naive, 0)
    _, p_vs_mean = stats.ttest_1samp(rmse_diffs_vs_mean, 0)
    print(f"P-value vs Naive:         {p_vs_naive:.4f}")
    print(f"P-value vs Mean-Baseline: {p_vs_mean:.4f}")
    print("="*60)

    if avg_model_rmse < avg_mean_baseline_rmse and p_vs_mean < 0.05:
        print("\n✅ Model genuinely kuch seekh raha hai — sirf 'average predict karo' se BEHTAR hai!")
    else:
        print("\n⚠️  IMPORTANT: Model 'mean baseline' se better NAHI hai (ya barely better).")
        print("   -> Iska matlab: model ka 'naive ko beat karna' sirf isliye hai kyunki")
        print("      persistence baseline khud bahut noisy/erratic hai (kal ke spike ko copy karta hai).")
        print("      Model shayad sirf 'safe average ke paas predict karo' seekh raha hai,")
        print("      genuine pattern nahi. Ye alag hai 'real predictive skill' se.")

    print("\nFeature Importance:")
    importances = sorted(zip(feature_columns, model.feature_importances_), key=lambda x: -x[1])
    for feature, imp in importances:
        print(f"  {feature:<20} {imp:.3f}")


if __name__ == "__main__":
    main()
