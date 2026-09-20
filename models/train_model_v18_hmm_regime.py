#!/usr/bin/env python3
"""
STEP 20: Hidden Markov Model -- Apples-to-Apples Volatility Regime Detection
---------------------------------------------------------------------------
Apples-to-Apples benchmark matching Step 12:
- Dataset: Modern 5-Year Regime (2021-2026) across all 11 NSE stocks
- Observed Variable: Intraday_Spread_Pct ((High - Low) / Open * 100)
- Model: 2-State Gaussian Hidden Markov Model
- Compares: HMM High-Vol Persistence State Probability vs Step 12 Empirical Rule (57.29%)

Usage:
    python3 models/train_model_v18_hmm_regime.py
"""

import os
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings("ignore")

if os.path.exists("ml_dataset.csv"):
    CSV_FILE = "ml_dataset.csv"
elif os.path.exists("../ml_dataset.csv"):
    CSV_FILE = "../ml_dataset.csv"
elif os.path.exists("ml_dataset_bigdata.csv"):
    CSV_FILE = "ml_dataset_bigdata.csv"
else:
    CSV_FILE = "/Users/adarshvashistha/Desktop/EZ/ml_dataset.csv"


def prepare_apples_to_apples_data(df, start_date="2021-01-01"):
    """Prepares pooled 11-stock dataset using Intraday_Spread_Pct in modern era"""
    df = df.sort_values(["Symbol", "Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df["Intraday_Spread_Pct"] = ((df["High"] - df["Low"]) / df["Open"]) * 100
    df["Daily_Return_Pct"] = ((df["Close"] - df["Open"]) / df["Open"]) * 100

    # Filter strictly for modern regime (same as Step 12)
    modern_df = df[df["Date"] >= start_date].dropna(subset=["Intraday_Spread_Pct", "Daily_Return_Pct"]).copy()
    return modern_df


def fit_hmm_on_spread(spread_series, n_regimes=2):
    """Fits Gaussian HMM on Intraday Spread Pct"""
    X = spread_series.values.reshape(-1, 1)

    model = GaussianHMM(n_components=n_regimes, covariance_type="full",
                         n_iter=1000, random_state=42)
    model.fit(X)

    hidden_states = model.predict(X)
    state_probs = model.predict_proba(X)

    return model, hidden_states, state_probs


def characterize_regimes(data_df, hidden_states, n_regimes=2):
    summary = []
    for state in range(n_regimes):
        mask = hidden_states == state
        state_spreads = data_df["Intraday_Spread_Pct"].values[mask]
        state_returns = data_df["Daily_Return_Pct"].values[mask]
        summary.append({
            "Regime": state,
            "Mean_Spread_Pct": state_spreads.mean(),
            "Spread_Std_Pct": state_spreads.std(),
            "Mean_Return_Pct": state_returns.mean(),
            "Return_Vol_Pct": state_returns.std(),
            "Frequency_Pct": mask.mean() * 100,
            "Total_Candles": mask.sum()
        })
    res_df = pd.DataFrame(summary)
    # Sort by Mean_Spread_Pct so Regime 0 is Low Vol, Regime 1 is High Vol
    res_df = res_df.sort_values("Mean_Spread_Pct").reset_index(drop=True)
    res_df["Label"] = ["Low Volatility (Calm)", "High Volatility (Turbulent)"]
    return res_df


def main():
    print("=" * 80)
    print(" STEP 20: APPLES-TO-APPLES HIDDEN MARKOV MODEL VOLATILITY AUDIT")
    print("=" * 80)
    print(f"Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)

    # 1. Primary Apples-to-Apples Test: Modern 5-Year (2021-2026) Pooled 11 Stocks on Intraday_Spread_Pct
    modern_df = prepare_apples_to_apples_data(df, start_date="2021-01-01")
    n_stocks = modern_df["Symbol"].nunique()
    n_candles = len(modern_df)
    start_d = modern_df["Date"].min().strftime("%Y-%m-%d")
    end_d = modern_df["Date"].max().strftime("%Y-%m-%d")

    print(f"\n📊 PRIMARY TEST: Modern Era ({start_d} to {end_d}) | {n_stocks} Stocks Pooled | {n_candles} Candles")
    print("Target Feature: Intraday_Spread_Pct ((High - Low) / Open * 100)\n")

    model, hidden_states, state_probs = fit_hmm_on_spread(modern_df["Intraday_Spread_Pct"], n_regimes=2)
    regime_summary = characterize_regimes(modern_df, hidden_states, n_regimes=2)

    print("Learned Regime Characteristics:")
    print(regime_summary[["Regime", "Label", "Mean_Spread_Pct", "Spread_Std_Pct", "Return_Vol_Pct", "Frequency_Pct", "Total_Candles"]].to_string(index=False))

    # Identify High Vol Regime Index in model.transmat_
    high_vol_regime_id = regime_summary.loc[regime_summary["Label"] == "High Volatility (Turbulent)", "Regime"].values[0]
    low_vol_regime_id = regime_summary.loc[regime_summary["Label"] == "Low Volatility (Calm)", "Regime"].values[0]

    p_stay_high_vol = model.transmat_[high_vol_regime_id, high_vol_regime_id] * 100
    p_stay_low_vol = model.transmat_[low_vol_regime_id, low_vol_regime_id] * 100

    print("\nTransition Probability Matrix (Row = Today, Column = Tomorrow):")
    trans_matrix_df = pd.DataFrame(
        model.transmat_,
        columns=[f"To State {i}" for i in range(2)],
        index=[f"From State {i}" for i in range(2)]
    )
    print(trans_matrix_df.round(4))

    print("\n" + "-" * 80)
    print(f"🎯 APPLES-TO-APPLES COMPARISON WITH STEP 12 (Modern 2021-2026 Sample):")
    print(f"   • HMM Learned High-Vol Persistence Probability (P(High -> High)):  {p_stay_high_vol:.2f}%")
    print(f"   • HMM Learned Low-Vol Persistence Probability (P(Low -> Low)):    {p_stay_low_vol:.2f}%")
    print(f"   • Step 12 Simple 1-Lag Persistence Rule Accuracy:                 57.29%")
    print(f"   • Step 12 Majority-Class Baseline:                                49.21%")
    print("-" * 80)

    # 2. Per-Stock Breakdown across all 11 assets
    print("\n📈 Per-Stock High-Vol Persistence Rate in Modern Era:")
    print(f"{'Symbol':<18} | {'High-Vol Persistence (%)':<26} | {'Calm Persistence (%)':<22} | {'High-Vol Frequency'}")
    print("-" * 80)

    per_stock_persist = []
    for sym, sgroup in modern_df.groupby("Symbol"):
        try:
            s_model, s_states, _ = fit_hmm_on_spread(sgroup["Intraday_Spread_Pct"], n_regimes=2)
            s_sum = characterize_regimes(sgroup, s_states, n_regimes=2)
            s_high_id = s_sum.loc[s_sum["Label"] == "High Volatility (Turbulent)", "Regime"].values[0]
            s_low_id = s_sum.loc[s_sum["Label"] == "Low Volatility (Calm)", "Regime"].values[0]
            s_p_high = s_model.transmat_[s_high_id, s_high_id] * 100
            s_p_low = s_model.transmat_[s_low_id, s_low_id] * 100
            s_freq = s_sum.loc[s_sum["Label"] == "High Volatility (Turbulent)", "Frequency_Pct"].values[0]
            per_stock_persist.append(s_p_high)
            print(f"{sym:<18} | {s_p_high:>22.2f}% | {s_p_low:>18.2f}% | {s_freq:>14.1f}%")
        except Exception as e:
            print(f"{sym:<18} | Error: {e}")

    print("-" * 80)
    print(f"{'Average Across 11 Stocks':<18} | {np.mean(per_stock_persist):>22.2f}% |")
    print("=" * 80)


if __name__ == "__main__":
    main()
