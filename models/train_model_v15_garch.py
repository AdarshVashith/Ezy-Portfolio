#!/usr/bin/env python3
"""
STEP 17: GARCH(1,1) Model — Formal Volatility Clustering
--------------------------------------------------------------------
GARCH equation:
    sigma_t^2 = omega + alpha * epsilon_(t-1)^2 + beta * sigma_(t-1)^2

Matlab: AAJ ka volatility (sigma_t^2) depend karta hai:
    - omega: long-run average volatility (baseline)
    - alpha * epsilon_(t-1)^2: KAL ka "shock" (bada unexpected return kal hua tha kya)
    - beta * sigma_(t-1)^2: KAL ka volatility level

Ye Step 12 (simple persistence rule) ka FORMAL, probabilistic upgrade hai --
persistence rule sirf "aaj jaisa kal" bolta tha, GARCH batata hai EXACTLY
kitna % weight shocks ko aur kitna % weight purani volatility ko dena hai.

Chalane ka tarika:
    python3 train_model_v15_garch.py
"""

import os
import pandas as pd
import numpy as np
from arch import arch_model
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


def fit_garch_for_symbol(returns_series):
    """
    Ek symbol ke returns pe GARCH(1,1) fit karta hai.
    returns_series: daily returns in PERCENT (jaise 1.5 matlab 1.5%)
    """
    model = arch_model(returns_series, vol="Garch", p=1, q=1, dist="normal")
    fitted = model.fit(disp="off")
    return fitted


def evaluate_garch_vs_persistence(df, symbol, test_fraction=0.2):
    """
    GARCH ki forecast volatility ko:
    1. Simple persistence rule (Step 12 wala) se compare karta hai
    2. Actual realized volatility (|actual return|) se compare karta hai (RMSE)

    IMPORTANT: Ye MANUAL recursive forecast use karta hai (static .forecast() call
    repeat karne ki jagah), taaki har din ka forecast naye actual return se
    genuinely update ho -- no look-ahead.
    """
    symbol_df = df[df["Symbol"] == symbol].sort_values("Date").copy()
    symbol_df["Daily_Return_Pct"] = ((symbol_df["Close"] - symbol_df["Open"]) / symbol_df["Open"]) * 100

    returns = symbol_df["Daily_Return_Pct"].dropna().reset_index(drop=True)

    split_idx = int(len(returns) * (1 - test_fraction))
    train_returns = returns.iloc[:split_idx]
    test_returns = returns.iloc[split_idx:].reset_index(drop=True)

    # Fit GARCH ONCE on training data
    fitted = fit_garch_for_symbol(train_returns)
    params = fitted.params
    omega = params["omega"]
    alpha = params["alpha[1]"]
    beta = params["beta[1]"]
    mu = params.get("mu", 0.0)

    # Starting point: training period ka last known residual^2 aur variance
    last_resid2 = fitted.resid.iloc[-1] ** 2
    last_var = fitted.conditional_volatility.iloc[-1] ** 2

    garch_forecasts = []
    persistence_forecasts = []
    actual_vol_proxy = []

    prev_actual_abs_return = abs(train_returns.iloc[-1] - mu)

    for i in range(len(test_returns)):
        # MANUAL RECURSION: is din ka forecast, PEHLE se pata cheezon se
        predicted_var = omega + alpha * last_resid2 + beta * last_var
        predicted_vol = np.sqrt(predicted_var)
        garch_forecasts.append(predicted_vol)

        # Persistence baseline: "kal jaisa aaj"
        persistence_forecasts.append(prev_actual_abs_return)

        # Ab AAJ ka actual return dekh lo -- update state
        actual_return_today = test_returns.iloc[i]
        actual_resid_today = actual_return_today - mu
        actual_vol_proxy.append(abs(actual_resid_today))

        # Agla iteration ke liye recursion update karo
        last_resid2 = actual_resid_today ** 2
        last_var = predicted_var
        prev_actual_abs_return = abs(actual_resid_today)

    garch_forecasts = np.array(garch_forecasts)
    persistence_forecasts = np.array(persistence_forecasts)
    actual_vol_proxy = np.array(actual_vol_proxy)

    garch_rmse = np.sqrt(np.mean((garch_forecasts - actual_vol_proxy) ** 2))
    persistence_rmse = np.sqrt(np.mean((persistence_forecasts - actual_vol_proxy) ** 2))

    return fitted, garch_rmse, persistence_rmse


def main():
    print(f"Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    df["Date"] = pd.to_datetime(df["Date"])

    symbols = df["Symbol"].unique()[:5]  # Top 5 symbols for demonstration

    print("="*70)
    print("GARCH(1,1) vs Simple Persistence Rule -- Volatility Forecast Comparison")
    print("="*70)

    all_garch_rmse, all_persistence_rmse = [], []

    for symbol in symbols:
        try:
            fitted, garch_rmse, persistence_rmse = evaluate_garch_vs_persistence(df, symbol)
            all_garch_rmse.append(garch_rmse)
            all_persistence_rmse.append(persistence_rmse)

            print(f"\n{symbol}:")
            print(f"  GARCH RMSE:       {garch_rmse:.4f}")
            print(f"  Persistence RMSE: {persistence_rmse:.4f}")
            winner = "GARCH" if garch_rmse < persistence_rmse else "Persistence"
            print(f"  Winner: {winner}")

            # Show fitted GARCH parameters
            params = fitted.params
            print(f"  Fitted equation: omega={params['omega']:.4f}, "
                  f"alpha={params['alpha[1]']:.4f}, beta={params['beta[1]']:.4f}")
            print(f"  Interpretation: {params['alpha[1]']*100:.1f}% weight to yesterday's shock, "
                  f"{params['beta[1]']*100:.1f}% weight to yesterday's volatility level")

        except Exception as e:
            print(f"\n{symbol}: FAILED -- {e}")

    print("\n" + "="*70)
    print(f"Average GARCH RMSE:       {np.mean(all_garch_rmse):.4f}")
    print(f"Average Persistence RMSE: {np.mean(all_persistence_rmse):.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
