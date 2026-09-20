#!/usr/bin/env python3
"""
STEP 19: Kelly Criterion -- Model Confidence se Optimal Bet Size
--------------------------------------------------------------------
Kelly formula:
    f* = (b*p - q) / b

Jahan:
    f* = capital ka kitna FRACTION invest karna chahiye (0 to 1)
    p  = winning probability (MODEL se aata hai -- predict_proba)
    q  = 1 - p (losing probability)
    b  = odds (gain/loss ratio)

Chalane ka tarika:
    python3 train_model_v17_kelly_criterion.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

if os.path.exists("ml_dataset.csv"):
    CSV_FILE = "ml_dataset.csv"
elif os.path.exists("../ml_dataset.csv"):
    CSV_FILE = "../ml_dataset.csv"
elif os.path.exists("ml_dataset_bigdata.csv"):
    CSV_FILE = "ml_dataset_bigdata.csv"
else:
    CSV_FILE = "/Users/adarshvashistha/Desktop/EZ/ml_dataset.csv"


def calculate_kelly_fraction(win_prob, win_amount_pct, loss_amount_pct):
    """
    win_prob: model ki probability ki trade jeetega (0 to 1)
    win_amount_pct: agar jeeta to kitna % gain (positive number)
    loss_amount_pct: agar hara to kitna % loss (positive number, magnitude)
    """
    if loss_amount_pct == 0:
        return 0

    b = win_amount_pct / loss_amount_pct  # odds ratio
    p = win_prob
    q = 1 - p

    kelly_fraction = (b * p - q) / b
    return max(0, kelly_fraction)


def simulate_kelly_vs_fixed_sizing(df, feature_columns, symbol):
    symbol_df = df[df["Symbol"] == symbol].sort_values("Date").copy()
    symbol_df["Daily_Return_Pct"] = ((symbol_df["Close"] - symbol_df["Open"]) / symbol_df["Open"]) * 100
    symbol_df["Next_Day_Return_Pct"] = symbol_df["Close"].pct_change().shift(-1) * 100
    symbol_df["Next_Day_Direction"] = (symbol_df["Next_Day_Return_Pct"] > 0).astype(int)

    symbol_df = symbol_df.dropna(subset=feature_columns + ["Next_Day_Direction", "Next_Day_Return_Pct"])

    split_idx = int(len(symbol_df) * 0.8)
    train_df = symbol_df.iloc[:split_idx]
    test_df = symbol_df.iloc[split_idx:].reset_index(drop=True)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(train_df[feature_columns], train_df["Next_Day_Direction"])

    avg_win_pct = train_df.loc[train_df["Next_Day_Return_Pct"] > 0, "Next_Day_Return_Pct"].mean()
    avg_loss_pct = abs(train_df.loc[train_df["Next_Day_Return_Pct"] < 0, "Next_Day_Return_Pct"].mean())

    probs = model.predict_proba(test_df[feature_columns])[:, 1]

    fixed_capital = 1.0
    kelly_capital = 1.0
    fixed_bet_size = 0.5

    fixed_history, kelly_history = [fixed_capital], [kelly_capital]

    for i in range(len(test_df)):
        actual_return = test_df["Next_Day_Return_Pct"].iloc[i] / 100
        win_prob = probs[i]

        # Fixed sizing
        fixed_capital *= (1 + fixed_bet_size * actual_return)
        fixed_history.append(fixed_capital)

        # Kelly sizing
        kelly_frac = calculate_kelly_fraction(win_prob, avg_win_pct, avg_loss_pct)
        kelly_frac = min(kelly_frac, 1.0)
        kelly_capital *= (1 + kelly_frac * actual_return)
        kelly_history.append(kelly_capital)

    return fixed_history, kelly_history, avg_win_pct, avg_loss_pct


def main():
    print(f"Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    df["Date"] = pd.to_datetime(df["Date"])

    symbol = "RELIANCE.NS"
    feature_columns = ["Daily_Return_Pct"]

    symbol_df = df[df["Symbol"] == symbol].sort_values("Date").copy()
    symbol_df["Daily_Return_Pct"] = ((symbol_df["Close"] - symbol_df["Open"]) / symbol_df["Open"]) * 100
    df.loc[df["Symbol"] == symbol, "Daily_Return_Pct"] = symbol_df["Daily_Return_Pct"]

    print(f"\nKelly Criterion Simulation for {symbol}")
    print("="*60)

    fixed_hist, kelly_hist, avg_win, avg_loss = simulate_kelly_vs_fixed_sizing(df, feature_columns, symbol)

    print(f"Historical avg WIN day: +{avg_win:.2f}%")
    print(f"Historical avg LOSS day: -{avg_loss:.2f}%")
    print(f"\nStarting capital: 1.00 (representing 100% of initial capital)")
    print(f"Final capital (Fixed 50% sizing):  {fixed_hist[-1]:.4f}")
    print(f"Final capital (Kelly sizing):       {kelly_hist[-1]:.4f}")

    print("\n" + "="*60)
    print("IMPORTANT CAVEAT:")
    print("Kelly sizing is optimal ONLY when model predict_proba is genuinely calibrated.")
    print("In absence of genuine edge, treat this strictly as a theoretical sizing demo.")


if __name__ == "__main__":
    main()
