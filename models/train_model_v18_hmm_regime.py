#!/usr/bin/env python3
"""
STEP 20: Hidden Markov Model -- Probabilistic Market Regime Detection
--------------------------------------------------------------------
Concept: Market switches between "Low Volatility / Calm" and
"High Volatility / Turbulent" hidden regimes.
HMM estimates:
    1. Current hidden regime probability
    2. Transition probability matrix between regimes

Chalane ka tarika:
    python3 train_model_v18_hmm_regime.py
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


def fit_hmm_regimes(returns, n_regimes=2):
    X = returns.values.reshape(-1, 1)

    model = GaussianHMM(n_components=n_regimes, covariance_type="full",
                         n_iter=1000, random_state=42)
    model.fit(X)

    hidden_states = model.predict(X)
    state_probs = model.predict_proba(X)

    return model, hidden_states, state_probs


def characterize_regimes(returns, hidden_states, n_regimes):
    summary = []
    for state in range(n_regimes):
        mask = hidden_states == state
        state_returns = returns[mask]
        summary.append({
            "Regime": state,
            "Avg_Return_Pct": state_returns.mean(),
            "Volatility_Pct": state_returns.std(),
            "Frequency_Pct": mask.mean() * 100,
            "Num_Days": mask.sum()
        })
    return pd.DataFrame(summary)


def main():
    print(f"Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    df["Date"] = pd.to_datetime(df["Date"])

    symbol = "RELIANCE.NS"
    symbol_df = df[df["Symbol"] == symbol].sort_values("Date").copy()
    symbol_df["Daily_Return_Pct"] = symbol_df["Close"].pct_change() * 100
    returns = symbol_df["Daily_Return_Pct"].dropna().reset_index(drop=True)

    print(f"\nFitting Hidden Markov Model (2 regimes) for {symbol}")
    print("="*60)

    model, hidden_states, state_probs = fit_hmm_regimes(returns, n_regimes=2)

    regime_summary = characterize_regimes(returns, hidden_states, n_regimes=2)

    regime_summary = regime_summary.sort_values("Volatility_Pct")
    regime_summary["Label"] = ["Low Volatility (Calm)", "High Volatility (Turbulent)"]

    print("\nRegime Characteristics:")
    print(regime_summary.to_string(index=False))

    print("\nTransition Probability Matrix:")
    print("(Row = Current Regime, Column = Tomorrow's Regime Probability)")
    transition_df = pd.DataFrame(
        model.transmat_,
        columns=[f"To Regime {i}" for i in range(2)],
        index=[f"From Regime {i}" for i in range(2)]
    )
    print(transition_df.round(4))

    current_state = hidden_states[-1]
    current_probs = state_probs[-1]

    print(f"\n{'='*60}")
    print(f"CURRENT REGIME (as of latest data point):")
    for i, prob in enumerate(current_probs):
        label = regime_summary[regime_summary["Regime"] == i]["Label"].values
        label_str = label[0] if len(label) > 0 else f"Regime {i}"
        print(f"  {label_str}: {prob*100:.1f}% probability")

    high_vol_regime_idx = regime_summary.sort_values("Volatility_Pct", ascending=False).iloc[0]["Regime"]
    high_vol_regime_idx = int(high_vol_regime_idx)
    stay_in_high_vol_prob = model.transmat_[high_vol_regime_idx, high_vol_regime_idx]

    print(f"\nProbability of STAYING in High-Volatility regime (day to day): "
          f"{stay_in_high_vol_prob*100:.1f}%")
    print("(Compare this to Step 12's empirical finding: persistence rule")
    print(" achieved ~57% accuracy predicting high-vol continuation)")

    print("\n" + "="*60)
    print("INTERPRETATION NOTE:")
    print("Stay probability significantly >50% confirms that volatility clustering")
    print("is a formal, statistically robust feature of market microstructure.")


if __name__ == "__main__":
    main()
