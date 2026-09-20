#!/usr/bin/env python3
"""
STEP 18: Monte Carlo Simulation -- Future Price Probability Fan
--------------------------------------------------------------------
Geometric Brownian Motion equation:
    S_(t+1) = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

Jahan:
    S_t = aaj ka price
    mu = historical average daily return (drift)
    sigma = historical daily volatility
    Z = random number from standard normal distribution (ye "randomness" hai)
    dt = 1 (ek din ka time-step)

Concept: Hum HAZAARON alag "possible futures" simulate karte hain (har ek mein
Z alag random value leta hai), aur dekhte hain in sab paths mein price kahan
kahan pahuncha. Isse ek PROBABILITY DISTRIBUTION milti hai future price ki.

Chalane ka tarika:
    python3 train_model_v16_monte_carlo.py
"""

import os
import pandas as pd
import numpy as np

if os.path.exists("ml_dataset.csv"):
    CSV_FILE = "ml_dataset.csv"
elif os.path.exists("../ml_dataset.csv"):
    CSV_FILE = "../ml_dataset.csv"
elif os.path.exists("ml_dataset_bigdata.csv"):
    CSV_FILE = "ml_dataset_bigdata.csv"
else:
    CSV_FILE = "/Users/adarshvashistha/Desktop/EZ/ml_dataset.csv"


def run_monte_carlo(symbol_df, num_simulations=10000, num_days=30):
    """
    Historical returns se drift aur volatility estimate karta hai,
    phir num_simulations alag future paths simulate karta hai.
    """
    symbol_df = symbol_df.sort_values("Date").copy()
    symbol_df["Daily_Return_Pct"] = symbol_df["Close"].pct_change() * 100

    daily_returns = symbol_df["Daily_Return_Pct"].dropna() / 100  # decimal form mein

    mu = daily_returns.mean()       # average daily drift
    sigma = daily_returns.std()     # daily volatility

    last_price = symbol_df["Close"].iloc[-1]

    # Random shocks generate karo: shape = (num_days, num_simulations)
    np.random.seed(42)  # reproducibility ke liye
    random_shocks = np.random.normal(0, 1, size=(num_days, num_simulations))

    # GBM equation apply karo, har din ke liye, sab simulations ek saath (vectorized)
    daily_multipliers = np.exp(
        (mu - 0.5 * sigma ** 2) + sigma * random_shocks
    )

    # Cumulative product se price paths banao
    price_paths = last_price * np.cumprod(daily_multipliers, axis=0)

    return price_paths, last_price, mu, sigma


def summarize_simulation(price_paths, last_price):
    """Simulation ke results se probability statements nikaalta hai"""
    final_prices = price_paths[-1, :]  # sab simulations ka last din ka price

    percentiles = {
        "5th (Bearish)": np.percentile(final_prices, 5),
        "25th": np.percentile(final_prices, 25),
        "50th (Median)": np.percentile(final_prices, 50),
        "75th": np.percentile(final_prices, 75),
        "95th (Bullish)": np.percentile(final_prices, 95),
    }

    prob_above_current = (final_prices > last_price).mean() * 100
    prob_10pct_gain = (final_prices > last_price * 1.10).mean() * 100
    prob_10pct_loss = (final_prices < last_price * 0.90).mean() * 100

    return percentiles, prob_above_current, prob_10pct_gain, prob_10pct_loss


def main():
    print(f"Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    df["Date"] = pd.to_datetime(df["Date"])

    symbol = "RELIANCE.NS"
    symbol_df = df[df["Symbol"] == symbol]

    if symbol_df.empty:
        print(f"{symbol} ka data nahi mila.")
        return

    print(f"\nRunning Monte Carlo Simulation for {symbol}")
    print("="*60)

    price_paths, last_price, mu, sigma = run_monte_carlo(
        symbol_df, num_simulations=10000, num_days=30
    )

    print(f"Current Price: Rs {last_price:.2f}")
    print(f"Historical Daily Drift (mu): {mu*100:.4f}%")
    print(f"Historical Daily Volatility (sigma): {sigma*100:.4f}%")
    print(f"Simulating 10,000 possible paths, 30 trading days ahead...\n")

    percentiles, prob_above, prob_gain, prob_loss = summarize_simulation(price_paths, last_price)

    print("30-Day Price Probability Distribution:")
    for label, value in percentiles.items():
        change_pct = ((value - last_price) / last_price) * 100
        print(f"  {label:<20} Rs {value:>10.2f}  ({change_pct:+.1f}%)")

    print(f"\nProbability price is ABOVE current level in 30 days: {prob_above:.1f}%")
    print(f"Probability of 10%+ GAIN in 30 days: {prob_gain:.1f}%")
    print(f"Probability of 10%+ LOSS in 30 days: {prob_loss:.1f}%")

    print("\n" + "="*60)
    print("IMPORTANT CAVEAT:")
    print("Ye simulation assume karta hai ki future returns PAST ke")
    print("statistical distribution (mu, sigma) se aayenge -- ye ek")
    print("simplification hai. Real markets mein regime changes, fat-tails,")
    print("aur jumps hote hain jo simple GBM model capture nahi karta.")
    print("Isko 'guaranteed forecast' ki tarah mat treat karo -- ye ek")
    print("probability-based RISK COMMUNICATION tool hai, prediction nahi.")

    # Save summary paths for dashboard fan chart export
    output_filename = f"monte_carlo_{symbol.replace('.', '_')}.csv"
    output_df = pd.DataFrame(price_paths)
    output_df.to_csv(output_filename, index=False)
    print(f"\nRaw simulation paths saved to {output_filename}")


if __name__ == "__main__":
    main()
