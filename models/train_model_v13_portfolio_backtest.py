#!/usr/bin/env python3
"""
STEP 15: Walk-Forward Portfolio Backtest (Economic Alpha vs Statistical Accuracy)
---------------------------------------------------------------------------------
Translates ML predictions into real-world investable portfolio simulation:
1. Walk-Forward Expanding/Rolling Window Validation (zero lookahead bias)
2. Daily Cross-Sectional Ranking: Top 3 Long vs Bottom 3 Short / Long-Only vs Equal-Weight Benchmark
3. Realistic Friction Modeling: 10 bps (0.10%) transaction cost & slippage per trade
4. Institutional Risk & Performance Metrics:
   - Cumulative Return (%) & Annualized CAGR (%)
   - Sharpe Ratio & Sortino Ratio
   - Maximum Drawdown (Max DD %)
   - Strategy vs Buy-and-Hold Benchmark comparison
   - Hit Rate & Annualized Volatility

Chalane ka tarika:
    python3 train_model_v13_portfolio_backtest.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Support both running from `models/` directory or root workspace
if os.path.exists("ml_dataset.csv"):
    CSV_FILE = "ml_dataset.csv"
elif os.path.exists("../ml_dataset.csv"):
    CSV_FILE = "../ml_dataset.csv"
elif os.path.exists("ml_dataset_bigdata.csv"):
    CSV_FILE = "ml_dataset_bigdata.csv"
else:
    CSV_FILE = "/Users/adarshvashistha/Desktop/EZ/ml_dataset.csv"

TRANSACTION_COST_BPS = 10  # 10 bps = 0.10% per turnover
ANNUAL_RISK_FREE_RATE = 0.06  # 6.0% Indian 91-day T-Bill risk-free rate


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


def engineer_features_and_targets(df):
    """Generates clean multi-factor features and next-day real percentage returns"""
    df = df.sort_values(["Symbol", "Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"])

    engineered_rows = []
    for symbol, group in df.groupby("Symbol"):
        group = group.sort_values("Date").copy()

        # Real price returns
        group["Daily_Return_Pct"] = ((group["Close"] - group["Open"]) / group["Open"]) * 100
        group["Intraday_Spread_Pct"] = ((group["High"] - group["Low"]) / group["Open"]) * 100
        group["Volume_Change_Pct"] = group["Volume"].pct_change() * 100
        group["Return_MA_3"] = group["Daily_Return_Pct"].rolling(window=3).mean()
        group["Return_MA_7"] = group["Daily_Return_Pct"].rolling(window=7).mean()

        group["RSI_14"] = calculate_rsi(group["Close"], period=14)
        group["Bollinger_PercentB"] = calculate_bollinger_percent_b(group["Close"], window=20)

        # NEXT DAY's actual close-to-close return for economic PnL calculation
        group["Next_Day_Real_Return_Pct"] = group["Close"].pct_change().shift(-1) * 100
        group["Next_Day_Direction"] = (group["Next_Day_Real_Return_Pct"] > 0).astype(int)

        engineered_rows.append(group)

    result = pd.concat(engineered_rows, ignore_index=True)

    feature_cols = [
        "Daily_Return_Pct", "Intraday_Spread_Pct", "Volume_Change_Pct",
        "Return_MA_3", "Return_MA_7", "RSI_14", "Bollinger_PercentB",
        "Next_Day_Real_Return_Pct", "Next_Day_Direction"
    ]
    result = result.dropna(subset=feature_cols)
    return result


def run_walk_forward_portfolio_backtest(df, feature_cols, top_k=3, train_years=5, test_days=60):
    """
    Simulates a walk-forward portfolio:
    - Every `test_days` (approx quarterly), the model re-trains on the preceding `train_years`.
    - Every single trading day, stocks are ranked cross-sectionally by Model Predicted Probability.
    - Long Top K, Short / Zero Bottom K.
    - Tracks daily portfolio return, turnover, transaction cost drag, and benchmark performance.
    """
    df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    dates = sorted(df["Date"].unique())

    min_train_days = int(train_years * 252)
    if len(dates) <= min_train_days + test_days:
        raise ValueError("Insufficient trading days for walk-forward portfolio backtest.")

    portfolio_records = []
    prev_long_holdings = set()

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)

    print(f"Starting Walk-Forward Simulation from {dates[min_train_days].date()} to {dates[-1].date()}...")
    print(f"Holding Top {top_k} Longs vs Equal-Weighted Universe Benchmark (Transaction Drag: {TRANSACTION_COST_BPS} bps/turnover)\n")

    current_train_end_idx = min_train_days
    retrain_needed = True

    for t_idx in range(min_train_days, len(dates) - 1):
        current_date = dates[t_idx]

        # Re-train model periodically
        if (t_idx - min_train_days) % test_days == 0:
            train_dates = dates[max(0, t_idx - min_train_days): t_idx]
            train_mask = df["Date"].isin(train_dates)
            X_train = df.loc[train_mask, feature_cols].values
            y_train = df.loc[train_mask, "Next_Day_Direction"].values
            model.fit(X_train, y_train)

        # Predict today's cross-sectional probabilities for tomorrow's return
        day_mask = df["Date"] == current_date
        day_df = df[day_mask].copy()

        if len(day_df) < top_k:
            continue

        X_today = day_df[feature_cols].values
        # Probability of UP direction (class 1)
        prob_up = model.predict_proba(X_today)[:, 1]
        day_df["Prob_UP"] = prob_up

        # Rank stocks descending by probability
        ranked_df = day_df.sort_values("Prob_UP", ascending=False)
        long_stocks = set(ranked_df.head(top_k)["Symbol"].tolist())
        short_stocks = set(ranked_df.tail(top_k)["Symbol"].tolist())

        # Returns on day t+1
        long_returns = ranked_df.head(top_k)["Next_Day_Real_Return_Pct"].mean()
        short_returns = ranked_df.tail(top_k)["Next_Day_Real_Return_Pct"].mean()
        benchmark_return = day_df["Next_Day_Real_Return_Pct"].mean()  # Equal-weight all 11 stocks

        # Turnover calculation: how many stocks changed in the top K
        turnover_count = len(long_stocks - prev_long_holdings)
        turnover_pct = turnover_count / top_k
        fee_drag_pct = turnover_pct * 2 * (TRANSACTION_COST_BPS / 100.0)  # Buy new + sell old

        net_long_only_return = long_returns - fee_drag_pct
        net_long_short_return = (long_returns - short_returns) / 2.0 - fee_drag_pct

        prev_long_holdings = long_stocks

        portfolio_records.append({
            "Date": current_date,
            "Long_Gross_Return_Pct": long_returns,
            "Long_Net_Return_Pct": net_long_only_return,
            "Long_Short_Net_Return_Pct": net_long_short_return,
            "Benchmark_Return_Pct": benchmark_return,
            "Fee_Drag_Pct": fee_drag_pct,
            "Top_Picks": ",".join([s.replace(".NS", "") for s in long_stocks])
        })

    return pd.DataFrame(portfolio_records)


def compute_metrics(series, risk_free_rate=0.06):
    """Calculates CAGR, Sharpe, Sortino, Max Drawdown, and Win Rate"""
    daily_returns = series / 100.0
    cum_returns = (1 + daily_returns).cumprod()
    total_return_pct = (cum_returns.iloc[-1] - 1.0) * 100

    trading_days = len(daily_returns)
    years = trading_days / 252.0
    cagr_pct = ((cum_returns.iloc[-1]) ** (1.0 / years) - 1.0) * 100 if years > 0 and cum_returns.iloc[-1] > 0 else 0.0

    ann_vol_pct = daily_returns.std() * np.sqrt(252) * 100
    daily_rf = risk_free_rate / 252.0
    excess_returns = daily_returns - daily_rf
    sharpe = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0.0

    # Downside volatility for Sortino
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-6
    sortino = (excess_returns.mean() * np.sqrt(252) * 100) / (downside_std * 100) if downside_std > 0 else 0.0

    # Max Drawdown
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_dd_pct = drawdown.min() * 100

    # Hit Rate / Win Rate
    win_rate_pct = (daily_returns > 0).mean() * 100

    return {
        "Total_Return_Pct": total_return_pct,
        "CAGR_Pct": cagr_pct,
        "Ann_Vol_Pct": ann_vol_pct,
        "Sharpe_Ratio": sharpe,
        "Sortino_Ratio": sortino,
        "Max_Drawdown_Pct": max_dd_pct,
        "Win_Rate_Pct": win_rate_pct
    }


def main():
    print("="*75)
    print(" STEP 15: Quantitative Walk-Forward Portfolio Backtesting Engine")
    print("="*75)
    print("Loading multi-decade historical dataset...")
    df = pd.read_csv(CSV_FILE)

    print("Engineering technical factors and forward returns...")
    df_clean = engineer_features_and_targets(df)

    feature_cols = [
        "Daily_Return_Pct", "Intraday_Spread_Pct", "Volume_Change_Pct",
        "Return_MA_3", "Return_MA_7", "RSI_14", "Bollinger_PercentB"
    ]

    print(f"Total clean rows across 11 stocks: {len(df_clean)}")
    print(f"Running Institutional Walk-Forward Backtest (Top 3 Long Strategy)...")

    results_df = run_walk_forward_portfolio_backtest(df_clean, feature_cols, top_k=3, train_years=5, test_days=60)

    strat_net = compute_metrics(results_df["Long_Net_Return_Pct"], risk_free_rate=ANNUAL_RISK_FREE_RATE)
    strat_gross = compute_metrics(results_df["Long_Gross_Return_Pct"], risk_free_rate=ANNUAL_RISK_FREE_RATE)
    bench = compute_metrics(results_df["Benchmark_Return_Pct"], risk_free_rate=ANNUAL_RISK_FREE_RATE)
    long_short = compute_metrics(results_df["Long_Short_Net_Return_Pct"], risk_free_rate=ANNUAL_RISK_FREE_RATE)

    start_date = results_df['Date'].min().strftime('%Y-%m-%d')
    end_date = results_df['Date'].max().strftime('%Y-%m-%d')
    n_days = len(results_df)

    print("="*75)
    print(f" 📊 PORTFOLIO BACKTEST PERFORMANCE REPORT ({start_date} to {end_date} | {n_days} Trading Days)")
    print("="*75)
    print(f"{'Performance Metric':<28} | {'ML Long-Only (Net)':<20} | {'ML Long-Only (Gross)':<20} | {'Universe Benchmark':<20}")
    print("-" * 95)
    print(f"{'Cumulative Total Return':<28} | {strat_net['Total_Return_Pct']:>17.2f}% | {strat_gross['Total_Return_Pct']:>17.2f}% | {bench['Total_Return_Pct']:>17.2f}%")
    print(f"{'Annualized Return (CAGR)':<28} | {strat_net['CAGR_Pct']:>17.2f}% | {strat_gross['CAGR_Pct']:>17.2f}% | {bench['CAGR_Pct']:>17.2f}%")
    print(f"{'Annualized Volatility':<28} | {strat_net['Ann_Vol_Pct']:>17.2f}% | {strat_gross['Ann_Vol_Pct']:>17.2f}% | {bench['Ann_Vol_Pct']:>17.2f}%")
    print(f"{'Sharpe Ratio (Rf=6%)':<28} | {strat_net['Sharpe_Ratio']:>17.2f}  | {strat_gross['Sharpe_Ratio']:>17.2f}  | {bench['Sharpe_Ratio']:>17.2f} ")
    print(f"{'Sortino Ratio':<28} | {strat_net['Sortino_Ratio']:>17.2f}  | {strat_gross['Sortino_Ratio']:>17.2f}  | {bench['Sortino_Ratio']:>17.2f} ")
    print(f"{'Maximum Drawdown (Max DD)':<28} | {strat_net['Max_Drawdown_Pct']:>17.2f}% | {strat_gross['Max_Drawdown_Pct']:>17.2f}% | {bench['Max_Drawdown_Pct']:>17.2f}%")
    print(f"{'Daily Win Rate':<28} | {strat_net['Win_Rate_Pct']:>17.2f}% | {strat_gross['Win_Rate_Pct']:>17.2f}% | {bench['Win_Rate_Pct']:>17.2f}%")
    print("="*75)

    print("\n💡 Key Quantitative Findings:")
    cagr_diff = strat_net['CAGR_Pct'] - bench['CAGR_Pct']
    sharpe_diff = strat_net['Sharpe_Ratio'] - bench['Sharpe_Ratio']
    total_fee_drag = strat_gross['Total_Return_Pct'] - strat_net['Total_Return_Pct']

    if cagr_diff > 0 and sharpe_diff > 0:
        print(f"✅ Strategy generates POSITIVE ECONOMIC ALPHA over the Buy-and-Hold Benchmark (+{cagr_diff:.2f}% CAGR, +{sharpe_diff:.2f} Sharpe)!")
    else:
        print(f"⚠️ Strategy did not beat the passive Buy-and-Hold Benchmark ({cagr_diff:+.2f}% CAGR diff).")

    print(f"💸 Cumulative Transaction Cost & Slippage Drag: -{total_fee_drag:.2f}% total return over simulation horizon.")
    print("="*75)


if __name__ == "__main__":
    main()
