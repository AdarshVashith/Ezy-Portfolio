#!/usr/bin/env python3
"""
STEP 16: Execution Timing & Realistic Slippage Audit
---------------------------------------------------
Systematic audit to test the Execution Leakage & Timing Hypotheses:
1. Mode A: Naive Close(T) -> Close(T+1) (Old model with overnight gap leakage)
2. Mode B: Realistic Tradeable Open(T+1) -> Close(T+1) (Intraday execution)
3. Mode C: Realistic Tradeable Open(T+1) -> Open(T+2) (Next-day holding)

Tracks:
- Exact Actual Daily Turnover (%)
- Modern Era (2021-2026) as primary control baseline
- Sub-period breakdown (2001-2010, 2011-2020, 2021-2026)
- Impact of Overnight Gap vs Intraday Drift

Usage:
    python3 models/train_model_v14_execution_audit.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# File path resolution
if os.path.exists("ml_dataset.csv"):
    CSV_FILE = "ml_dataset.csv"
elif os.path.exists("../ml_dataset.csv"):
    CSV_FILE = "../ml_dataset.csv"
else:
    CSV_FILE = "/Users/adarshvashistha/Desktop/EZ/ml_dataset.csv"

TRANSACTION_COST_BPS = 10  # 10 bps = 0.10% per transaction leg
ANNUAL_RISK_FREE_RATE = 0.06  # 6.0% risk-free rate


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


def prepare_execution_dataset(df):
    """
    Builds clean features computed strictly up to Date T Close,
    along with three distinct forward execution return targets.
    """
    df = df.sort_values(["Symbol", "Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"])

    engineered = []
    for symbol, group in df.groupby("Symbol"):
        group = group.sort_values("Date").copy()

        # Features known strictly at Date T Close
        group["Daily_Return_Pct"] = ((group["Close"] - group["Open"]) / group["Open"]) * 100
        group["Intraday_Spread_Pct"] = ((group["High"] - group["Low"]) / group["Open"]) * 100
        group["Volume_Change_Pct"] = group["Volume"].pct_change() * 100
        group["Return_MA_3"] = group["Daily_Return_Pct"].rolling(window=3).mean()
        group["Return_MA_7"] = group["Daily_Return_Pct"].rolling(window=7).mean()
        group["RSI_14"] = calculate_rsi(group["Close"], period=14)
        group["Bollinger_PercentB"] = calculate_bollinger_percent_b(group["Close"], window=20)

        # Execution Targets:
        # 1. Mode A (Naive): Close(T) -> Close(T+1) (contains overnight gap)
        group["Ret_CloseT_to_CloseT1"] = group["Close"].pct_change().shift(-1) * 100

        # 2. Mode B (Tradeable Intraday): Open(T+1) -> Close(T+1)
        # Shift Open and Close of T+1 back to row T
        open_t1 = group["Open"].shift(-1)
        close_t1 = group["Close"].shift(-1)
        group["Ret_OpenT1_to_CloseT1"] = ((close_t1 - open_t1) / open_t1) * 100

        # 3. Mode C (Tradeable 1-Day Hold): Open(T+1) -> Open(T+2)
        open_t2 = group["Open"].shift(-2)
        group["Ret_OpenT1_to_OpenT2"] = ((open_t2 - open_t1) / open_t1) * 100

        # 4. Overnight Gap: Close(T) -> Open(T+1)
        group["Ret_Overnight_Gap"] = ((open_t1 - group["Close"]) / group["Close"]) * 100

        # Binary training label (based on Close-to-Close or Intraday)
        group["Next_Day_Direction"] = (group["Ret_CloseT_to_CloseT1"] > 0).astype(int)

        engineered.append(group)

    result = pd.concat(engineered, ignore_index=True)
    feature_cols = [
        "Daily_Return_Pct", "Intraday_Spread_Pct", "Volume_Change_Pct",
        "Return_MA_3", "Return_MA_7", "RSI_14", "Bollinger_PercentB"
    ]
    all_needed = feature_cols + [
        "Ret_CloseT_to_CloseT1", "Ret_OpenT1_to_CloseT1",
        "Ret_OpenT1_to_OpenT2", "Ret_Overnight_Gap", "Next_Day_Direction"
    ]
    result = result.dropna(subset=all_needed)
    return result, feature_cols


def run_execution_simulation(df, feature_cols, top_k=3, train_years=5, test_days=60):
    df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    dates = sorted(df["Date"].unique())

    min_train_days = int(train_years * 252)
    if len(dates) <= min_train_days + test_days:
        raise ValueError("Insufficient data for walk-forward simulation.")

    records = []
    prev_longs = set()
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)

    for t_idx in range(min_train_days, len(dates) - 2):
        current_date = dates[t_idx]

        # Re-train quarterly
        if (t_idx - min_train_days) % test_days == 0:
            train_dates = dates[max(0, t_idx - min_train_days): t_idx]
            train_mask = df["Date"].isin(train_dates)
            X_train = df.loc[train_mask, feature_cols].values
            y_train = df.loc[train_mask, "Next_Day_Direction"].values
            model.fit(X_train, y_train)

        day_mask = df["Date"] == current_date
        day_df = df[day_mask].copy()

        if len(day_df) < top_k:
            continue

        X_today = day_df[feature_cols].values
        day_df["Prob_UP"] = model.predict_proba(X_today)[:, 1]

        ranked = day_df.sort_values("Prob_UP", ascending=False)
        top_picks = ranked.head(top_k)
        bottom_picks = ranked.tail(top_k)

        # Actual turnover
        current_longs = set(top_picks["Symbol"].tolist())
        turnover_count = len(current_longs - prev_longs)
        turnover_pct = turnover_count / top_k
        fee_drag_pct = turnover_pct * 2 * (TRANSACTION_COST_BPS / 100.0)
        prev_longs = current_longs

        # Mode A: Naive Close(T) -> Close(T+1)
        ret_mode_a_gross = top_picks["Ret_CloseT_to_CloseT1"].mean()
        ret_mode_a_net = ret_mode_a_gross - fee_drag_pct
        bench_mode_a = day_df["Ret_CloseT_to_CloseT1"].mean()

        # Mode B: Tradeable Open(T+1) -> Close(T+1)
        ret_mode_b_gross = top_picks["Ret_OpenT1_to_CloseT1"].mean()
        ret_mode_b_net = ret_mode_b_gross - fee_drag_pct
        bench_mode_b = day_df["Ret_OpenT1_to_CloseT1"].mean()

        # Mode C: Tradeable Open(T+1) -> Open(T+2)
        ret_mode_c_gross = top_picks["Ret_OpenT1_to_OpenT2"].mean()
        ret_mode_c_net = ret_mode_c_gross - fee_drag_pct
        bench_mode_c = day_df["Ret_OpenT1_to_OpenT2"].mean()

        # Overnight Gap
        ret_gap_picks = top_picks["Ret_Overnight_Gap"].mean()
        ret_gap_bench = day_df["Ret_Overnight_Gap"].mean()

        records.append({
            "Date": current_date,
            "Turnover_Pct": turnover_pct,
            "Fee_Drag_Pct": fee_drag_pct,
            # Mode A
            "ModeA_Gross": ret_mode_a_gross,
            "ModeA_Net": ret_mode_a_net,
            "Bench_ModeA": bench_mode_a,
            # Mode B
            "ModeB_Gross": ret_mode_b_gross,
            "ModeB_Net": ret_mode_b_net,
            "Bench_ModeB": bench_mode_b,
            # Mode C
            "ModeC_Gross": ret_mode_c_gross,
            "ModeC_Net": ret_mode_c_net,
            "Bench_ModeC": bench_mode_c,
            # Overnight Gap
            "Gap_Picks": ret_gap_picks,
            "Gap_Bench": ret_gap_bench,
        })

    return pd.DataFrame(records)


def compute_metrics(series, risk_free_rate=0.06):
    daily_returns = series / 100.0
    cum_returns = (1 + daily_returns).cumprod()
    total_return_pct = (cum_returns.iloc[-1] - 1.0) * 100 if len(cum_returns) > 0 else 0.0

    trading_days = len(daily_returns)
    years = trading_days / 252.0
    cagr_pct = ((cum_returns.iloc[-1]) ** (1.0 / years) - 1.0) * 100 if years > 0 and cum_returns.iloc[-1] > 0 else 0.0

    ann_vol_pct = daily_returns.std() * np.sqrt(252) * 100
    daily_rf = risk_free_rate / 252.0
    excess_returns = daily_returns - daily_rf
    sharpe = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0.0

    # Max Drawdown
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_dd_pct = drawdown.min() * 100

    win_rate_pct = (daily_returns > 0).mean() * 100

    return {
        "CAGR_Pct": cagr_pct,
        "Total_Return_Pct": total_return_pct,
        "Ann_Vol_Pct": ann_vol_pct,
        "Sharpe_Ratio": sharpe,
        "Max_Drawdown_Pct": max_dd_pct,
        "Win_Rate_Pct": win_rate_pct,
    }


def print_era_report(era_name, era_df):
    m_a_net = compute_metrics(era_df["ModeA_Net"])
    m_a_gross = compute_metrics(era_df["ModeA_Gross"])
    b_a = compute_metrics(era_df["Bench_ModeA"])

    m_b_net = compute_metrics(era_df["ModeB_Net"])
    m_b_gross = compute_metrics(era_df["ModeB_Gross"])
    b_b = compute_metrics(era_df["Bench_ModeB"])

    m_c_net = compute_metrics(era_df["ModeC_Net"])
    b_c = compute_metrics(era_df["Bench_ModeC"])

    avg_turnover = era_df["Turnover_Pct"].mean() * 100
    avg_gap_picks = era_df["Gap_Picks"].mean()
    avg_gap_bench = era_df["Gap_Bench"].mean()

    start_d = era_df["Date"].min().strftime("%Y-%m-%d")
    end_d = era_df["Date"].max().strftime("%Y-%m-%d")
    n_days = len(era_df)

    print("\n" + "=" * 85)
    print(f" 📈 {era_name.upper()} ({start_d} to {end_d} | {n_days} Trading Days)")
    print(f"    Average Daily Portfolio Turnover: {avg_turnover:.1f}%")
    print(f"    Avg Daily Overnight Gap: Model Picks = {avg_gap_picks:+.3f}% | Benchmark = {avg_gap_bench:+.3f}%")
    print("=" * 85)
    print(f"{'Execution Mode / Strategy':<35} | {'CAGR (%)':<10} | {'Sharpe':<8} | {'Max DD (%)':<12} | {'Win Rate':<8}")
    print("-" * 85)
    print(f"{'Mode A: Naive Close(T)->Close(T+1) [NET]':<35} | {m_a_net['CAGR_Pct']:>8.2f}% | {m_a_net['Sharpe_Ratio']:>8.2f} | {m_a_net['Max_Drawdown_Pct']:>10.2f}% | {m_a_net['Win_Rate_Pct']:>7.1f}%")
    print(f"{'Mode A: Naive Close(T)->Close(T+1) [GROSS]':<35} | {m_a_gross['CAGR_Pct']:>8.2f}% | {m_a_gross['Sharpe_Ratio']:>8.2f} | {m_a_gross['Max_Drawdown_Pct']:>10.2f}% | {m_a_gross['Win_Rate_Pct']:>7.1f}%")
    print(f"{'Mode A: Benchmark Equal-Weight':<35} | {b_a['CAGR_Pct']:>8.2f}% | {b_a['Sharpe_Ratio']:>8.2f} | {b_a['Max_Drawdown_Pct']:>10.2f}% | {b_a['Win_Rate_Pct']:>7.1f}%")
    print("-" * 85)
    print(f"{'Mode B: Tradeable Open(T+1)->Close(T+1) [NET]':<35} | {m_b_net['CAGR_Pct']:>8.2f}% | {m_b_net['Sharpe_Ratio']:>8.2f} | {m_b_net['Max_Drawdown_Pct']:>10.2f}% | {m_b_net['Win_Rate_Pct']:>7.1f}%")
    print(f"{'Mode B: Tradeable Open(T+1)->Close(T+1) [GROSS]':<35} | {m_b_gross['CAGR_Pct']:>8.2f}% | {m_b_gross['Sharpe_Ratio']:>8.2f} | {m_b_gross['Max_Drawdown_Pct']:>10.2f}% | {m_b_gross['Win_Rate_Pct']:>7.1f}%")
    print(f"{'Mode B: Benchmark Intraday':<35} | {b_b['CAGR_Pct']:>8.2f}% | {b_b['Sharpe_Ratio']:>8.2f} | {b_b['Max_Drawdown_Pct']:>10.2f}% | {b_b['Win_Rate_Pct']:>7.1f}%")
    print("-" * 85)
    print(f"{'Mode C: Tradeable Open(T+1)->Open(T+2) [NET]':<35} | {m_c_net['CAGR_Pct']:>8.2f}% | {m_c_net['Sharpe_Ratio']:>8.2f} | {m_c_net['Max_Drawdown_Pct']:>10.2f}% | {m_c_net['Win_Rate_Pct']:>7.1f}%")
    print(f"{'Mode C: Benchmark Open-to-Open':<35} | {b_c['CAGR_Pct']:>8.2f}% | {b_c['Sharpe_Ratio']:>8.2f} | {b_c['Max_Drawdown_Pct']:>10.2f}% | {b_c['Win_Rate_Pct']:>7.1f}%")
    print("=" * 85)


def main():
    print("=" * 85)
    print(" STEP 16: RIGOROUS EXECUTION TIMING & SLIPPAGE AUDIT")
    print("=" * 85)
    print("Loading historical dataset...")
    df = pd.read_csv(CSV_FILE)

    print("Building execution dataset with Mode A (Naive), Mode B (Intraday), Mode C (Open-to-Open)...")
    df_clean, feature_cols = prepare_execution_dataset(df)

    print(f"Total clean rows across universe: {len(df_clean)}")
    print(f"Running full walk-forward multi-execution simulation...\n")

    sim_df = run_execution_simulation(df_clean, feature_cols, top_k=3, train_years=5, test_days=60)

    # 1. Primary Control Test: Modern Era (2021-2026) FIRST
    modern_mask = sim_df["Date"] >= "2021-01-01"
    modern_df = sim_df[modern_mask].copy()
    print_era_report("AUDIT 1 (PRIMARY CONTROL): Modern Era (2021 - 2026)", modern_df)

    # 2. Early Era (2001-2010)
    early_mask = (sim_df["Date"] >= "2001-01-01") & (sim_df["Date"] < "2011-01-01")
    early_df = sim_df[early_mask].copy()
    print_era_report("AUDIT 2: Early Era (2001 - 2010) [Peak Survivorship & Regime]", early_df)

    # 3. Middle Era (2011-2020)
    mid_mask = (sim_df["Date"] >= "2011-01-01") & (sim_df["Date"] < "2021-01-01")
    mid_df = sim_df[mid_mask].copy()
    print_era_report("AUDIT 3: Middle Era (2011 - 2020)", mid_df)

    # 4. Full 25-Year Multi-Decade Horizon
    print_era_report("AUDIT 4: Full Multi-Decade Horizon (2001 - 2026)", sim_df)


if __name__ == "__main__":
    main()
