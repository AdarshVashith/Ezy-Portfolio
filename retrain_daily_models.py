"""
Daily Model Retraining Script
--------------------------------------------------------------------
Ye script har din chalna chahiye (cron/Task Scheduler se) taaki models
HAMESHA latest data pe trained rahein. Trained models disk pe save hote
hain (joblib se), aur chatbot inhe load karke live predictions deta hai.

Setup (Linux/Mac cron -- roz raat 2 baje chalega):
    crontab -e
    0 2 * * * cd /path/to/project && python3 retrain_daily_models.py

Chalane ka tarika (manual test ke liye):
    python3 retrain_daily_models.py
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

DB_FILE = "stock_data.db"
CSV_FILE = "ml_dataset.csv"
MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)


def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def engineer_features(group):
    group = group.sort_values("Date").copy()
    group["Daily_Return_Pct"] = ((group["Close"] - group["Open"]) / group["Open"]) * 100
    group["Intraday_Spread_Pct"] = ((group["High"] - group["Low"]) / group["Open"]) * 100
    group["Volume_Change_Pct"] = group["Volume"].pct_change() * 100
    group["Return_MA_3"] = group["Daily_Return_Pct"].rolling(window=3).mean()
    group["RSI_14"] = calculate_rsi(group["Close"], period=14)
    group["Next_Day_Direction"] = (group["Close"].pct_change().shift(-1) > 0).astype(int)
    return group


def load_dataset():
    """Load dataset from stock_data.db or ml_dataset.csv"""
    if os.path.exists(DB_FILE):
        try:
            conn = sqlite3.connect(DB_FILE)
            df = pd.read_sql("SELECT symbol as Symbol, date as Date, open as Open, high as High, low as Low, close as Close, volume as Volume FROM ohlc_data", conn)
            conn.close()
            if not df.empty:
                df["Date"] = pd.to_datetime(df["Date"])
                return df
        except Exception as e:
            print(f"Warning: Failed to load from DB ({e}), trying CSV...")

    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    raise FileNotFoundError("Neither stock_data.db nor ml_dataset.csv found.")


def train_and_save_direction_model(df, symbol):
    """Ek symbol ke liye direction classifier train karta hai aaj tak ke saare data pe"""
    symbol_df = df[df["Symbol"] == symbol].copy()
    symbol_df = engineer_features(symbol_df)

    feature_cols = ["Daily_Return_Pct", "Intraday_Spread_Pct", "Volume_Change_Pct",
                    "Return_MA_3", "RSI_14"]
    symbol_df = symbol_df.dropna(subset=feature_cols + ["Next_Day_Direction"])

    if len(symbol_df) < 50:
        return None

    X = symbol_df[feature_cols]
    y = symbol_df["Next_Day_Direction"]

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)

    model_path = os.path.join(MODEL_DIR, f"{symbol.replace('.', '_')}_direction_model.joblib")
    joblib.dump({
        "model": model,
        "feature_cols": feature_cols,
        "trained_on": datetime.now().isoformat(),
        "training_rows": len(symbol_df),
        # IMPORTANT: honest metadata saved WITH the model, taaki chatbot ise
        # hamesha disclose kar sake -- ye humare poore project ki integrity
        # ka core part hai
        "known_reliability": "No statistically significant edge found in modern "
                             "regime testing (p=0.418). Use as one data point, not a directive."
    }, model_path)

    return model_path


def main():
    print(f"[{datetime.now()}] Starting daily model retraining...")
    df = load_dataset()
    symbols = df["Symbol"].unique()
    trained_count = 0

    for symbol in sorted(symbols):
        path = train_and_save_direction_model(df, symbol)
        if path:
            trained_count += 1
            print(f"  Trained and saved: {symbol} -> {path}")
        else:
            print(f"  Skipped {symbol} (insufficient data)")

    print(f"[{datetime.now()}] Retraining complete. {trained_count}/{len(symbols)} models updated.")


if __name__ == "__main__":
    main()
