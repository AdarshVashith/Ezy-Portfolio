#!/usr/bin/env python3
"""
STEP 2: Yahoo Finance Scraper — OHLC + Multiple Stocks
---------------------------------------------------------
Ye script:
1. Yahoo Finance se OHLC (Open, High, Low, Close) + Volume nikaalta hai
2. Multiple stocks ek saath handle karta hai (e.g. RELIANCE.NS, TCS.NS, INFY.NS)
3. SQLite database mein save karta hai (ohlc_data table)
4. Non-trading days (Volume = 0) ko skip karta hai
5. Same date par duplicate records nahi aane deta (UNIQUE(symbol, date) constraint)

Chalane ka tarika:
    python3 scraper_step2.py
"""

import re
import time
import sqlite3
import requests
from datetime import datetime

# ---------- CONFIG ----------
SYMBOLS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"]
DB_FILE = "stock_data.db"

# Session cache
_session = None
_crumb = None


def get_session():
    """Yahoo Finance session aur crumb token maintain karta hai"""
    global _session, _crumb
    if _session is not None and _crumb is not None:
        return _session, _crumb

    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    try:
        res = session.get("https://finance.yahoo.com/quote/RELIANCE.NS", headers=headers, timeout=10)
        crumb_match = re.search(r'\"crumb\":\"(.*?)\"', res.text)
        if crumb_match:
            _crumb = crumb_match.group(1)
        else:
            api_headers = {
                "User-Agent": headers["User-Agent"],
                "Referer": "https://finance.yahoo.com/quote/RELIANCE.NS",
            }
            crumb_res = session.get("https://query1.finance.yahoo.com/v1/test/getcrumb", headers=api_headers, timeout=10)
            if crumb_res.status_code == 200:
                _crumb = crumb_res.text
            else:
                _crumb = None
    except Exception as e:
        print(f"Warning: Session init error ({e}), proceeding with standard headers.")

    _session = session
    return _session, _crumb


def _get_last_valid(lst, fallback=0.0):
    """List me se aakhiri non-None value dhoondta hai"""
    if not lst:
        return fallback
    for val in reversed(lst):
        if val is not None:
            return float(val)
    return fallback


def fetch_ohlc(symbol):
    """
    Ek stock ka latest OHLC (Open, High, Low, Close) + Volume nikaalta hai.
    Returns: dict with symbol, date, open, high, low, close, volume, last_updated
    """
    session, crumb = get_session()
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"
    if crumb:
        url += f"&crumb={crumb}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Referer": f"https://finance.yahoo.com/quote/{symbol}",
    }

    response = session.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()

    result = data["chart"]["result"][0]
    meta = result.get("meta", {})
    quote = result["indicators"]["quote"][0]
    timestamps = result.get("timestamp", [])

    # Market price ya previous close fallback
    current_price = meta.get("regularMarketPrice") or meta.get("chartPreviousClose", 0.0)

    # OHLC arrays me se safe non-None values nikalna
    open_val = _get_last_valid(quote.get("open"), fallback=current_price)
    high_val = _get_last_valid(quote.get("high"), fallback=meta.get("regularMarketDayHigh", current_price))
    low_val = _get_last_valid(quote.get("low"), fallback=meta.get("regularMarketDayLow", current_price))
    close_val = _get_last_valid(quote.get("close"), fallback=current_price)
    volume_val = int(_get_last_valid(quote.get("volume"), fallback=meta.get("regularMarketVolume", 0)))

    # Timestamp date (YYYY-MM-DD)
    if timestamps:
        trade_date = datetime.fromtimestamp(timestamps[-1]).strftime("%Y-%m-%d")
    else:
        trade_date = datetime.now().strftime("%Y-%m-%d")

    return {
        "symbol": symbol,
        "date": trade_date,
        "open": round(open_val, 2),
        "high": round(high_val, 2),
        "low": round(low_val, 2),
        "close": round(close_val, 2),
        "volume": volume_val,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def setup_database():
    """Database aur ohlc_data table banata hai with UNIQUE(symbol, date) constraint"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ohlc_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            last_updated TEXT,
            UNIQUE(symbol, date) ON CONFLICT REPLACE
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlc_symbol ON ohlc_data(symbol);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlc_date ON ohlc_data(date);")
    conn.commit()
    conn.close()


def save_ohlc(record):
    """Ek OHLC record database mein save ya update karta hai"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO ohlc_data (symbol, date, open, high, low, close, volume, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record["symbol"], record["date"], record["open"],
        record["high"], record["low"], record["close"],
        record["volume"], record["last_updated"]
    ))
    conn.commit()
    conn.close()


def collect_all_symbols():
    """Saare configured symbols ke liye data collect karta hai"""
    print(f"\n--- [Data Collection Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ---")
    for symbol in SYMBOLS:
        try:
            record = fetch_ohlc(symbol)
            
            # Non-trading day ya zero volume check
            if record["volume"] <= 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol:<14} -> SKIPPED (Non-trading day / Volume = 0)")
                continue

            save_ohlc(record)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol:<14} [{record['date']}] -> O: {record['open']:>8.2f} | H: {record['high']:>8.2f} | L: {record['low']:>8.2f} | C: {record['close']:>8.2f} | Vol: {record['volume']}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol:<14} -> FAILED: {e}")

        time.sleep(1.5)  # Rate-limit safety gap
    print("--- [Cycle Complete] ---\n")


def run_once():
    setup_database()
    collect_all_symbols()


def run_forever(interval_seconds=60):
    """Har `interval_seconds` me saare symbols ka data fetch karta rehta hai"""
    setup_database()
    print(f"Starting Multi-Stock Continuous Scraper. Polling interval: {interval_seconds}s")
    print("Press Ctrl+C to stop.\n")
    while True:
        collect_all_symbols()
        print(f"Waiting {interval_seconds} seconds for next cycle...")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    run_once()
    # run_forever(60)  # Continuous chalane ke liye isko uncomment karein
