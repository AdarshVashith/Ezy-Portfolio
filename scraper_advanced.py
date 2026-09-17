#!/usr/bin/env python3
"""
Advanced Stock Data Scraper & Backfill Engine
---------------------------------------------
Features:
- Live Price & Daily OHLC Scraper
- Historical Backfill Engine (fetch 1mo, 6mo, 1y, etc. past data)
- Multi-stock support with rate-limit safe session handling
- Auto-recovering Yahoo Finance Crumb/Cookie authentication
- SQLite persistence with `UNIQUE(symbol, date)` to prevent duplicate entries
- Zero-Volume / Non-trading day filtering
- CSV Export option
- File logging (`scraper.log`)

Usage:
    python3 scraper_advanced.py                       # Single cycle live data
    python3 scraper_advanced.py --continuous          # Continuous loop mode (default 60s)
    python3 scraper_advanced.py --continuous --interval 30
    python3 scraper_advanced.py --backfill --range 1mo # Fetch past 1 month historical data
    python3 scraper_advanced.py --backfill --range 1y  # Fetch past 1 year historical data
    python3 scraper_advanced.py --symbols RELIANCE.NS TCS.NS INFY.NS
    python3 scraper_advanced.py --export-csv output.csv
"""

import os
import re
import csv
import time
import sqlite3
import logging
import argparse
import requests
from datetime import datetime

# Default configuration
DEFAULT_SYMBOLS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"]
DB_FILE = "stock_data.db"
LOG_FILE = "scraper.log"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)


class YahooFinanceClient:
    """Robust client handling Yahoo Finance session, cookies, and crumb tokens."""
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        self.crumb = None
        self._init_session()

    def _init_session(self):
        try:
            res = self.session.get("https://finance.yahoo.com/quote/RELIANCE.NS", headers=self.headers, timeout=10)
            crumb_match = re.search(r'\"crumb\":\"(.*?)\"', res.text)
            if crumb_match:
                self.crumb = crumb_match.group(1)
            else:
                api_headers = {
                    "User-Agent": self.headers["User-Agent"],
                    "Referer": "https://finance.yahoo.com/quote/RELIANCE.NS",
                }
                crumb_res = self.session.get("https://query1.finance.yahoo.com/v1/test/getcrumb", headers=api_headers, timeout=10)
                if crumb_res.status_code == 200:
                    self.crumb = crumb_res.text
        except Exception as e:
            logging.warning(f"Could not initialize Yahoo session token: {e}")

    def fetch_chart(self, symbol, interval="1d", range_str="5d"):
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_str}"
        if self.crumb:
            url += f"&crumb={self.crumb}"

        headers = {
            "User-Agent": self.headers["User-Agent"],
            "Referer": f"https://finance.yahoo.com/quote/{symbol}",
        }

        res = self.session.get(url, headers=headers, timeout=12)
        if res.status_code in (401, 429):
            logging.info("Refreshing expired crumb session...")
            self._init_session()
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_str}"
            if self.crumb:
                url += f"&crumb={self.crumb}"
            res = self.session.get(url, headers=headers, timeout=12)

        res.raise_for_status()
        return res.json()


class StockDatabase:
    def __init__(self, db_path=DB_FILE):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON ohlc_data(symbol);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON ohlc_data(date);")
            conn.commit()

    def insert_records(self, records):
        if not records:
            return 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO ohlc_data (symbol, date, open, high, low, close, volume, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (r["symbol"], r["date"], r["open"], r["high"], r["low"], r["close"], r["volume"], r.get("last_updated", r["date"]))
                for r in records
            ])
            conn.commit()
            return len(records)


def extract_latest_ohlc(client, symbol):
    data = client.fetch_chart(symbol, interval="1d", range_str="5d")
    result = data["chart"]["result"][0]
    meta = result.get("meta", {})
    quote = result["indicators"]["quote"][0]
    timestamps = result.get("timestamp", [])

    current_price = meta.get("regularMarketPrice") or meta.get("chartPreviousClose", 0.0)

    def _last_val(lst, fallback=0.0):
        if not lst:
            return fallback
        for v in reversed(lst):
            if v is not None:
                return float(v)
        return fallback

    volume_val = int(_last_val(quote.get("volume"), meta.get("regularMarketVolume", 0)))
    
    # Check for non-trading day / zero volume
    if volume_val <= 0:
        return None

    if timestamps:
        trade_date = datetime.fromtimestamp(timestamps[-1]).strftime("%Y-%m-%d")
    else:
        trade_date = datetime.now().strftime("%Y-%m-%d")

    return {
        "symbol": symbol,
        "date": trade_date,
        "open": round(_last_val(quote.get("open"), current_price), 2),
        "high": round(_last_val(quote.get("high"), meta.get("regularMarketDayHigh", current_price)), 2),
        "low": round(_last_val(quote.get("low"), meta.get("regularMarketDayLow", current_price)), 2),
        "close": round(_last_val(quote.get("close"), current_price), 2),
        "volume": volume_val,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def extract_historical_ohlc(client, symbol, range_str="1mo"):
    data = client.fetch_chart(symbol, interval="1d", range_str=range_str)
    result = data["chart"]["result"][0]
    timestamps = result.get("timestamp", [])
    quote = result["indicators"]["quote"][0]

    opens = quote.get("open", [])
    highs = quote.get("high", [])
    lows = quote.get("low", [])
    closes = quote.get("close", [])
    volumes = quote.get("volume", [])

    records = []
    skipped_count = 0

    for i, ts in enumerate(timestamps):
        o = opens[i] if i < len(opens) else None
        h = highs[i] if i < len(highs) else None
        l = lows[i] if i < len(lows) else None
        c = closes[i] if i < len(closes) else None
        v = volumes[i] if i < len(volumes) else None

        if None in (o, h, l, c):
            continue

        volume_val = int(v) if v else 0

        # Skip zero volume / non-trading days
        if volume_val <= 0:
            skipped_count += 1
            continue

        records.append({
            "symbol": symbol,
            "date": datetime.fromtimestamp(ts).strftime("%Y-%m-%d"),
            "open": round(float(o), 2),
            "high": round(float(h), 2),
            "low": round(float(l), 2),
            "close": round(float(c), 2),
            "volume": volume_val,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    if skipped_count > 0:
        logging.info(f"[{symbol}] Skipped {skipped_count} zero-volume / non-trading day records.")

    return records


def export_to_csv(db_path, output_csv):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, date, open, high, low, close, volume, last_updated FROM ohlc_data ORDER BY symbol, date DESC")
        rows = cursor.fetchall()

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Symbol", "Date", "Open", "High", "Low", "Close", "Volume", "LastUpdated"])
        writer.writerows(rows)

    logging.info(f"Exported {len(rows)} rows to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Advanced Stock Scraper & Backfill Tool")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="List of ticker symbols")
    parser.add_argument("--continuous", action="store_true", help="Run continuously in a loop")
    parser.add_argument("--interval", type=int, default=60, help="Interval in seconds for continuous mode (default: 60)")
    parser.add_argument("--backfill", action="store_true", help="Fetch historical past candles")
    parser.add_argument("--range", default="1mo", choices=["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], help="Historical range for backfill")
    parser.add_argument("--export-csv", help="Export DB data to specified CSV filename")

    args = parser.parse_args()

    db = StockDatabase(DB_FILE)

    if args.export_csv:
        export_to_csv(DB_FILE, args.export_csv)
        return

    client = YahooFinanceClient()

    if args.backfill:
        logging.info(f"Starting Historical Backfill for {args.symbols} with range={args.range}")
        for symbol in args.symbols:
            try:
                records = extract_historical_ohlc(client, symbol, range_str=args.range)
                count = db.insert_records(records)
                logging.info(f"[{symbol}] Saved/Updated {count} valid candles in DB.")
            except Exception as e:
                logging.error(f"[{symbol}] Backfill failed: {e}")
            time.sleep(1.0)
        logging.info("Historical backfill completed.")
        return

    def run_cycle():
        for symbol in args.symbols:
            try:
                record = extract_latest_ohlc(client, symbol)
                if record is None:
                    logging.info(f"{symbol:<14} -> SKIPPED (Non-trading day / Zero Volume)")
                    continue
                db.insert_records([record])
                logging.info(f"{symbol:<14} [{record['date']}] -> O:{record['open']:>8.2f} | H:{record['high']:>8.2f} | L:{record['low']:>8.2f} | C:{record['close']:>8.2f} | Vol:{record['volume']}")
            except Exception as e:
                logging.error(f"{symbol:<14} -> FAILED: {e}")
            time.sleep(1.0)

    if args.continuous:
        logging.info(f"Starting Continuous Scraper. Interval: {args.interval}s. Press Ctrl+C to stop.")
        while True:
            run_cycle()
            logging.info(f"Cycle completed. Waiting {args.interval}s...\n")
            time.sleep(args.interval)
    else:
        run_cycle()


if __name__ == "__main__":
    main()
