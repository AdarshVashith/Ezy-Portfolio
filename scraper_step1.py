#!/usr/bin/env python3
"""
STEP 1: Basic Stock Data Scraper (Learning Version)
------------------------------------------------------
Ye script:
1. Yahoo Finance se ek stock ka current price aur volume nikaalta hai
2. Use SQLite database (stock_data.db) mein save karta hai
3. Har baar chalane par ek naya row add hota hai

Chalane ka tarika:
    python3 scraper_step1.py
"""

import re
import time
import sqlite3
import requests
from datetime import datetime

# ---------- CONFIG ----------
SYMBOL = "RELIANCE.NS"   # NSE stock symbol (.NS = National Stock Exchange India)
DB_FILE = "stock_data.db"

# Global session to keep cookies and crumb
_session = None
_crumb = None


def get_session():
    """Yahoo Finance ke liye session aur crumb token tayyar karta hai"""
    global _session, _crumb
    if _session is not None and _crumb is not None:
        return _session, _crumb

    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    # Step 1: Initial page request to get cookies
    res = session.get(f"https://finance.yahoo.com/quote/{SYMBOL}", headers=headers, timeout=10)
    
    # Step 2: Extract or fetch crumb
    crumb_match = re.search(r'\"crumb\":\"(.*?)\"', res.text)
    if crumb_match:
        _crumb = crumb_match.group(1)
    else:
        api_headers = {
            "User-Agent": headers["User-Agent"],
            "Referer": f"https://finance.yahoo.com/quote/{SYMBOL}",
        }
        crumb_res = session.get("https://query1.finance.yahoo.com/v1/test/getcrumb", headers=api_headers, timeout=10)
        if crumb_res.status_code == 200:
            _crumb = crumb_res.text
        else:
            _crumb = None

    _session = session
    return _session, _crumb


def fetch_price(symbol):
    """Website se latest price aur volume nikaalta hai"""
    session, crumb = get_session()
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"
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
    meta = result["meta"]
    price = meta.get("regularMarketPrice") or meta.get("chartPreviousClose")
    
    # Volume extraction with safe fallback
    quote = result["indicators"]["quote"][0]
    volumes = quote.get("volume", [])
    volume = next((v for v in reversed(volumes) if v is not None), meta.get("regularMarketVolume", 0))

    return price, volume


def setup_database():
    """Database aur table banata hai (agar pehle se nahi hai)"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            price REAL,
            volume INTEGER,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_price(symbol, price, volume):
    """Ek naya price record database mein save karta hai"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO prices (symbol, price, volume, timestamp) VALUES (?, ?, ?, ?)",
        (symbol, price, volume, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def main():
    setup_database()
    price, volume = fetch_price(SYMBOL)
    save_price(SYMBOL, price, volume)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved -> {SYMBOL}: ₹{price} (Volume: {volume})")


def run_forever():
    """
    Ye function har 60 second mein data collect karta rehta hai.
    """
    setup_database()
    print(f"Starting continuous scraper for {SYMBOL}... Press Ctrl+C to stop.\n")
    while True:
        try:
            price, volume = fetch_price(SYMBOL)
            save_price(SYMBOL, price, volume)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved -> {SYMBOL}: ₹{price} (Volume: {volume})")
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: {e}")
        time.sleep(60)


if __name__ == "__main__":
    main()
    # run_forever()  # Uncomment karein continuous chalane ke liye
