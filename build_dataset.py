#!/usr/bin/env python3
"""
Price + Sentiment Dataset Builder (ML-Ready Pipeline)
------------------------------------------------------
Ye script:
1. News headlines se company names identify karke Stock Symbols assign karta hai
2. Daily level par symbol-wise average sentiment aur news counts aggregate karta hai
3. `ohlc_data` (price) aur `news` (sentiment) ko date-wise merge karta hai
4. Target column (`next_day_direction` & `next_day_return_pct`) calculate karta hai
5. Final `ml_dataset` table database (`stock_data.db`) mein save karta hai

Chalane ka tarika:
    python3 build_dataset.py
    python3 build_dataset.py --export-csv ml_dataset.csv
"""

import re
import csv
import sqlite3
import argparse
import email.utils
from datetime import datetime
from collections import defaultdict

DB_FILE = "stock_data.db"

# -------------------------------------------------------------
# 🏢 Comprehensive Company Name -> Stock Symbol Mapping
# -------------------------------------------------------------
COMPANY_SYMBOL_MAP = {
    # Reliance Industries
    "reliance": "RELIANCE.NS",
    "ril": "RELIANCE.NS",
    "reliance industries": "RELIANCE.NS",
    "jio": "RELIANCE.NS",

    # Tata Consultancy Services
    "tcs": "TCS.NS",
    "tata consultancy": "TCS.NS",
    "tata consultancy services": "TCS.NS",

    # Infosys
    "infosys": "INFY.NS",
    "infy": "INFY.NS",

    # HDFC Bank
    "hdfc bank": "HDFCBANK.NS",
    "hdfc": "HDFCBANK.NS",

    # ICICI Bank
    "icici bank": "ICICIBANK.NS",
    "icici": "ICICIBANK.NS",

    # State Bank of India
    "sbin": "SBIN.NS",
    "sbi": "SBIN.NS",
    "state bank of india": "SBIN.NS",
    "state bank": "SBIN.NS",

    # Wipro & Tata Motors
    "wipro": "WIPRO.NS",
    "tata motors": "TATAMTRDVR.NS",
}


def tag_news_with_symbol(title):
    """
    Headline mein company name dhoondh ke uska symbol return karta hai.
    Word boundary (\\b) use karta hai taaki 'sbi' kisi doosre word ke andar match na ho.
    """
    if not title:
        return None

    title_lower = title.lower()
    matched_symbols = set()

    for company_name, symbol in COMPANY_SYMBOL_MAP.items():
        if re.search(r'\b' + re.escape(company_name) + r'\b', title_lower):
            matched_symbols.add(symbol)

    return list(matched_symbols) if matched_symbols else None


def parse_date_only(date_str):
    """
    Robust Date Parser:
    - ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
    - RFC-822 / RSS Date format ('Thu, 17 Sep 2026 12:04:50 +0530') via email.utils
    - Fallback strptime patterns
    """
    if not date_str:
        return datetime.now().strftime("%Y-%m-%d")

    date_str = str(date_str).strip()

    # 1. ISO format check: '2026-09-17...'
    if len(date_str) >= 10 and date_str[4] == '-' and date_str[7] == '-':
        return date_str[:10]

    # 2. RFC-822 / RSS feed format with timezone (e.g. 'Thu, 17 Sep 2026 12:04:50 +0530')
    try:
        dt = email.utils.parsedate_to_datetime(date_str)
        if dt:
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass

    # 3. Standard strptime patterns on full string
    for fmt in [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S",
        "%d %b %Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d"
    ]:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass

    return date_str[:10]


def get_daily_sentiment(db_file=DB_FILE):
    """
    News table se saari headlines leke, symbol-wise aur date-wise aggregate karta hai.
    Returns: dict like { ('RELIANCE.NS', '2026-09-17'): {avg_sentiment, news_count, ...} }
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='news'")
    if not cursor.fetchone():
        conn.close()
        return {}

    cursor.execute("SELECT title, published, sentiment, label FROM news")
    rows = cursor.fetchall()
    conn.close()

    grouped = defaultdict(list)
    tagged_count = 0

    for title, published, sentiment, label in rows:
        symbols = tag_news_with_symbol(title)
        if not symbols:
            continue

        date_only = parse_date_only(published)
        tagged_count += 1

        for symbol in symbols:
            grouped[(symbol, date_only)].append({
                "sentiment": sentiment if sentiment is not None else 0.0,
                "label": label or "NEUTRAL"
            })

    daily_sentiment = {}
    for (symbol, date), items in grouped.items():
        scores = [item["sentiment"] for item in items]
        bullish_count = sum(1 for item in items if item["label"] == "BULLISH")
        bearish_count = sum(1 for item in items if item["label"] == "BEARISH")

        daily_sentiment[(symbol, date)] = {
            "avg_sentiment": round(sum(scores) / len(scores), 3),
            "news_count": len(items),
            "bullish_count": bullish_count,
            "bearish_count": bearish_count
        }

    return daily_sentiment


def build_combined_dataset(db_file=DB_FILE):
    """
    Price data + sentiment data ko combine karke ML dataset banata hai.
    """
    daily_sentiment = get_daily_sentiment(db_file)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ohlc_data'")
    if not cursor.fetchone():
        print("Error: Table 'ohlc_data' database mein nahi mili.")
        conn.close()
        return []

    # Get column info
    cursor.execute("PRAGMA table_info(ohlc_data)")
    cols = [c[1] for c in cursor.fetchall()]
    date_col = "date" if "date" in cols else "timestamp"

    cursor.execute(f"SELECT symbol, {date_col}, open, high, low, close, volume FROM ohlc_data ORDER BY symbol, {date_col} ASC")
    price_rows = cursor.fetchall()
    conn.close()

    combined_data = []

    for symbol, date_val, open_p, high_p, low_p, close_p, volume in price_rows:
        date_only = parse_date_only(date_val)
        key = (symbol, date_only)

        # Matched sentiment or neutral default
        sentiment_data = daily_sentiment.get(key, {
            "avg_sentiment": 0.0,
            "news_count": 0,
            "bullish_count": 0,
            "bearish_count": 0
        })

        combined_data.append({
            "symbol": symbol,
            "date": date_only,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": volume,
            "avg_sentiment": sentiment_data["avg_sentiment"],
            "news_count": sentiment_data["news_count"],
            "bullish_count": sentiment_data["bullish_count"],
            "bearish_count": sentiment_data["bearish_count"],
        })

    return combined_data


def save_and_compute_targets(combined_data, db_file=DB_FILE):
    """
    ml_dataset table banata hai aur next_day_direction & next_day_return_pct targets compute karta hai.
    """
    if not combined_data:
        print("No combined data available to save.")
        return 0

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ml_dataset (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            avg_sentiment REAL,
            news_count INTEGER,
            bullish_count INTEGER,
            bearish_count INTEGER,
            next_day_direction INTEGER DEFAULT NULL,
            next_day_return_pct REAL DEFAULT NULL,
            UNIQUE(symbol, date) ON CONFLICT REPLACE
        )
    """)

    # Insert raw rows
    for row in combined_data:
        cursor.execute("""
            INSERT OR REPLACE INTO ml_dataset
            (symbol, date, open, high, low, close, volume, avg_sentiment, news_count, bullish_count, bearish_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["symbol"], row["date"], row["open"], row["high"], row["low"],
            row["close"], row["volume"], row["avg_sentiment"], row["news_count"],
            row["bullish_count"], row["bearish_count"]
        ))

    conn.commit()

    # Compute target labels per symbol
    cursor.execute("SELECT DISTINCT symbol FROM ml_dataset")
    symbols = [r[0] for r in cursor.fetchall()]

    for symbol in symbols:
        cursor.execute("""
            SELECT id, close, date FROM ml_dataset
            WHERE symbol = ? ORDER BY date ASC
        """, (symbol,))
        rows = cursor.fetchall()

        for i in range(len(rows) - 1):
            curr_id, curr_close, _ = rows[i]
            _, next_close, _ = rows[i + 1]

            direction = 1 if next_close > curr_close else 0
            ret_pct = round(((next_close - curr_close) / curr_close) * 100, 2)

            cursor.execute("""
                UPDATE ml_dataset 
                SET next_day_direction = ?, next_day_return_pct = ?
                WHERE id = ?
            """, (direction, ret_pct, curr_id))

    conn.commit()
    conn.close()
    return len(combined_data)


def export_to_csv(db_file=DB_FILE, output_csv="ml_dataset.csv"):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT symbol, date, open, high, low, close, volume, 
               avg_sentiment, news_count, bullish_count, bearish_count, 
               next_day_direction, next_day_return_pct 
        FROM ml_dataset ORDER BY symbol, date DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No rows found in ml_dataset to export.")
        return

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Symbol", "Date", "Open", "High", "Low", "Close", "Volume",
            "AvgSentiment", "NewsCount", "BullishCount", "BearishCount",
            "NextDayDirection", "NextDayReturnPct"
        ])
        writer.writerows(rows)

    print(f"Exported {len(rows)} rows to '{output_csv}' successfully.")


def preview_dataset(db_file=DB_FILE, limit=12):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT symbol, date, close, volume, avg_sentiment, news_count, next_day_direction, next_day_return_pct
        FROM ml_dataset ORDER BY date DESC, symbol ASC LIMIT {limit}
    """)
    rows = cursor.fetchall()
    
    cursor.execute("SELECT COUNT(*), COUNT(next_day_direction), SUM(CASE WHEN next_day_direction=1 THEN 1 ELSE 0 END), SUM(CASE WHEN next_day_direction=0 THEN 1 ELSE 0 END), SUM(CASE WHEN news_count > 0 THEN 1 ELSE 0 END) FROM ml_dataset")
    total, labeled, up_days, down_days, news_linked = cursor.fetchone()
    conn.close()

    print(f"\n=======================================================================================================")
    print(f" 🤖 ML-READY DATASET PREVIEW | Total Rows: {total} | News Linked: {news_linked or 0} | Labeled: {labeled} (📈 UP: {up_days or 0} | 📉 DOWN: {down_days or 0})")
    print(f"=======================================================================================================")
    print(f"{'SYMBOL':<14} | {'DATE':<11} | {'CLOSE':<9} | {'VOLUME':<10} | {'SENTIMENT':<10} | {'NEWS#':<6} | {'TARGET (UP/DOWN)':<16} | {'RETURN %'}")
    print("-" * 103)

    for row in rows:
        sym, dt, cl, vol, sent, n_cnt, target, ret = row
        tgt_str = "🟢 1 (UP)" if target == 1 else ("🔴 0 (DOWN)" if target == 0 else "⏳ Pending")
        ret_str = f"{ret:+.2f}%" if ret is not None else "N/A"
        print(f"{sym:<14} | {dt:<11} | {cl:>9.2f} | {vol:>10} | {sent:>10.3f} | {n_cnt:>6} | {tgt_str:<16} | {ret_str}")

    print(f"=======================================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Combine Price and Sentiment into ML-Ready Dataset")
    parser.add_argument("--export-csv", nargs="?", const="ml_dataset.csv", help="Export dataset to CSV file")
    parser.add_argument("--preview", action="store_true", help="Preview the ml_dataset table")
    args = parser.parse_args()

    print("\n--- [Step 1: Building Combined Price + Sentiment Dataset] ---")
    combined = build_combined_dataset(DB_FILE)
    saved_count = save_and_compute_targets(combined, DB_FILE)
    print(f"Successfully processed {saved_count} rows in 'ml_dataset' table.")

    if args.export_csv:
        export_to_csv(DB_FILE, args.export_csv)

    preview_dataset(DB_FILE)


if __name__ == "__main__":
    main()
