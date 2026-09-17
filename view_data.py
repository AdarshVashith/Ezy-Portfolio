#!/usr/bin/env python3
"""
Database Viewer Utility
------------------------
Ye script `stock_data.db` ke saare tables (prices, ohlc_data, news, ml_dataset) ko clean formatted tables mein display karta hai.

Chalane ke tarike:
    python3 view_data.py                                    # Default: ohlc_data table
    python3 view_data.py --table ml_dataset                 # ML-ready price + sentiment + targets
    python3 view_data.py --table news                       # News headlines with sentiment
    python3 view_data.py --table news --sentiment bullish   # Sirf Bullish news
    python3 view_data.py --table prices                     # Real-time tick prices
    python3 view_data.py --symbol RELIANCE.NS --limit 10    # Specific stock ka data
"""

import sys
import sqlite3
import argparse

DB_FILE = "stock_data.db"


def view_news(sentiment_filter=None, source_filter=None, limit=15):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='news'")
        if not cursor.fetchone():
            print("Error: Table 'news' database mein nahi mili. Pehle 'python3 news_scraper.py' chalayein.")
            return

        query = "SELECT id, source, label, sentiment, title, published FROM news"
        params = []
        conditions = []

        if sentiment_filter:
            conditions.append("UPPER(label) = ?")
            params.append(sentiment_filter.upper())
        if source_filter:
            conditions.append("LOWER(source) LIKE ?")
            params.append(f"%{source_filter.lower()}%")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        cursor.execute("SELECT COUNT(*), SUM(CASE WHEN label='BULLISH' THEN 1 ELSE 0 END), SUM(CASE WHEN label='BEARISH' THEN 1 ELSE 0 END), SUM(CASE WHEN label='NEUTRAL' THEN 1 ELSE 0 END) FROM news")
        total, bullish, bearish, neutral = cursor.fetchone()

        conn.close()

        print(f"\n========================================================================================================")
        print(f" 📰 NEWS & SENTIMENT FEED | Total: {total} (🟢 Bullish: {bullish or 0} | 🔴 Bearish: {bearish or 0} | ⚪ Neutral: {neutral or 0}) | Showing: {len(rows)}")
        print(f"========================================================================================================")

        if not rows:
            print("No news records found matching criteria.")
            return

        print(f"{'ID':<5} | {'SOURCE':<18} | {'SENTIMENT':<9} | {'SCORE':<6} | {'HEADLINE':<55}")
        print("-" * 104)

        for row in rows:
            news_id, source, label, score, title, published = row
            if label == "BULLISH":
                lbl_str = "🟢 BULLISH"
            elif label == "BEARISH":
                lbl_str = "🔴 BEARISH"
            else:
                lbl_str = "⚪ NEUTRAL"

            title_clean = title.replace("\n", " ").strip()
            if len(title_clean) > 52:
                title_clean = title_clean[:49] + "..."

            print(f"{news_id:<5} | {source:<18} | {lbl_str:<11} | {score:>5.2f} | {title_clean:<55}")

        print(f"========================================================================================================\n")

    except Exception as e:
        print(f"Error: {e}")


def view_table(table_name="ohlc_data", symbol=None, limit=15):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cursor.fetchone():
            print(f"Error: Table '{table_name}' database mein nahi mili.")
            return

        cursor.execute(f"PRAGMA table_info({table_name})")
        col_info = cursor.fetchall()
        columns = [col[1] for col in col_info]

        sort_col = "date" if "date" in columns else "id"

        query = f"SELECT * FROM {table_name}"
        params = []

        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)

        query += f" ORDER BY {sort_col} DESC, id DESC LIMIT {limit}"
        cursor.execute(query, params)
        rows = cursor.fetchall()

        count_query = f"SELECT COUNT(*) FROM {table_name}"
        if symbol:
            count_query += " WHERE symbol = ?"
            cursor.execute(count_query, (symbol,))
        else:
            cursor.execute(count_query)
        total_rows = cursor.fetchone()[0]

        conn.close()

        title_info = f" Table: '{table_name}'"
        if symbol:
            title_info += f" | Symbol: {symbol}"
        title_info += f" | Total Records: {total_rows} | Showing: {len(rows)}"

        print(f"\n=========================================================================================")
        print(f"{title_info}")
        print(f"=========================================================================================")

        if not rows:
            print("No records found.")
            return

        header_line = " | ".join([f"{col.upper():<11}" for col in columns])
        print(header_line)
        print("-" * len(header_line))

        for row in rows:
            formatted_row = []
            for item in row:
                if isinstance(item, float):
                    formatted_row.append(f"{item:<11.2f}")
                elif isinstance(item, int):
                    formatted_row.append(f"{item:<11}")
                else:
                    val_str = str(item) if item is not None else ""
                    if len(val_str) > 19:
                        val_str = val_str[:19]
                    formatted_row.append(f"{val_str:<11}")
            print(" | ".join(formatted_row))

        print(f"=========================================================================================\n")

    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="View stock, news, and ML dataset from SQLite database.")
    parser.add_argument("--table", choices=["prices", "ohlc_data", "news", "ml_dataset"], default="ohlc_data", help="Table name to view")
    parser.add_argument("--symbol", help="Filter by stock symbol (e.g., RELIANCE.NS)")
    parser.add_argument("--sentiment", choices=["bullish", "bearish", "neutral"], help="Filter news by sentiment")
    parser.add_argument("--source", help="Filter news by RSS source name")
    parser.add_argument("--limit", type=int, default=15, help="Number of rows to display (default: 15)")
    args = parser.parse_args()

    if args.table == "news" or args.sentiment or args.source:
        view_news(sentiment_filter=args.sentiment, source_filter=args.source, limit=args.limit)
    else:
        view_table(table_name=args.table, symbol=args.symbol, limit=args.limit)


if __name__ == "__main__":
    main()
