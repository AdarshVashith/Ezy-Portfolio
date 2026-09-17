#!/usr/bin/env python3
"""
STEP 3: Financial News Scraper — General + Targeted Company RSS Feeds
----------------------------------------------------------------------
Features:
1. General Financial Feeds: Economic Times, Moneycontrol, LiveMint, NDTV Profit, Yahoo Finance
2. Targeted Company Feeds: Google News RSS for Reliance, TCS, Infosys, HDFC Bank, ICICI Bank, SBI
3. Robust Sentiment Engine: `financial_lexicon.py` (Word boundaries, span deduplication, clause negation)
4. Duplicate-free SQLite Storage: 'news' table with `UNIQUE(title, source)`

Chalane ka tarika:
    python3 news_scraper.py
    python3 news_scraper.py --targeted-only
"""

import time
import sqlite3
import argparse
import urllib.parse
import feedparser
from datetime import datetime
from financial_lexicon import analyze_financial_sentiment

# ---------- CONFIG ----------
DB_FILE = "stock_data.db"

# 1. General Financial Feeds
GENERAL_RSS_FEEDS = {
    "Economic Times": "https://economictimes.indiatimes.com/rssfeedsdefault.cms",
    "Moneycontrol": "https://www.moneycontrol.com/rss/business.xml",
    "LiveMint Markets": "https://www.livemint.com/rss/markets",
    "NDTV Profit": "https://feeds.feedburner.com/ndtvprofit-latest",
    "Yahoo Finance News": "https://finance.yahoo.com/news/rssindex",
}

# 2. Targeted Company Google News Queries
TARGETED_COMPANY_QUERIES = {
    "RELIANCE.NS": "Reliance Industries stock OR RIL share price",
    "TCS.NS": "TCS stock OR Tata Consultancy Services share",
    "INFY.NS": "Infosys stock OR INFY share price",
    "HDFCBANK.NS": "HDFC Bank stock OR HDFCBANK share",
    "ICICIBANK.NS": "ICICI Bank stock OR ICICIBANK share",
    "SBIN.NS": "State Bank of India stock OR SBI share price",
}


def build_google_news_url(query):
    encoded = urllib.parse.quote(query)
    return f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"


def fetch_news_from_feed(source_name, feed_url):
    """Ek RSS feed se headlines aur sentiment parse karta hai"""
    feed = feedparser.parse(feed_url, agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    news_items = []

    for entry in feed.entries:
        title = entry.get("title", "").strip()
        if not title:
            continue

        link = entry.get("link", "").strip()
        published = entry.get("published", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Comprehensive financial lexicon se score aur label
        score, label = analyze_financial_sentiment(title)

        news_items.append({
            "source": source_name,
            "title": title,
            "link": link,
            "published": published,
            "sentiment": score,
            "label": label,
            "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    return news_items


def setup_database():
    """News table banata hai (agar pehle se nahi hai)"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            title TEXT,
            link TEXT,
            published TEXT,
            sentiment REAL,
            label TEXT,
            scraped_at TEXT,
            UNIQUE(title, source)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_source ON news(source);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_label ON news(label);")
    conn.commit()
    conn.close()


def save_news(news_items):
    """News items ko database mein save karta hai (duplicates auto-skip)"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    saved_count = 0
    for item in news_items:
        try:
            cursor.execute("""
                INSERT INTO news (source, title, link, published, sentiment, label, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                item["source"], item["title"], item["link"],
                item["published"], item["sentiment"], item["label"],
                item["scraped_at"]
            ))
            saved_count += 1
        except sqlite3.IntegrityError:
            pass  # Duplicate news headline, skip

    conn.commit()
    conn.close()
    return saved_count


def collect_all_news(include_general=True, include_targeted=True):
    """Saare RSS feeds (General + Targeted) se news collect karta hai"""
    setup_database()
    total_fetched = 0
    total_saved = 0

    print(f"\n=======================================================")
    print(f" [News Scraping Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print(f"=======================================================")

    # 1. Scrape General Feeds
    if include_general:
        print("--- [General Financial Feeds] ---")
        for source_name, feed_url in GENERAL_RSS_FEEDS.items():
            try:
                news_items = fetch_news_from_feed(source_name, feed_url)
                saved = save_news(news_items)
                total_fetched += len(news_items)
                total_saved += saved
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {source_name:<22} -> {len(news_items):>3} fetched | {saved:>3} new saved")
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {source_name:<22} -> FAILED: {e}")
            time.sleep(0.8)

    # 2. Scrape Targeted Company Feeds
    if include_targeted:
        print("\n--- [Targeted Stock News Feeds] ---")
        for symbol, query in TARGETED_COMPANY_QUERIES.items():
            source_name = f"GoogleNews ({symbol})"
            feed_url = build_google_news_url(query)
            try:
                news_items = fetch_news_from_feed(source_name, feed_url)
                saved = save_news(news_items)
                total_fetched += len(news_items)
                total_saved += saved
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {source_name:<22} -> {len(news_items):>3} fetched | {saved:>3} new saved")
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {source_name:<22} -> FAILED: {e}")
            time.sleep(0.8)

    print(f"-------------------------------------------------------")
    print(f"Summary: {total_fetched} total headlines processed, {total_saved} new entries added to DB.")
    print(f"=======================================================\n")


def run_forever(interval_minutes=15):
    """Har `interval_minutes` mein news collect karta rehta hai"""
    while True:
        collect_all_news()
        print(f"Waiting {interval_minutes} minutes for next news cycle...\n")
        time.sleep(interval_minutes * 60)


def main():
    parser = argparse.ArgumentParser(description="Financial News & Sentiment Scraper")
    parser.add_argument("--targeted-only", action="store_true", help="Scrape only targeted stock news feeds")
    parser.add_argument("--continuous", action="store_true", help="Run continuously in background")
    parser.add_argument("--interval", type=int, default=15, help="Interval in minutes for continuous mode (default: 15)")
    args = parser.parse_args()

    if args.continuous:
        run_forever(args.interval)
    else:
        collect_all_news(include_general=not args.targeted_only, include_targeted=True)


if __name__ == "__main__":
    main()
