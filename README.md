# Stock & News Data Pipeline + ML-Ready Dataset

Complete End-to-End Stock Market Analysis & ML Pipeline:
1. **Live & Historical Stock Data (NSE)**: Real-time price tracking, historical OHLC backfilling, SQLite persistence.
2. **Financial News & Sentiment Scoring**: RSS news feeds (ET, Moneycontrol, Mint, NDTV, Yahoo), exact word boundaries, negation handling, sentiment scoring.
3. **ML Dataset Pipeline**: Automatic news-to-company symbol tagging, daily sentiment aggregation, and target label creation (`next_day_direction` & `next_day_return_pct`).

---

## 📁 Files Overview

1. **`scraper_step1.py`**: Basic single stock price scraper (Learning version).
2. **`scraper_step2.py`**: Live multi-stock OHLC scraper (Duplicate-free & weekend filtered).
3. **`scraper_advanced.py`**: Historical Backfill (1mo, 1y, etc.) & continuous scraper.
4. **`financial_lexicon.py`**: Pre-compiled regex sentiment engine with span deduplication & clause-aware negation.
5. **`news_scraper.py`**: Financial news scraper across 5 active RSS feeds.
6. **`build_dataset.py`**: Merges Price (OHLC) + Sentiment into the final `ml_dataset` table and exports CSV.
7. **`view_data.py`**: Formatted CLI database viewer for all tables.
8. **`stock_data.db`**: Local SQLite database storing all collected data.

---

## 🚀 Execution Workflow (Step-by-Step)

### Step 1: Stock Data Collect / Backfill Karein
```bash
# 1 Month historical backfill
python3 scraper_advanced.py --backfill --range 1mo

# Ya Live OHLC prices fetch karein
python3 scraper_step2.py
```

### Step 2: Financial News & Sentiment Scrape Karein
```bash
python3 news_scraper.py
```

### Step 3: ML Dataset Build & Export Karein
```bash
# Combine Price + Sentiment into 'ml_dataset' table and export CSV:
python3 build_dataset.py --export-csv ml_dataset.csv
```

---

## 📊 Data Inspect Karna (`view_data.py`)

```bash
# 1. ML-Ready Dataset Preview:
python3 view_data.py --table ml_dataset --limit 15

# 2. Specific Stock ka ML Dataset:
python3 view_data.py --table ml_dataset --symbol RELIANCE.NS --limit 10

# 3. News Headlines & Sentiment Feed:
python3 view_data.py --table news --limit 10

# 4. Stock OHLC Table:
python3 view_data.py --table ohlc_data
```
