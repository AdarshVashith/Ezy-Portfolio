#!/usr/bin/env python3
"""
Continuous Automated Data Pipeline Runner
-----------------------------------------
Runs Stock OHLC Scraper, Financial News Scraper & ML Dataset Builder
in a continuous background loop with proper error handling and logging.
"""

import time
import logging
import subprocess
import threading
from datetime import datetime

LOG_FILE = "pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def run_stock_loop(interval_sec=60):
    logging.info(f"Stock Scraper Loop started (Interval: {interval_sec}s)")
    while True:
        try:
            logging.info("Running Stock OHLC Scraper...")
            result = subprocess.run(["python3", "scraper_advanced.py"], capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                logging.info("Stock OHLC Scraper finished cycle successfully.")
            else:
                logging.warning(f"Stock Scraper error output: {result.stderr.strip()}")
        except Exception as e:
            logging.error(f"Error in Stock Scraper: {e}")
        time.sleep(interval_sec)

def run_news_and_dataset_loop(interval_sec=900): # 15 minutes
    logging.info(f"News & Dataset Pipeline Loop started (Interval: {interval_sec}s)")
    while True:
        try:
            logging.info("Running News & Sentiment Scraper...")
            news_res = subprocess.run(["python3", "news_scraper.py"], capture_output=True, text=True, timeout=180)
            if news_res.returncode == 0:
                logging.info("News Scraper cycle completed.")
            else:
                logging.warning(f"News Scraper stderr: {news_res.stderr.strip()}")

            logging.info("Updating ML Dataset...")
            dataset_res = subprocess.run(["python3", "build_dataset.py", "--export-csv", "ml_dataset.csv"], capture_output=True, text=True, timeout=120)
            if dataset_res.returncode == 0:
                logging.info("ML Dataset & CSV updated successfully.")
            else:
                logging.warning(f"Dataset Builder stderr: {dataset_res.stderr.strip()}")
        except Exception as e:
            logging.error(f"Error in News/Dataset pipeline: {e}")
        time.sleep(interval_sec)

def main():
    logging.info("=== Starting Master Continuous Data Pipeline ===")
    
    t_stock = threading.Thread(target=run_stock_loop, args=(60,), daemon=True)
    t_news = threading.Thread(target=run_news_and_dataset_loop, args=(900,), daemon=True)
    
    t_stock.start()
    t_news.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Master Pipeline stopped by user.")

if __name__ == "__main__":
    main()
