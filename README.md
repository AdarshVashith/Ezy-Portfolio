# Ezy-Portfolio: A Quantitative Research Journey into Indian Equity Predictability

## Overview

This repository documents an end-to-end quantitative research project built from scratch: a real-time data collection pipeline, a financial news sentiment engine, and sixteen iterative machine learning experiments testing whether Indian equity markets (NSE) contain exploitable predictive signal.

The project does not claim to have found a profitable trading strategy. Instead, it demonstrates a rigorous, statistically grounded research process: forming hypotheses, testing them against proper baselines, cross-validating results, and specifically hunting for and eliminating false positives (data leakage, survivorship bias, and look-ahead bias) before drawing conclusions. Several early "positive" results were later proven to be statistical artifacts under closer scrutiny — this process of self-correction is the central contribution of the project.

## Motivation

The goal was to learn, from first principles, how to build a complete quantitative research pipeline: data engineering, NLP-based sentiment scoring, feature engineering, supervised learning, and rigorous statistical validation — the same skill set increasingly in demand across quant research, quant trading, and applied ML roles in 2026, where hybrid profiles combining engineering, statistics, and domain judgment are valued over narrow specialists.

## System Architecture

```
[Yahoo Finance OHLC Scraper] ----\
                                   >---> [SQLite Database: stock_data.db] ---> [ML Dataset Builder]
[RSS News Scraper + Sentiment] --/                                                    |
                                                                                        v
                                                                          [Feature Engineering Layer]
                                                                                        |
                                                                                        v
                                                                    [Model Training + Time-Series CV]
                                                                                        |
                                                                                        v
                                                                    [Statistical Significance Testing]
```

## Tech Stack

- **Data collection:** Python, `requests`, `feedparser` (RSS parsing)
- **Storage:** SQLite
- **Sentiment analysis:** Custom rule-based financial lexicon (300+ terms) with negation handling and span-based deduplication
- **Feature engineering / ML:** `pandas`, `numpy`, `scikit-learn` (Random Forest, Logistic Regression)
- **Statistics:** `scipy.stats` (paired t-tests, significance testing)

## Data Sources

- **Price data:** Yahoo Finance historical OHLCV data for 11 major NSE-listed stocks (RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, BHARTIARTL, KOTAKBANK, WIPRO, ITC, LT), spanning up to 30 years (1996-2026) depending on listing date.
- **News data:** RSS feeds from Economic Times, Moneycontrol, LiveMint, NDTV Profit, and Yahoo Finance, scored using a custom-built financial sentiment lexicon.

## Research Journey: Summary of All Experiments

| Step | Experiment | Method | Key Result |
|---|---|---|---|
| 1-3 | Data pipeline construction | Scraper, sentiment engine, dataset builder | Working end-to-end ETL pipeline |
| 4 | Baseline direction model | Logistic Regression on raw prices | Underperformed baseline (47.8% vs 54.5%) |
| 5 | Feature engineering | Returns/ratios instead of raw prices | Marginal change (48.6%) |
| 6 | Non-linear model test | Random Forest, same features | Result within statistical noise of baseline |
| 7 | Time-series cross-validation | 5-fold CV instead of single split | Confirmed no reliable signal (avg -4.6 pts vs baseline) |
| 8 | Advanced technical features | RSI, Bollinger Bands, multi-day sentiment | No improvement over basic features (fair comparison test) |
| 9 | Big-data test | 76,000+ candles, 30 years, 11 stocks | Statistically significant edge found (+1.46%, p=0.028) |
| 9b | Modern-regime test | Same features, last 5 years only | Edge did not replicate (p=0.418) — historical signal had decayed |
| 10 | Deep feature set (14 features) | Added MACD, momentum, gap, day-of-week | No improvement in modern regime (p=0.418) |
| 11 | Volatility regression | Predict next-day volatility magnitude | Beat naive baseline (+20.3%, p=0.018) but failed to beat a constant-mean baseline (p=0.735) — revealed as mean-shrinkage artifact, not genuine skill |
| 12 | Volatility regime classification | Binary high/low volatility classification | Confirmed volatility clustering is real (persistence beats majority class, p=0.026); ML model could not exceed the simple persistence rule |
| 13 | Cross-sectional relative momentum (daily) | Predict relative outperformance vs peer average | No signal found; daily-frequency relative returns behave as noise |
| 14 | Academic momentum test (monthly) | 12-month formation period, monthly resampling | Directional test consistent with literature (small-sample, exploratory) |
| 15-16 | Portfolio backtest and execution audit | Walk-forward backtest with realistic execution timing | Initial extreme returns (CAGR > 80%) traced to overnight-gap look-ahead leakage and survivorship bias in stock universe selection; corrected for realistic execution, the modern-era strategy showed no positive alpha, consistent with Step 9b |

## Key Findings

**1. Daily-frequency directional prediction shows no reliable signal in the modern market (2021-2026).**
Multiple feature sets (raw price, returns, technical indicators, momentum, day-of-week effects) were tested with Random Forest and Logistic Regression models under rigorous time-series cross-validation. None produced a statistically significant edge over a majority-class baseline in the recent market regime, despite a significant edge appearing in the full 30-year historical sample. This is consistent with the concept of alpha decay: patterns that existed in a less efficient, less automated market appear to have been arbitraged away as algorithmic and institutional participation increased.

**2. Volatility clustering is real and statistically confirmed, but is not further exploitable with the tested feature set.**
A simple rule — "if today was a high-volatility day, tomorrow is also likely to be one" — significantly outperformed a naive majority-class baseline (57.3% vs 49.2%, p = 0.026). However, a Random Forest model given the same and additional features could not reliably outperform this simple persistence rule, indicating that the exploitable signal is concentrated in the most recent volatility state rather than in more complex feature interactions.

**3. A large apparent backtested return (CAGR above 80%) was diagnosed as a methodological artifact, not real alpha.**
A full portfolio backtest initially showed extreme, implausible returns. Rather than accepting this result, it was systematically audited and traced to two causes: (a) execution-timing leakage, where the backtest implicitly captured unexecutable overnight price gaps, and (b) survivorship bias, from testing on a stock universe selected using hindsight (today's market leaders projected back to 2001). Once corrected for realistic trade execution timing, the strategy's performance in the modern era showed no positive alpha, corroborating the earlier direction-prediction findings through a completely independent methodology.

**4. Volume, price-range, and momentum-based features consistently rank as the most informative, though modest in absolute effect.**
Across nearly every experiment, `Volume_Change_Pct`, `Intraday_Spread_Pct`, and short-window return moving averages ranked highest in feature importance, even in experiments that failed to beat baseline overall. This is a plausible area for future investigation with richer data.

## Methodological Practices Applied

- **Time-series cross-validation** was used throughout instead of random train/test splits, to avoid look-ahead bias inherent in financial time series.
- **Baseline selection was treated as seriously as model selection.** Several apparent "wins" against a weak baseline were shown to disappear when tested against a more appropriate baseline (e.g., constant-mean baseline for regression, persistence baseline for classification).
- **Statistical significance testing** (paired t-tests across folds) was applied to every reported result rather than relying on single point-estimates of accuracy.
- **Fair, controlled comparisons** were used when testing whether new features added value — always on identical data with only the variable of interest changed, to avoid conflating dataset changes with genuine feature effects.
- **Apparent breakthroughs were treated with heightened suspicion, not excitement.** Both the volatility-regression result and the portfolio backtest result initially looked like strong positive findings and were both later shown, through deliberate follow-up testing, to be artifacts rather than genuine predictive skill.

## Repository Structure

```
EZ/
├── continuous_pipeline.py                  Background scraper for live data collection
├── scraper_advanced.py                     Multi-decade historical OHLC backfill engine
├── news_scraper.py                         RSS-based financial news sentiment scraper
├── financial_lexicon.py                    Custom financial sentiment lexicon and scorer
├── build_dataset.py                        ETL script combining price and sentiment data
├── ml_dataset.csv                          Combined, labeled dataset (76,000+ rows, 1996-2026)
├── stock_data.db                           SQLite database (raw price and news data)
└── models/
    ├── train_model.py                      Step 4: Baseline linear model
    ├── train_model_v2.py                   Step 5: Feature engineering (returns, ratios)
    ├── train_model_v3.py                   Step 6: Non-linear model (Random Forest)
    ├── train_model_v4_cv.py                Step 7: Time-series cross-validation
    ├── train_model_v5_advanced.py          Step 8: Technical indicators, fair comparison
    ├── train_model_v6_bigdata.py           Step 9: Full 30-year, 76k-row dataset
    ├── train_model_v7_recent_only.py       Step 9b: Modern-regime-only test
    ├── train_model_v8_recent_deep.py       Step 10: Extended feature set
    ├── train_model_v9_volatility.py        Step 11: Volatility regression
    ├── train_model_v10_vol_regime.py       Step 12: Volatility regime classification
    ├── train_model_v11_relative_momentum.py Step 13: Cross-sectional daily momentum
    ├── train_model_v12_monthly_momentum.py Step 14: Academic monthly momentum test
    └── train_model_v13_portfolio_backtest.py / v14_execution_audit.py
                                              Step 15-16: Portfolio backtest and leakage audit
```

## How to Run

```bash
# 1. Collect data
python3 scraper_advanced.py --backfill --range max --symbols RELIANCE.NS TCS.NS INFY.NS ...
python3 news_scraper.py

# 2. Build the combined ML dataset
python3 build_dataset.py --export-csv ml_dataset.csv

# 3. Run any experiment
cd models
python3 train_model_v6_bigdata.py
```

## Limitations and Honest Caveats

- This project tests a specific, limited set of features (technical indicators derived from OHLCV data, plus RSS-based sentiment) on a specific universe of 11 large-cap NSE stocks. Absence of signal in these experiments does not prove markets are entirely unpredictable — it demonstrates that this particular feature set, on this data, under rigorous testing, did not yield a reliable edge.
- Sentiment coverage was sparse for several stocks on several days, limiting the strength of any conclusions about news-driven effects.
- The monthly momentum test (Step 14) used a limited number of monthly observations and should be treated as directional/exploratory rather than conclusive.
- All findings pertain to the historical data on which they were tested and do not constitute investment advice.

## Future Work

- Extend the volatility clustering finding using proper realized-volatility measures (e.g., from intraday data) and formal GARCH-family models for a stronger baseline comparison.
- Test formal momentum strategies at the standard 3-12 month academic horizon with a larger and more representative stock universe to address survivorship bias.
- Explore ensemble approaches that combine the volatility-regime signal with relative-momentum signals.
- Incorporate a properly point-in-time stock universe (avoiding survivorship bias) for any future backtesting work.

## Disclaimer

This project is for educational and research purposes only. It does not constitute financial or investment advice. Historical backtested performance, where shown, does not guarantee future results, and several of the most extreme results in this repository's history were later shown to be methodological artifacts rather than genuine trading edges — a reminder to treat any backtested performance claim, from any source, with appropriate skepticism until independently audited.
