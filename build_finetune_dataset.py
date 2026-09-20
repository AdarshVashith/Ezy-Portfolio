"""
Step 1: Training Dataset Builder -- Finance/Quant Domain Knowledge
--------------------------------------------------------------------
Ye script instruction-tuning format mein training examples banata hai --
"instruction" (question) aur "output" (answer) pairs, jo tumhare project
ke apne concepts aur 20-step research findings se banaye gaye hain.

Chalane ka tarika:
    python3 build_finetune_dataset.py
"""

import json

training_examples = [
    {
        "instruction": "What is RSI in technical analysis?",
        "output": "RSI (Relative Strength Index) is a momentum indicator calculated over a period (typically 14 days) that measures the speed and magnitude of recent price changes. It ranges from 0 to 100. Values above 70 typically indicate 'overbought' conditions (price may be due for a pullback), while values below 30 indicate 'oversold' conditions (price may be due for a bounce). Formula: RSI = 100 - (100 / (1 + RS)), where RS = Average Gain / Average Loss."
    },
    {
        "instruction": "Explain GARCH model in simple terms.",
        "output": "GARCH (Generalized Autoregressive Conditional Heteroskedasticity) is a statistical time-series model for forecasting volatility. Its core GARCH(1,1) equation is: sigma_t^2 = omega + alpha * epsilon_(t-1)^2 + beta * sigma_(t-1)^2. In plain terms: tomorrow's expected variance depends on a baseline long-run average (omega), yesterday's shock size (alpha term), and yesterday's persistent volatility level (beta term). It formalizes 'volatility clustering' -- calm periods tend to stay calm, turbulent periods tend to stay turbulent."
    },
    {
        "instruction": "What is the Kelly Criterion and how does it apply to trading?",
        "output": "The Kelly Criterion is a mathematical formula for optimal position sizing given a known edge: f* = (b*p - q) / b, where p is the win probability, q = 1 - p, and b is the win/loss payoff ratio. It determines the mathematically optimal fraction of capital to allocate to maximize long-term geometric growth. Critically, Kelly only works if your probability estimate (p) is well-calibrated -- if your model's stated confidence is uncalibrated, Kelly sizing can lead to excessive drawdown and underperform fixed-fraction sizing."
    },
    {
        "instruction": "What did this project find about daily stock direction prediction?",
        "output": "This project tested daily UP/DOWN price direction prediction across 11 Indian NSE stocks using Random Forest, Logistic Regression, and Gradient Boosting with walk-forward time-series cross-validation. While a full 30-year historical backtest (1996-2026) showed a statistically significant edge (p=0.028), this edge completely decayed when tested on the modern market regime alone (2021-2026, p=0.418). This demonstrates 'alpha decay' -- simple directional technical patterns have been arbitraged away by modern institutional algorithms."
    },
    {
        "instruction": "What did this project find about volatility clustering in Indian equities?",
        "output": "This project confirmed strong statistical evidence for volatility clustering in NSE stocks. A 1-lag persistence rule -- 'if today exhibited high intraday spread, tomorrow is likely high volatility too' -- achieved 57.3% accuracy versus a 49.2% majority-class baseline (p=0.026, statistically significant). This was further cross-validated using a 2-state Gaussian Hidden Markov Model, which showed high state persistence for both Calm (93.2%) and Turbulent (45.9%-65.9%) regimes."
    },
    {
        "instruction": "What is a p-value and why does it matter in quantitative finance?",
        "output": "A p-value measures the probability of observing a backtest result at least as extreme as the one calculated, assuming the null hypothesis (no true edge, purely random noise) is true. A p-value below 0.05 indicates statistical significance. In quantitative finance, tracking p-values protects researchers from p-hacking and false discoveries -- distinguishing real market phenomena (volatility clustering, p=0.026) from spurious overfitting (modern directional prediction, p=0.418)."
    },
    {
        "instruction": "What is Monte Carlo simulation and how is it used in this project?",
        "output": "Monte Carlo simulation generates a probability distribution of future stock prices by running thousands (e.g., 5,000) of randomized price paths using Geometric Brownian Motion: S_t = S_0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t). Instead of giving a single deterministic target, it produces probability fan charts (5th, 25th, 50th, 75th, 95th percentiles) to communicate risk and horizon uncertainty objectively."
    },
    {
        "instruction": "Why did the initial backtest show an unrealistic 80%+ CAGR?",
        "output": "The initial 80%+ CAGR was an execution-timing artifact caused by two methodological flaws: (1) Execution timing leakage -- features were calculated using Day T's Close price, but trades were assumed to execute at that exact same Close, capturing unexecutable overnight price gaps. (2) Survivorship bias -- testing only modern blue-chip winners. When corrected to realistic Next-Day Open or Close execution, the excess return vanished, proving the edge was an artifact."
    },
    {
        "instruction": "What is a Hidden Markov Model (HMM) in quantitative finance?",
        "output": "A Hidden Markov Model (HMM) is a probabilistic model that assumes the market transitions between unobservable latent states ('Calm' and 'Turbulent' regimes) with specific transition probabilities. Observed emissions (such as intraday spread or returns) follow state-dependent Gaussian distributions. In this project, a 2-state HMM validates market regime shifts and persistence without lookahead bias."
    },
    {
        "instruction": "Should I buy or sell a stock based on this research dashboard?",
        "output": "This research platform does not provide buy or sell recommendations. Rigorous empirical validation on modern NSE data (2021-2026) proved that daily directional predictions behave as a Martingale random walk without statistical edge (p=0.418). The platform provides transparent, grounded statistical snapshots (RSI, moving averages, volatility regime, news sentiment, Monte Carlo fan charts) so users can conduct objective, evidence-based research."
    },
    {
        "instruction": "What is Martingale Difference Sequence in market efficiency?",
        "output": "A Martingale Difference Sequence (MDS) implies that given all historical information up to time t, the expected future return is zero: E[R_(t+1) | I_t] = 0. In our empirical testing on 2021-2026 NSE data, price returns conformed to an MDS, meaning past technical indicators provided no predictive information about future price direction after transaction costs."
    },
    {
        "instruction": "How was news sentiment scored in this project?",
        "output": "News sentiment is scored using a custom 300+ term Financial NLP Lexicon tailored to Indian equity headlines. Each headline is tokenized, matched against weighted financial keywords (positive/bullish, negative/bearish, neutral), and assigned a normalized sentiment score between -1.0 and +1.0. This avoids external black-box LLM API dependencies and provides transparent, reproducible scoring."
    },
    {
        "instruction": "What is Alpha Decay?",
        "output": "Alpha Decay refers to the gradual loss of predictive power or profitability in a quantitative trading strategy over time. As market participants discover similar signals, computational resources increase, and institutional capital flows in, market efficiency rises and historical mispricings are arbitraged away."
    },
    {
        "instruction": "What is Sharpe Ratio and how is it calculated?",
        "output": "The Sharpe Ratio measures risk-adjusted return: Sharpe = (R_p - R_f) / sigma_p, where R_p is the portfolio annualized return, R_f is the risk-free rate, and sigma_p is the annualized standard deviation of excess returns. A Sharpe ratio above 1.0 is considered good, above 2.0 very good, while negative Sharpe indicates underperformance relative to cash."
    },
    {
        "instruction": "What is Maximum Drawdown?",
        "output": "Maximum Drawdown (MDD) is the largest peak-to-trough percentage drop in portfolio equity before a new peak is attained: MDD = (Trough Value - Peak Value) / Peak Value. It is a critical risk metric representing the worst-case capital loss an investor would have experienced over the evaluation period."
    },
    {
        "instruction": "What is Walk-Forward Cross-Validation in quantitative research?",
        "output": "Walk-forward cross-validation (or time-series split) trains a model strictly on past historical data up to time T, and evaluates it on out-of-sample data from time T+1 to T+k. It rolls forward sequentially through time, ensuring zero lookahead bias and preventing data leakage from future timestamps."
    },
    {
        "instruction": "What is Geometric Brownian Motion (GBM)?",
        "output": "Geometric Brownian Motion is a continuous-time stochastic process where the logarithm of the randomly varying quantity follows a Brownian motion with drift. It is standardly expressed as dS_t = mu*S_t*dt + sigma*S_t*dW_t, where mu is the drift rate, sigma is volatility, and W_t is a Wiener process (standard Brownian motion)."
    },
    {
        "instruction": "Why is Volatility Clustering easier to predict than Price Direction?",
        "output": "Price direction (sign of return) in liquid markets is rapidly arbitraged towards a random walk by competitive traders. However, volatility (magnitude of return) reflects information arrival rates and market participant risk adjustments, creating strong autocorrelation (memory) that persists over days and weeks."
    },
    {
        "instruction": "What is Survivorship Bias in backtesting?",
        "output": "Survivorship bias occurs when a backtest is run only on companies that have survived until today (e.g. current NIFTY 50 constituents), omitting companies that went bankrupt, merged, or were delisted during the historical window. This artificially inflates historical returns."
    },
    {
        "instruction": "What stocks are tracked in this research platform?",
        "output": "This research system tracks 11 core Indian NSE equities across major economic sectors: RELIANCE.NS (Energy/Conglomerate), TCS.NS (IT), INFY.NS (IT), WIPRO.NS (IT), HDFCBANK.NS (Banking), ICICIBANK.NS (Banking), SBIN.NS (Banking), KOTAKBANK.NS (Banking), BHARTIARTL.NS (Telecom), ITC.NS (FMCG), and LT.NS (Infrastructure)."
    }
]


def save_dataset(examples, filename="finance_training_data.jsonl"):
    with open(filename, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} training examples to {filename}")
    print("Dataset generated successfully in Alpaca instruction format.")


if __name__ == "__main__":
    save_dataset(training_examples)
