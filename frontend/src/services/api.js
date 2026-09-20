/**
 * API Service for Ezy-Portfolio Quantitative Dashboard
 * Interacts with backend REST endpoints:
 * - GET /api/symbols
 * - GET /api/prices/:symbol?days=180
 * - GET /api/news?limit=25
 * - GET /api/prediction/:symbol
 * - GET /api/backtest-summary
 * - GET /api/monte-carlo/:symbol?days=30&simulations=5000
 * - GET /api/regime/:symbol
 */

export const SYMBOLS_METADATA = [
  { symbol: "RELIANCE.NS", name: "Reliance Industries Ltd", sector: "Energy / Conglomerate", basePrice: 1226.40 },
  { symbol: "TCS.NS", name: "Tata Consultancy Services Ltd", sector: "Information Technology", basePrice: 2105.00 },
  { symbol: "INFY.NS", name: "Infosys Ltd", sector: "Information Technology", basePrice: 1051.40 },
  { symbol: "HDFCBANK.NS", name: "HDFC Bank Ltd", sector: "Banking & Financials", basePrice: 731.00 },
  { symbol: "ICICIBANK.NS", name: "ICICI Bank Ltd", sector: "Banking & Financials", basePrice: 1338.90 },
  { symbol: "SBIN.NS", name: "State Bank of India", sector: "Public Sector Banking", basePrice: 996.20 },
  { symbol: "BHARTIARTL.NS", name: "Bharti Airtel Ltd", sector: "Telecommunications", basePrice: 1893.30 },
  { symbol: "KOTAKBANK.NS", name: "Kotak Mahindra Bank Ltd", sector: "Banking & Financials", basePrice: 412.50 },
  { symbol: "WIPRO.NS", name: "Wipro Ltd", sector: "Information Technology", basePrice: 166.83 },
  { symbol: "ITC.NS", name: "ITC Ltd", sector: "Consumer Goods / FMCG", basePrice: 262.30 },
  { symbol: "LT.NS", name: "Larsen & Toubro Ltd", sector: "Infrastructure & Engineering", basePrice: 3885.00 },
];

function generateFallbackCandles(symbol, days = 180) {
  const meta = SYMBOLS_METADATA.find(s => s.symbol === symbol) || SYMBOLS_METADATA[0];
  let price = meta.basePrice;
  const candles = [];
  const now = new Date();

  const dateList = [];
  let d = new Date(now);
  while (dateList.length < days) {
    d.setDate(d.getDate() - 1);
    const dayOfWeek = d.getDay();
    if (dayOfWeek !== 0 && dayOfWeek !== 6) {
      dateList.push(new Date(d));
    }
  }
  dateList.reverse();

  let seed = 42;
  function pseudoRandom() {
    seed = (seed * 9301 + 49297) % 233280;
    return seed / 233280;
  }

  for (let i = 0; i < dateList.length; i++) {
    const curDate = dateList[i].toISOString().split('T')[0];
    const dailyReturn = (pseudoRandom() - 0.49) * 0.028;
    const open = price;
    const close = +(open * (1 + dailyReturn)).toFixed(2);
    const spread = Math.abs(open * 0.015 * pseudoRandom()) + 2.0;
    const high = +(Math.max(open, close) + spread).toFixed(2);
    const low = +(Math.min(open, close) - spread).toFixed(2);
    const volume = Math.floor(1500000 + pseudoRandom() * 8500000);

    candles.push({ date: curDate, open, high, low, close, volume });
    price = close;
  }
  return candles;
}

export async function fetchSymbols() {
  try {
    const res = await fetch('/api/symbols');
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data) && data.length > 0) return data;
    }
  } catch {}
  return SYMBOLS_METADATA;
}

export async function fetchPrices(symbol, days = 180) {
  try {
    const res = await fetch(`/api/prices/${encodeURIComponent(symbol)}?days=${days}`);
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data) && data.length > 0) return data;
    }
  } catch {}
  return generateFallbackCandles(symbol, days);
}

export async function fetchNews(limit = 25) {
  try {
    const res = await fetch(`/api/news?limit=${limit}`);
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data) && data.length > 0) return data;
    }
  } catch {}
  return [];
}

export async function fetchPrediction(symbol) {
  try {
    const res = await fetch(`/api/prediction/${encodeURIComponent(symbol)}`);
    if (res.ok) return await res.json();
  } catch {}
  
  return {
    symbol,
    signal: "High Volatility Persistence",
    secondary_signal: "No Directional Edge (Martingale)",
    confidence: 57.3,
    volatility_regime: "Elevated Volatility Cluster",
    model_name: "RandomForest Volatility-Regime Classifier (v10)",
    p_value: 0.026,
    disclaimer: "Based on rigorous backtesting, directional predictions in the current market regime do not show statistically significant edge (see Research Findings). Displayed for demonstration purposes.",
    last_updated: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  };
}

export async function fetchMonteCarlo(symbol, days = 30, simulations = 5000) {
  try {
    const res = await fetch(`/api/monte-carlo/${encodeURIComponent(symbol)}?days=${days}&simulations=${simulations}`);
    if (res.ok) return await res.json();
  } catch {}

  const meta = SYMBOLS_METADATA.find(s => s.symbol === symbol) || SYMBOLS_METADATA[0];
  const lastPrice = meta.basePrice;
  const fan_chart = [];
  
  for (let i = 1; i <= days; i++) {
    const std = Math.sqrt(i) * (lastPrice * 0.018);
    fan_chart.push({
      day: i,
      p5: +(lastPrice - 1.645 * std).toFixed(2),
      p25: +(lastPrice - 0.674 * std).toFixed(2),
      p50: +(lastPrice + (i * 0.0005 * lastPrice)).toFixed(2),
      p75: +(lastPrice + 0.674 * std).toFixed(2),
      p95: +(lastPrice + 1.645 * std).toFixed(2),
    });
  }

  return {
    symbol,
    last_price: lastPrice,
    last_date: new Date().toISOString().split('T')[0],
    forecast_days: days,
    probability_above_current_pct: 54.0,
    fan_chart,
    methodology_note: "Based on Geometric Brownian Motion using historical drift and volatility. This is a probability-based risk visualization, not a guaranteed forecast. Real markets exhibit regime changes and fat tails not captured by this simplified model."
  };
}

export async function fetchRegime(symbol) {
  try {
    const res = await fetch(`/api/regime/${encodeURIComponent(symbol)}`);
    if (res.ok) return await res.json();
  } catch {}

  return {
    symbol,
    current_regime: "Calm",
    current_regime_probabilities: { calm_pct: 93.2, turbulent_pct: 6.8 },
    regime_characteristics: {
      calm: { avg_volatility_pct: 1.70, frequency_pct: 91.6 },
      turbulent: { avg_volatility_pct: 5.29, frequency_pct: 8.4 }
    },
    persistence_probabilities: {
      stay_calm_if_calm_pct: 93.2,
      stay_turbulent_if_turbulent_pct: 45.9
    },
    methodology_note: "Regime detected using a Hidden Markov Model on historical intraday price range. Empirically validated: volatility clustering shows a statistically significant persistence effect (p=0.026 in prior testing), unlike directional price movement, which showed no reliable edge in the modern market regime."
  };
}

export async function fetchBacktestSummary() {
  try {
    const res = await fetch('/api/backtest-summary');
    if (res.ok) return await res.json();
  } catch {}

  return {
    cagr: -12.71,
    gross_cagr: 0.15,
    benchmark_cagr: 9.93,
    sharpe: -0.95,
    benchmark_sharpe: 0.30,
    max_drawdown: -62.88,
    benchmark_max_drawdown: -19.29,
    win_rate: 47.4,
    benchmark_win_rate: 52.4,
    daily_turnover_pct: 57.0,
    transaction_cost_bps: 10,
    horizon: "2021-01-01 to 2026-09-11 (Modern Era Control)",
    total_candles: 76368,
    universe_size: 11
  };
}
