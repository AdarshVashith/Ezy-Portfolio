import React, { useState, useEffect } from 'react';
import { fetchBacktestSummary } from '../services/api';
import { ShieldCheck, TrendingDown, Layers, CheckCircle2, AlertTriangle, FileText, Check } from 'lucide-react';

export default function ResearchPage() {
  const [summary, setSummary] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      setIsLoading(true);
      const data = await fetchBacktestSummary();
      setSummary(data);
      setIsLoading(false);
    }
    loadData();
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      
      {/* Header & Mission */}
      <div className="bg-white rounded border border-slate-200 p-6">
        <div className="flex items-center space-x-2 text-xs font-bold text-[#1E5FBF] uppercase tracking-wider mb-1">
          <ShieldCheck className="w-4 h-4" />
          <span>Empirical Research Summary</span>
        </div>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
          Systematic Predictability & Execution Timing Audit on NSE Equities
        </h1>
        <p className="mt-2 text-sm text-slate-600 leading-relaxed max-w-4xl">
          This study evaluated 76,368 daily candles across 11 core Indian equity assets spanning 1996–2026. 
          Through sixteen iterative hypotheses, the research tested directional machine learning, 
          volatility clustering, cross-sectional relative momentum, and walk-forward portfolio execution.
        </p>
      </div>

      {/* Backtest Statistics Cards */}
      <div className="space-y-3">
        <h2 className="text-xs font-bold text-slate-500 uppercase tracking-wider">
          Walk-Forward Execution Audit Performance (Modern Regime: 2021–2026 Control)
        </h2>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white rounded border border-slate-200 p-4">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
              Strategy Net CAGR
            </div>
            <div className="text-2xl font-bold font-mono tabular-nums text-rose-600">
              {summary ? `${summary.cagr.toFixed(2)}%` : '—'}
            </div>
            <div className="mt-2 text-[11px] text-slate-500">
              Benchmark CAGR: <span className="font-mono font-semibold text-slate-800">{summary ? `+${summary.benchmark_cagr.toFixed(2)}%` : '—'}</span>
            </div>
          </div>

          <div className="bg-white rounded border border-slate-200 p-4">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
              Realized Sharpe Ratio
            </div>
            <div className="text-2xl font-bold font-mono tabular-nums text-slate-900">
              {summary ? summary.sharpe.toFixed(2) : '—'}
            </div>
            <div className="mt-2 text-[11px] text-slate-500">
              Benchmark Sharpe: <span className="font-mono font-semibold text-slate-800">{summary ? summary.benchmark_sharpe.toFixed(2) : '—'}</span> (Rf = 6.0%)
            </div>
          </div>

          <div className="bg-white rounded border border-slate-200 p-4">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
              Maximum Drawdown
            </div>
            <div className="text-2xl font-bold font-mono tabular-nums text-rose-600">
              {summary ? `${summary.max_drawdown.toFixed(2)}%` : '—'}
            </div>
            <div className="mt-2 text-[11px] text-slate-500">
              Benchmark Max DD: <span className="font-mono font-semibold text-slate-800">{summary ? `${summary.benchmark_max_drawdown.toFixed(2)}%` : '—'}</span>
            </div>
          </div>

          <div className="bg-white rounded border border-slate-200 p-4">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
              Daily Win Rate & Turnover
            </div>
            <div className="text-2xl font-bold font-mono tabular-nums text-slate-900">
              {summary ? `${summary.win_rate.toFixed(1)}%` : '—'}
            </div>
            <div className="mt-2 text-[11px] text-slate-500">
              Avg Daily Turnover: <span className="font-mono font-semibold text-slate-800">{summary ? `${summary.daily_turnover_pct.toFixed(1)}%` : '—'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Readable Institutional Prose Blocks */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Finding 1 */}
        <div className="bg-white rounded border border-slate-200 p-6 space-y-3">
          <div className="flex items-center space-x-2 text-xs font-bold text-[#1E5FBF] uppercase tracking-wider">
            <CheckCircle2 className="w-4 h-4 text-[#1E5FBF]" />
            <span>Finding 1: Martingale Difference & Alpha Decay</span>
          </div>
          <h3 className="text-base font-bold text-slate-900">
            Daily Directional Prediction Shows No Edge in Modern Regime (2021–2026)
          </h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            Across multiple iterations testing raw price levels, engineered return ratios, non-linear Random Forests, 
            and 14 multi-factor features, cross-validated accuracy remained statistically indistinguishable from a 
            majority-class baseline in the modern 5-year sample (<span className="font-mono font-semibold text-slate-800">50.85% vs 50.34%, p = 0.418</span>). 
            While historical 30-year data exhibited a minor technical edge (<span className="font-mono font-semibold text-slate-800">+1.46%, p = 0.028</span>), 
            that advantage has fully decayed as electronic and institutional liquidity absorbed daily inefficiencies.
          </p>
        </div>

        {/* Finding 2 */}
        <div className="bg-white rounded border border-slate-200 p-6 space-y-3">
          <div className="flex items-center space-x-2 text-xs font-bold text-[#1E5FBF] uppercase tracking-wider">
            <CheckCircle2 className="w-4 h-4 text-[#1E5FBF]" />
            <span>Finding 2: Statistically Significant Volatility Clustering</span>
          </div>
          <h3 className="text-base font-bold text-slate-900">
            1-Lag Volatility State Outperforms Majority Class (p = 0.026)
          </h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            Unlike directional returns, intraday spread and volatility regimes exhibit robust autocorrelation. 
            A 1-lag volatility persistence rule achieved <span className="font-mono font-semibold text-slate-800">57.29% accuracy</span> across 
            all 5 cross-validation folds, significantly beating the random majority baseline (<span className="font-mono font-semibold text-slate-800">49.21%, p = 0.026</span>). 
            However, complex machine learning ensembles could not exceed this simple 1-lag rule, confirming that exploitable structure 
            is concentrated in the most recent regime state.
          </p>
        </div>

        {/* Finding 3 */}
        <div className="bg-white rounded border border-slate-200 p-6 space-y-3">
          <div className="flex items-center space-x-2 text-xs font-bold text-[#1E5FBF] uppercase tracking-wider">
            <CheckCircle2 className="w-4 h-4 text-[#1E5FBF]" />
            <span>Finding 3: Execution-Timing Leakage Diagnosis</span>
          </div>
          <h3 className="text-base font-bold text-slate-900">
            Initial 82%+ CAGR Diagnosed as Overnight Gap Artifact
          </h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            An initial backtest indicating an 82.63% CAGR was systematically audited rather than accepted. 
            The audit revealed that calculating returns from <span className="font-mono font-semibold text-slate-800">Close(T) to Close(T+1)</span> captured 
            an unexecutable <span className="font-mono font-semibold text-slate-800">+0.12% to +0.38% daily overnight jump</span> before market open. 
            When simulated with realistic execution (<span className="font-mono font-semibold text-slate-800">Open(T+1) to Open(T+2)</span>), 
            the strategy underperformed the passive buy-and-hold benchmark due to a <span className="font-mono font-semibold text-slate-800">57% daily turnover friction</span>.
          </p>
        </div>

        {/* Finding 4 */}
        <div className="bg-white rounded border border-slate-200 p-6 space-y-3">
          <div className="flex items-center space-x-2 text-xs font-bold text-[#1E5FBF] uppercase tracking-wider">
            <CheckCircle2 className="w-4 h-4 text-[#1E5FBF]" />
            <span>Finding 4: Survivorship Bias in Backfilled Universes</span>
          </div>
          <h3 className="text-base font-bold text-slate-900">
            Universe Selection Flattens Early-Era Benchmark to ~40% CAGR
          </h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            Backfilling today's 11 large-cap market leaders to 2001 inherently excludes companies that subsequently delisted or decayed. 
            Consequently, the universe benchmark itself compounded at 39.88% CAGR during 2001–2010. 
            This highlights the vital necessity of point-in-time constituent data when modeling multi-decade systematic equities.
          </p>
        </div>

      </div>

    </div>
  );
}
