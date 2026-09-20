import React from 'react';
import { Database, ShieldAlert, Cpu, Layers, Terminal, BookOpen } from 'lucide-react';

export default function AboutPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      
      {/* Overview Card */}
      <div className="bg-white rounded border border-slate-200 p-6 space-y-3">
        <div className="flex items-center space-x-2 text-xs font-bold text-[#1E5FBF] uppercase tracking-wider">
          <BookOpen className="w-4 h-4" />
          <span>Project Overview & Motivation</span>
        </div>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
          About the Ezy-Portfolio Quantitative Research Platform
        </h1>
        <p className="text-sm text-slate-600 leading-relaxed">
          Ezy-Portfolio is a systematic quantitative research pipeline designed to investigate market efficiency, 
          alpha decay, and machine learning predictability within Indian equity markets (NSE). 
          The project emphasizes rigorous empirical standards: hypothesis testing against strong naive baselines, 
          time-series cross-validation without look-ahead bias, and forensic auditing of statistical anomalies.
        </p>
      </div>

      {/* Architecture & Stack */}
      <div className="bg-white rounded border border-slate-200 p-6 space-y-4">
        <div className="flex items-center space-x-2 text-xs font-bold text-[#1E5FBF] uppercase tracking-wider">
          <Layers className="w-4 h-4" />
          <span>System Architecture & Technology Stack</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
          <div className="p-4 bg-slate-50 rounded border border-slate-200">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 font-mono mb-2">
              Data & Feature Engineering
            </h3>
            <ul className="text-xs text-slate-600 space-y-1.5 list-disc list-inside">
              <li>Yahoo Finance multi-decade OHLCV ingestion (1996–2026)</li>
              <li>SQLite database with unique constraints</li>
              <li>RSS scrapers: ET, Mint, Moneycontrol, NDTV Profit</li>
              <li>Custom 300+ term financial sentiment lexicon</li>
            </ul>
          </div>

          <div className="p-4 bg-slate-50 rounded border border-slate-200">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 font-mono mb-2">
              Machine Learning & Validation
            </h3>
            <ul className="text-xs text-slate-600 space-y-1.5 list-disc list-inside">
              <li>Scikit-Learn (Random Forest, Logistic Regression)</li>
              <li>5-Fold TimeSeriesSplit Cross-Validation</li>
              <li>Paired Student t-tests (<code className="font-mono text-slate-800">scipy.stats</code>)</li>
              <li>Walk-Forward simulation with realistic 10 bps friction</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Target Asset Universe */}
      <div className="bg-white rounded border border-slate-200 p-6 space-y-3">
        <div className="flex items-center space-x-2 text-xs font-bold text-[#1E5FBF] uppercase tracking-wider">
          <Database className="w-4 h-4" />
          <span>Tracked NSE Universe (11 Assets)</span>
        </div>
        <p className="text-xs text-slate-600 leading-relaxed">
          The dataset encompasses 76,368 total daily candles across the following 11 constituents:
        </p>
        <div className="flex flex-wrap gap-2 pt-1">
          {[
            'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
            'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'WIPRO.NS', 'ITC.NS', 'LT.NS'
          ].map((sym) => (
            <span
              key={sym}
              className="px-2.5 py-1 rounded bg-slate-100 border border-slate-200 text-xs font-mono font-medium text-slate-800"
            >
              {sym.replace('.NS', '')}
            </span>
          ))}
        </div>
      </div>

      {/* Academic / Regulatory Disclaimer */}
      <div className="bg-amber-50 rounded border border-amber-200 p-6 space-y-2">
        <div className="flex items-center space-x-2 text-xs font-bold text-amber-800 uppercase tracking-wider">
          <ShieldAlert className="w-4 h-4 text-amber-700" />
          <span>Regulatory & Educational Disclaimer</span>
        </div>
        <p className="text-xs text-amber-900 leading-relaxed">
          This platform and repository are created strictly for academic, educational, and computational quantitative research. 
          None of the information, model predictions, backtest results, or sentiment metrics presented on this dashboard 
          constitute financial, investment, legal, or tax advice. 
          Historical backtests and model signals do not guarantee future performance.
        </p>
      </div>

    </div>
  );
}
