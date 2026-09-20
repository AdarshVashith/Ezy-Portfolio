import React, { useState } from 'react';
import { Activity, Info, ChevronDown, ChevronUp } from 'lucide-react';

export default function VolatilityRegimeCard({ regimeData, isLoading }) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (isLoading) {
    return (
      <div className="bg-white rounded border border-slate-200 p-4 animate-pulse">
        <div className="h-4 bg-slate-200 rounded w-1/3 mb-2"></div>
        <div className="h-6 bg-slate-200 rounded w-1/2 mb-2"></div>
        <div className="h-2 bg-slate-200 rounded w-full"></div>
      </div>
    );
  }

  if (!regimeData) {
    return (
      <div className="bg-white rounded border border-slate-200 p-4 text-xs text-slate-400">
        Volatility Regime: Data unavailable
      </div>
    );
  }

  const isCalm = regimeData.current_regime === 'Calm';
  const calmPct = regimeData.current_regime_probabilities?.calm_pct ?? 90;
  const turbPct = regimeData.current_regime_probabilities?.turbulent_pct ?? 10;

  const stayCalm = regimeData.persistence_probabilities?.stay_calm_if_calm_pct ?? 93.2;
  const stayTurb = regimeData.persistence_probabilities?.stay_turbulent_if_turbulent_pct ?? 45.9;

  return (
    <div className="bg-white rounded border border-slate-200 p-4 transition-all hover:border-slate-300">
      {/* Card Header */}
      <div className="flex items-center justify-between text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
        <div className="flex items-center space-x-1.5">
          <Activity className="w-4 h-4 text-[#1E5FBF]" />
          <span>HMM Volatility Regime</span>
        </div>
        <span
          className={`px-2 py-0.5 rounded text-[11px] font-bold tracking-normal uppercase ${
            isCalm
              ? 'bg-blue-50 text-[#1E5FBF] border border-blue-200'
              : 'bg-amber-50 text-amber-800 border border-amber-200'
          }`}
        >
          {regimeData.current_regime} State
        </span>
      </div>

      {/* Prominent State & Probabilities */}
      <div className="flex items-baseline justify-between mt-1">
        <div className="text-xl font-bold font-mono tracking-tight text-slate-900">
          {isCalm ? `${calmPct}% Calm Probability` : `${turbPct}% Turbulent Probability`}
        </div>
        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-xs text-slate-400 hover:text-[#1E5FBF] flex items-center transition-colors"
          title="Toggle Model Methodology Details"
        >
          {isExpanded ? <ChevronUp className="w-3.5 h-3.5 ml-1" /> : <ChevronDown className="w-3.5 h-3.5 ml-1" />}
        </button>
      </div>

      {/* Two-Segment Regime Probability Bar */}
      <div className="mt-2.5">
        <div className="w-full bg-slate-100 rounded-full h-2 flex overflow-hidden">
          <div
            className="bg-[#1E5FBF] h-2 transition-all duration-500"
            style={{ width: `${calmPct}%` }}
            title={`Calm: ${calmPct}%`}
          />
          <div
            className="bg-amber-400 h-2 transition-all duration-500"
            style={{ width: `${turbPct}%` }}
            title={`Turbulent: ${turbPct}%`}
          />
        </div>
        <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-1">
          <span>Calm: {calmPct}%</span>
          <span>Turbulent: {turbPct}%</span>
        </div>
      </div>

      {/* Plain Language Persistence Statement */}
      <div className="mt-2 text-[11px] text-slate-600 font-medium leading-snug">
        {isCalm ? (
          <span>
            <strong className="text-slate-900 font-mono">{stayCalm}%</strong> persistence probability of remaining in Calm state tomorrow.
          </span>
        ) : (
          <span>
            <strong className="text-slate-900 font-mono">{stayTurb}%</strong> persistence probability of staying Turbulent tomorrow.
          </span>
        )}
      </div>

      {/* Expandable Methodology Note & Validation Disclaimer */}
      {isExpanded && (
        <div className="mt-3 pt-2.5 border-t border-slate-100 text-[10px] text-slate-500 leading-relaxed space-y-1.5">
          <div className="flex items-start space-x-1.5">
            <Info className="w-3.5 h-3.5 text-[#1E5FBF] shrink-0 mt-0.5" />
            <p>{regimeData.methodology_note}</p>
          </div>
        </div>
      )}
    </div>
  );
}
