import React from 'react';

export default function DateRangeSelector({ selectedRange, onSelectRange }) {
  const ranges = [
    { label: '30D', value: 30 },
    { label: '90D', value: 90 },
    { label: '180D', value: 180 },
    { label: '365D', value: 365 },
  ];

  return (
    <div>
      <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">
        Lookback Horizon
      </label>
      <div className="grid grid-cols-4 gap-1 bg-slate-100 p-1 rounded border border-slate-200">
        {ranges.map((r) => {
          const isActive = selectedRange === r.value;
          return (
            <button
              key={r.value}
              type="button"
              onClick={() => onSelectRange(r.value)}
              className={`py-1.5 text-xs font-semibold rounded text-center transition-all ${
                isActive
                  ? 'bg-[#1E5FBF] text-white shadow-sm'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-white/50'
              }`}
            >
              {r.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
