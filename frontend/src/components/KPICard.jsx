import React from 'react';
import { ArrowUpRight, ArrowDownRight, TrendingUp, Cpu, BarChart } from 'lucide-react';

export default function KPICard({ title, value, change, isPositive, subtext, icon: Icon, isSignal = false, badgeText }) {
  return (
    <div className="bg-white rounded border border-slate-200 p-4 transition-all hover:border-slate-300">
      <div className="flex items-center justify-between text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
        <span>{title}</span>
        {Icon && <Icon className="w-4 h-4 text-slate-400" />}
      </div>

      <div className="flex items-baseline space-x-2">
        <div className={`text-2xl font-bold tracking-tight tabular-nums ${isSignal ? 'text-[#1E5FBF]' : 'text-slate-900'}`}>
          {value}
        </div>
      </div>

      <div className="mt-2.5 flex items-center justify-between text-xs">
        {change !== undefined && (
          <div className={`flex items-center font-semibold tabular-nums ${isPositive ? 'text-emerald-600' : 'text-rose-600'}`}>
            {isPositive ? (
              <ArrowUpRight className="w-4 h-4 mr-0.5 stroke-[2.5]" />
            ) : (
              <ArrowDownRight className="w-4 h-4 mr-0.5 stroke-[2.5]" />
            )}
            <span>{change}</span>
          </div>
        )}

        {badgeText && (
          <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-blue-50 text-[#1E5FBF] border border-blue-200">
            {badgeText}
          </span>
        )}

        {subtext && (
          <span className="text-slate-500 text-[11px] font-medium">
            {subtext}
          </span>
        )}
      </div>
    </div>
  );
}
