import React, { useState } from 'react';
import { TrendingUp, AlertCircle, Compass, HelpCircle } from 'lucide-react';

export default function MonteCarloFanChart({ mcData, isLoading }) {
  const [hoveredDay, setHoveredDay] = useState(null);

  if (isLoading) {
    return (
      <div className="bg-white rounded border border-slate-200 p-6 mt-4 animate-pulse">
        <div className="h-5 bg-slate-200 rounded w-1/4 mb-4"></div>
        <div className="h-64 bg-slate-100 rounded w-full mb-3"></div>
        <div className="h-4 bg-slate-200 rounded w-2/3"></div>
      </div>
    );
  }

  if (!mcData || !mcData.fan_chart || mcData.fan_chart.length === 0) {
    return (
      <div className="bg-white rounded border border-slate-200 p-6 mt-4 text-xs text-slate-400 text-center">
        Monte Carlo Probability Forecast unavailable for this symbol.
      </div>
    );
  }

  const { last_price, probability_above_current_pct, fan_chart, methodology_note, forecast_days } = mcData;

  // Chart dimensions & scaling
  const width = 800;
  const height = 280;
  const padding = { top: 20, right: 75, bottom: 35, left: 65 };

  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Find min and max price across all percentiles
  let minPrice = last_price;
  let maxPrice = last_price;

  fan_chart.forEach(pt => {
    if (pt.p5 < minPrice) minPrice = pt.p5;
    if (pt.p95 > maxPrice) maxPrice = pt.p95;
  });

  // Add 5% buffer to Y-axis
  const priceRange = maxPrice - minPrice || 1;
  const yMin = minPrice - priceRange * 0.05;
  const yMax = maxPrice + priceRange * 0.05;

  const getX = (day) => padding.left + ((day - 1) / (fan_chart.length - 1)) * chartWidth;
  const getY = (price) => padding.top + chartHeight - ((price - yMin) / (yMax - yMin)) * chartHeight;

  // Build SVG polygon paths for 5-95 and 25-75 confidence cones
  const topP95 = fan_chart.map(pt => `${getX(pt.day)},${getY(pt.p95)}`);
  const botP5 = [...fan_chart].reverse().map(pt => `${getX(pt.day)},${getY(pt.p5)}`);
  const cone95Path = `M ${getX(1)},${getY(last_price)} ` + topP95.join(' L ') + ' L ' + botP5.join(' L ') + ' Z';

  const topP75 = fan_chart.map(pt => `${getX(pt.day)},${getY(pt.p75)}`);
  const botP25 = [...fan_chart].reverse().map(pt => `${getX(pt.day)},${getY(pt.p25)}`);
  const cone75Path = `M ${getX(1)},${getY(last_price)} ` + topP75.join(' L ') + ' L ' + botP25.join(' L ') + ' Z';

  const lineP50 = fan_chart.map((pt, i) => `${i === 0 ? 'M' : 'L'} ${getX(pt.day)},${getY(pt.p50)}`).join(' ');

  const currentPriceY = getY(last_price);

  // Day 30 endpoint metrics
  const finalPoint = fan_chart[fan_chart.length - 1];
  const activePoint = hoveredDay !== null ? fan_chart[hoveredDay] : finalPoint;

  // Y-axis tick marks
  const yTicks = [
    yMin + (yMax - yMin) * 0.1,
    yMin + (yMax - yMin) * 0.35,
    last_price,
    yMin + (yMax - yMin) * 0.7,
    yMin + (yMax - yMin) * 0.9,
  ];

  return (
    <div className="bg-white rounded border border-slate-200 p-5 mt-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-100 mb-3">
        <div className="flex items-center space-x-2">
          <Compass className="w-4 h-4 text-[#1E5FBF]" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-900">
            {forecast_days}-Day Price Probability Forecast (Monte Carlo Fan Chart)
          </h3>
        </div>

        {/* Live Hover Inspection Info */}
        <div className="flex items-center space-x-3 text-xs font-mono tabular-nums text-slate-600">
          <span className="text-slate-400 font-sans">Day {activePoint.day}:</span>
          <span className="text-slate-500">5th: ₹{activePoint.p5.toFixed(1)}</span>
          <span className="font-bold text-[#1E5FBF]">50th (Med): ₹{activePoint.p50.toFixed(1)}</span>
          <span className="text-slate-500">95th: ₹{activePoint.p95.toFixed(1)}</span>
        </div>
      </div>

      {/* SVG Fan Chart */}
      <div className="relative w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto max-h-[300px]"
          onMouseLeave={() => setHoveredDay(null)}
        >
          {/* Background Grid */}
          {yTicks.map((val, idx) => (
            <g key={idx}>
              <line
                x1={padding.left}
                y1={getY(val)}
                x2={width - padding.right}
                y2={getY(val)}
                stroke="#F1F5F9"
                strokeWidth="1"
              />
              <text
                x={padding.left - 8}
                y={getY(val) + 3.5}
                textAnchor="end"
                className="text-[10px] font-mono fill-slate-400 tabular-nums"
              >
                ₹{val.toFixed(0)}
              </text>
            </g>
          ))}

          {/* Current Price Reference Line */}
          <line
            x1={padding.left}
            y1={currentPriceY}
            x2={width - padding.right}
            y2={currentPriceY}
            stroke="#94A3B8"
            strokeDasharray="4 3"
            strokeWidth="1.2"
          />
          <text
            x={width - padding.right + 6}
            y={currentPriceY + 3.5}
            className="text-[10px] font-mono font-semibold fill-slate-700 tabular-nums"
          >
            ₹{last_price.toFixed(0)} (Base)
          </text>

          {/* 5th to 95th Percentile Fan (Outer Cone) */}
          <path
            d={cone95Path}
            fill="#EBF3FC"
            stroke="none"
            opacity="0.85"
          />

          {/* 25th to 75th Percentile Fan (Inner Cone) */}
          <path
            d={cone75Path}
            fill="#BFDBFE"
            stroke="none"
            opacity="0.9"
          />

          {/* Median 50th Percentile Path */}
          <path
            d={lineP50}
            fill="none"
            stroke="#1E5FBF"
            strokeWidth="2.2"
            strokeLinecap="round"
          />

          {/* Interactive Day Vertical Hover Trackers */}
          {fan_chart.map((pt, i) => (
            <rect
              key={pt.day}
              x={getX(pt.day) - (chartWidth / fan_chart.length) / 2}
              y={padding.top}
              width={chartWidth / fan_chart.length}
              height={chartHeight}
              fill="transparent"
              onMouseEnter={() => setHoveredDay(i)}
              className="cursor-pointer"
            />
          ))}

          {/* Hover Indicator Crosshair */}
          {hoveredDay !== null && (
            <g>
              <line
                x1={getX(fan_chart[hoveredDay].day)}
                y1={padding.top}
                x2={getX(fan_chart[hoveredDay].day)}
                y2={padding.top + chartHeight}
                stroke="#1E5FBF"
                strokeDasharray="2 2"
                strokeWidth="1"
              />
              <circle
                cx={getX(fan_chart[hoveredDay].day)}
                cy={getY(fan_chart[hoveredDay].p50)}
                r="3.5"
                fill="#1E5FBF"
              />
            </g>
          )}

          {/* X-axis Day labels */}
          {[1, 5, 10, 15, 20, 25, 30].map(day => (
            <text
              key={day}
              x={getX(day)}
              y={height - 10}
              textAnchor="middle"
              className="text-[10px] font-mono fill-slate-400 tabular-nums"
            >
              Day {day}
            </text>
          ))}
        </svg>
      </div>

      {/* Probability Summary Stat Bar */}
      <div className="mt-3 pt-3 border-t border-slate-100 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-semibold text-slate-700">
            30-Day Directional Probability:
          </span>
          <span className="px-2.5 py-0.5 rounded text-xs font-bold font-mono bg-blue-50 text-[#1E5FBF] border border-blue-200">
            {probability_above_current_pct}% probability price is ABOVE current level in 30 days
          </span>
        </div>

        {/* Legend */}
        <div className="flex items-center space-x-3 text-[11px] text-slate-500 font-mono">
          <span className="flex items-center">
            <span className="w-3 h-2 bg-[#EBF3FC] border border-blue-200 inline-block mr-1.5" /> 5th–95th (90% Conf)
          </span>
          <span className="flex items-center">
            <span className="w-3 h-2 bg-[#BFDBFE] inline-block mr-1.5" /> 25th–75th
          </span>
          <span className="flex items-center text-[#1E5FBF] font-semibold">
            <span className="w-3 h-0.5 bg-[#1E5FBF] inline-block mr-1.5" /> Median Path
          </span>
        </div>
      </div>

      {/* Prominent Visible Caveat & Methodology Note (No Tooltip) */}
      <div className="mt-3 p-3 bg-slate-50 rounded border border-slate-200 text-[11px] text-slate-500 leading-relaxed">
        <div className="flex items-start space-x-2">
          <AlertCircle className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
          <p>
            <strong className="text-slate-700 font-semibold">Methodology Caveat: </strong>
            {methodology_note}
          </p>
        </div>
      </div>
    </div>
  );
}
