import React, { useState, useEffect, useRef } from 'react';
import { createChart, ColorType, LineSeries } from 'lightweight-charts';

function calculateSMA(data, period) {
  const result = [];
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) continue;
    let sum = 0;
    for (let j = 0; j < period; j++) {
      sum += Number(data[i - j].close);
    }
    result.push({
      time: data[i].date,
      value: +(sum / period).toFixed(2),
    });
  }
  return result;
}

function calculateRSI(data, period = 14) {
  const result = [];
  if (data.length <= period) return result;

  let gains = 0;
  let losses = 0;

  for (let i = 1; i <= period; i++) {
    const diff = Number(data[i].close) - Number(data[i - 1].close);
    if (diff >= 0) gains += diff;
    else losses -= diff;
  }

  let avgGain = gains / period;
  let avgLoss = losses / period;

  let rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
  let rsi = 100 - 100 / (1 + rs);
  result.push({ time: data[period].date, value: +rsi.toFixed(2) });

  for (let i = period + 1; i < data.length; i++) {
    const diff = Number(data[i].close) - Number(data[i - 1].close);
    const gain = diff > 0 ? diff : 0;
    const loss = diff < 0 ? -diff : 0;

    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;

    rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    rsi = 100 - 100 / (1 + rs);
    result.push({ time: data[i].date, value: +rsi.toFixed(2) });
  }

  return result;
}

export default function IndicatorChart({ data }) {
  const [activeTab, setActiveTab] = useState('ma');
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    if (!data || data.length === 0) return;

    const container = chartContainerRef.current;
    const chart = createChart(container, {
      width: container.clientWidth,
      height: 180,
      layout: {
        background: { type: ColorType.Solid, color: '#FFFFFF' },
        textColor: '#64748B',
        fontSize: 10,
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      },
      grid: {
        vertLines: { color: '#F8FAFC' },
        horzLines: { color: '#F1F5F9' },
      },
      crosshair: {
        vertLine: { color: '#1E5FBF', width: 1, style: 3 },
        horzLine: { color: '#1E5FBF', width: 1, style: 3 },
      },
      rightPriceScale: {
        borderColor: '#E2E8F0',
      },
      timeScale: {
        borderColor: '#E2E8F0',
        timeVisible: true,
      },
    });

    if (activeTab === 'ma') {
      const closeLine = chart.addSeries(LineSeries, {
        color: '#94A3B8',
        lineWidth: 1,
        title: 'Close',
      });
      closeLine.setData(
        data.map((d) => ({ time: d.date, value: Number(d.close) }))
      );

      const sma20 = chart.addSeries(LineSeries, {
        color: '#1E5FBF',
        lineWidth: 2,
        title: 'SMA 20',
      });
      sma20.setData(calculateSMA(data, 20));

      const sma50 = chart.addSeries(LineSeries, {
        color: '#F59E0B',
        lineWidth: 1.5,
        title: 'SMA 50',
      });
      sma50.setData(calculateSMA(data, 50));
    } else {
      const rsiSeries = chart.addSeries(LineSeries, {
        color: '#1E5FBF',
        lineWidth: 2,
        title: 'RSI 14',
      });
      const rsiData = calculateRSI(data, 14);
      rsiSeries.setData(rsiData);

      // Overbought 70
      const obLine = chart.addSeries(LineSeries, {
        color: '#EF4444',
        lineWidth: 1,
        lineStyle: 2,
        title: '70 Overbought',
      });
      obLine.setData(rsiData.map((d) => ({ time: d.time, value: 70 })));

      // Oversold 30
      const osLine = chart.addSeries(LineSeries, {
        color: '#10B981',
        lineWidth: 1,
        lineStyle: 2,
        title: '30 Oversold',
      });
      osLine.setData(rsiData.map((d) => ({ time: d.time, value: 30 })));
    }

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartContainerRef.current && chart) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);
    chartRef.current = chart;

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [data, activeTab]);

  return (
    <div className="bg-white rounded border border-slate-200 p-4 mt-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-3">
        {/* Tab Controls */}
        <div className="flex space-x-2">
          <button
            type="button"
            onClick={() => setActiveTab('ma')}
            className={`px-3 py-1.5 text-xs font-bold rounded transition-colors ${
              activeTab === 'ma'
                ? 'bg-blue-50 text-[#1E5FBF] border border-blue-200'
                : 'text-slate-600 hover:text-slate-900 bg-slate-50 border border-slate-200'
            }`}
          >
            Moving Averages (SMA 20 / SMA 50)
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('rsi')}
            className={`px-3 py-1.5 text-xs font-bold rounded transition-colors ${
              activeTab === 'rsi'
                ? 'bg-blue-50 text-[#1E5FBF] border border-blue-200'
                : 'text-slate-600 hover:text-slate-900 bg-slate-50 border border-slate-200'
            }`}
          >
            Relative Strength Index (RSI 14)
          </button>
        </div>

        {/* Legend */}
        <div className="text-[11px] font-mono text-slate-500 hidden sm:flex items-center space-x-3">
          {activeTab === 'ma' ? (
            <>
              <span className="flex items-center">
                <span className="w-2.5 h-1 bg-[#1E5FBF] mr-1.5 inline-block" /> SMA 20
              </span>
              <span className="flex items-center">
                <span className="w-2.5 h-1 bg-[#F59E0B] mr-1.5 inline-block" /> SMA 50
              </span>
              <span className="flex items-center">
                <span className="w-2.5 h-1 bg-[#94A3B8] mr-1.5 inline-block" /> Close
              </span>
            </>
          ) : (
            <>
              <span className="flex items-center text-rose-600">70 OB</span>
              <span className="flex items-center text-emerald-600">30 OS</span>
              <span className="flex items-center text-[#1E5FBF]">RSI 14</span>
            </>
          )}
        </div>
      </div>

      <div className="relative w-full h-[180px]">
        <div ref={chartContainerRef} className="w-full h-full" />
      </div>
    </div>
  );
}
