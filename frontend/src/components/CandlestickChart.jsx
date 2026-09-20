import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, CrosshairMode, CandlestickSeries, HistogramSeries } from 'lightweight-charts';
import { RefreshCw } from 'lucide-react';

export default function CandlestickChart({ data, symbol, isLoading }) {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const [hoverData, setHoverData] = useState(null);

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
      height: 380,
      layout: {
        background: { type: ColorType.Solid, color: '#FFFFFF' },
        textColor: '#64748B',
        fontSize: 11,
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      },
      grid: {
        vertLines: { color: '#F1F5F9' },
        horzLines: { color: '#F1F5F9' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: '#1E5FBF',
          width: 1,
          style: 3,
          labelBackgroundColor: '#1E5FBF',
        },
        horzLine: {
          color: '#1E5FBF',
          width: 1,
          style: 3,
          labelBackgroundColor: '#1E5FBF',
        },
      },
      rightPriceScale: {
        borderColor: '#E2E8F0',
        scaleMargins: {
          top: 0.1,
          bottom: 0.25,
        },
      },
      timeScale: {
        borderColor: '#E2E8F0',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    // Candlestick series using lightweight-charts v5 API
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#16A34A',
      downColor: '#DC2626',
      borderVisible: false,
      wickUpColor: '#16A34A',
      wickDownColor: '#DC2626',
    });

    const formattedCandles = data.map((d) => ({
      time: d.date,
      open: Number(d.open),
      high: Number(d.high),
      low: Number(d.low),
      close: Number(d.close),
    })).sort((a, b) => (a.time > b.time ? 1 : -1));

    candleSeries.setData(formattedCandles);

    // Volume histogram on bottom pane
    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '', // overlay
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });

    const formattedVolume = data.map((d) => ({
      time: d.date,
      value: Number(d.volume) || 0,
      color: Number(d.close) >= Number(d.open) ? '#DCFCE7' : '#FEE2E2',
    })).sort((a, b) => (a.time > b.time ? 1 : -1));

    volumeSeries.setData(formattedVolume);

    chart.timeScale().fitContent();

    // Crosshair move tooltip
    chart.subscribeCrosshairMove((param) => {
      if (
        param.point === undefined ||
        !param.time ||
        param.point.x < 0 ||
        param.point.x > container.clientWidth ||
        param.point.y < 0 ||
        param.point.y > 380
      ) {
        setHoverData(null);
      } else {
        const candle = param.seriesData.get(candleSeries);
        const volume = param.seriesData.get(volumeSeries);
        if (candle) {
          setHoverData({
            time: param.time,
            open: candle.open,
            high: candle.high,
            low: candle.low,
            close: candle.close,
            volume: volume ? volume.value : null,
          });
        }
      }
    });

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
  }, [data]);

  const latestCandle = data && data.length > 0 ? data[data.length - 1] : null;
  const displayPoint = hoverData || latestCandle;

  return (
    <div className="bg-white rounded border border-slate-200 p-4">
      {/* Chart Header & Live OHLC Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-100 mb-3">
        <div className="flex items-center space-x-3">
          <span className="text-xs font-bold uppercase tracking-wider text-[#1E5FBF]">
            OHLCV Candlestick
          </span>
          <span className="text-xs text-slate-400 font-mono">|</span>
          <span className="text-xs font-mono font-semibold text-slate-700">
            {symbol.replace('.NS', '')} (NSE)
          </span>
        </div>

        {/* Tabular OHLC Figures */}
        {displayPoint && (
          <div className="flex items-center space-x-3 text-xs font-mono tabular-nums text-slate-600">
            <span>
              <span className="text-slate-400 font-sans">O:</span> {Number(displayPoint.open).toFixed(2)}
            </span>
            <span>
              <span className="text-slate-400 font-sans">H:</span> {Number(displayPoint.high).toFixed(2)}
            </span>
            <span>
              <span className="text-slate-400 font-sans">L:</span> {Number(displayPoint.low).toFixed(2)}
            </span>
            <span className="font-bold text-slate-900">
              <span className="text-slate-400 font-sans">C:</span> {Number(displayPoint.close).toFixed(2)}
            </span>
            {displayPoint.volume && (
              <span className="hidden sm:inline text-slate-500">
                <span className="text-slate-400 font-sans">Vol:</span> {(Number(displayPoint.volume) / 1000000).toFixed(2)}M
              </span>
            )}
          </div>
        )}
      </div>

      {/* Chart Canvas */}
      <div className="relative w-full h-[380px]">
        {isLoading && (
          <div className="absolute inset-0 bg-white/80 flex items-center justify-center z-10">
            <div className="flex items-center space-x-2 text-xs font-semibold text-[#1E5FBF]">
              <RefreshCw className="w-4 h-4 animate-spin" />
              <span>Updating Candle Buffer...</span>
            </div>
          </div>
        )}

        {!isLoading && (!data || data.length === 0) && (
          <div className="w-full h-full flex flex-col items-center justify-center text-xs text-slate-400">
            <p>No candlestick data available for this range.</p>
          </div>
        )}

        <div ref={chartContainerRef} className="w-full h-full" />
      </div>
    </div>
  );
}
