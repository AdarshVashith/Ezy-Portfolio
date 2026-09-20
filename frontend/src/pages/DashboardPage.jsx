import React, { useState, useEffect, useCallback } from 'react';
import KPICard from '../components/KPICard';
import SymbolSelector from '../components/SymbolSelector';
import DateRangeSelector from '../components/DateRangeSelector';
import CandlestickChart from '../components/CandlestickChart';
import IndicatorChart from '../components/IndicatorChart';
import NewsFeed from '../components/NewsFeed';
import { fetchSymbols, fetchPrices, fetchNews, fetchPrediction } from '../services/api';
import { DollarSign, Percent, BarChart3, Cpu } from 'lucide-react';

export default function DashboardPage() {
  const [symbols, setSymbols] = useState([]);
  const [selectedSymbol, setSelectedSymbol] = useState('RELIANCE.NS');
  const [rangeDays, setRangeDays] = useState(180);
  const [prices, setPrices] = useState([]);
  const [news, setNews] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [isLoadingPrices, setIsLoadingPrices] = useState(true);
  const [isLoadingNews, setIsLoadingNews] = useState(true);
  const [lastNewsUpdated, setLastNewsUpdated] = useState('');

  useEffect(() => {
    async function loadSymbols() {
      const symList = await fetchSymbols();
      setSymbols(symList);
    }
    loadSymbols();
  }, []);

  useEffect(() => {
    async function loadPricesAndPred() {
      setIsLoadingPrices(true);
      const [pData, predData] = await Promise.all([
        fetchPrices(selectedSymbol, rangeDays),
        fetchPrediction(selectedSymbol)
      ]);
      setPrices(pData);
      setPrediction(predData);
      setIsLoadingPrices(false);
    }
    loadPricesAndPred();
  }, [selectedSymbol, rangeDays]);

  const loadNews = useCallback(async () => {
    setIsLoadingNews(true);
    const nData = await fetchNews(30);
    setNews(nData);
    setLastNewsUpdated(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
    setIsLoadingNews(false);
  }, []);

  // Initial load and 30-second live polling interval
  useEffect(() => {
    loadNews();
    const interval = setInterval(() => {
      loadNews();
    }, 30000);
    return () => clearInterval(interval);
  }, [loadNews]);

  // Compute KPI values from price data
  const latestCandle = prices.length > 0 ? prices[prices.length - 1] : null;
  const prevCandle = prices.length > 1 ? prices[prices.length - 2] : null;

  let latestCloseStr = "—";
  let changeStr = "—";
  let isPositive = true;
  let volumeStr = "—";

  if (latestCandle) {
    const close = Number(latestCandle.close);
    latestCloseStr = `₹${close.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    
    if (prevCandle) {
      const prevClose = Number(prevCandle.close);
      const diff = close - prevClose;
      const pct = (diff / prevClose) * 100;
      isPositive = diff >= 0;
      changeStr = `${isPositive ? '+' : ''}${pct.toFixed(2)}% (${isPositive ? '+' : ''}₹${diff.toFixed(2)})`;
    }

    const vol = Number(latestCandle.volume);
    volumeStr = vol >= 1000000 ? `${(vol / 1000000).toFixed(2)}M` : `${(vol / 1000).toFixed(1)}K`;
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      
      {/* Top Filter Bar */}
      <div className="bg-white rounded border border-slate-200 p-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
          <div className="md:col-span-2">
            <SymbolSelector
              symbols={symbols}
              selectedSymbol={selectedSymbol}
              onSelectSymbol={setSelectedSymbol}
            />
          </div>
          <div>
            <DateRangeSelector
              selectedRange={rangeDays}
              onSelectRange={setRangeDays}
            />
          </div>
        </div>
      </div>

      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard
          title="Latest Close"
          value={latestCloseStr}
          subtext={latestCandle ? `Date: ${latestCandle.date}` : ""}
          icon={DollarSign}
        />
        <KPICard
          title="Day's Change"
          value={changeStr}
          isPositive={isPositive}
          subtext={prevCandle ? `vs Prev Close: ₹${Number(prevCandle.close).toFixed(2)}` : ""}
          icon={Percent}
        />
        <KPICard
          title="Day's Volume"
          value={volumeStr}
          subtext="NSE Cash Market Volume"
          icon={BarChart3}
        />
        <KPICard
          title="Model Signal"
          value={prediction ? prediction.signal : "Analyzing..."}
          isSignal={true}
          badgeText={prediction ? `${prediction.confidence}% Confidence (p=${prediction.p_value})` : ""}
          subtext={prediction ? prediction.model_name : "Quant Engine"}
          icon={Cpu}
        />
      </div>

      {/* Main Grid: Chart & Indicators (Left/Center) + News Feed (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left 2 Columns: Candlestick + Technical Indicators */}
        <div className="lg:col-span-2 space-y-4">
          <CandlestickChart
            data={prices}
            symbol={selectedSymbol}
            isLoading={isLoadingPrices}
          />
          <IndicatorChart data={prices} />
        </div>

        {/* Right Column: Financial News Sentiment Feed */}
        <div className="lg:col-span-1">
          <NewsFeed
            news={news}
            isLoading={isLoadingNews}
            onRefresh={loadNews}
            lastUpdated={lastNewsUpdated}
          />
        </div>

      </div>

    </div>
  );
}
