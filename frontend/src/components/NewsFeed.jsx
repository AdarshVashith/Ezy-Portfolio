import React, { useState } from 'react';
import { Newspaper, RotateCw, Clock, Filter, ExternalLink } from 'lucide-react';

export default function NewsFeed({ news, isLoading, onRefresh, lastUpdated }) {
  const [filter, setFilter] = useState('ALL');

  const getSentimentPill = (label, score) => {
    const formattedScore = score !== undefined ? ` (${score > 0 ? '+' : ''}${Number(score).toFixed(2)})` : '';
    
    if (label === 'Bullish' || score > 0.25) {
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
          Bullish{formattedScore}
        </span>
      );
    }
    if (label === 'Bearish' || score < -0.25) {
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold bg-rose-50 text-rose-700 border border-rose-200">
          Bearish{formattedScore}
        </span>
      );
    }
    return (
      <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold bg-blue-50 text-[#1E5FBF] border border-blue-200">
        Neutral{formattedScore}
      </span>
    );
  };

  const filteredNews = (news || []).filter(item => {
    if (filter === 'ALL') return true;
    if (filter === 'BULLISH') return item.label === 'Bullish' || item.sentiment > 0.25;
    if (filter === 'BEARISH') return item.label === 'Bearish' || item.sentiment < -0.25;
    if (filter === 'NEUTRAL') return item.label === 'Neutral' || (item.sentiment >= -0.25 && item.sentiment <= 0.25);
    return true;
  });

  return (
    <div className="bg-white rounded border border-slate-200 flex flex-col h-full">
      {/* Header */}
      <div className="p-3.5 border-b border-slate-100 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Newspaper className="w-4 h-4 text-[#1E5FBF]" />
          <span className="text-xs font-bold uppercase tracking-wider text-slate-900">
            Financial News Sentiment
          </span>
          <span className="flex h-2 w-2 relative">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
        </div>

        <button
          type="button"
          onClick={onRefresh}
          disabled={isLoading}
          title="Refresh Live News Feed"
          className="p-1 rounded hover:bg-slate-100 text-slate-500 hover:text-[#1E5FBF] transition-colors"
        >
          <RotateCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-[#1E5FBF]' : ''}`} />
        </button>
      </div>

      {/* Filter Tabs */}
      <div className="px-3 py-2 bg-slate-50 border-b border-slate-100 flex items-center justify-between text-[11px]">
        <span className="text-slate-400 font-mono uppercase text-[10px]">Filter:</span>
        <div className="flex space-x-1">
          {['ALL', 'BULLISH', 'BEARISH', 'NEUTRAL'].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-2 py-0.5 rounded text-[10px] font-semibold transition-colors ${
                filter === f
                  ? 'bg-[#1E5FBF] text-white shadow-xs'
                  : 'bg-white text-slate-600 hover:text-slate-900 border border-slate-200'
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Feed List */}
      <div className="flex-1 overflow-y-auto divide-y divide-slate-100 p-2 max-h-[580px]">
        {isLoading && (!news || news.length === 0) ? (
          <div className="p-8 text-center text-xs text-slate-400">
            Loading real-time news headlines...
          </div>
        ) : filteredNews.length === 0 ? (
          <div className="p-8 text-center text-xs text-slate-400">
            No headlines matching '{filter}' filter.
          </div>
        ) : (
          filteredNews.map((item, idx) => (
            <div
              key={idx}
              className="p-3 hover:bg-slate-50 transition-colors rounded group"
            >
              <div className="flex items-center justify-between mb-1.5 gap-2">
                <span className="text-[10px] font-bold text-slate-700 uppercase tracking-wider font-mono truncate max-w-[160px]">
                  {item.source}
                </span>
                {getSentimentPill(item.label, item.sentiment)}
              </div>

              <h4 className="text-xs font-medium text-slate-900 leading-snug line-clamp-2 group-hover:text-[#1E5FBF] transition-colors">
                {item.title}
              </h4>

              <div className="mt-2 flex items-center justify-between text-[10px] text-slate-400 font-mono">
                <div className="flex items-center">
                  <Clock className="w-3 h-3 mr-1" />
                  <span className="truncate max-w-[200px]">{item.published}</span>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Footer Info */}
      <div className="p-2.5 bg-slate-50 border-t border-slate-100 text-[10px] text-slate-500 flex justify-between items-center">
        <span>{filteredNews.length} Headlines Loaded</span>
        <span className="font-mono text-[#1E5FBF] font-semibold">
          {lastUpdated ? `Updated ${lastUpdated}` : 'Auto-polling (30s)'}
        </span>
      </div>
    </div>
  );
}
