import React from 'react';
import { Newspaper, ExternalLink, Clock } from 'lucide-react';

export default function NewsFeed({ news, isLoading }) {
  const getSentimentPill = (label, score) => {
    const formattedScore = score !== undefined ? ` (${score > 0 ? '+' : ''}${Number(score).toFixed(2)})` : '';
    
    if (label === 'Bullish' || score > 0.25) {
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
          Bullish{formattedScore}
        </span>
      );
    }
    if (label === 'Bearish' || score < -0.25) {
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold bg-rose-50 text-rose-700 border border-rose-200">
          Bearish{formattedScore}
        </span>
      );
    }
    return (
      <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold bg-blue-50 text-[#1E5FBF] border border-blue-200">
        Neutral{formattedScore}
      </span>
    );
  };

  return (
    <div className="bg-white rounded border border-slate-200 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-slate-100 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Newspaper className="w-4 h-4 text-[#1E5FBF]" />
          <span className="text-xs font-bold uppercase tracking-wider text-slate-900">
            Financial News Sentiment Feed
          </span>
        </div>
        <span className="text-[11px] font-mono text-slate-500">
          RSS / NLP Lexicon
        </span>
      </div>

      {/* Feed List */}
      <div className="flex-1 overflow-y-auto divide-y divide-slate-100 p-2 max-h-[640px]">
        {isLoading ? (
          <div className="p-8 text-center text-xs text-slate-400">
            Loading real-time news headlines...
          </div>
        ) : !news || news.length === 0 ? (
          <div className="p-8 text-center text-xs text-slate-400">
            No news sentiment items available.
          </div>
        ) : (
          news.map((item, idx) => (
            <div
              key={idx}
              className="p-3 hover:bg-slate-50 transition-colors rounded group"
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[11px] font-bold text-slate-700 uppercase tracking-wider font-mono">
                  {item.source}
                </span>
                {getSentimentPill(item.label, item.sentiment)}
              </div>

              <h4 className="text-xs font-medium text-slate-900 leading-snug line-clamp-2 group-hover:text-[#1E5FBF] transition-colors">
                {item.title}
              </h4>

              <div className="mt-2 flex items-center text-[11px] text-slate-400 font-mono">
                <Clock className="w-3 h-3 mr-1" />
                <span>{item.published}</span>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Footer Info */}
      <div className="p-3 bg-slate-50 border-t border-slate-100 text-[11px] text-slate-500 flex justify-between items-center">
        <span>ET • Mint • Moneycontrol • NDTV</span>
        <span className="font-mono text-[#1E5FBF] font-semibold">300+ Term FinLexicon</span>
      </div>
    </div>
  );
}
