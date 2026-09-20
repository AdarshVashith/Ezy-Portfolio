import React, { useState, useRef, useEffect } from 'react';
import { Search, ChevronDown, Check } from 'lucide-react';

export default function SymbolSelector({ symbols, selectedSymbol, onSelectSymbol }) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const dropdownRef = useRef(null);

  const selectedMeta = symbols.find(s => s.symbol === selectedSymbol) || {
    symbol: selectedSymbol,
    name: selectedSymbol,
    sector: "NSE Equity"
  };

  const filteredSymbols = symbols.filter(s =>
    s.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (s.name && s.name.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="relative" ref={dropdownRef}>
      <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">
        Target Asset
      </label>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3.5 py-2.5 bg-white border border-slate-300 rounded text-left hover:border-[#1E5FBF] focus:outline-none focus:ring-1 focus:ring-[#1E5FBF] transition-colors"
      >
        <div className="truncate">
          <div className="flex items-center space-x-2">
            <span className="font-bold text-slate-900 text-sm font-mono">
              {selectedMeta.symbol.replace('.NS', '')}
            </span>
            <span className="text-[11px] px-1.5 py-0.2 rounded bg-slate-100 text-slate-600 font-medium">
              NSE
            </span>
          </div>
          <div className="text-xs text-slate-500 truncate mt-0.5 font-medium">
            {selectedMeta.name}
          </div>
        </div>
        <ChevronDown className="w-4 h-4 text-slate-400 ml-2 shrink-0" />
      </button>

      {isOpen && (
        <div className="absolute left-0 right-0 mt-1.5 bg-white border border-slate-200 rounded shadow-lg z-50 overflow-hidden max-h-80 flex flex-col">
          {/* Search box */}
          <div className="p-2 border-b border-slate-100 bg-slate-50">
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-3" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search symbol or name..."
                className="w-full pl-8 pr-3 py-1.5 text-xs bg-white border border-slate-200 rounded focus:outline-none focus:border-[#1E5FBF]"
                autoFocus
              />
            </div>
          </div>

          {/* Symbol List */}
          <div className="overflow-y-auto divide-y divide-slate-50 p-1">
            {filteredSymbols.length > 0 ? (
              filteredSymbols.map((item) => {
                const isSelected = item.symbol === selectedSymbol;
                return (
                  <button
                    key={item.symbol}
                    type="button"
                    onClick={() => {
                      onSelectSymbol(item.symbol);
                      setIsOpen(false);
                      setSearchTerm('');
                    }}
                    className={`w-full text-left px-3 py-2 rounded flex items-center justify-between text-xs transition-colors ${
                      isSelected
                        ? 'bg-blue-50 text-[#1E5FBF]'
                        : 'hover:bg-slate-50 text-slate-800'
                    }`}
                  >
                    <div>
                      <div className="font-bold font-mono text-xs">
                        {item.symbol.replace('.NS', '')}
                      </div>
                      <div className="text-[11px] text-slate-500 truncate max-w-[200px]">
                        {item.name}
                      </div>
                    </div>
                    {isSelected && <Check className="w-4 h-4 text-[#1E5FBF]" />}
                  </button>
                );
              })
            ) : (
              <div className="p-3 text-xs text-slate-500 text-center">
                No matching symbols
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
