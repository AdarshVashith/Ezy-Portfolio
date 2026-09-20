import React from 'react';
import { Activity, Database, BookOpen, Info, BarChart2 } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab }) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: BarChart2 },
    { id: 'research', label: 'Research Findings', icon: BookOpen },
    { id: 'about', label: 'About', icon: Info },
  ];

  return (
    <header className="sticky top-0 z-40 bg-white border-b border-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Logo & Title */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-9 h-9 rounded bg-[#1E5FBF] text-white">
              <Activity className="w-5 h-5 stroke-[2.2]" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-lg font-bold tracking-tight text-[#1E5FBF]">
                  EZY-PORTFOLIO
                </span>
                <span className="text-xs font-semibold px-2 py-0.5 rounded bg-blue-50 text-[#1E5FBF] border border-blue-200 uppercase tracking-wider">
                  Quant Research
                </span>
              </div>
              <p className="text-[11px] text-slate-500 font-medium hidden sm:block">
                NSE India Systematic Market Predictability & Execution Engine
              </p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="flex space-x-1 sm:space-x-4">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`relative flex items-center space-x-2 px-3.5 py-2 text-sm font-medium transition-colors ${
                    isActive
                      ? 'text-[#1E5FBF] font-semibold'
                      : 'text-slate-600 hover:text-[#1E5FBF] hover:bg-slate-50'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-[#1E5FBF]' : 'text-slate-500'}`} />
                  <span>{item.label}</span>
                  {isActive && (
                    <span className="absolute bottom-[-17px] left-0 right-0 h-[2.5px] bg-[#1E5FBF]" />
                  )}
                </button>
              );
            })}
          </nav>

          {/* System Telemetry Pill */}
          <div className="hidden lg:flex items-center space-x-2 text-xs text-slate-600 bg-slate-50 px-3 py-1.5 rounded border border-slate-200">
            <Database className="w-3.5 h-3.5 text-[#1E5FBF]" />
            <span className="font-mono text-[11px] text-slate-700 font-medium">11 NSE ASSETS / 76K+ CANDLES</span>
          </div>

        </div>
      </div>
    </header>
  );
}
