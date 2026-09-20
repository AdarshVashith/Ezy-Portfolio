import React, { useState } from 'react';
import Navbar from './components/Navbar';
import DashboardPage from './pages/DashboardPage';
import ResearchPage from './pages/ResearchPage';
import AboutPage from './pages/AboutPage';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col">
      {/* Institutional Top Navbar */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Content Area */}
      <main className="flex-1">
        {activeTab === 'dashboard' && <DashboardPage />}
        {activeTab === 'research' && <ResearchPage />}
        {activeTab === 'about' && <AboutPage />}
      </main>

      {/* Institutional Footer */}
      <footer className="bg-white border-t border-slate-200 py-4 mt-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-2">
          <div className="flex items-center space-x-2">
            <span className="font-semibold text-[#1E5FBF]">EZY-PORTFOLIO QUANT DASHBOARD</span>
            <span>•</span>
            <span>NSE Research Pipeline</span>
          </div>
          <div className="font-mono text-[11px] text-slate-400">
            Strictly for academic & research evaluation. No financial advice.
          </div>
        </div>
      </footer>
    </div>
  );
}
