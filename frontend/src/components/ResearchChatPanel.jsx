import React, { useState, useRef, useEffect } from 'react';
import { MessageSquare, Send, X, ShieldAlert, Cpu, ExternalLink, RefreshCw, ChevronDown, Bot, User } from 'lucide-react';
import { sendChatMessage } from '../services/api';

const STARTER_PROMPTS = [
  "What's the latest news on Reliance?",
  "Is this model reliable for predictions?",
  "What's TCS's current volatility regime?",
  "Should I buy Infosys?"
];

export default function ResearchChatPanel({ currentSymbol }) {
  const [isOpen, setIsOpen] = useState(false);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      sender: 'assistant',
      text: "Hello! I am the rule-based Research Assistant for this quantitative platform. You can ask me about live quotes, news sentiment, volatility regimes, or our empirical findings (e.g., alpha decay & volatility clustering).",
      intent: "GREETING",
      symbol: null,
      sources: [],
      grounded: true,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (isOpen) {
      scrollToBottom();
      inputRef.current?.focus();
    }
  }, [messages, isOpen]);

  const handleSend = async (textToSend) => {
    const query = (textToSend || inputMessage).trim();
    if (!query || loading) return;

    const userMsg = {
      id: 'user-' + Date.now(),
      sender: 'user',
      text: query,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMsg]);
    setInputMessage('');
    setLoading(true);

    try {
      const response = await sendChatMessage(query, currentSymbol);
      const assistantMsg = {
        id: 'asst-' + Date.now(),
        sender: 'assistant',
        text: response.answer,
        intent: response.intent_detected,
        symbol: response.symbol_detected,
        sources: response.sources || [],
        grounded: response.grounded,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (err) {
      const errorMsg = {
        id: 'err-' + Date.now(),
        sender: 'assistant',
        text: "Error connecting to the local research assistant. Please verify the Python server is running.",
        intent: "ERROR",
        symbol: null,
        sources: [],
        grounded: false,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <>
      {/* Floating Toggle Button */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 z-40 flex items-center gap-2.5 bg-[#1E5FBF] hover:bg-[#184E9E] text-white px-4 py-3 rounded-full shadow-lg transition-all duration-200 hover:shadow-xl font-medium text-sm group"
          aria-label="Open Research Assistant"
        >
          <Bot className="w-5 h-5 text-white transition-transform group-hover:scale-110" />
          <span>Research Assistant</span>
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        </button>
      )}

      {/* Slide-in Chat Drawer / Panel */}
      {isOpen && (
        <div className="fixed bottom-6 right-6 z-50 w-[420px] max-w-[calc(100vw-2rem)] h-[620px] max-h-[calc(100vh-4rem)] bg-white rounded-xl shadow-2xl border border-slate-200 flex flex-col overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-200">
          
          {/* Header */}
          <div className="bg-[#1E5FBF] text-white px-4 py-3 flex items-center justify-between shadow-sm">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-blue-700 flex items-center justify-center">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-semibold tracking-tight">Research Assistant</h3>
                  <span className="text-[10px] bg-blue-800 text-blue-100 px-1.5 py-0.5 rounded font-mono">
                    RULE-BASED NLP
                  </span>
                </div>
                <p className="text-[11px] text-blue-100 font-normal">
                  Deterministic SQLite RAG • No External LLM
                </p>
              </div>
            </div>
            
            <button
              onClick={() => setIsOpen(false)}
              className="text-blue-200 hover:text-white hover:bg-blue-700/50 p-1.5 rounded-lg transition-colors"
              title="Close Panel"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Pinned Disclaimer Banner */}
          <div className="bg-amber-50 border-b border-amber-200/80 px-3.5 py-2 flex items-start gap-2 text-[11px] text-amber-900 leading-tight">
            <ShieldAlert className="w-3.5 h-3.5 text-amber-700 shrink-0 mt-0.5" />
            <span>
              <strong>Research Guardrail:</strong> This is a rule-based assistant built entirely for this project -- it retrieves and summarizes data from the project's own database using keyword matching, with no external AI service involved. It does not provide investment recommendations.
            </span>
          </div>

          {/* Messages Feed */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3.5 bg-slate-50/50">
            {messages.map((msg) => {
              const isUser = msg.sender === 'user';
              return (
                <div
                  key={msg.id}
                  className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}
                >
                  {/* Grounded warning badge if false on assistant message */}
                  {!isUser && !msg.grounded && msg.intent !== 'GREETING' && (
                    <span className="text-[10px] text-slate-500 italic mb-1 px-1">
                      No specific data found for this query
                    </span>
                  )}

                  {/* Bubble */}
                  <div
                    className={`max-w-[88%] rounded-xl px-3.5 py-2.5 text-xs leading-relaxed whitespace-pre-line shadow-xs ${
                      isUser
                        ? 'bg-[#1E5FBF] text-white rounded-br-none'
                        : 'bg-white border border-slate-200 text-slate-800 rounded-bl-none'
                    }`}
                  >
                    {msg.text}
                  </div>

                  {/* Assistant Message Metadata & Intent Badge */}
                  {!isUser && (
                    <div className="mt-1 flex flex-wrap items-center gap-1.5 px-1">
                      {msg.intent && (
                        <span className="text-[10px] font-mono text-slate-500 bg-slate-100 border border-slate-200 px-1.5 py-0.5 rounded">
                          Detected: {msg.intent} {msg.symbol ? `(${msg.symbol.replace('.NS', '')})` : ''}
                        </span>
                      )}
                      <span className="text-[10px] text-slate-400">{msg.time}</span>
                    </div>
                  )}

                  {/* Citations / Sources Chips */}
                  {!isUser && msg.sources && msg.sources.length > 0 && (
                    <div className="mt-2 w-full max-w-[88%] space-y-1">
                      <p className="text-[10px] font-medium text-slate-500 uppercase tracking-wider">
                        Cited Database Sources:
                      </p>
                      <div className="space-y-1">
                        {msg.sources.map((src, idx) => (
                          <div
                            key={idx}
                            className="bg-white border border-blue-100 rounded p-1.5 text-[10px] text-slate-700 shadow-2xs hover:border-blue-300 transition-colors"
                          >
                            <div className="font-medium text-[#1E5FBF] truncate">
                              {src.title}
                            </div>
                            <div className="flex items-center justify-between text-slate-400 mt-0.5">
                              <span>{src.source}</span>
                              <span>{src.published}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}

            {loading && (
              <div className="flex items-start gap-2">
                <div className="bg-white border border-slate-200 rounded-xl px-3.5 py-2 text-xs text-slate-500 flex items-center gap-2">
                  <RefreshCw className="w-3.5 h-3.5 animate-spin text-[#1E5FBF]" />
                  <span>Scanning local lexicon & SQLite database...</span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Starter Chips */}
          <div className="px-3 pt-2 pb-1.5 bg-white border-t border-slate-100">
            <p className="text-[10px] font-medium text-slate-400 uppercase tracking-wider mb-1.5">
              Suggested Research Questions:
            </p>
            <div className="flex flex-wrap gap-1">
              {STARTER_PROMPTS.map((prompt, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSend(prompt)}
                  disabled={loading}
                  className="text-[10px] bg-blue-50/80 hover:bg-blue-100 text-[#1E5FBF] border border-blue-200/60 px-2 py-1 rounded-md transition-colors text-left truncate max-w-full disabled:opacity-50"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>

          {/* Input Box */}
          <div className="p-3 bg-white border-t border-slate-200 flex items-center gap-2">
            <input
              ref={inputRef}
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about TCS sentiment, Reliance news, volatility..."
              disabled={loading}
              className="flex-1 text-xs border border-slate-300 rounded-lg px-3 py-2 text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-[#1E5FBF] focus:border-[#1E5FBF] transition-all disabled:bg-slate-50"
            />
            <button
              onClick={() => handleSend()}
              disabled={loading || !inputMessage.trim()}
              className="bg-[#1E5FBF] hover:bg-[#184E9E] disabled:bg-slate-200 text-white p-2 rounded-lg transition-colors shadow-xs disabled:cursor-not-allowed"
              aria-label="Send message"
            >
              <Send className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}
    </>
  );
}
