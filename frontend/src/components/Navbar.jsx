import React, { useEffect, useState } from 'react';
import { Shield, Activity, Globe, Cpu, AlertCircle } from 'lucide-react';
import { getHealth } from '../services/api';

export default function Navbar({ activeTab, setActiveTab }) {
  const [health, setHealth] = useState({ status: 'CHECKING', device: '...' });

  useEffect(() => {
    let isMounted = true;
    async function checkBackend() {
      try {
        const data = await getHealth();
        if (isMounted) setHealth(data);
      } catch (e) {
        if (isMounted) setHealth({ status: 'OFFLINE', device: 'unreachable' });
      }
    }
    checkBackend();
    const interval = setInterval(checkBackend, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const getStatusPill = () => {
    if (health.status === 'OPERATIONAL') {
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-950/80 text-emerald-400 border border-emerald-500/30">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
          Operational ({health.device?.toUpperCase() || 'CPU'})
        </span>
      );
    }
    if (health.status === 'DEGRADED') {
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-950/80 text-amber-400 border border-amber-500/30">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
          Degraded Models
        </span>
      );
    }
    if (health.status === 'OFFLINE') {
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-red-950/80 text-red-400 border border-red-500/30">
          <span className="w-1.5 h-1.5 rounded-full bg-red-400"></span>
          Backend Offline
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-800 text-slate-400 border border-slate-700">
        <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-pulse"></span>
        Connecting...
      </span>
    );
  };

  return (
    <header className="border-b border-surface-border bg-surface/80 backdrop-blur sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div 
          onClick={() => setActiveTab('scan')} 
          className="flex items-center gap-3 cursor-pointer group"
        >
          <div className="w-9 h-9 rounded-lg bg-cyan-950 border border-cyan-500/40 flex items-center justify-center text-cyan-400 shadow-sm group-hover:border-cyan-400 transition-colors">
            <Shield className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-base tracking-tight text-white group-hover:text-cyan-300 transition-colors">PhishVision</span>
              <span className="text-xs px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">AI v1.0</span>
            </div>
            <p className="text-[10px] text-slate-400 hidden sm:block tracking-wide">Multimodal Forensic Defense</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1 sm:gap-2">
          <button
            onClick={() => setActiveTab('scan')}
            className={`px-3 py-1.5 rounded-md text-xs sm:text-sm font-medium transition-colors ${
              activeTab === 'scan' || activeTab === 'progress' || activeTab === 'results'
                ? 'bg-slate-800 text-cyan-400 border border-cyan-500/30'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
          >
            Scanner
          </button>

          <button
            onClick={() => setActiveTab('models')}
            className={`px-3 py-1.5 rounded-md text-xs sm:text-sm font-medium transition-colors ${
              activeTab === 'models'
                ? 'bg-slate-800 text-cyan-400 border border-cyan-500/30'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
          >
            Models & Architecture
          </button>
        </nav>

        {/* System Health */}
        <div className="flex items-center gap-3">
          {getStatusPill()}
        </div>
      </div>
    </header>
  );
}
