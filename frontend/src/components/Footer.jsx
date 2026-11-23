import React from 'react';
import { Shield, Terminal, BookOpen, Lock } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="border-t border-surface-border bg-surface/50 mt-auto py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-slate-400">
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-cyan-400" />
            <span className="font-semibold text-slate-300">PhishVision AI</span>
            <span>— Seven-Model Forensic Phishing Defense Platform</span>
          </div>

          <div className="flex items-center gap-6">
            <span className="inline-flex items-center gap-1 text-slate-400">
              <Lock className="w-3.5 h-3.5 text-emerald-400" />
              Anti-SSRF & DNS Pinning Active
            </span>
            <span className="inline-flex items-center gap-1 text-slate-400">
              <Terminal className="w-3.5 h-3.5 text-cyan-400" />
              Phase 9 Six-Stage Pipeline
            </span>
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-slate-800/60 flex flex-col sm:flex-row items-center justify-between text-[11px] text-slate-500 gap-2">
          <p>© 2026 PhishVision AI Research. Preserving verified ground-truth and calibrated decision boundaries.</p>
          <p className="font-mono">Backend: FastAPI 0.115 / PyTorch 2.6 CUDA / Playwright</p>
        </div>
      </div>
    </footer>
  );
}
