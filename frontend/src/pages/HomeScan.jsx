import React, { useState } from 'react';
import { 
  ShieldAlert, 
  Search, 
  Zap, 
  Globe, 
  Eye, 
  Server, 
  Lock, 
  Layers,
  ArrowRight,
  AlertCircle
} from 'lucide-react';
import { submitScan } from '../services/api';

export default function HomeScan({ onScanSubmitted }) {
  const [url, setUrl] = useState('');
  const [fastMode, setFastMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    const trimmedUrl = url.trim();
    if (!trimmedUrl) {
      setError('Please provide a target HTTP or HTTPS URL to scan.');
      return;
    }

    if (!trimmedUrl.startsWith('http://') && !trimmedUrl.startsWith('https://')) {
      setError('Invalid URL scheme. The target URL must explicitly begin with http:// or https://');
      return;
    }

    setLoading(true);
    try {
      const data = await submitScan({
        url: trimmedUrl,
        fast_mode: fastMode,
      });
      if (data && data.scan_id) {
        onScanSubmitted(data.scan_id, trimmedUrl, fastMode);
      } else {
        throw new Error('Backend failed to return a scan identifier.');
      }
    } catch (err) {
      setError(err.message || 'Failed to submit scan request.');
    } finally {
      setLoading(false);
    }
  };

  const sampleTargets = [
    { label: 'Legitimate Portal', url: 'https://example.com' },
    { label: 'Telegram Preview', url: 'https://t.me' },
    { label: 'Synthetic 404 Target', url: 'http://cloudsupport-knowledge.org' },
  ];

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 sm:py-12">
      {/* Hero section */}
      <div className="text-center space-y-4 mb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-cyan-950/70 text-cyan-400 border border-cyan-500/30">
          <ShieldAlert className="w-3.5 h-3.5" />
          Zero-Day Phishing & Brand Impersonation Defense
        </div>

        <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-white tracking-tight">
          Forensic Website Intelligence & <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
            Multimodal Threat Analysis
          </span>
        </h1>

        <p className="text-sm sm:text-base text-slate-400 max-w-2xl mx-auto leading-relaxed">
          PhishVision AI evaluates suspicious websites across six sequential forensic stages—fusing lexical URL heuristics, cryptographic TLS inspection, headless browser rendering, and deep computer vision.
        </p>
      </div>

      {/* Main Scan Input Card */}
      <div className="bg-surface border border-surface-border rounded-xl p-6 shadow-2xl shadow-cyan-950/20 mb-12">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="target-url" className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
              Target URL to Scan
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-500">
                <Globe className="w-5 h-5 text-slate-400" />
              </div>
              <input
                id="target-url"
                type="text"
                placeholder="https://secure-login.portal.net/auth"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                disabled={loading}
                className="w-full pl-11 pr-32 py-3.5 bg-slate-900 border border-slate-700/80 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-colors font-mono"
              />
              <div className="absolute inset-y-1.5 right-1.5 flex items-center">
                <button
                  type="submit"
                  disabled={loading}
                  className="h-full px-5 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold rounded-md shadow flex items-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <>
                      <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Submitting...
                    </>
                  ) : (
                    <>
                      <span>Scan Target</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Options Row */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pt-2 border-t border-slate-800/80">
            <label className="flex items-center gap-2.5 cursor-pointer group">
              <input
                type="checkbox"
                checked={fastMode}
                onChange={(e) => setFastMode(e.target.checked)}
                className="rounded border-slate-700 bg-slate-900 text-cyan-500 focus:ring-cyan-500 h-4 w-4"
              />
              <span className="text-xs text-slate-300 font-medium group-hover:text-cyan-300 transition-colors flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-amber-400" />
                Fast Mode (Offline Lexical & Sequence ML only; skips browser rendering)
              </span>
            </label>

            {/* Quick pre-fill buttons */}
            <div className="flex items-center gap-1.5 text-xs text-slate-400">
              <span className="text-[11px] text-slate-500">Quick Test:</span>
              {sampleTargets.map((item, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => setUrl(item.url)}
                  className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] font-mono transition-colors border border-slate-700/60"
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          {/* Error Banner */}
          {error && (
            <div className="p-3.5 rounded-lg bg-red-950/60 border border-red-500/40 text-red-300 text-xs flex items-start gap-2.5 animate-fadeIn">
              <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="font-semibold text-red-200">Scan Ingress Error</p>
                <p className="mt-0.5 text-red-300 font-mono text-[11px]">{error}</p>
              </div>
            </div>
          )}
        </form>
      </div>

      {/* Feature Grid / Architectural Highlights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="p-5 rounded-xl border border-slate-800 bg-surface/50 space-y-2.5">
          <div className="w-8 h-8 rounded-lg bg-cyan-950 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <Lock className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-semibold text-slate-200">Anti-SSRF Egress Security</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Connection-level DNS resolution pinning and IP classification block private ranges (RFC 1918 / RFC 4193 / link-local / cloud metadata) before socket initialization.
          </p>
        </div>

        <div className="p-5 rounded-xl border border-slate-800 bg-surface/50 space-y-2.5">
          <div className="w-8 h-8 rounded-lg bg-cyan-950 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <Eye className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-semibold text-slate-200">Visual & Spatial Consensus</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Fuses M-02 ResNet-18 visual layout classification, M-03 Siamese brand similarity, and spatial consensus across YOLOv8n, Faster R-CNN, and YOLO-Seg.
          </p>
        </div>

        <div className="p-5 rounded-xl border border-slate-800 bg-surface/50 space-y-2.5">
          <div className="w-8 h-8 rounded-lg bg-cyan-950 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <Layers className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-semibold text-slate-200">Six-Stage Deterministic Pipeline</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Real-time execution telemetry tracks URL checks, redirect investigation, TLS inspection, browser rendering, visual models, and calibrated evidence fusion.
          </p>
        </div>
      </div>
    </div>
  );
}
