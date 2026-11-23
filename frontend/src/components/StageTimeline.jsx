import React from 'react';
import { 
  CheckCircle2, 
  CircleDashed, 
  XCircle, 
  MinusCircle, 
  AlertCircle,
  Clock,
  Layers
} from 'lucide-react';

const STAGE_METADATA = {
  URL_CHECK: {
    title: 'URL & Lexical Check',
    desc: 'Anti-SSRF validation, DNS pinning, and pre-browser preliminary risk scoring (M-01 & M-07)',
  },
  REDIRECT_INVESTIGATION: {
    title: 'Redirect Investigation',
    desc: 'Multi-hop redirect chain tracking, cross-domain hop analysis, and open-redirect parameter detection',
  },
  TLS_INSPECTION: {
    title: 'TLS/SSL Inspection',
    desc: 'Probing certificate validity, issuer DN, SANs, expiry, and cryptographic SHA-256 fingerprint',
  },
  PLAYWRIGHT_CAPTURE: {
    title: 'Playwright Capture',
    desc: 'Headless browser rendering, full viewport screenshot capture, and DOM credential input extraction',
  },
  VISUAL_ANALYSIS: {
    title: 'Visual Intelligence Suite',
    desc: 'Multi-model layout classification (M-02), Siamese logo matching (M-03), and object detection consensus (M-04/05/06)',
  },
  EVIDENCE_FUSION: {
    title: 'Evidence Risk Fusion',
    desc: 'Probabilistic fusion synthesizing all 7 model outputs, TLS context, and behavioral signals',
  },
};

const STAGE_ORDER = [
  'URL_CHECK',
  'REDIRECT_INVESTIGATION',
  'TLS_INSPECTION',
  'PLAYWRIGHT_CAPTURE',
  'VISUAL_ANALYSIS',
  'EVIDENCE_FUSION',
];

export default function StageTimeline({ stages = {}, currentStage }) {
  return (
    <div className="space-y-3">
      {STAGE_ORDER.map((stageName, index) => {
        const stage = stages[stageName] || {
          status: 'PENDING',
          message: 'Stage pending execution.',
          duration_ms: null,
        };
        const meta = STAGE_METADATA[stageName];
        const isCurrent = currentStage === stageName || stage.status === 'IN_PROGRESS';

        let statusIcon = <Clock className="w-4 h-4 text-slate-500" />;
        let statusBadge = <span className="text-xs text-slate-500 bg-slate-800/60 px-2 py-0.5 rounded border border-slate-700/50">Pending</span>;
        let cardBorder = 'border-slate-800 bg-surface/50';

        if (stage.status === 'IN_PROGRESS') {
          statusIcon = <CircleDashed className="w-4 h-4 text-cyan-400 animate-spin" />;
          statusBadge = <span className="text-xs text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-500/40 animate-pulse">Running</span>;
          cardBorder = 'border-cyan-500/40 bg-surface/90 shadow-lg shadow-cyan-950/30';
        } else if (stage.status === 'COMPLETED') {
          statusIcon = <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
          statusBadge = <span className="text-xs text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-500/40">Complete</span>;
          cardBorder = 'border-slate-800 bg-surface/70';
        } else if (stage.status === 'FAILED') {
          statusIcon = <XCircle className="w-4 h-4 text-red-400" />;
          statusBadge = <span className="text-xs text-red-400 bg-red-950/60 px-2 py-0.5 rounded border border-red-500/40">Failed</span>;
          cardBorder = 'border-red-900/50 bg-red-950/10';
        } else if (stage.status === 'SKIPPED') {
          statusIcon = <MinusCircle className="w-4 h-4 text-slate-500" />;
          statusBadge = <span className="text-xs text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700">Skipped</span>;
          cardBorder = 'border-slate-800/50 bg-surface/30 opacity-60';
        } else if (stage.status === 'UNAVAILABLE') {
          statusIcon = <AlertCircle className="w-4 h-4 text-amber-400" />;
          statusBadge = <span className="text-xs text-amber-400 bg-amber-950/60 px-2 py-0.5 rounded border border-amber-500/40">Unavailable</span>;
          cardBorder = 'border-amber-900/40 bg-surface/70';
        }

        return (
          <div 
            key={stageName} 
            className={`p-3.5 rounded-lg border transition-all ${cardBorder} flex items-start gap-3`}
          >
            <div className="mt-0.5 flex-shrink-0">
              {statusIcon}
            </div>

            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-slate-400">0{index + 1}.</span>
                  <h4 className="text-sm font-semibold text-slate-200 truncate">{meta.title}</h4>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                  {stage.duration_ms !== null && stage.duration_ms !== undefined && (
                    <span className="text-xs font-mono text-slate-400">
                      {stage.duration_ms > 1000 ? `${(stage.duration_ms / 1000).toFixed(2)}s` : `${stage.duration_ms}ms`}
                    </span>
                  )}
                  {statusBadge}
                </div>
              </div>

              <p className="text-xs text-slate-400 mt-1 line-clamp-1">{meta.desc}</p>

              {stage.message && (
                <div className="mt-2 text-xs font-mono bg-slate-900/70 border border-slate-800 rounded px-2.5 py-1.5 text-slate-300">
                  {stage.message}
                </div>
              )}

              {stage.error && (
                <div className="mt-2 text-xs font-mono bg-red-950/40 border border-red-900/60 rounded px-2.5 py-1.5 text-red-300">
                  Error: {stage.error}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
