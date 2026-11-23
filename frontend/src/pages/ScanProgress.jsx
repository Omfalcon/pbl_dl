import React, { useEffect, useState } from 'react';
import { 
  CircleDashed, 
  AlertTriangle, 
  CheckCircle2, 
  XCircle, 
  ShieldAlert, 
  ExternalLink,
  RotateCw
} from 'lucide-react';
import { getScanStatus, getScanReport } from '../services/api';
import StageTimeline from '../components/StageTimeline';
import RiskBadge from '../components/RiskBadge';

export default function ScanProgress({ scanId, targetUrl, fastMode, onScanCompleted, onBackToScan }) {
  const [scanState, setScanState] = useState(null);
  const [error, setError] = useState(null);
  const [pollingActive, setPollingActive] = useState(true);

  useEffect(() => {
    let isCancelled = false;
    let timerId = null;

    async function poll() {
      try {
        const data = await getScanStatus(scanId);
        if (isCancelled) return;

        setScanState(data);

        // Terminal states
        if (data.status === 'COMPLETED') {
          setPollingActive(false);
          // Fetch final report and notify parent
          try {
            const report = await getScanReport(scanId);
            if (!isCancelled) {
              onScanCompleted(report);
            }
          } catch (reportErr) {
            setError(`Scan marked complete, but failed to retrieve forensic report: ${reportErr.message}`);
          }
          return;
        }

        if (data.status === 'FAILED') {
          setPollingActive(false);
          setError(data.error || 'The scan failed during pipeline execution.');
          return;
        }

        if (data.status === 'BLOCKED_SSRF') {
          setPollingActive(false);
          setError('Scan blocked by Anti-SSRF protection: Target resolved to private or reserved network address.');
          return;
        }

        // Keep polling every 1.5 seconds while IN_PROGRESS
        timerId = setTimeout(poll, 1500);
      } catch (err) {
        if (!isCancelled) {
          setError(err.message || 'Error communicating with scan status endpoint.');
          timerId = setTimeout(poll, 3000); // retry with backoff
        }
      }
    }

    poll();

    return () => {
      isCancelled = true;
      if (timerId) clearTimeout(timerId);
    };
  }, [scanId, onScanCompleted]);

  const progressPercent = scanState?.progress_percent ?? 0.0;
  const currentStage = scanState?.current_stage ?? 'URL_CHECK';
  const stages = scanState?.stages ?? {};

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      {/* Header card */}
      <div className="bg-surface border border-surface-border rounded-xl p-5 mb-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-mono text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-500/30">
                Scan ID: {scanId.slice(0, 8)}...{scanId.slice(-4)}
              </span>
              {fastMode && (
                <span className="text-xs font-medium text-amber-400 bg-amber-950/60 px-2 py-0.5 rounded border border-amber-500/30">
                  Fast Mode Active
                </span>
              )}
            </div>
            <h2 className="text-base font-bold text-white font-mono break-all flex items-center gap-2">
              <span className="truncate max-w-md sm:max-w-xl">{targetUrl}</span>
            </h2>
          </div>

          <div className="flex items-center gap-3">
            {scanState?.status === 'IN_PROGRESS' && (
              <div className="flex items-center gap-2 text-xs font-medium text-cyan-400">
                <CircleDashed className="w-4 h-4 animate-spin" />
                <span>Scanning Website...</span>
              </div>
            )}
            {scanState?.status && scanState.status !== 'IN_PROGRESS' && (
              <RiskBadge category={scanState.status} />
            )}
          </div>
        </div>

        {/* Real Progress Bar */}
        <div className="mt-5 space-y-1.5">
          <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
            <span>Stage: {currentStage}</span>
            <span>{progressPercent.toFixed(0)}% Complete</span>
          </div>
          <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
            <div 
              className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-500 rounded-full"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>
      </div>

      {/* Error banner if failed */}
      {error && (
        <div className="p-4 rounded-xl bg-red-950/60 border border-red-500/50 mb-6 flex items-start gap-3">
          <XCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1 text-xs">
            <h4 className="font-bold text-red-200">Execution Error</h4>
            <p className="text-red-300 font-mono mt-1">{error}</p>
            <button
              onClick={onBackToScan}
              className="mt-3 px-3 py-1 bg-red-900/60 hover:bg-red-800 text-red-200 font-medium rounded border border-red-700/60 transition-colors"
            >
              Back to Scanner
            </button>
          </div>
        </div>
      )}

      {/* Six-Stage Live Progression Timeline */}
      <div className="bg-surface border border-surface-border rounded-xl p-5 shadow-xl">
        <h3 className="text-sm font-semibold text-slate-200 mb-4 flex items-center justify-between">
          <span>Live Six-Stage Pipeline Telemetry</span>
          <span className="text-xs font-normal text-slate-400">Deterministic Backend Tracking</span>
        </h3>

        <StageTimeline stages={stages} currentStage={currentStage} />
      </div>
    </div>
  );
}
