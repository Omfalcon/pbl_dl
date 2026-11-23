import React, { useState } from 'react';
import { 
  ShieldCheck, 
  ShieldAlert, 
  AlertTriangle, 
  Lock, 
  Unlock, 
  ExternalLink, 
  Layers, 
  FileText, 
  Image as ImageIcon,
  CheckCircle,
  XCircle,
  HelpCircle,
  RotateCcw,
  Network
} from 'lucide-react';
import RiskBadge from '../components/RiskBadge';
import { getScreenshotUrl } from '../services/api';

export default function ScanResults({ report, onNewScan }) {
  const [activeTab, setActiveTab] = useState('summary');

  if (!report) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-12 text-center text-slate-400">
        <p>No report data loaded.</p>
        <button onClick={onNewScan} className="mt-4 px-4 py-2 bg-cyan-600 rounded text-white text-xs">Start New Scan</button>
      </div>
    );
  }

  const risk = report.risk_assessment || {};
  const urlIntel = report.url_intelligence || {};
  const prelimRisk = report.preliminary_url_risk || null;
  const visIntel = report.visual_intelligence || {};
  const behavIntel = report.behavioral_intelligence || {};
  const tls = report.tls_evidence || null;
  const redirectIntel = report.redirect_intelligence || null;
  const modelAvail = report.model_availability || [];
  const mitigatingEvidence = report.mitigating_evidence || [];
  const unavailableEvidence = report.unavailable_evidence || [];

  const screenshotSrc = visIntel.screenshot_captured 
    ? getScreenshotUrl(report.scan_id) 
    : null;

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
      {/* Top Bar with Verdict & Risk Score */}
      <div className="bg-surface border border-surface-border rounded-xl p-6 shadow-xl flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <RiskBadge category={risk.category} size="lg" />
            <span className="text-xs font-mono text-slate-400 bg-slate-900 px-2 py-1 rounded border border-slate-800">
              Confidence: {((risk.confidence || 0) * 100).toFixed(1)}%
            </span>
          </div>

          <h2 className="text-xl sm:text-2xl font-bold text-white font-mono break-all">
            {report.target_url}
          </h2>

          <p className="text-xs text-slate-400 flex items-center gap-4">
            <span>Elapsed: {report.elapsed_seconds}s</span>
            <span>•</span>
            <span>Scan ID: {report.scan_id}</span>
          </p>
        </div>

        {/* Risk Score Gauge Pill */}
        <div className="flex items-center gap-4 bg-slate-900 border border-slate-800 p-4 rounded-xl flex-shrink-0">
          <div className="text-right">
            <div className="text-xs uppercase font-semibold text-slate-400">Fused Risk Score</div>
            <div className="text-3xl font-extrabold text-white font-mono">
              {(risk.risk_score || 0).toFixed(4)}
            </div>
            <div className="text-[10px] text-slate-500">Scale: 0.00 (Safe) - 1.00 (Malicious)</div>
          </div>
          <button
            onClick={onNewScan}
            className="p-3 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg border border-slate-700 transition-colors"
            title="Scan another website"
          >
            <RotateCcw className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Main Forensic Summary Explanation */}
      {risk.summary && (
        <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 text-slate-300 text-xs font-mono leading-relaxed">
          <span className="font-semibold text-cyan-400 uppercase tracking-wider block mb-1">Forensic Threat Summary</span>
          {risk.summary}
        </div>
      )}

      {/* Detailed Navigation Tabs */}
      <div className="flex border-b border-surface-border gap-2 text-xs font-medium">
        {['summary', 'visual', 'network_tls', 'models'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`pb-3 px-3 transition-colors capitalize ${
              activeTab === tab 
                ? 'border-b-2 border-cyan-400 text-cyan-400 font-bold' 
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {tab.replace('_', ' & ')}
          </button>
        ))}
      </div>

      {/* Tab: Forensic Summary */}
      {activeTab === 'summary' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Contributing & Mitigating Factors */}
          <div className="space-y-4">
            <div className="bg-surface border border-surface-border rounded-xl p-5 space-y-3">
              <h3 className="text-sm font-semibold text-slate-200 flex items-center justify-between">
                <span>Contributing Threat Factors</span>
                <span className="text-xs text-slate-400 font-mono font-normal">
                  {risk.contributing_factors?.length || 0} factors
                </span>
              </h3>

              {risk.contributing_factors && risk.contributing_factors.length > 0 ? (
                <div className="space-y-2">
                  {risk.contributing_factors.map((factor, idx) => (
                    <div key={idx} className="p-3 rounded-lg bg-slate-900 border border-slate-800 space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-mono font-bold text-red-400">{factor.vector}</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded font-mono bg-red-950 text-red-300 border border-red-900">
                          {factor.severity}
                        </span>
                      </div>
                      <p className="text-xs text-slate-300">{factor.description}</p>
                      {factor.evidence_ref && (
                        <p className="text-[10px] font-mono text-slate-500">Ref: {factor.evidence_ref}</p>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-slate-500">No active threat factors identified.</p>
              )}
            </div>

            {/* Mitigating Evidence */}
            {mitigatingEvidence.length > 0 && (
              <div className="bg-surface border border-emerald-950/40 rounded-xl p-5 space-y-2">
                <h3 className="text-sm font-semibold text-emerald-400 flex items-center gap-1.5">
                  <CheckCircle className="w-4 h-4" />
                  <span>Mitigating Evidence (Risk Reducers)</span>
                </h3>
                <ul className="space-y-1.5 text-xs text-slate-300 list-disc list-inside">
                  {mitigatingEvidence.map((item, idx) => (
                    <li key={idx} className="text-slate-300">{item}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* URL Intelligence & Pre-Browser Risk Comparison */}
          <div className="space-y-4">
            <div className="bg-surface border border-surface-border rounded-xl p-5 space-y-4">
              <h3 className="text-sm font-semibold text-slate-200">Pre-Browser vs Final Fusion Scores</h3>

              <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                  <div className="text-slate-400 text-[11px]">M-01 Lexical MLP</div>
                  <div className="text-lg font-bold text-white mt-1">
                    {urlIntel.mlp_lexical_score !== null && urlIntel.mlp_lexical_score !== undefined
                      ? Number(urlIntel.mlp_lexical_score).toFixed(4)
                      : 'N/A'}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">Threshold: 0.050</div>
                </div>

                <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                  <div className="text-slate-400 text-[11px]">M-07 URL Sequence LSTM</div>
                  <div className="text-lg font-bold text-white mt-1">
                    {urlIntel.lstm_sequence_score !== null && urlIntel.lstm_sequence_score !== undefined
                      ? Number(urlIntel.lstm_sequence_score).toFixed(4)
                      : 'N/A'}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">Threshold: 0.010</div>
                </div>

                <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 col-span-2">
                  <div className="text-slate-400 text-[11px]">Stage 1 Preliminary URL Risk (Pre-Browser)</div>
                  <div className="text-base font-bold text-cyan-400 mt-1">
                    {prelimRisk?.composite_url_risk !== null && prelimRisk?.composite_url_risk !== undefined
                      ? Number(prelimRisk.composite_url_risk).toFixed(4)
                      : 'N/A'}
                  </div>
                  <p className="text-[10px] text-slate-500 mt-1">
                    {prelimRisk?.note || 'Computed prior to dynamic rendering; separated from final fusion.'}
                  </p>
                </div>
              </div>

              {/* Domain Breakdown */}
              <div className="space-y-2 pt-2 border-t border-slate-800 text-xs">
                <div className="flex justify-between text-slate-400">
                  <span>Registrable Domain:</span>
                  <span className="font-mono text-slate-200">{urlIntel.registrable_domain || 'N/A'}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>IP Address Host:</span>
                  <span className="font-mono text-slate-200">{urlIntel.is_ip_address ? 'Yes (Anomalous)' : 'No'}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Punycode Present:</span>
                  <span className="font-mono text-slate-200">{urlIntel.has_punycode ? 'Yes' : 'No'}</span>
                </div>
              </div>
            </div>

            {/* DOM Form Findings */}
            <div className="bg-surface border border-surface-border rounded-xl p-5 space-y-3">
              <h3 className="text-sm font-semibold text-slate-200">DOM Form Credentials Telemetry</h3>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between text-slate-400">
                  <span>DOM Password Field:</span>
                  <span className={`font-mono font-bold ${behavIntel.has_dom_password_field ? 'text-red-400' : 'text-emerald-400'}`}>
                    {behavIntel.has_dom_password_field ? 'Detected' : 'Not Present'}
                  </span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Forms Extracted:</span>
                  <span className="font-mono text-slate-200">{behavIntel.forms?.length || 0}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab: Visual Intelligence & Screenshot */}
      {activeTab === 'visual' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Screenshot Display */}
          <div className="bg-surface border border-surface-border rounded-xl p-5 space-y-3">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center justify-between">
              <span>Captured Viewport (Playwright)</span>
              {screenshotSrc && (
                <a 
                  href={screenshotSrc} 
                  target="_blank" 
                  rel="noreferrer" 
                  className="text-xs text-cyan-400 hover:underline flex items-center gap-1"
                >
                  Full Viewport <ExternalLink className="w-3 h-3" />
                </a>
              )}
            </h3>

            {screenshotSrc ? (
              <div className="rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
                <img 
                  src={screenshotSrc} 
                  alt="Captured website viewport" 
                  className="w-full h-auto object-cover max-h-96"
                />
              </div>
            ) : (
              <div className="h-64 rounded-lg border border-dashed border-slate-800 bg-slate-900/50 flex flex-col items-center justify-center text-slate-500 text-xs">
                <ImageIcon className="w-8 h-8 mb-2 text-slate-600" />
                <span>No visual capture available (fast mode or headless error).</span>
              </div>
            )}
          </div>

          {/* Visual Model Findings */}
          <div className="space-y-4">
            {/* Brand Impersonation (Siamese CNN) */}
            <div className="bg-surface border border-surface-border rounded-xl p-5 space-y-3">
              <h3 className="text-sm font-semibold text-slate-200">M-03 Siamese Brand Impersonation</h3>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between text-slate-400">
                  <span>Matched Brand:</span>
                  <span className="font-mono text-slate-200">{visIntel.brand_similarity?.matched_brand || 'None'}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Cosine Similarity:</span>
                  <span className="font-mono text-slate-200">
                    {visIntel.brand_similarity?.similarity_score !== undefined
                      ? Number(visIntel.brand_similarity.similarity_score).toFixed(4)
                      : '0.0000'}
                  </span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Unauthorized Impersonation:</span>
                  <span className={`font-mono font-bold ${visIntel.brand_similarity?.is_unauthorized_brand_use ? 'text-red-400' : 'text-emerald-400'}`}>
                    {visIntel.brand_similarity?.is_unauthorized_brand_use ? 'CONFIRMED' : 'False'}
                  </span>
                </div>
              </div>
            </div>

            {/* Global Classifier & Localized UI Detections */}
            <div className="bg-surface border border-surface-border rounded-xl p-5 space-y-3">
              <h3 className="text-sm font-semibold text-slate-200">M-02 ResNet-18 & Spatial Consensus Detections</h3>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between text-slate-400">
                  <span>M-02 Global Phishing Prob:</span>
                  <span className="font-mono text-slate-200">
                    {visIntel.resnet_phishing_probability !== null && visIntel.resnet_phishing_probability !== undefined
                      ? Number(visIntel.resnet_phishing_probability).toFixed(4)
                      : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Fused UI Elements Detected:</span>
                  <span className="font-mono text-slate-200">{visIntel.detections?.length || 0}</span>
                </div>
              </div>

              {/* Detections Pill List */}
              {visIntel.detections && visIntel.detections.length > 0 && (
                <div className="mt-3 space-y-1.5 max-h-48 overflow-y-auto pr-1">
                  {visIntel.detections.map((det, idx) => (
                    <div key={idx} className="p-2 rounded bg-slate-900 border border-slate-800 text-[11px] font-mono flex items-center justify-between">
                      <span className="text-cyan-400">{det.class_name}</span>
                      <span className="text-slate-400">Conf: {(det.confidence * 100).toFixed(1)}%</span>
                      <span className="text-slate-500 text-[10px]">[{det.consensus_detectors?.join(', ')}]</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Tab: Network & TLS Telemetry */}
      {activeTab === 'network_tls' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Cryptographic TLS Inspector */}
          <div className="bg-surface border border-surface-border rounded-xl p-5 space-y-3">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Lock className="w-4 h-4 text-cyan-400" />
              <span>TLS / SSL Certificate Findings</span>
            </h3>

            {tls && tls.tls_available ? (
              <div className="space-y-2.5 text-xs">
                <div className="flex justify-between text-slate-400">
                  <span>Verification Status:</span>
                  <span className={`font-mono font-bold ${tls.certificate_valid ? 'text-emerald-400' : 'text-red-400'}`}>
                    {tls.certificate_valid ? 'VALID CERTIFICATE' : 'VERIFICATION FAILED'}
                  </span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Common Name (CN):</span>
                  <span className="font-mono text-slate-200 truncate max-w-xs">{tls.common_name || 'N/A'}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Issuer DN:</span>
                  <span className="font-mono text-slate-200 truncate max-w-xs">{tls.issuer || 'N/A'}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Expired Status:</span>
                  <span className={`font-mono ${tls.is_expired ? 'text-red-400 font-bold' : 'text-emerald-400'}`}>
                    {tls.is_expired ? 'EXPIRED' : `Valid (${tls.days_until_expiry} days remaining)`}
                  </span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Protocol & Cipher:</span>
                  <span className="font-mono text-slate-200">{tls.protocol_version || 'N/A'} ({tls.cipher_suite || 'N/A'})</span>
                </div>
                {tls.sha256_fingerprint && (
                  <div className="pt-2 border-t border-slate-800 text-[10px] font-mono text-slate-500 break-all">
                    SHA-256: {tls.sha256_fingerprint}
                  </div>
                )}
              </div>
            ) : (
              <p className="text-xs text-slate-500">
                {tls?.error || 'TLS inspection not available (HTTP target or probe skipped).'}
              </p>
            )}
          </div>

          {/* Redirect Chain & Behavioral Telemetry */}
          <div className="bg-surface border border-surface-border rounded-xl p-5 space-y-3">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Network className="w-4 h-4 text-cyan-400" />
              <span>Redirect Intelligence</span>
            </h3>

            {redirectIntel ? (
              <div className="space-y-3 text-xs">
                <div className="flex justify-between text-slate-400">
                  <span>Total Hops:</span>
                  <span className="font-mono text-slate-200">{redirectIntel.hop_count}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Cross-Domain Hops:</span>
                  <span className={`font-mono font-bold ${redirectIntel.has_cross_domain_redirects ? 'text-amber-400' : 'text-slate-200'}`}>
                    {redirectIntel.cross_domain_hop_count}
                  </span>
                </div>

                {/* Redirect Steps */}
                {behavIntel.redirect_chain && behavIntel.redirect_chain.length > 0 && (
                  <div className="space-y-2 mt-2 pt-2 border-t border-slate-800">
                    <span className="text-[11px] font-semibold text-slate-400 uppercase">Redirect Chain:</span>
                    {behavIntel.redirect_chain.map((hop, i) => (
                      <div key={i} className="p-2 rounded bg-slate-900 border border-slate-800 font-mono text-[11px] flex items-center gap-2">
                        <span className="text-cyan-400">[{hop.step}]</span>
                        <span className="text-slate-400">{hop.status_code}</span>
                        <span className="text-slate-200 truncate">{hop.url}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <p className="text-xs text-slate-500">No redirect telemetry available.</p>
            )}
          </div>
        </div>
      )}

      {/* Tab: Model Availability & Disclosures */}
      {activeTab === 'models' && (
        <div className="space-y-6">
          <div className="bg-surface border border-surface-border rounded-xl p-5 space-y-4">
            <h3 className="text-sm font-semibold text-slate-200">Seven-Model Pipeline Availability</h3>
            <p className="text-xs text-slate-400">
              PhishVision AI strictly requires that missing or unserviceable models are reported honestly as UNAVAILABLE or MISSING, rather than assigning fabricated placeholder scores.
            </p>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="border-b border-slate-800 text-slate-400 text-[11px]">
                  <tr>
                    <th className="py-2 px-3">Model ID</th>
                    <th className="py-2 px-3">Architecture Role</th>
                    <th className="py-2 px-3">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {modelAvail.map((m, idx) => (
                    <tr key={idx} className="hover:bg-slate-900/40">
                      <td className="py-2.5 px-3 font-bold text-cyan-400">{m.model_id}</td>
                      <td className="py-2.5 px-3">{m.name}</td>
                      <td className="py-2.5 px-3">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          m.status === 'LOADED' 
                            ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' 
                            : 'bg-amber-950 text-amber-400 border border-amber-800'
                        }`}>
                          {m.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Research & Synthetic Dataset Disclosures */}
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-400 space-y-2">
            <span className="font-semibold text-slate-300 uppercase tracking-wider block">Mandatory Academic & Research Disclosure</span>
            <p className="leading-relaxed">
              {report.disclaimer || (
                "Notice: M-04, M-05, and M-06 were trained on synthetic UI elements. High synthetic mAP does not establish real-world zero-day detection accuracy. M-07 URL Sequence Bi-LSTM operates in fallback mode as multi-hop redirect ground-truth was unavailable in historical feeds."
              )}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
