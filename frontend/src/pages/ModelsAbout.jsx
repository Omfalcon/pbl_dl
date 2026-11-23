import React from 'react';
import { Shield, Brain, Layers, Cpu, Eye, FileCode, CheckCircle, AlertTriangle } from 'lucide-react';

const MODELS_DATA = [
  {
    id: 'M-01',
    name: 'URL Lexical MLP',
    type: 'Multi-Layer Perceptron (from Scratch / PyTorch)',
    input: '40 extracted lexical & information-theoretic features (lengths, counts, Shannon entropy, brand tokens)',
    threshold: 'Calibrated threshold τ = 0.050 (Cost-Sensitive BCE with Class-Weighting Strategy B)',
    performance: 'Holdout Test: ROC-AUC = 0.9773, PR-AUC = 0.9995, Macro F1 = 0.8239, Accuracy = 98.66%',
    role: 'Fast-path pre-browser scoring; evaluates URL structure independently before rendering.',
  },
  {
    id: 'M-02',
    name: 'Global Screenshot Classifier',
    type: 'Deep Convolutional Network (ResNet-18 Backbone)',
    input: 'Full rendered viewport screenshot (aspect-ratio preserved, padded & normalized)',
    threshold: 'Decision threshold τ = 0.500',
    performance: 'Evaluated across 262 validation screenshots and multi-region viewport cropping diagnostics',
    role: 'Evaluates global webpage visual aesthetics, layout coherence, and visual template profiles.',
  },
  {
    id: 'M-03',
    name: 'Siamese Brand Matcher',
    type: 'Siamese ResNet-18 with Triplet / Contrastive Loss',
    input: 'Query logo crop vs 18-brand Curated Reference Gallery (256-d L2 normalized embeddings)',
    threshold: 'Calibrated Cosine Similarity threshold τ = 0.820',
    performance: 'Independent Test: Accuracy = 85.00%, Precision = 1.0000 (0 false positives), ROC-AUC = 0.8750',
    role: 'Detects unauthorized brand impersonation by matching cropped visual logos against protected brand assets.',
  },
  {
    id: 'M-04',
    name: 'YOLOv8n UI Component Detector',
    type: 'Single-Stage Anchor-Free Object Detector (Ultralytics YOLOv8n)',
    input: 'Rendered webpage screenshot (640x640 resolution)',
    threshold: 'IoU Threshold = 0.45, Confidence = 0.25',
    performance: 'Untouched Test: Box mAP50 = 0.9950, mAP50-95 = 0.9699, Precision = 0.9973, Latency = 7.2 ms',
    role: 'Localizes 6 standardized UI classes: brand_logo, login_form, username_field, password_field, submit_button, payment_element.',
  },
  {
    id: 'M-05',
    name: 'Faster R-CNN Comparative Detector',
    type: 'Two-Stage Detector (ResNet-50-FPN Backbone)',
    input: 'Webpage screenshot processed through Region Proposal Network (RPN)',
    threshold: 'Score threshold = 0.05, NMS IoU = 0.50',
    performance: 'Comparative evaluation baseline across identical 7-class synthetic dataset splits',
    role: 'Provides dense comparative detection proposals for spatial consensus with M-04 and M-06.',
  },
  {
    id: 'M-06',
    name: 'YOLOv8-Seg Instance Segmenter',
    type: 'Instance Segmentation Network (YOLOv8n-seg with ProtoNet)',
    input: 'Webpage screenshot (640x640 resolution)',
    threshold: 'Mask confidence = 0.25, Box threshold = 0.25',
    performance: 'Test Split: Box mAP50 = 0.9950, Mask mAP50 = 0.9950',
    role: 'Extracts exact pixel-level geometric polygon masks for functional UI regions and input boundaries.',
  },
  {
    id: 'M-07',
    name: 'URL Sequence Bi-LSTM',
    type: 'Bidirectional LSTM with AttentionPooling',
    input: 'Sequential character tokens & multi-hop redirect hostnames (vocab size = 256, max len = 200)',
    threshold: 'Calibrated threshold τ = 0.010',
    performance: 'Holdout Test: Accuracy = 98.69%, ROC-AUC = 0.9666, PR-AUC = 0.9991, Phishing F1 = 0.9933',
    role: 'Analyzes sequential character and hostname token patterns to detect evasive obfuscation.',
  },
];

export default function ModelsAbout() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-8">
      {/* Title */}
      <div className="space-y-2">
        <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded text-xs font-mono bg-cyan-950 text-cyan-400 border border-cyan-800">
          Architecture Reference
        </div>
        <h1 className="text-3xl font-extrabold text-white">
          Seven-Model Multimodal Defense Suite
        </h1>
        <p className="text-sm text-slate-400 leading-relaxed max-w-3xl">
          PhishVision AI avoids arbitrary black-box averaging by decoupling lexical, visual, spatial, and behavioral intelligence. Each model specializes in a distinct attack surface.
        </p>
      </div>

      {/* Model Cards */}
      <div className="space-y-5">
        {MODELS_DATA.map((model) => (
          <div 
            key={model.id}
            className="p-5 rounded-xl border border-surface-border bg-surface/70 space-y-3 hover:border-slate-700 transition-colors"
          >
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2.5">
                <span className="px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 font-mono font-bold text-xs border border-cyan-500/20">
                  {model.id}
                </span>
                <h3 className="text-base font-bold text-white">{model.name}</h3>
              </div>
              <span className="text-xs font-mono text-slate-400">{model.type}</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
              <div className="space-y-1">
                <span className="text-slate-500 uppercase font-semibold text-[10px]">Input Signals:</span>
                <p className="text-slate-300 font-mono text-[11px]">{model.input}</p>
              </div>

              <div className="space-y-1">
                <span className="text-slate-500 uppercase font-semibold text-[10px]">Decision Calibration:</span>
                <p className="text-slate-300 font-mono text-[11px]">{model.threshold}</p>
              </div>
            </div>

            <div className="pt-2 border-t border-slate-800/60 flex flex-col sm:flex-row sm:items-center justify-between text-xs gap-2">
              <p className="text-slate-400">
                <span className="font-semibold text-slate-300">Role: </span>
                {model.role}
              </p>
            </div>

            <div className="p-2 rounded bg-slate-900/80 border border-slate-800/80 text-[11px] font-mono text-cyan-300/90">
              Benchmark: {model.performance}
            </div>
          </div>
        ))}
      </div>

      {/* Explicit Academic & Synthetic Dataset Disclosure */}
      <div className="p-6 rounded-xl border border-amber-500/40 bg-amber-950/20 space-y-3">
        <h3 className="text-sm font-bold text-amber-300 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-amber-400" />
          <span>Research Integrity & Synthetic Dataset Disclosure</span>
        </h3>
        <p className="text-xs text-slate-300 leading-relaxed">
          In strict adherence to empirical research standards, PhishVision AI maintains that <strong>high synthetic mAP does not establish real-world detection accuracy</strong>. M-04, M-05, and M-06 were trained on controlled synthetic webpage layouts to isolate UI component bounding boxes and segmentation boundaries.
        </p>
        <p className="text-xs text-slate-300 leading-relaxed">
          Furthermore, M-07 (URL Sequence Bi-LSTM) operates in single-hop fallback mode because multi-hop redirect sequence ground-truth was unavailable in historical threat feeds. PhishVision AI never fabricates scores or mocks synthetic performance metrics.
        </p>
      </div>
    </div>
  );
}
