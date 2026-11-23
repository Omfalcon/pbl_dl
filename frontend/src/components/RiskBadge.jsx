import React from 'react';
import { ShieldCheck, AlertTriangle, AlertOctagon, HelpCircle, ShieldX } from 'lucide-react';

const CATEGORY_STYLES = {
  LEGITIMATE: {
    bg: 'bg-emerald-950/60',
    border: 'border-emerald-500/40',
    text: 'text-emerald-400',
    icon: ShieldCheck,
    label: 'Legitimate',
  },
  SUSPICIOUS: {
    bg: 'bg-amber-950/60',
    border: 'border-amber-500/40',
    text: 'text-amber-400',
    icon: AlertTriangle,
    label: 'Suspicious',
  },
  PHISHING: {
    bg: 'bg-red-950/60',
    border: 'border-red-500/40',
    text: 'text-red-400',
    icon: AlertOctagon,
    label: 'Phishing',
  },
  CRITICAL_PHISHING: {
    bg: 'bg-rose-950/80',
    border: 'border-rose-500/60',
    text: 'text-rose-300 font-bold',
    icon: AlertOctagon,
    label: 'Critical Phishing (Impersonation)',
  },
  'UNKNOWN / INSUFFICIENT EVIDENCE': {
    bg: 'bg-slate-900',
    border: 'border-slate-700',
    text: 'text-slate-400',
    icon: HelpCircle,
    label: 'Insufficient Evidence',
  },
  BLOCKED_SSRF: {
    bg: 'bg-purple-950/60',
    border: 'border-purple-500/40',
    text: 'text-purple-400',
    icon: ShieldX,
    label: 'Blocked (Anti-SSRF)',
  },
};

export default function RiskBadge({ category, size = 'md' }) {
  const normCat = category || 'UNKNOWN / INSUFFICIENT EVIDENCE';
  const style = CATEGORY_STYLES[normCat] || CATEGORY_STYLES['UNKNOWN / INSUFFICIENT EVIDENCE'];
  const Icon = style.icon;

  const sizeClasses = size === 'lg' 
    ? 'px-4 py-2 text-base gap-2' 
    : 'px-2.5 py-1 text-xs gap-1.5';

  return (
    <span className={`inline-flex items-center rounded-md border font-medium ${style.bg} ${style.border} ${style.text} ${sizeClasses}`}>
      <Icon className={size === 'lg' ? 'w-5 h-5' : 'w-3.5 h-3.5'} />
      <span>{style.label}</span>
    </span>
  );
}
