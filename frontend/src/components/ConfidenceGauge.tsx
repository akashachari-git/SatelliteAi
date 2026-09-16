import React from 'react';
import { ShieldCheck, CheckCircle, Info, Zap } from 'lucide-react';
import { ConfidenceProfile } from '../types';

interface ConfidenceGaugeProps {
  confidence?: ConfidenceProfile | null;
}

export const ConfidenceGauge: React.FC<ConfidenceGaugeProps> = ({ confidence }) => {
  if (!confidence) return null;

  const pct = confidence.percentage || Math.round((confidence.score || 0.85) * 100);
  
  // Color tier
  const getBadgeColor = () => {
    if (pct >= 85) return 'from-emerald-500 to-cyan-500 text-emerald-400 border-emerald-500/30';
    if (pct >= 70) return 'from-cyan-500 to-blue-500 text-cyan-400 border-cyan-500/30';
    return 'from-amber-500 to-orange-500 text-amber-400 border-amber-500/30';
  };

  return (
    <div className="rounded-xl border border-space-700/80 bg-space-900/60 p-4 backdrop-blur-sm">
      <div className="flex items-center justify-between pb-2 mb-3 border-b border-space-800">
        <div className="flex items-center space-x-2">
          <ShieldCheck className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-semibold text-slate-200">Confidence Assessment</span>
        </div>
        <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 font-mono border border-cyan-500/20">
          {confidence.calibrated ? 'Calibrated Multi-Factor' : 'Estimated'}
        </span>
      </div>

      {/* Main Score Bar */}
      <div className="space-y-2 mb-4">
        <div className="flex items-end justify-between">
          <div>
            <span className="text-2xl font-bold font-mono tracking-tight text-white">{pct}%</span>
            <span className="text-xs text-slate-400 ml-2 font-medium">{confidence.label}</span>
          </div>
          <span className="text-[11px] font-mono text-cyan-400">Score: {confidence.score.toFixed(2)} / 1.00</span>
        </div>
        
        <div className="w-full h-2.5 rounded-full bg-space-950 overflow-hidden border border-space-800 p-0.5">
          <div 
            className={`h-full rounded-full bg-gradient-to-r ${getBadgeColor()} transition-all duration-700`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Evidence Factors Grid */}
      <div className="grid grid-cols-2 gap-2 text-[11px]">
        {confidence.factors && Object.entries(confidence.factors).map(([key, val]) => (
          <div key={key} className="p-2 rounded-lg bg-space-950/70 border border-space-800/80 flex items-center justify-between">
            <span className="text-slate-400 capitalize font-mono text-[10px]">
              {key.replace(/_/g, ' ')}
            </span>
            <span className="text-slate-200 font-medium font-mono">
              {val}
            </span>
          </div>
        ))}
      </div>

      <div className="mt-3 pt-2.5 border-t border-space-800 flex items-start space-x-1.5 text-[10.5px] text-slate-400">
        <Info className="w-3.5 h-3.5 text-cyan-400 shrink-0 mt-0.5" />
        <p>
          Derived from BigEarthNet spectral signature congruence, spatial co-registration overlap, and specialist tool detection margin.
        </p>
      </div>
    </div>
  );
};
