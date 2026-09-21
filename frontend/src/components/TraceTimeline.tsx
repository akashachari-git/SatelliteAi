import React from 'react';
import { 
  CheckCircle2, 
  AlertTriangle, 
  XCircle, 
  Clock, 
  Cpu, 
  ChevronRight,
  ShieldCheck,
  Zap
} from 'lucide-react';
import { TraceStep } from '../types';

interface TraceTimelineProps {
  trace: TraceStep[];
  task?: string;
  toolName?: string;
  totalDuration?: number;
}

export const TraceTimeline: React.FC<TraceTimelineProps> = ({
  trace = [],
  task,
  toolName,
  totalDuration
}) => {
  if (trace.length === 0) {
    return (
      <div className="rounded-xl border border-space-800 bg-space-900/40 p-6 text-center text-slate-400">
        <Cpu className="w-8 h-8 mx-auto text-space-700 mb-2" />
        <p className="text-xs">No active execution trace recorded.</p>
      </div>
    );
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'COMPLETED':
        return <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />;
      case 'WARNING':
        return <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />;
      case 'FAILED':
      case 'ERROR':
        return <XCircle className="w-4 h-4 text-rose-400 shrink-0" />;
      default:
        return <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0" />;
    }
  };

  return (
    <div className="rounded-2xl border border-space-700/80 bg-space-900/60 p-5 backdrop-blur-sm shadow-xl">
      
      {/* Header */}
      <div className="flex items-center justify-between pb-3 mb-4 border-b border-space-800">
        <div className="flex items-center space-x-2">
          <div className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></div>
          <span className="text-xs font-bold font-mono tracking-wider uppercase text-slate-200">
            Auditable Execution Trace
          </span>
        </div>
        
        <div className="flex items-center space-x-3 text-xs font-mono">
          {task && (
            <span className="px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 text-[11px]">
              Task: {task}
            </span>
          )}
          {totalDuration !== undefined && (
            <span className="flex items-center text-slate-400 text-[11px]">
              <Clock className="w-3 h-3 mr-1 text-slate-400" />
              {totalDuration}s total
            </span>
          )}
        </div>
      </div>

      {/* Step by Step Timeline */}
      <div className="relative pl-6 space-y-4 before:absolute before:left-2.5 before:top-2 before:bottom-2 before:w-0.5 before:bg-gradient-to-b before:from-cyan-500 before:via-blue-500 before:to-emerald-500">
        {trace.map((step, index) => (
          <div key={step.stage_id || index} className="relative group">
            {/* Step dot indicator */}
            <div className="absolute -left-6 top-1 w-5 h-5 rounded-full bg-space-950 border border-space-700 flex items-center justify-center shadow-md">
              {getStatusIcon(step.status)}
            </div>

            {/* Step Body */}
            <div className="p-3 rounded-xl bg-space-950/80 border border-space-800/80 hover:border-space-700 transition-colors">
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center space-x-2">
                  <span className="font-mono text-[10px] text-cyan-400 font-semibold">
                    STEP {index + 1}
                  </span>
                  <span className="font-semibold text-slate-200">{step.stage_name}</span>
                </div>
                <span className="text-[10px] font-mono text-slate-400">
                  {step.duration_ms} ms
                </span>
              </div>

              <p className="text-xs text-slate-300 mt-1 font-mono leading-relaxed">
                {step.description}
              </p>

              {/* Observable Details (Observable only, no hidden thoughts) */}
              {step.details && Object.keys(step.details).length > 0 && (
                <div className="mt-2 pt-2 border-t border-space-900/60 flex flex-wrap gap-1.5">
                  {Object.entries(step.details).map(([k, v]) => {
                    if (typeof v === 'object' && v !== null) return null;
                    return (
                      <span key={k} className="text-[10px] font-mono px-2 py-0.5 rounded bg-space-900 text-slate-400 border border-space-800">
                        <span className="text-cyan-400">{k}:</span> {String(v)}
                      </span>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

    </div>
  );
};
