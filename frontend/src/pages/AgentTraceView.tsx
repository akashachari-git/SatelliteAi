import React, { useState } from 'react';
import { Cpu, CheckCircle2, Clock, ShieldCheck, Layers, FileCode, ArrowDown } from 'lucide-react';
import { AnalysisResult } from '../types';
import { TraceTimeline } from '../components/TraceTimeline';

interface AgentTraceViewProps {
  currentAnalysis: AnalysisResult | null;
  onGoToWorkspace: () => void;
}

export const AgentTraceView: React.FC<AgentTraceViewProps> = ({
  currentAnalysis,
  onGoToWorkspace
}) => {
  const [showRawJson, setShowRawJson] = useState(false);

  if (!currentAnalysis) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-16 text-center space-y-4">
        <Cpu className="w-12 h-12 text-space-700 mx-auto" />
        <h2 className="text-base font-semibold text-slate-200">No Active Trace Recorded</h2>
        <p className="text-xs text-slate-400 max-w-md mx-auto">
          Execute a query in Single image VQA or load a demo scenario to inspect the complete 13-stage agent execution trace.
        </p>
        <button
          onClick={onGoToWorkspace}
          className="px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-white text-xs font-semibold"
        >
          Go to Single image VQA
        </button>
      </div>
    );
  }

  const stages = [
    { label: "Query Inspection", sub: "Linguistic Normalization", color: "border-cyan-500 text-cyan-400" },
    { label: "Metadata Inspection", sub: "GeoTIFF & Modality", color: "border-blue-500 text-blue-400" },
    { label: "Task Router", sub: "Classifier Intent", color: "border-purple-500 text-purple-400" },
    { label: "Co-Registration", sub: "Spatial & CRS Check", color: "border-amber-500 text-amber-400" },
    { label: "Specialist Tool", sub: currentAnalysis.selected_tool, color: "border-emerald-500 text-emerald-400" },
    { label: "Evidence & Answer", sub: "Dossier Assembly", color: "border-rose-500 text-rose-400" },
  ];

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      
      {/* Header */}
      <div className="flex items-center justify-between p-4 rounded-2xl bg-space-900/80 border border-space-800">
        <div>
          <div className="flex items-center space-x-2">
            <Cpu className="w-5 h-5 text-cyan-400" />
            <h2 className="text-sm font-bold text-white uppercase font-mono tracking-wider">
              Autonomous Agent Execution Architecture
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Real-time auditable step-by-step observable stages of the remote-sensing reasoning pipeline.
          </p>
        </div>

        <button
          onClick={() => setShowRawJson(!showRawJson)}
          className="px-3 py-1.5 rounded-lg bg-space-800 hover:bg-space-700 text-slate-300 text-xs font-mono transition-colors border border-space-700 flex items-center space-x-1.5"
        >
          <FileCode className="w-3.5 h-3.5 text-cyan-400" />
          <span>{showRawJson ? 'Hide JSON Trace' : 'View JSON Trace'}</span>
        </button>
      </div>

      {/* Visual Pipeline Flowchart */}
      <div className="p-5 rounded-2xl bg-space-900/40 border border-space-800">
        <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider block mb-3 font-semibold">
          High-Level Pipeline Graph:
        </span>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
          {stages.map((st, idx) => (
            <div key={idx} className={`p-3 rounded-xl bg-space-950 border ${st.color} shadow-sm text-center relative`}>
              <span className="text-[9px] font-mono opacity-70 block mb-1">STAGE 0{idx + 1}</span>
              <span className="text-xs font-bold text-slate-100 block truncate">{st.label}</span>
              <span className="text-[10px] font-mono opacity-80 block truncate mt-0.5">{st.sub}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Raw JSON Trace Modal/Drawer */}
      {showRawJson && (
        <div className="p-4 rounded-2xl bg-space-950 border border-space-800 font-mono text-xs overflow-x-auto max-h-96">
          <pre className="text-cyan-300">
            {JSON.stringify(currentAnalysis.execution_trace, null, 2)}
          </pre>
        </div>
      )}

      {/* Detailed Trace Timeline Component */}
      <TraceTimeline
        trace={currentAnalysis.execution_trace}
        task={currentAnalysis.task}
        toolName={currentAnalysis.selected_tool}
        totalDuration={currentAnalysis.total_duration_seconds}
      />

    </div>
  );
};
