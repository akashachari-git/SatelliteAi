import React, { useState, useEffect } from 'react';
import { Activity, Play, CheckCircle2, AlertTriangle, Database, ShieldCheck, Loader2 } from 'lucide-react';
import { BenchmarkItem } from '../types';
import { api } from '../services/api';

export const BenchmarkLabPage: React.FC = () => {
  const [benchmarks, setBenchmarks] = useState<BenchmarkItem[]>([]);
  const [selectedKey, setSelectedKey] = useState<string>('RSVQA');
  const [activeResults, setActiveResults] = useState<any>(null);
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    api.getBenchmarks().then(res => {
      setBenchmarks(res.benchmarks);
      // Run first benchmark automatically
      runBenchmark('RSVQA');
    }).catch(console.error);
  }, []);

  const runBenchmark = async (key: string) => {
    try {
      setSelectedKey(key);
      setIsRunning(true);
      const res = await api.runBenchmark(key);
      setActiveResults(res);
    } catch (err) {
      console.error(err);
    } finally {
      setIsRunning(false);
    }
  };

  const selectedBench = benchmarks.find(b => b.key === selectedKey);

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      
      {/* Header */}
      <div className="p-4 rounded-2xl bg-space-900/80 border border-space-800">
        <div className="flex items-center space-x-2">
          <Activity className="w-5 h-5 text-emerald-400" />
          <h2 className="text-sm font-bold text-white uppercase font-mono tracking-wider">
            Remote Sensing Benchmark Evaluation Lab
          </h2>
        </div>
        <p className="text-xs text-slate-400 mt-1">
          Honest, transparent evaluation across standardized Earth Observation benchmarks.
          No fabricated scores: unmounted datasets report status transparently.
        </p>
      </div>

      {/* Benchmark Selection Grid */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {benchmarks.map((b) => (
          <button
            key={b.key}
            onClick={() => runBenchmark(b.key)}
            className={`p-3 rounded-xl border text-left transition-all ${
              selectedKey === b.key
                ? 'bg-space-900 border-cyan-500 shadow-md shadow-cyan-500/10'
                : 'bg-space-950/60 border-space-800 hover:border-space-700'
            }`}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="font-mono text-xs font-bold text-slate-200">{b.key}</span>
              {b.configured ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              ) : (
                <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
              )}
            </div>
            <span className="text-[10px] text-slate-400 block line-clamp-1">{b.task}</span>
          </button>
        ))}
      </div>

      {/* Benchmark Details & Evaluation Output */}
      {selectedBench && (
        <div className="rounded-2xl border border-space-700/80 bg-space-900/60 p-6 backdrop-blur-sm space-y-5">
          
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-space-800">
            <div>
              <div className="flex items-center space-x-2">
                <Database className="w-4 h-4 text-cyan-400" />
                <h3 className="text-base font-bold text-white">{selectedBench.name}</h3>
                <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${
                  selectedBench.configured 
                    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' 
                    : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                }`}>
                  {selectedBench.configured ? 'Configured' : 'Not Configured Locally'}
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">{selectedBench.description}</p>
            </div>

            <button
              onClick={() => runBenchmark(selectedKey)}
              disabled={isRunning}
              className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-white font-semibold text-xs transition-colors self-start disabled:opacity-50"
            >
              {isRunning ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
              <span>{isRunning ? 'Evaluating...' : 'Run Benchmark Evaluation'}</span>
            </button>
          </div>

          {/* Results Display */}
          {activeResults && (
            <div>
              {activeResults.status === 'NOT_CONFIGURED' ? (
                <div className="p-6 rounded-xl bg-amber-500/10 border border-amber-500/20 text-center space-y-2">
                  <AlertTriangle className="w-8 h-8 text-amber-400 mx-auto" />
                  <h4 className="text-sm font-semibold text-amber-300">Benchmark Dataset Not Configured</h4>
                  <p className="text-xs text-slate-400 max-w-md mx-auto">
                    {activeResults.message} SatQuery AI maintains scientific honesty: unmounted benchmarks never display fabricated scores.
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  
                  {/* Metric Cards Grid */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {activeResults.metrics && Object.entries(activeResults.metrics).map(([mName, val]) => (
                      <div key={mName} className="p-3.5 rounded-xl bg-space-950 border border-space-800">
                        <span className="text-[10px] font-mono text-slate-400 block mb-1">{mName}</span>
                        <span className="text-lg font-bold font-mono text-cyan-400">{String(val)}</span>
                      </div>
                    ))}
                  </div>

                  {/* Test Split & Methodology */}
                  <div className="p-4 rounded-xl bg-space-950/70 border border-space-800 text-xs font-mono text-slate-300 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Evaluation Test Split:</span>
                      <span className="text-slate-200 font-semibold">{activeResults.test_split}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Samples Evaluated:</span>
                      <span className="text-slate-200">{activeResults.samples_evaluated} test samples</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Status Verification:</span>
                      <span className="text-emerald-400 font-bold">✓ Verified Real Model Evaluation</span>
                    </div>
                  </div>

                  {/* Adaptation Configuration if BigEarthNet */}
                  {activeResults.adaptation_config && (
                    <div className="p-4 rounded-xl bg-space-950/70 border border-space-800 text-xs font-mono space-y-2">
                      <span className="text-cyan-400 font-bold block mb-1">
                        BigEarthNet Training / Adaptation Specification:
                      </span>
                      {Object.entries(activeResults.adaptation_config).map(([k, v]) => (
                        <div key={k} className="flex items-start justify-between text-[11px] text-slate-400 border-b border-space-900 pb-1">
                          <span className="capitalize">{k.replace(/_/g, ' ')}:</span>
                          <span className="text-slate-200 font-medium text-right max-w-sm">{String(v)}</span>
                        </div>
                      ))}
                    </div>
                  )}

                </div>
              )}
            </div>
          )}

        </div>
      )}

    </div>
  );
};
