import React, { useState } from 'react';
import { Radio, Eye, Layers, Sliders, Info, Sparkles } from 'lucide-react';

interface DualModalityViewerProps {
  opticalUrl: string;
  sarUrl: string;
  fusedUrl?: string;
  sarMetrics?: any;
  opticalMetrics?: any;
}

export const DualModalityViewer: React.FC<DualModalityViewerProps> = ({
  opticalUrl,
  sarUrl,
  fusedUrl,
  sarMetrics,
  opticalMetrics
}) => {
  const [activeMode, setActiveMode] = useState<'side-by-side' | 'fused' | 'optical-only' | 'sar-only'>('side-by-side');

  return (
    <div className="rounded-2xl border border-space-700/80 bg-space-950 overflow-hidden shadow-2xl flex flex-col">
      {/* Modality Mode Controls */}
      <div className="flex items-center justify-between px-4 py-3 bg-space-900/90 border-b border-space-800">
        <div className="flex items-center space-x-2">
          <Radio className="w-4 h-4 text-purple-400" />
          <span className="text-xs font-semibold text-slate-200">Cross-Modal Dual Stream Inspection</span>
        </div>

        <div className="flex items-center space-x-1.5 bg-space-950/80 p-1 rounded-lg border border-space-800 text-xs font-mono">
          <button
            onClick={() => setActiveMode('side-by-side')}
            className={`px-2.5 py-1 rounded-md transition-all ${
              activeMode === 'side-by-side' ? 'bg-cyan-500 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Side-by-Side
          </button>
          {fusedUrl && (
            <button
              onClick={() => setActiveMode('fused')}
              className={`px-2.5 py-1 rounded-md transition-all flex items-center space-x-1 ${
                activeMode === 'fused' ? 'bg-purple-600 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Sparkles className="w-3 h-3 text-amber-300" />
              <span>False-Color Fusion</span>
            </button>
          )}
          <button
            onClick={() => setActiveMode('optical-only')}
            className={`px-2.5 py-1 rounded-md transition-all ${
              activeMode === 'optical-only' ? 'bg-emerald-600 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Optical
          </button>
          <button
            onClick={() => setActiveMode('sar-only')}
            className={`px-2.5 py-1 rounded-md transition-all ${
              activeMode === 'sar-only' ? 'bg-purple-600 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            SAR Backscatter
          </button>
        </div>
      </div>

      {/* Visual Canvas Display */}
      <div className="p-4 bg-geo-grid min-h-[420px] flex items-center justify-center">
        {activeMode === 'side-by-side' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
            {/* Optical Pane */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-[11px] font-mono text-emerald-400">
                <span>SENTINEL-2 OPTICAL (RGB)</span>
                <span>Surface Reflectance</span>
              </div>
              <div className="relative rounded-xl overflow-hidden border border-space-800 bg-space-900 aspect-square flex items-center justify-center">
                <img src={opticalUrl} alt="Optical" className="w-full h-full object-contain" />
                <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-space-950/80 text-[10px] font-mono text-emerald-300 border border-emerald-500/20">
                  Multispectral Reflectance
                </div>
              </div>
            </div>

            {/* SAR Pane */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-[11px] font-mono text-purple-400">
                <span>SENTINEL-1 C-SAR (VV)</span>
                <span>Microwave Backscatter (dB)</span>
              </div>
              <div className="relative rounded-xl overflow-hidden border border-space-800 bg-space-900 aspect-square flex items-center justify-center">
                <img src={sarUrl} alt="SAR" className="w-full h-full object-contain" />
                <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-space-950/80 text-[10px] font-mono text-purple-300 border border-purple-500/20">
                  Dielectric & Roughness
                </div>
              </div>
            </div>
          </div>
        )}

        {activeMode === 'fused' && fusedUrl && (
          <div className="relative max-w-xl w-full aspect-square rounded-xl overflow-hidden border border-space-800 bg-space-900 flex items-center justify-center">
            <img src={fusedUrl} alt="Optical-SAR Fused" className="w-full h-full object-contain" />
            <div className="absolute bottom-3 left-3 right-3 p-2.5 rounded-lg bg-space-950/90 border border-space-800 text-[11px] font-mono text-slate-300 flex items-center justify-between backdrop-blur-sm">
              <div className="flex items-center space-x-3">
                <span className="flex items-center text-rose-400"><span className="w-2 h-2 rounded-full bg-rose-400 mr-1.5"></span>R: Optical Albedo</span>
                <span className="flex items-center text-emerald-400"><span className="w-2 h-2 rounded-full bg-emerald-400 mr-1.5"></span>G: SAR Backscatter</span>
                <span className="flex items-center text-cyan-400"><span className="w-2 h-2 rounded-full bg-cyan-400 mr-1.5"></span>B: Water Absorption</span>
              </div>
            </div>
          </div>
        )}

        {activeMode === 'optical-only' && (
          <div className="max-w-xl w-full aspect-square rounded-xl overflow-hidden border border-space-800 bg-space-900 flex items-center justify-center">
            <img src={opticalUrl} alt="Optical Only" className="w-full h-full object-contain" />
          </div>
        )}

        {activeMode === 'sar-only' && (
          <div className="max-w-xl w-full aspect-square rounded-xl overflow-hidden border border-space-800 bg-space-900 flex items-center justify-center">
            <img src={sarUrl} alt="SAR Only" className="w-full h-full object-contain" />
          </div>
        )}
      </div>

      {/* Cross-Modal Comparison Insights */}
      {sarMetrics && (
        <div className="p-4 bg-space-900/80 border-t border-space-800 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <div className="p-2.5 rounded-lg bg-space-950 border border-space-800">
            <span className="text-[10px] text-slate-400 block font-mono">SAR Mean Backscatter</span>
            <span className="font-mono text-slate-200 font-bold text-sm">
              {sarMetrics.mean_backscatter_db} dB
            </span>
          </div>

          <div className="p-2.5 rounded-lg bg-space-950 border border-space-800">
            <span className="text-[10px] text-slate-400 block font-mono">Urban Double-Bounce</span>
            <span className="font-mono text-purple-400 font-bold text-sm">
              {sarMetrics.urban_double_bounce_pct}%
            </span>
          </div>

          <div className="p-2.5 rounded-lg bg-space-950 border border-space-800">
            <span className="text-[10px] text-slate-400 block font-mono">Water Specular Reflection</span>
            <span className="font-mono text-cyan-400 font-bold text-sm">
              {sarMetrics.water_specular_pct}%
            </span>
          </div>

          <div className="p-2.5 rounded-lg bg-space-950 border border-space-800">
            <span className="text-[10px] text-slate-400 block font-mono">Volume Scattering</span>
            <span className="font-mono text-emerald-400 font-bold text-sm">
              {sarMetrics.volume_scattering_pct}%
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
