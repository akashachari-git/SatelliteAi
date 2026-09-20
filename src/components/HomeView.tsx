import React from 'react';
import {
  Satellite,
  Upload,
  ArrowRight,
  GitCompare,
  Layers,
  Sparkles,
  Search,
  FileText,
  Shield,
  Eye,
} from 'lucide-react';

interface HomeViewProps {
  onStartAnalysis: (mode?: 'single' | 'compare' | 'optical-sar') => void;
  onViewHistory: () => void;
}

export const HomeView: React.FC<HomeViewProps> = ({
  onStartAnalysis,
  onViewHistory,
}) => {
  return (
    <div className="max-w-6xl mx-auto space-y-12 py-6 animate-fadeIn">
      {/* Hero Section */}
      <section className="relative rounded-3xl bg-gradient-to-b from-cyan-950/40 via-slate-900/60 to-slate-950/80 border border-slate-800/90 p-8 sm:p-12 lg:p-16 text-center overflow-hidden shadow-2xl">
        {/* Ambient background glow */}
        <div className="absolute -top-24 left-1/2 -translate-x-1/2 w-96 h-96 bg-cyan-500/15 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute -bottom-24 right-10 w-80 h-80 bg-blue-600/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-3xl mx-auto space-y-6">
          {/* Logo & Tag badge */}
          <div className="inline-flex items-center gap-2.5 px-4 py-1.5 rounded-full bg-cyan-950/70 border border-cyan-500/30 text-cyan-300 text-xs font-medium tracking-wide">
            <Satellite className="w-4 h-4 text-cyan-400 animate-pulse" />
            <span>AI-Powered Earth Observation</span>
          </div>

          {/* Main Title & Tagline */}
          <div className="space-y-3">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-slate-100 tracking-tight font-sans">
              SatQuery <span className="text-cyan-400">AI</span>
            </h1>
            <p className="text-xl sm:text-2xl text-slate-300 font-medium leading-relaxed">
              Ask questions about satellite imagery using natural language.
            </p>
          </div>

          {/* Simple Explanation */}
          <p className="text-sm sm:text-base text-slate-400 leading-relaxed max-w-2xl mx-auto">
            SatQuery AI enables natural-language analysis across single satellite images, past vs present imagery, and cross-modal optical + SAR radar imagery with verifiable evidence grounding.
          </p>

          {/* Primary Action Button */}
          <div className="pt-4 flex flex-col sm:flex-row items-center justify-center gap-4">
            <button
              id="home-btn-add-images"
              onClick={() => onStartAnalysis('single')}
              className="w-full sm:w-auto px-8 py-4 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-base shadow-[0_0_25px_rgba(6,182,212,0.35)] hover:shadow-[0_0_35px_rgba(6,182,212,0.5)] transition-all flex items-center justify-center gap-3 cursor-pointer group"
            >
              <Upload className="w-5 h-5 text-slate-950 group-hover:-translate-y-0.5 transition-transform" />
              <span>Start Analysis</span>
              <ArrowRight className="w-5 h-5 text-slate-950 group-hover:translate-x-1 transition-transform" />
            </button>

            <button
              id="home-btn-view-history"
              onClick={onViewHistory}
              className="w-full sm:w-auto px-6 py-4 rounded-xl bg-slate-900/90 hover:bg-slate-850 border border-slate-700 text-slate-200 font-semibold text-sm transition flex items-center justify-center gap-2 cursor-pointer"
            >
              <FileText className="w-4 h-4 text-cyan-400" />
              <span>View Past Analyses</span>
            </button>
          </div>
        </div>
      </section>

      {/* What the application does - 3 Simple Analysis Modes */}
      <section className="space-y-6">
        <div className="text-center space-y-2">
          <h2 className="text-2xl font-bold text-slate-100 font-sans">
            How SatQuery AI Works
          </h2>
          <p className="text-sm text-slate-400 max-w-xl mx-auto">
            Choose from three simple workflows to analyze your satellite imagery.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Card 1: Single Image Analysis */}
          <div
            onClick={() => onStartAnalysis('single')}
            className="group p-6 sm:p-7 rounded-2xl bg-slate-900/60 hover:bg-slate-900/90 border border-slate-800 hover:border-cyan-500/50 transition-all duration-200 cursor-pointer flex flex-col justify-between space-y-4 shadow-lg hover:shadow-cyan-500/10"
          >
            <div className="space-y-4">
              <div className="w-12 h-12 rounded-xl bg-cyan-950/80 border border-cyan-500/30 flex items-center justify-center text-cyan-400 group-hover:scale-110 transition-transform">
                <Search className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <h3 className="text-lg font-bold text-slate-100 group-hover:text-cyan-300 transition-colors">
                  1. Single Image Analysis
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Upload any satellite or aerial image. Ask natural questions about land cover, infrastructure, vessels, or environmental features.
                </p>
              </div>
              <ul className="text-xs text-slate-400 space-y-1.5 pt-2 border-t border-slate-800/80">
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                  <span>Instant image validation</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                  <span>Natural-language question answering</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                  <span>Visual detection highlights</span>
                </li>
              </ul>
            </div>
            <div className="pt-4 flex items-center gap-2 text-xs font-semibold text-cyan-400 group-hover:text-cyan-300">
              <span>Start Single Image</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>

          {/* Card 2: Past & Present Comparison */}
          <div
            onClick={() => onStartAnalysis('compare')}
            className="group p-6 sm:p-7 rounded-2xl bg-slate-900/60 hover:bg-slate-900/90 border border-slate-800 hover:border-cyan-500/50 transition-all duration-200 cursor-pointer flex flex-col justify-between space-y-4 shadow-lg hover:shadow-cyan-500/10"
          >
            <div className="space-y-4">
              <div className="w-12 h-12 rounded-xl bg-cyan-950/80 border border-cyan-500/30 flex items-center justify-center text-cyan-400 group-hover:scale-110 transition-transform">
                <GitCompare className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <h3 className="text-lg font-bold text-slate-100 group-hover:text-cyan-300 transition-colors">
                  2. Past & Present Comparison
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Upload two images of the same location (Past vs Present). Automatically identify urban development, water shrinkage, or forest loss.
                </p>
              </div>
              <ul className="text-xs text-slate-400 space-y-1.5 pt-2 border-t border-slate-800/80">
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                  <span>Side-by-side adjacent display</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                  <span>Synchronized pan & zoom</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                  <span>Increased vs decreased areas</span>
                </li>
              </ul>
            </div>
            <div className="pt-4 flex items-center gap-2 text-xs font-semibold text-cyan-400 group-hover:text-cyan-300">
              <span>Compare Past & Present</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>

          {/* Card 3: Optical + SAR Analysis */}
          <div
            onClick={() => onStartAnalysis('optical-sar')}
            className="group p-6 sm:p-7 rounded-2xl bg-slate-900/60 hover:bg-slate-900/90 border border-slate-800 hover:border-cyan-500/50 transition-all duration-200 cursor-pointer flex flex-col justify-between space-y-4 shadow-lg hover:shadow-cyan-500/10"
          >
            <div className="space-y-4">
              <div className="w-12 h-12 rounded-xl bg-cyan-950/80 border border-cyan-500/30 flex items-center justify-center text-cyan-400 group-hover:scale-110 transition-transform">
                <Layers className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <h3 className="text-lg font-bold text-slate-100 group-hover:text-cyan-300 transition-colors">
                  3. Optical + SAR Analysis
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Combine multispectral optical imagery with radar (SAR) backscatter to penetrate clouds, distinguish water from shadows, and verify structures.
                </p>
              </div>
              <ul className="text-xs text-slate-400 space-y-1.5 pt-2 border-t border-slate-800/80">
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                  <span>Cloud & haze penetration</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                  <span>Radar microwave reflection</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                  <span>Joint cross-sensor reasoning</span>
                </li>
              </ul>
            </div>
            <div className="pt-4 flex items-center gap-2 text-xs font-semibold text-cyan-400 group-hover:text-cyan-300">
              <span>Start Optical + SAR</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>
        </div>
      </section>

      {/* The Linear User Workflow Banner */}
      <section className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="space-y-1 text-center md:text-left">
          <h3 className="text-base font-bold text-slate-200">
            Simple 5-Step Workflow
          </h3>
          <p className="text-xs text-slate-400">
            Every analysis follows an intuitive, transparent progression without technical confusion.
          </p>
        </div>
        <div className="flex items-center gap-2 sm:gap-4 text-xs font-semibold text-slate-300 flex-wrap justify-center">
          <span className="px-3 py-1.5 rounded-lg bg-cyan-950/70 border border-cyan-500/40 text-cyan-300">1. Add Satellite Image</span>
          <ArrowRight className="w-3.5 h-3.5 text-slate-600 hidden sm:block" />
          <span className="px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-300">2. Preview Images</span>
          <ArrowRight className="w-3.5 h-3.5 text-slate-600 hidden sm:block" />
          <span className="px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-300">3. Ask / Analyze</span>
          <ArrowRight className="w-3.5 h-3.5 text-slate-600 hidden sm:block" />
          <span className="px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-300">4. Results</span>
          <ArrowRight className="w-3.5 h-3.5 text-slate-600 hidden sm:block" />
          <span className="px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-300">5. Report</span>
        </div>
      </section>
    </div>
  );
};
