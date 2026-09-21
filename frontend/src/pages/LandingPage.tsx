import React, { useRef, useState, useEffect } from 'react';
import { 
  Globe2, 
  ArrowRight, 
  Sparkles, 
  Layers, 
  GitCompare, 
  Radio, 
  ChevronRight,
  ChevronDown
} from 'lucide-react';
import { DemoScenario } from '../types';

interface LandingPageProps {
  onStartAnalysis: () => void;
  onSelectScenario: (scenarioId: string) => void;
  demoScenarios: Record<string, DemoScenario>;
}

// Continuous Scroll-Reveal: keeps animations active on every scroll (down or up)
const useScrollReveal = (threshold = 0.08, rootMargin = '10px 0px -40px 0px') => {
  const ref = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        // Keep animation active: triggers visible when in viewport, resets when scrolled away
        setIsVisible(entry.isIntersecting);
      },
      { 
        threshold,
        rootMargin
      }
    );

    observer.observe(node);

    return () => {
      observer.disconnect();
    };
  }, [threshold, rootMargin]);

  return [ref, isVisible] as const;
};

// Reusable Animated Reveal Component with buttery-smooth float and gentle fade
const Reveal: React.FC<{
  children: React.ReactNode;
  className?: string;
  delay?: number;
}> = ({ children, className = '', delay = 0 }) => {
  const [ref, isVisible] = useScrollReveal();

  return (
    <div
      ref={ref}
      style={{ 
        transitionDelay: isVisible ? `${delay}ms` : '0ms',
        transitionDuration: isVisible ? '700ms' : '350ms',
        transitionTimingFunction: 'cubic-bezier(0.16, 1, 0.3, 1)',
      }}
      className={`transition-all transform will-change-[opacity,transform] ${
        isVisible 
          ? 'opacity-100 translate-y-0 scale-100' 
          : 'opacity-0 translate-y-7 scale-[0.98] pointer-events-none'
      } ${className}`}
    >
      {children}
    </div>
  );
};

export const LandingPage: React.FC<LandingPageProps> = ({
  onStartAnalysis,
  onSelectScenario,
  demoScenarios
}) => {
  return (
    <div className="relative min-h-screen text-slate-100 flex flex-col bg-transparent selection:bg-cyan-500/30">
      <div className="relative z-10 flex flex-col flex-1 bg-transparent">

        {/* 1. Hero Section */}
        <section className="relative pt-24 pb-20 md:pt-32 md:pb-28 bg-transparent">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
            
            <Reveal delay={100}>
              <div className="inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-full bg-black/80 backdrop-blur-md border border-neutral-800 text-neutral-300 text-xs font-sans font-semibold tracking-wide mb-6 shadow-xl hover:border-neutral-700 transition-colors">
                <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                <span>Earth Observation AI Innovation Track</span>
              </div>
            </Reveal>

            <Reveal delay={200}>
              <h1 className="text-4xl sm:text-6xl md:text-7xl font-extrabold tracking-tight text-white mb-6 drop-shadow-2xl leading-[1.1]">
                A smarter Way to <br />
                <span className="text-white">
                  Analyze Our Planet
                </span>
              </h1>
            </Reveal>

            <Reveal delay={300}>
              <p className="max-w-2xl mx-auto text-base sm:text-lg text-slate-200 font-normal leading-relaxed mb-10 drop-shadow">
                An interactive vision-language assistant for multimodal remote sensing analysis.
                Engineered with agentic model orchestration, BigEarthNet spectral adaptation, 
                bi-temporal change detection, and cross-modal optical + SAR fusion.
              </p>
            </Reveal>

            <Reveal delay={400}>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4 max-w-md mx-auto">
                <button
                  onClick={onStartAnalysis}
                  className="w-full sm:w-auto flex items-center justify-center space-x-2 px-6 py-3.5 rounded-xl bg-black/90 hover:bg-neutral-900 text-white font-semibold border border-neutral-700 hover:border-neutral-500 shadow-2xl transition-all active:scale-95 text-sm backdrop-blur-md"
                >
                  <span>Start Analysis</span>
                  <ArrowRight className="w-4 h-4 ml-1" />
                </button>

                <button
                  onClick={() => onSelectScenario('scenario_1_urban')}
                  className="w-full sm:w-auto flex items-center justify-center space-x-2 px-6 py-3.5 rounded-xl bg-black/80 hover:bg-neutral-900 text-slate-200 border border-neutral-800 hover:border-neutral-700 font-medium transition-all text-sm backdrop-blur-md shadow-xl"
                >
                  <Sparkles className="w-4 h-4 text-cyan-400" />
                  <span>Explore Demo</span>
                </button>
              </div>
            </Reveal>


            {/* Subtle Scroll Down Prompt */}
            <Reveal delay={500}>
              <div 
                onClick={() => {
                  const target = document.getElementById('capabilities-section');
                  target?.scrollIntoView({ behavior: 'smooth' });
                }}
                className="pt-14 flex flex-col items-center justify-center space-y-2 opacity-50 hover:opacity-100 transition-opacity cursor-pointer select-none group"
              >
                <span className="text-[10px] font-mono tracking-widest uppercase text-neutral-400 group-hover:text-cyan-400 transition-colors">Scroll to explore</span>
                <ChevronDown className="w-4 h-4 text-neutral-400 group-hover:text-cyan-400 animate-bounce transition-colors" />
              </div>
            </Reveal>

          </div>
        </section>

        {/* 2. Core Capabilities Section (Seamless Continuous Flow) */}
        <section id="capabilities-section" className="relative py-20 md:py-28 bg-transparent">
          <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            
            <Reveal>
              <div className="text-center max-w-2xl mx-auto mb-14">
                <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider font-semibold block mb-2">
                  Specialist Architecture
                </span>
                <h2 className="text-2xl sm:text-4xl font-bold text-white tracking-tight">
                  Beyond Generic VLMs: Domain-Adapted Remote Sensing AI
                </h2>
                <p className="text-xs sm:text-sm text-slate-300 mt-3 leading-relaxed">
                  Standard commercial VLMs hallucinate satellite geometries and confuse radiometric backscatter. 
                  SatQuery AI integrates specialized earth-observation models and spectral-spatial pipelines.
                </p>
              </div>
            </Reveal>

            {/* Sequential Feature Showcase */}
            <div className="space-y-10 md:space-y-14 mt-12 max-w-3xl mx-auto">
              {[
                {
                  title: "Single-Image VQA & Grounding",
                  desc: "Ask natural language inquiries regarding land-cover classification, vegetation vitality (NDVI), water indices (NDWI), and automated bounding-box localization of rivers, reservoirs, and built-up infrastructure.",
                  action: () => onSelectScenario('scenario_2_water'),
                  actionLabel: "Try Water Grounding",
                  icon: Layers
                },
                {
                  title: "Bi-Temporal Change Analysis",
                  desc: "Rigorous co-registration and radiometric flux analysis between earlier (T1) and later (T2) satellite passes. Generates difference heatmaps, Otsu threshold masks, and directional quadrant flux metrics.",
                  action: () => onSelectScenario('scenario_3_change'),
                  actionLabel: "Try Urban Change Detection",
                  icon: GitCompare
                },
                {
                  title: "Optical + SAR Cross-Modal Fusion",
                  desc: "Jointly synthesizes Sentinel-2 optical multispectral surface reflectance with Sentinel-1 C-band SAR microwave backscatter to isolate double-bounce urban structures from bare soil and water surfaces.",
                  action: () => onSelectScenario('scenario_4_optical_sar'),
                  actionLabel: "Try Cross-Modal Fusion",
                  icon: Radio
                }
              ].map((feat) => {
                const Icon = feat.icon;
                return (
                  <Reveal key={feat.title} className="flex items-center justify-center">
                    <div className="w-full rounded-2xl md:rounded-3xl border border-neutral-800/80 bg-black/75 backdrop-blur-2xl p-7 sm:p-9 md:p-10 shadow-2xl hover:border-neutral-700 transition-all duration-500 group space-y-5 text-left">
                      
                      <div className="flex items-center space-x-4">
                        <div className="w-12 h-12 rounded-2xl bg-neutral-900 border border-neutral-800 flex items-center justify-center text-white shrink-0 shadow-lg group-hover:scale-105 transition-transform">
                          <Icon className="w-6 h-6 text-cyan-400" />
                        </div>
                        <h3 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
                          {feat.title}
                        </h3>
                      </div>

                      <p className="text-sm sm:text-base text-slate-300 leading-relaxed font-normal">
                        {feat.desc}
                      </p>

                      <div className="pt-2">
                        <button
                          onClick={feat.action}
                          className="inline-flex items-center space-x-2 px-5 py-3 rounded-xl bg-black hover:bg-neutral-900 text-white font-mono text-xs border border-neutral-700 hover:border-neutral-500 transition-all shadow-xl active:scale-95 group/btn"
                        >
                          <span>{feat.actionLabel}</span>
                          <ChevronRight className="w-4 h-4 ml-1 group-hover/btn:translate-x-1 transition-transform text-cyan-400" />
                        </button>
                      </div>

                    </div>
                  </Reveal>
                );
              })}
            </div>
          </div>
        </section>

        {/* 3. How SatQuery Works Flowchart (Seamless Continuous Flow) */}
        <section className="relative py-20 md:py-28 bg-transparent">
          <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            
            <Reveal>
              <div className="text-center max-w-2xl mx-auto mb-14">
                <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider font-semibold block mb-2">
                  Transparent & Auditable
                </span>
                <h2 className="text-2xl sm:text-4xl font-bold text-white tracking-tight">
                  How SatQuery AI Works
                </h2>
                <p className="text-xs sm:text-sm text-slate-300 mt-3 leading-relaxed">
                  From raw satellite raster bytes to formal intelligence dossiers in sub-second execution.
                </p>
              </div>
            </Reveal>

            <div className="grid grid-cols-1 md:grid-cols-6 gap-3">
              {[
                { step: "01", title: "Natural Query", desc: "User asks in plain language without GIS jargon." },
                { step: "02", title: "Input Inspection", desc: "GeoTIFF tags, CRS, bounds & resolution checked." },
                { step: "03", title: "Task Routing", desc: "Agent classifies intent to specialist tool registry." },
                { step: "04", title: "Specialist AI", desc: "BigEarthNet / CDVQA model processes imagery." },
                { step: "05", title: "Evidence Fusion", desc: "Heatmaps, masks & bounding boxes validated." },
                { step: "06", title: "Grounded Answer", desc: "Interactive evidence maps + downloadable PDF dossier." },
              ].map((s, idx) => (
                <Reveal key={s.step} delay={idx * 80}>
                  <div className="p-4 rounded-xl bg-black/65 backdrop-blur-xl border border-neutral-800/80 hover:border-neutral-600 hover:bg-black/85 transition-all shadow-xl h-full">
                    <span className="text-[10px] font-mono font-bold text-neutral-400 block mb-1">
                      STAGE {s.step}
                    </span>
                    <h4 className="text-xs font-semibold text-white mb-1">{s.title}</h4>
                    <p className="text-[11px] text-slate-300 leading-normal">{s.desc}</p>
                  </div>
                </Reveal>
              ))}
            </div>
          </div>
        </section>

        {/* 4. Footer (Floating seamlessly over the background) */}
        <footer className="mt-auto py-12 bg-transparent text-xs text-slate-400">
          <Reveal>
            <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="flex items-center space-x-2">
                <Globe2 className="w-4 h-4 text-neutral-400" />
                <span className="font-semibold text-slate-200">SATQUERY AI</span>
                <span>— Multimodal Geospatial Intelligence</span>
              </div>
              <p className="font-mono text-[11px] text-slate-400">
                GeoTIFF • Sentinel-1 SAR • Sentinel-2 MSI • BigEarthNet-19
              </p>
            </div>
          </Reveal>
        </footer>

      </div>

    </div>
  );
};
