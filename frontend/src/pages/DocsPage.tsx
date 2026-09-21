import React, { useState } from 'react';
import { BookOpen, Cpu, Database, Layers, Radio, FileText, CheckCircle2 } from 'lucide-react';

export const DocsPage: React.FC = () => {
  const [activeDoc, setActiveDoc] = useState<'arch' | 'models' | 'adaptation' | 'api' | 'standards'>('arch');

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      
      {/* Header */}
      <div className="p-4 rounded-2xl bg-space-900/80 border border-space-800">
        <div className="flex items-center space-x-2">
          <BookOpen className="w-5 h-5 text-cyan-400" />
          <h2 className="text-sm font-bold text-white uppercase font-mono tracking-wider">
            SatQuery AI Technical Documentation & Architecture Dossier
          </h2>
        </div>
        <p className="text-xs text-slate-400 mt-1">
          Full technical specifications, model adaptation methodologies, and geospatial standards compliance.
        </p>
      </div>

      {/* Docs Tabs */}
      <div className="flex flex-wrap gap-2 border-b border-space-800 pb-3">
        {[
          { id: 'arch', label: 'Architecture & Agent' },
          { id: 'models', label: 'Specialist Model Registry' },
          { id: 'adaptation', label: 'BigEarthNet Adaptation' },
          { id: 'api', label: 'REST API Reference' },
          { id: 'standards', label: 'Geospatial AI Standards' }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveDoc(tab.id as any)}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-mono transition-all ${
              activeDoc === tab.id
                ? 'bg-cyan-500 text-white font-bold shadow'
                : 'bg-space-900 text-slate-400 hover:text-white border border-space-800'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Doc Content */}
      <div className="rounded-2xl border border-space-700/80 bg-space-900/60 p-6 backdrop-blur-sm space-y-6 text-xs text-slate-300 leading-relaxed font-sans">
        
        {activeDoc === 'arch' && (
          <div className="space-y-4">
            <h3 className="text-base font-bold text-white font-mono">1. System Architecture & Autonomous Agent</h3>
            <p>
              SatQuery AI addresses the core friction non-expert users face when querying complex satellite imagery: 
              lack of GIS familiarity, unfamiliar sensor physics (e.g. SAR vs Optical), and opaque model selection.
            </p>
            <div className="p-4 rounded-xl bg-space-950 font-mono text-cyan-300 border border-space-800">
              <pre className="text-[11px] leading-relaxed">
{`User Query + Satellite Raster(s)
      │
      ▼
Stage 1: Linguistic Normalization & Query Inspection
Stage 2: Raster Metadata & GeoTIFF Tag Extraction (CRS, GSD, Modality)
Stage 3: Agentic Task Classification (Deterministic + Rule Classifier)
Stage 4: Multi-Image Co-Registration & Spatial Compatibility Check
Stage 5: Specialist Tool Registry Selection
Stage 6: Model Inference Execution (BigEarthNet / CDVQA / Optical+SAR)
Stage 7: Visual Evidence Extraction (Heatmaps, Masks, Bounding Boxes)
Stage 8: Calibrated Multi-Factor Confidence Estimation
Stage 9: Auditable Observable Trace & PDF Intelligence Dossier`}
              </pre>
            </div>
            <p>
              Unlike generic LLM wrappers that hallucinate geospatial geometries, SatQuery AI relies on an autonomous 
              Python controller that verifies preconditions. For example, if a user requests change analysis with only one image,
              the agent immediately returns a friendly diagnostic explanation rather than an unhandled model crash.
            </p>
          </div>
        )}

        {activeDoc === 'models' && (
          <div className="space-y-4">
            <h3 className="text-base font-bold text-white font-mono">2. Specialist Model & Tool Registry</h3>
            <p>
              Every tool implements the standard <code>BaseRemoteSensingModel</code> contract: <code>load()</code>, <code>predict()</code>, 
              <code>validate_input()</code>, <code>get_confidence()</code>, and <code>get_metadata()</code>.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {[
                { name: "RemoteSensingVQA", task: "SINGLE_VQA", desc: "Answers natural language queries about scene composition, dominance, and land cover using 19-class BigEarthNet probabilities." },
                { name: "RemoteSensingCaptioner", task: "CAPTIONING", desc: "Generates comprehensive narrative descriptions aligned with VRSBench/RSICD benchmark standards." },
                { name: "GroundingModel", task: "GROUNDING", desc: "Text-guided spatial localization of water bodies, urban areas, and vegetation with bounding boxes and segmentation masks." },
                { name: "ChangeDetectionModel", task: "CHANGE_ANALYSIS", desc: "Bi-temporal change differencing using log-ratio for SAR and Euclidean spectral delta for optical, paired with Otsu adaptive thresholding." },
                { name: "ChangeVQAModel", task: "CHANGE_VQA", desc: "Answers directional change questions (increased, decreased, stable) and quantifies flux across geographic quadrants." },
                { name: "OpticalSARFusion", task: "OPTICAL_SAR_ANALYSIS", desc: "Joint cross-modal reasoning pairing optical surface reflectance with microwave backscatter double-bounce." }
              ].map(tool => (
                <div key={tool.name} className="p-3.5 rounded-xl bg-space-950 border border-space-800">
                  <span className="font-mono text-cyan-400 font-bold block">{tool.name}</span>
                  <span className="text-[10px] font-mono text-slate-400 block mb-1">Task: {tool.task}</span>
                  <p className="text-slate-300 text-[11px]">{tool.desc}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeDoc === 'adaptation' && (
          <div className="space-y-4">
            <h3 className="text-base font-bold text-white font-mono">3. Remote Sensing Adaptation via BigEarthNet</h3>
            <p>
              A fundamental mandate of the platform requirements is that the system must NOT be presented as a generic VLM. 
              SatQuery AI integrates a dedicated remote-sensing adaptation layer built upon the standardized 19-class 
              CORINE Land Cover nomenclature (BigEarthNet-S2 / BigEarthNet.txt).
            </p>
            <div className="p-4 rounded-xl bg-space-950 border border-space-800 space-y-2 font-mono text-[11px]">
              <div className="text-cyan-400 font-bold">19-Class Standard Taxonomy:</div>
              <p className="text-slate-300">
                Urban fabric, Industrial or commercial units, Arable land, Permanent crops, Pastures, Complex cultivation patterns,
                Agricultural land with natural vegetation, Agro-forestry areas, Broad-leaved forest, Coniferous forest, Mixed forest,
                Natural grassland, Moors/heathland, Transitional woodland-shrub, Beaches/dunes/sands, Bare rock, Inland wetlands,
                Coastal wetlands, Inland / Marine waters.
              </p>
              <div className="pt-2 text-slate-400">
                <strong>Radiometric Normalization:</strong> Bands B02, B03, B04, B08, B11, B12 normalized with precomputed Top-of-Atmosphere
                reflectance means and standard deviations from the BigEarthNet benchmark.
              </div>
            </div>
          </div>
        )}

        {activeDoc === 'api' && (
          <div className="space-y-4">
            <h3 className="text-base font-bold text-white font-mono">4. REST API Endpoints</h3>
            <div className="space-y-2 font-mono text-[11px]">
              {[
                { method: "POST", path: "/api/upload", desc: "Uploads GeoTIFF/TIFF/PNG, extracts metadata, creates web preview." },
                { method: "POST", path: "/api/validate", desc: "Evaluates spatial overlap, CRS match, and resolution ratio across image pairs." },
                { method: "POST", path: "/api/analyze", desc: "Executes autonomous 13-stage agent controller; returns answer, confidence, trace, evidence." },
                { method: "POST", path: "/api/report/generate-pdf", desc: "Compiles formal PDF intelligence report using ReportLab with embedded evidence." },
                { method: "POST", path: "/api/report/generate-json", desc: "Exports machine-readable JSON dossier." },
                { method: "GET", path: "/api/demo-scenarios", desc: "Lists pre-configured remote sensing demo scenarios." },
                { method: "POST", path: "/api/demo-scenarios/load/{id}", desc: "Mounts curated GeoTIFF scenario with sample queries." },
                { method: "GET", path: "/api/benchmarks", desc: "Returns benchmark evaluation harness status." }
              ].map(apiItem => (
                <div key={apiItem.path} className="p-2.5 rounded-lg bg-space-950 border border-space-800 flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span className="px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-400 font-bold">{apiItem.method}</span>
                    <span className="text-slate-200">{apiItem.path}</span>
                  </div>
                  <span className="text-slate-400 text-[10px] hidden md:inline">{apiItem.desc}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeDoc === 'standards' && (
          <div className="space-y-4">
            <h3 className="text-base font-bold text-white font-mono">5. Geospatial AI Standards & Compliance</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {[
                { title: "Agentic Orchestration", detail: "13-stage controller with Model/Tool Registry, task routing, and observable execution traces." },
                { title: "Remote-Sensing Specific AI", detail: "BigEarthNet-19 land cover taxonomy, NDVI/NDWI/NDBI spectral indices, and SAR backscatter physics." },
                { title: "Multimodal Optical + SAR", detail: "Cross-modal reasoning distinguishing specular water reflection and double-bounce urban structures." },
                { title: "Bi-Temporal Change Analysis", detail: "Log-ratio and spectral difference maps, Otsu segmentation masks, and sector-wise flux statistics." },
                { title: "Evidence-Grounded Answers", detail: "Interactive pan/zoom viewer with mask overlays, bounding boxes, and swipe curtain." },
                { title: "Scientific Honesty", detail: "Unmounted benchmark datasets state 'Not configured' with zero fabricated metrics." }
              ].map((item, i) => (
                <div key={i} className="p-3 rounded-xl bg-space-950 border border-space-800 flex items-start space-x-2.5">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-bold text-slate-200 block text-xs">{item.title}</span>
                    <span className="text-slate-400 text-[11px]">{item.detail}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>

    </div>
  );
};
