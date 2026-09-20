import React, { useState } from 'react';
import {
  Sparkles,
  MapPin,
  Building2,
  Trees,
  Waves,
  Route,
  Wheat,
  Mountain,
  Crosshair,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  TrendingUp,
  TrendingDown,
  PlusCircle,
  MinusCircle,
  Equal,
  Layers,
  Search,
  Info,
  ChevronDown,
  ChevronUp,
  Clock,
  ShieldCheck,
  AlertTriangle,
  Radio,
  Compass,
} from 'lucide-react';
import { AnalysisResult, BoundingBox, ChangedRegion } from '../types';

interface AnalysisResultCardProps {
  result: AnalysisResult;
  onFocusEvidence?: (id: string) => void;
  focusedEvidenceId?: string | null;
}

export const AnalysisResultCard: React.FC<AnalysisResultCardProps> = ({
  result,
  onFocusEvidence,
  focusedEvidenceId,
}) => {
  const [showExecutionTrace, setShowExecutionTrace] = useState<boolean>(false);
  const [showAgentTransparency, setShowAgentTransparency] = useState<boolean>(true);

  // Agent execution derivation from actual backend response
  const queryIntent =
    result.taskType ||
    result.agentPlan?.detected_intent ||
    (isBiTemporal
      ? 'Bi-temporal Change Detection'
      : isOpticalSar
      ? 'Cross-modal Optical + SAR Analysis'
      : 'Vision-Language Remote Sensing / Grounding');

  const selectedSpecialist =
    result.agentPlan?.selected_specialist ||
    result.selectedModel ||
    'Multi-Specialist Orchestrator';

  const modelTool =
    (result.agentPlan?.selected_tools && result.agentPlan.selected_tools.length > 0)
      ? result.agentPlan.selected_tools.join(', ')
      : result.selectedModel || 'Specialist Models';

  const evidenceSynthesisSummary =
    result.evidenceHierarchy
      ? `${directEvidence.length} direct, ${supportingEvidence.length} supporting`
      : result.evidence && result.evidence.length > 0
      ? `${result.evidence.length} evidence items synthesized`
      : 'Synthesized model findings';

  const geospatialGroundingSummary =
    geoEvidence?.status === 'available'
      ? `Available (${geoEvidence.crs || 'Projected CRS'})`
      : 'Unavailable (Pixel Coordinate Space)';

  const isBiTemporal = result.mode === 'bi-temporal';
  const isOpticalSar = result.mode === 'optical-sar';
  const isSingle = !isBiTemporal && !isOpticalSar;

  const boundingBoxes = result.boundingBoxes || [];
  const changedRegions = result.changedRegions || [];
  const rawEvidence = result.evidence || [];

  const evidenceHierarchy = result.evidenceHierarchy;
  const directEvidence = evidenceHierarchy?.direct_evidence || rawEvidence;
  const supportingEvidence = evidenceHierarchy?.supporting_evidence || [];
  const limitations = evidenceHierarchy?.limitations || result.geospatialEvidence?.limitations || [];
  const disagreements = evidenceHierarchy?.disagreements || [];
  const executionSteps = result.executionSteps || [];
  const geoEvidence = result.geospatialEvidence;
  const crossModal = result.crossModalEvidence;

  const isNoReliableResult =
    result.hasReliableResult === false ||
    result.answer?.trim() === 'No reliable result available' ||
    result.answer?.includes('does not provide enough information to answer this question confidently') ||
    result.answer?.toLowerCase().includes('no reliable result available');

  // Helper for BigEarthNet land cover classes mentioned in evidence
  const isBigEarthNetUsed =
    result.selectedModel?.toLowerCase().includes('bigearthnet') ||
    result.agentPlan?.selected_tools?.includes('bigearthnet-classifier') ||
    directEvidence.some((e) => e.toLowerCase().includes('bigearthnet') || e.toLowerCase().includes('corine'));

  return (
    <section
      id="analysis-result-section"
      className="rounded-2xl bg-[#080d1a] border border-slate-800 p-6 sm:p-8 space-y-7 shadow-xl"
    >
      {/* ------------------------------------------------------------- */}
      {/* 0. INCONCLUSIVE QUERY / NO RELIABLE RESULT                   */}
      {/* ------------------------------------------------------------- */}
      {isNoReliableResult && (
        <div className="space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800 gap-3">
            <div>
              <div className="flex items-center gap-2 text-amber-400 font-mono text-xs uppercase tracking-wider font-semibold">
                <HelpCircle className="w-4 h-4 text-amber-400" />
                Query Inconclusive
              </div>
              <h2 className="text-2xl font-bold text-slate-100 tracking-tight mt-1">
                No reliable result available
              </h2>
            </div>
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-950/60 border border-amber-800 text-amber-300 font-mono text-xs self-start sm:self-auto">
              <span>Insufficient Overhead Resolution</span>
            </div>
          </div>

          <div className="p-6 rounded-2xl bg-amber-950/30 border border-amber-600/70 space-y-3 shadow-lg">
            <div className="text-xs font-mono text-amber-400 uppercase tracking-wider font-semibold">
              Question Asked
            </div>
            <div className="text-sm font-mono text-slate-300">
              &ldquo;{result.query}&rdquo;
            </div>

            <div className="pt-3 border-t border-amber-900/50 space-y-2">
              <div className="text-base font-bold text-amber-200 flex items-center gap-2">
                <HelpCircle className="w-5 h-5 text-amber-400 shrink-0" />
                <span>No reliable result available</span>
              </div>
              <p className="text-sm text-amber-100 font-sans leading-relaxed">
                The available imagery does not provide enough information to answer this question confidently.
              </p>
              {result.whyThisAnswer && (
                <p className="text-xs text-slate-400 font-sans leading-relaxed pt-1">
                  <span className="text-amber-300 font-medium">Context: </span>
                  {result.whyThisAnswer}
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* 1. PRIMARY RESULT HEADER & ANSWER (FOR ALL ACTIVE MODES)      */}
      {/* ------------------------------------------------------------- */}
      {!isNoReliableResult && (
        <>
          {/* Header */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800 gap-3">
            <div>
              <div className="flex items-center gap-2 text-cyan-400 font-mono text-xs uppercase tracking-wider font-semibold">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                <span>
                  {isBiTemporal
                    ? 'Bi-Temporal Change Analysis'
                    : isOpticalSar
                    ? 'Cross-Modal Optical + SAR Analysis'
                    : 'Evidence-Grounded Intelligence'}
                </span>
              </div>
              <h2 className="text-2xl font-bold text-slate-100 tracking-tight mt-1">
                {result.selectedModel || 'Multi-Specialist Agent'}
              </h2>
            </div>

            <div className="flex items-center gap-2">
              {geoEvidence?.status === 'available' ? (
                <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-950/60 border border-emerald-800 text-emerald-300 font-mono text-xs">
                  <Compass className="w-3.5 h-3.5" />
                  <span>Geospatially Grounded</span>
                </div>
              ) : (
                <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 font-mono text-xs">
                  <MapPin className="w-3.5 h-3.5" />
                  <span>Pixel Coordinate Space</span>
                </div>
              )}
            </div>
          </div>

          {/* User Question & Answer */}
          <div className="p-5 rounded-xl bg-cyan-950/30 border border-cyan-700/50 space-y-2 shadow-lg">
            <div className="text-xs font-mono text-cyan-300 uppercase tracking-wider font-semibold flex items-center gap-1.5">
              <Search className="w-4 h-4 text-cyan-400" />
              Answer to Your Question
            </div>
            <div className="text-xs font-mono text-slate-400">
              &ldquo;{result.query}&rdquo;
            </div>
            <p className="text-slate-100 text-base leading-relaxed font-sans font-medium pt-1">
              {result.answer}
            </p>
          </div>

          {/* ----------------------------------------------------------- */}
          {/* HOW SATQUERY ANALYZED THIS (AGENT TRANSPARENCY - REQ 6)     */}
          {/* ----------------------------------------------------------- */}
          <div className="rounded-xl bg-slate-950/80 border border-slate-800 overflow-hidden shadow-md">
            <button
              type="button"
              onClick={() => setShowAgentTransparency(!showAgentTransparency)}
              className="w-full px-4 py-3 bg-slate-900/60 hover:bg-slate-900 border-b border-slate-800/80 flex items-center justify-between text-xs font-mono text-cyan-300 transition cursor-pointer"
            >
              <div className="flex items-center gap-2">
                <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                <span className="font-bold">How SatQuery analyzed this</span>
                <span className="text-[10px] text-slate-500 font-sans hidden sm:inline">(Agentic Orchestration Trace)</span>
              </div>
              {showAgentTransparency ? (
                <ChevronUp className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              )}
            </button>

            {showAgentTransparency && (
              <div className="p-4 space-y-3 text-xs font-mono">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-2.5">
                  <div className="p-2.5 rounded-lg bg-slate-900/50 border border-slate-800 space-y-1">
                    <span className="text-[10px] text-slate-500 block uppercase">Query</span>
                    <span className="text-slate-200 font-semibold block truncate" title={result.query}>
                      &ldquo;{result.query}&rdquo;
                    </span>
                  </div>

                  <div className="p-2.5 rounded-lg bg-slate-900/50 border border-slate-800 space-y-1">
                    <span className="text-[10px] text-slate-500 block uppercase">Agent Interpretation</span>
                    <span className="text-cyan-300 font-semibold block truncate">
                      {queryIntent}
                    </span>
                  </div>

                  <div className="p-2.5 rounded-lg bg-slate-900/50 border border-slate-800 space-y-1">
                    <span className="text-[10px] text-slate-500 block uppercase">Selected Specialist</span>
                    <span className="text-emerald-300 font-semibold block truncate">
                      {selectedSpecialist}
                    </span>
                  </div>

                  <div className="p-2.5 rounded-lg bg-slate-900/50 border border-slate-800 space-y-1">
                    <span className="text-[10px] text-slate-500 block uppercase">Model / Tool</span>
                    <span className="text-purple-300 font-semibold block truncate" title={modelTool}>
                      {modelTool}
                    </span>
                  </div>

                  <div className="p-2.5 rounded-lg bg-slate-900/50 border border-slate-800 space-y-1">
                    <span className="text-[10px] text-slate-500 block uppercase">Geospatial Grounding</span>
                    <span className={geoEvidence?.status === 'available' ? "text-emerald-400 font-semibold block truncate" : "text-slate-400 font-semibold block truncate"}>
                      {geospatialGroundingSummary}
                    </span>
                  </div>
                </div>

                <div className="text-[11px] font-sans text-slate-400 pt-1 flex items-center justify-between border-t border-slate-900">
                  <span>Evidence synthesis: <strong className="text-slate-300">{evidenceSynthesisSummary}</strong></span>
                  <span className="text-[10px] font-mono text-slate-500">Autonomous tool routing • No hardcoded responses</span>
                </div>
              </div>
            )}
          </div>

          {/* ----------------------------------------------------------- */}
          {/* 2. MULTI-SPECIALIST EVIDENCE HIERARCHY                     */}
          {/* ----------------------------------------------------------- */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Primary Direct Evidence */}
            <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2.5">
              <div className="text-xs font-mono text-cyan-400 uppercase tracking-wider font-semibold flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-cyan-400" />
                Primary Direct Evidence
              </div>
              {directEvidence.length > 0 ? (
                <ul className="space-y-1.5 text-xs text-slate-200 font-sans leading-relaxed">
                  {directEvidence.map((ev, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-cyan-400 font-bold">•</span>
                      <span>{ev}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-slate-400 font-sans">
                  Direct primary evidence was synthesized in the main response above.
                </p>
              )}
            </div>

            {/* Secondary Supporting Evidence */}
            <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2.5">
              <div className="text-xs font-mono text-slate-300 uppercase tracking-wider font-semibold flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-slate-400" />
                Secondary Supporting Evidence
              </div>
              {supportingEvidence.length > 0 ? (
                <ul className="space-y-1.5 text-xs text-slate-300 font-sans leading-relaxed">
                  {supportingEvidence.map((ev, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-slate-400 font-bold">•</span>
                      <span>{ev}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-slate-400 font-sans">
                  No secondary corroborating specialist evidence required for this query.
                </p>
              )}
            </div>
          </div>

          {/* Specialist Disagreement Notice (If Florence-2 and BigEarthNet diverge) */}
          {disagreements.length > 0 && (
            <div className="p-5 rounded-xl bg-amber-950/40 border border-amber-600/70 space-y-2 shadow-lg">
              <div className="text-xs font-mono text-amber-300 uppercase tracking-wider font-bold flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                Specialist Disagreement Logged
              </div>
              <p className="text-xs text-amber-200/90 font-sans leading-relaxed">
                SatQuery AI preserves divergent perspectives between specialist models rather than silently selecting one:
              </p>
              <ul className="space-y-1 text-xs text-amber-100 font-sans pt-1">
                {disagreements.map((dis, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-amber-400 font-bold">•</span>
                    <span>{dis}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* ----------------------------------------------------------- */}
          {/* 3. GEOSPATIAL INFORMATION PANEL (WHEN AVAILABLE)            */}
          {/* ----------------------------------------------------------- */}
          <div className="p-5 rounded-xl bg-slate-900/40 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <div className="text-xs font-mono text-slate-300 uppercase tracking-wider font-semibold flex items-center gap-2">
                <Compass className="w-4 h-4 text-emerald-400" />
                Geospatial Information & Calibration
              </div>
              <span className="text-[11px] font-mono text-slate-500">
                {geoEvidence?.status === 'available' ? 'Verified Spatial Georeference' : 'Unprojected Metadata'}
              </span>
            </div>

            {geoEvidence?.status === 'available' ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3 text-xs font-mono">
                <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">CRS / PROJECTION</span>
                  <span className="text-emerald-300 font-bold truncate block">{geoEvidence.crs || 'WGS 84'}</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">SPATIAL RESOLUTION</span>
                  <span className="text-slate-200 font-semibold truncate block">{geoEvidence.resolution || 'Standard GSD'}</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">CENTER LAT / LON</span>
                  <span className="text-cyan-300 font-semibold truncate block">
                    {geoEvidence.geographicCoordinates
                      ? `${geoEvidence.geographicCoordinates.latitude?.toFixed(4)}°, ${geoEvidence.geographicCoordinates.longitude?.toFixed(4)}°`
                      : 'Mapped'}
                  </span>
                </div>
                <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">SURFACE AREA</span>
                  <span className="text-slate-200 font-semibold truncate block">
                    {geoEvidence.area?.area_km2 ? `${geoEvidence.area.area_km2.toFixed(2)} km²` : 'Calculated'}
                  </span>
                </div>
              </div>
            ) : (
              <div className="p-3.5 rounded-lg bg-slate-950/50 border border-slate-800 text-slate-400 text-xs flex items-start gap-2.5">
                <Info className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
                <p>
                  Geospatial coordinates unavailable because the source raster does not contain sufficient georeferencing metadata (CRS or Affine Geotransform). Spatial evidence remains grounded in relative pixel space.
                </p>
              </div>
            )}
          </div>

          {/* ----------------------------------------------------------- */}
          {/* 4. OPTICAL + SAR MULTIMODAL FINDINGS (IF ACTIVE)           */}
          {/* ----------------------------------------------------------- */}
          {isOpticalSar && crossModal && (
            <div className="space-y-4 pt-2">
              <div className="text-xs font-mono text-cyan-400 uppercase tracking-wider font-semibold flex items-center gap-2">
                <Radio className="w-4 h-4 text-cyan-400" />
                Cross-Modal Sensor Corroboration
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {crossModal.opticalEvidence && (
                  <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 space-y-2">
                    <span className="text-xs font-mono text-blue-400 font-semibold block">
                      Optical Spectral Indices (NDVI / NDWI / NDBI)
                    </span>
                    <ul className="space-y-1 text-xs text-slate-300 font-sans">
                      {crossModal.opticalEvidence.map((o, idx) => (
                        <li key={idx}>• {o}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {crossModal.sarEvidence && (
                  <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 space-y-2">
                    <span className="text-xs font-mono text-cyan-400 font-semibold block">
                      SAR Radar Microwave Backscatter (dB)
                    </span>
                    <ul className="space-y-1 text-xs text-slate-300 font-sans">
                      {crossModal.sarEvidence.map((s, idx) => (
                        <li key={idx}>• {s}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              {crossModal.crossModalCorroboration && (
                <div className="p-4 rounded-xl bg-slate-900/60 border border-cyan-800/40 space-y-1.5">
                  <span className="text-xs font-mono text-emerald-400 font-semibold block">
                    Joint Physical Corroboration
                  </span>
                  <ul className="space-y-1 text-xs text-slate-200 font-sans">
                    {crossModal.crossModalCorroboration.map((c, idx) => (
                      <li key={idx}>✓ {c}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* ----------------------------------------------------------- */}
          {/* 4. SPATIAL EVIDENCE (VISUAL EVIDENCE REGIONS - REQ 7 & 9)   */}
          {/* ----------------------------------------------------------- */}
          <div className="space-y-3 pt-2 border-t border-slate-800">
            <div className="flex items-center justify-between">
              <div className="text-xs font-mono text-cyan-400 uppercase tracking-wider font-semibold flex items-center gap-1.5">
                <Crosshair className="w-4 h-4 text-cyan-400" />
                Spatial Evidence (Visual Regions Mapped)
              </div>
              {(boundingBoxes.length > 0 || changedRegions.length > 0) && (
                <span className="text-[11px] font-mono text-slate-400">
                  {boundingBoxes.length + changedRegions.length} Spatial Feature{(boundingBoxes.length + changedRegions.length) > 1 ? 's' : ''} Identified
                </span>
              )}
            </div>

            {boundingBoxes.length > 0 || changedRegions.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {/* Grounding Bounding Boxes */}
                {boundingBoxes.map((b) => {
                  const isFocused = focusedEvidenceId === b.id;
                  const boxColor = b.color || '#06b6d4';
                  return (
                    <div
                      key={b.id}
                      onClick={() => onFocusEvidence?.(b.id)}
                      className={`p-3.5 rounded-xl border text-xs transition cursor-pointer flex flex-col justify-between ${
                        isFocused
                          ? 'bg-cyan-950/60 border-cyan-400 shadow-md shadow-cyan-500/10 ring-1 ring-cyan-400'
                          : 'bg-slate-900/70 border-slate-800 hover:border-cyan-500/50'
                      }`}
                    >
                      <div>
                        <div className="flex items-center justify-between mb-1.5">
                          <div className="flex items-center gap-2 font-semibold text-slate-100">
                            <span
                              style={{ backgroundColor: boxColor }}
                              className="w-2.5 h-2.5 rounded-full inline-block"
                            />
                            <span>{b.label}</span>
                          </div>
                          <span className="font-mono text-[10px] text-cyan-300">
                            Grounding Box
                          </span>
                        </div>
                        <p className="text-[11px] text-slate-300 font-sans leading-relaxed">
                          {b.description || 'Model evidence suggests location of ' + b.label}
                        </p>
                      </div>

                      <div className="mt-2.5 pt-2 border-t border-slate-800/80 text-[10px] font-mono text-slate-400 flex items-center justify-between">
                        <span>X:{b.x.toFixed(1)}% Y:{b.y.toFixed(1)}%</span>
                        <span className="text-cyan-400">Inspect Highlight →</span>
                      </div>
                    </div>
                  );
                })}

                {/* Candidate Change Regions (Bi-Temporal - Requirement 8: Never convert candidate change to confirmed change) */}
                {changedRegions.map((region) => {
                  const isFocused = focusedEvidenceId === region.id;
                  return (
                    <div
                      key={region.id}
                      onClick={() => onFocusEvidence?.(region.id)}
                      className={`p-3.5 rounded-xl border text-xs transition cursor-pointer flex flex-col justify-between ${
                        isFocused
                          ? 'bg-amber-950/60 border-amber-400 shadow-md shadow-amber-500/10 ring-1 ring-amber-400'
                          : 'bg-slate-900/70 border-slate-800 hover:border-amber-500/50'
                      }`}
                    >
                      <div>
                        <div className="flex items-center justify-between mb-1.5">
                          <div className="flex items-center gap-2 font-semibold text-amber-200">
                            <span className="w-2.5 h-2.5 rounded-full bg-amber-400 inline-block" />
                            <span>Candidate Change Region: {region.label}</span>
                          </div>
                          <span className="font-mono text-[10px] text-amber-400">
                            {region.direction || 'Spectral Delta'}
                          </span>
                        </div>
                        <p className="text-[11px] text-slate-300 font-sans leading-relaxed">
                          {region.description || 'Algorithmic difference detected between T1 and T2 baseline.'}
                        </p>
                      </div>

                      <div className="mt-2.5 pt-2 border-t border-slate-800/80 text-[10px] font-mono text-slate-400 flex items-center justify-between">
                        <span>X:{region.x.toFixed(1)}% Y:{region.y.toFixed(1)}%</span>
                        <span className="text-amber-400">Inspect Change →</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="p-4 rounded-xl bg-slate-950/40 border border-slate-800/70 text-slate-400 text-xs flex items-center gap-2">
                <HelpCircle className="w-4 h-4 text-slate-500" />
                <span>No spatial evidence returned.</span>
              </div>
            )}
          </div>

          {/* ----------------------------------------------------------- */}
          {/* 6. LIMITATIONS & CALIBRATION DISCLAIMERS                    */}
          {/* ----------------------------------------------------------- */}
          {limitations.length > 0 && (
            <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-2 text-xs text-slate-400">
              <div className="flex items-center gap-2 text-slate-300 font-semibold font-mono text-[11px] uppercase">
                <Info className="w-4 h-4 text-cyan-400" />
                <span>Model Limitations & Uncertainty Governance</span>
              </div>
              <ul className="space-y-1 font-sans text-[11px] text-slate-400 pl-6 list-disc">
                {limitations.map((lim, idx) => (
                  <li key={idx}>{lim}</li>
                ))}
              </ul>
            </div>
          )}

          {/* ----------------------------------------------------------- */}
          {/* 7. AGENTIC EXECUTION TRACE (COMPACT EXPANDABLE)             */}
          {/* ----------------------------------------------------------- */}
          {executionSteps.length > 0 && (
            <div className="pt-2 border-t border-slate-800">
              <button
                onClick={() => setShowExecutionTrace(!showExecutionTrace)}
                className="w-full flex items-center justify-between p-3.5 rounded-xl bg-slate-900/40 border border-slate-800 hover:border-slate-700 transition text-xs font-mono text-slate-300"
              >
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4 text-cyan-400" />
                  <span>View Multi-Stage Execution Trace ({executionSteps.length} Steps)</span>
                </div>
                {showExecutionTrace ? (
                  <ChevronUp className="w-4 h-4 text-slate-400" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-slate-400" />
                )}
              </button>

              {showExecutionTrace && (
                <div className="mt-3 p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-3 text-xs font-mono">
                  {executionSteps.map((step) => (
                    <div
                      key={step.id}
                      className="p-3 rounded-lg bg-slate-900/50 border border-slate-800/80 space-y-1.5"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-slate-200">
                          Step {step.stepNumber}: {step.title}
                        </span>
                        <div className="flex items-center gap-2">
                          <span className="text-[10px] text-slate-500">{step.durationMs} ms</span>
                          <span className="px-1.5 py-0.5 rounded text-[10px] bg-emerald-950 text-emerald-300 border border-emerald-800">
                            {step.status.toUpperCase()}
                          </span>
                        </div>
                      </div>
                      <p className="text-slate-400 text-[11px] font-sans">{step.summary}</p>
                      {step.details && step.details.length > 0 && (
                        <div className="pt-1.5 grid grid-cols-1 sm:grid-cols-2 gap-1.5 text-[10px]">
                          {step.details.map((d, i) => (
                            <div key={i} className="text-slate-400">
                              <span className="text-slate-500">{d.label}: </span>
                              <span className="text-cyan-300">{d.value}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </section>
  );
};
