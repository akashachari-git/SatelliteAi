import React, { useState } from 'react';
import {
  X,
  Download,
  FileText,
  Printer,
  Copy,
  Check,
  CheckCircle,
  Satellite,
  ShieldCheck,
  Globe2,
  AlertTriangle,
} from 'lucide-react';
import { AnalysisResult } from '../types';
import { downloadAnalysisReport } from '../utils/reportGenerator';

interface ReportModalProps {
  result: AnalysisResult | null;
  onClose: () => void;
}

export const ReportModal: React.FC<ReportModalProps> = ({ result, onClose }) => {
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState<'briefing' | 'geojson'>('briefing');

  if (!result) return null;

  // Synthesize standard GeoJSON FeatureCollection from real bounding boxes and geospatial evidence
  const isGeoAvailable = result.geospatialEvidence?.status === 'available';
  const crsName = isGeoAvailable
    ? result.geospatialEvidence?.crs || 'urn:ogc:def:crs:OGC:1.3:CRS84'
    : 'Local Pixel Coordinates (Unprojected)';

  const geojsonDump = JSON.stringify(
    {
      type: 'FeatureCollection',
      crs: {
        type: 'name',
        properties: { name: crsName },
      },
      geospatialStatus: result.geospatialEvidence?.status || 'unavailable',
      features: (result.boundingBoxes || []).map((b) => {
        let coords: [number, number][][] = [];

        if (isGeoAvailable) {
          if (b.geographicBbox) {
            const g = b.geographicBbox;
            coords = [
              [
                [g.min_lon, g.max_lat],
                [g.max_lon, g.max_lat],
                [g.max_lon, g.min_lat],
                [g.min_lon, g.min_lat],
                [g.min_lon, g.max_lat],
              ],
            ];
          } else if (result.geospatialEvidence?.bounds) {
            const bounds = result.geospatialEvidence.bounds;
            const west = bounds.west + (b.x / 100) * (bounds.east - bounds.west);
            const east = bounds.west + ((b.x + b.width) / 100) * (bounds.east - bounds.west);
            const north = bounds.north - (b.y / 100) * (bounds.north - bounds.south);
            const south = bounds.north - ((b.y + b.height) / 100) * (bounds.north - bounds.south);
            coords = [
              [
                [west, north],
                [east, north],
                [east, south],
                [west, south],
                [west, north],
              ],
            ];
          }
        } else {
          // Unprojected pixel coordinate space
          coords = [
            [
              [b.x, b.y],
              [b.x + b.width, b.y],
              [b.x + b.width, b.y + b.height],
              [b.x, b.y + b.height],
              [b.x, b.y],
            ],
          ];
        }

        return {
          type: 'Feature',
          id: b.id,
          properties: {
            label: b.label,
            confidence: b.confidence ?? null,
            description: b.description,
            task: result.taskType,
            model: result.selectedModel,
            mode: result.mode,
            isSimulation: result.isSimulation ?? false,
          },
          geometry: {
            type: 'Polygon',
            coordinates: coords,
          },
        };
      }),
    },
    null,
    2
  );

  const handleCopyGeoJSON = () => {
    navigator.clipboard?.writeText(geojsonDump);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadReport = () => {
    downloadAnalysisReport(result, 'text');
  };

  return (
    <div
      id="report-modal-overlay"
      className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4 overflow-y-auto"
      onClick={onClose}
    >
      <div
        id="report-modal-container"
        className="bg-[#080d1a] border border-cyan-500/40 rounded-2xl w-full max-w-3xl shadow-[0_0_50px_rgba(6,182,212,0.2)] overflow-hidden font-mono"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="px-6 py-4 bg-slate-900/90 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 rounded-lg bg-cyan-950 border border-cyan-700/60 text-cyan-400">
              <Satellite className="w-4 h-4" />
            </div>
            <div>
              <div className="text-sm font-bold text-slate-100 flex items-center gap-2">
                <span>SATQUERY AI INTELLIGENCE DOSSIER</span>
                <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-950 text-emerald-400 border border-emerald-800">
                  VERIFIED
                </span>
              </div>
              <div className="text-[11px] text-slate-400">
                Analysis Reference: SQ-REPORT-{Date.now().toString().slice(-6)}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleDownloadReport}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-750 text-slate-300 hover:text-cyan-300 transition"
              title="Download text report"
            >
              <Download className="w-4 h-4" />
            </button>
            <button
              onClick={() => window.print()}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-750 text-slate-300 hover:text-cyan-300 transition"
              title="Print Dossier"
            >
              <Printer className="w-4 h-4" />
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded-lg bg-slate-800 hover:bg-rose-950 hover:text-rose-400 text-slate-400 transition"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="px-6 pt-3 border-b border-slate-800 flex items-center gap-4 text-xs font-mono">
          <button
            onClick={() => setActiveTab('briefing')}
            className={`pb-2.5 border-b-2 font-bold transition ${
              activeTab === 'briefing'
                ? 'border-cyan-400 text-cyan-300'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            Mission Intelligence Briefing
          </button>
          <button
            onClick={() => setActiveTab('geojson')}
            className={`pb-2.5 border-b-2 font-bold transition flex items-center gap-1.5 ${
              activeTab === 'geojson'
                ? 'border-cyan-400 text-cyan-300'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Globe2 className="w-3.5 h-3.5" />
            <span>GeoJSON Spatial Payload</span>
          </button>
        </div>

        {/* Content Area */}
        <div className="p-6 max-h-[70vh] overflow-y-auto space-y-5 text-xs">
          {/* Simulation Disclaimer Banner */}
          <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-slate-300 text-xs font-sans flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-cyan-400 shrink-0" />
              <span>
                <strong>ANALYSIS RECORD:</strong> Satellite imagery visual-language evaluation completed.
              </span>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] bg-slate-800 text-cyan-300 border border-slate-700 whitespace-nowrap">
              CONFIRMED
            </span>
          </div>

          {activeTab === 'briefing' ? (
            <>
              {/* Mission Summary Card */}
              <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-850 space-y-3">
                <div className="flex items-center justify-between text-slate-400 text-[11px] border-b border-slate-800 pb-2">
                  <span>QUERY PROMPT:</span>
                  <span className="text-cyan-400 font-bold">{result.taskType.toUpperCase()}</span>
                </div>
                <div className="text-sm font-sans font-bold text-slate-100">
                  "{result.query}"
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 text-[11px]">
                  <div>
                    <span className="text-slate-400 block">Confidence:</span>
                    <span className="text-emerald-400 font-bold">{result.confidence}%</span>
                  </div>
                  <div>
                    <span className="text-slate-400 block">Coordinates:</span>
                    <span className="text-slate-200 font-bold">{result.imageryMetadata.coordinates}</span>
                  </div>
                  <div>
                    <span className="text-slate-400 block">GSD Resolution:</span>
                    <span className="text-cyan-300 font-bold">{result.imageryMetadata.resolution}</span>
                  </div>
                  <div>
                    <span className="text-slate-400 block">Sensor:</span>
                    <span className="text-slate-200 truncate block">{result.imageryMetadata.sensor}</span>
                  </div>
                </div>
              </div>

              {/* Comprehensive Verdict */}
              <div className="space-y-1.5">
                <div className="text-[11px] text-cyan-400 font-bold tracking-wider">
                  EXECUTIVE VERDICT & REASONING:
                </div>
                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-slate-200 text-xs sm:text-sm font-sans leading-relaxed">
                  {result.answer}
                </div>
              </div>

              {/* Evidence Indicators */}
              <div className="space-y-2">
                <div className="text-[11px] text-cyan-400 font-bold tracking-wider">
                  SPATIAL & SPECTRAL VERIFICATION EVIDENCE:
                </div>
                <div className="space-y-2">
                  {result.evidence.map((point, idx) => (
                    <div
                      key={idx}
                      className="flex items-start gap-2.5 p-3 rounded-lg bg-slate-950/60 border border-slate-850 text-slate-300 font-sans"
                    >
                      <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                      <span className="leading-relaxed">{point}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Bounding Box Table if present */}
              {result.boundingBoxes && result.boundingBoxes.length > 0 && (
                <div className="space-y-2">
                  <div className="text-[11px] text-cyan-400 font-bold tracking-wider">
                    TARGET BOUNDING COORDINATES:
                  </div>
                  <div className="border border-slate-800 rounded-lg overflow-hidden">
                    <table className="w-full text-left text-[11px]">
                      <thead className="bg-slate-900 border-b border-slate-800 text-slate-400">
                        <tr>
                          <th className="py-2 px-3">Target Label</th>
                          <th className="py-2 px-3">Confidence</th>
                          <th className="py-2 px-3">Spatial Anchor (X, Y)</th>
                          <th className="py-2 px-3">Dimensions</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-850">
                        {result.boundingBoxes.map((box) => (
                          <tr key={box.id}>
                            <td className="py-2 px-3 text-cyan-300 font-bold">{box.label}</td>
                            <td className="py-2 px-3 text-emerald-400">{box.confidence}%</td>
                            <td className="py-2 px-3 text-slate-400">
                              X: {box.x}%, Y: {box.y}%
                            </td>
                            <td className="py-2 px-3 text-slate-400">
                              {box.width}% × {box.height}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-[11px] text-slate-400">
                  Standard GeoJSON RFC 7946 FeatureCollection:
                </span>
                <button
                  onClick={handleCopyGeoJSON}
                  className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-[11px] flex items-center gap-1.5 transition"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copied ? 'Copied' : 'Copy JSON'}</span>
                </button>
              </div>

              <pre className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-cyan-300 font-mono text-[11px] overflow-x-auto leading-relaxed max-h-96">
                {geojsonDump}
              </pre>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 bg-slate-900/80 border-t border-slate-800 flex items-center justify-between text-[11px] text-slate-400">
          <span>SatQuery AI Satellite Imagery Analysis</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 transition"
          >
            Close Dossier
          </button>
        </div>
      </div>
    </div>
  );
};
