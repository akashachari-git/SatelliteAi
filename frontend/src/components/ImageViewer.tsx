import React, { useState, useRef, useEffect } from 'react';
import { 
  ZoomIn, 
  ZoomOut, 
  RotateCcw, 
  Layers, 
  Eye, 
  EyeOff, 
  Sliders, 
  Maximize2,
  SlidersHorizontal,
  Compass
} from 'lucide-react';
import { EvidenceItem } from '../types';

interface ImageViewerProps {
  baseImageUrl?: string;
  evidenceItems?: EvidenceItem[];
  boundingBoxes?: Array<{
    box: [number, number, number, number];
    label: string;
    score?: number;
  }>;
  title?: string;
  subtitle?: string;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({
  baseImageUrl,
  evidenceItems = [],
  boundingBoxes = [],
  title = "Satellite View",
  subtitle = "Ready for Query"
}) => {
  const [zoom, setZoom] = useState(1);
  const [overlayOpacity, setOverlayOpacity] = useState<number>(1.0);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);

  // Find relevant evidence URLs
  const maskEvidence = evidenceItems.find(e => e.type === 'mask_overlay');
  const bboxEvidence = evidenceItems.find(e => e.type === 'bbox_overlay');
  const heatmapEvidence = evidenceItems.find(e => e.type === 'change_heatmap');
  const fusedEvidence = evidenceItems.find(e => e.type === 'optical_sar_fused');

  // Pick top display overlay raster
  const overlayUrl = fusedEvidence?.url || heatmapEvidence?.url || maskEvidence?.url;
  const bboxUrl = bboxEvidence?.url;

  // Active layer state: defaults to 'base' when no overlay exists yet
  const [activeLayer, setActiveLayer] = useState<string>(overlayUrl ? 'overlay' : 'base');

  // Automatically update active layer when an overlay is generated
  useEffect(() => {
    if (overlayUrl) {
      setActiveLayer('overlay');
    } else {
      setActiveLayer('base');
    }
  }, [overlayUrl]);

  // Determine which images are visible
  const showOverlay = activeLayer === 'overlay' && Boolean(overlayUrl);
  const showBase = Boolean(baseImageUrl) && (!showOverlay || overlayOpacity < 0.98);

  const hasBoxes = (boundingBoxes && boundingBoxes.length > 0) || Boolean(bboxUrl);

  const handleZoomIn = () => setZoom(prev => Math.min(Number((prev + 0.25).toFixed(2)), 3.5));
  const handleZoomOut = () => setZoom(prev => Math.max(Number((prev - 0.25).toFixed(2)), 0.5));
  const handleResetZoom = () => setZoom(1);

  return (
    <div className="relative rounded-2xl border border-space-800/80 bg-space-950/80 backdrop-blur-xl overflow-hidden shadow-2xl flex flex-col h-full min-h-[460px]">
      
      {/* Top Toolbar */}
      <div className="flex items-center justify-between px-5 py-3.5 bg-space-900/60 border-b border-space-800/80 backdrop-blur-md z-20">
        <div>
          <div className="flex items-center space-x-2">
            <div className="w-5 h-5 rounded-full bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center">
              <Compass className="w-3 h-3 text-cyan-400" />
            </div>
            <span className="text-xs font-bold text-slate-100 tracking-wide font-sans">{title}</span>
          </div>
          <span className="text-[11px] font-mono text-slate-400 block mt-0.5">{subtitle}</span>
        </div>

        {/* Layer Selector & Controls */}
        <div className="flex items-center space-x-2.5">
          {(evidenceItems.length > 0 || overlayUrl) && (
            <div className="flex items-center bg-space-950/80 rounded-lg p-0.5 border border-space-800 text-[11px] font-mono">
              <button
                onClick={() => setActiveLayer('base')}
                className={`px-2.5 py-1 rounded-md transition-all ${
                  activeLayer === 'base' ? 'bg-cyan-500 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Base
              </button>

              {overlayUrl && (
                <button
                  onClick={() => setActiveLayer('overlay')}
                  className={`px-2.5 py-1 rounded-md transition-all ${
                    activeLayer === 'overlay' ? 'bg-cyan-500 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Evidence Overlay
                </button>
              )}

              {hasBoxes && (
                <button
                  onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
                  className={`px-2.5 py-1 rounded-md transition-all flex items-center space-x-1 ${
                    showBoundingBoxes ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' : 'text-slate-400'
                  }`}
                >
                  <span>BBoxes</span>
                  {showBoundingBoxes ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                </button>
              )}
            </div>
          )}

          {/* Zoom controls */}
          <div className="flex items-center bg-space-950/90 rounded-lg p-0.5 border border-space-800 text-slate-400">
            <button 
              onClick={handleZoomOut}
              className="p-1.5 hover:text-cyan-300 transition-colors"
              title="Zoom Out"
              aria-label="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="text-[11px] font-mono px-2 text-slate-300 min-w-[42px] text-center select-none">
              {Math.round(zoom * 100)}%
            </span>
            <button 
              onClick={handleZoomIn}
              className="p-1.5 hover:text-cyan-300 transition-colors"
              title="Zoom In"
              aria-label="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button 
              onClick={handleResetZoom}
              className="p-1.5 hover:text-cyan-300 transition-colors border-l border-space-800"
              title="Reset Zoom"
              aria-label="Reset Zoom"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Main Canvas Area */}
      <div className="relative flex-1 overflow-hidden flex items-center justify-center p-4 bg-geo-grid select-none">
        
        {!baseImageUrl && !overlayUrl ? (
          <div className="text-center p-8 max-w-sm mx-auto">
            {/* 3D Isometric Stacked Layers Graphic */}
            <div className="relative w-16 h-16 mx-auto mb-4 flex items-center justify-center">
              <svg 
                className="w-16 h-16 text-slate-700/80 drop-shadow" 
                viewBox="0 0 64 64" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="2"
              >
                {/* Top Layer */}
                <path d="M32 8L54 20L32 32L10 20L32 8Z" stroke="currentColor" strokeLinejoin="round" />
                {/* Middle Layer */}
                <path d="M10 28L32 40L54 28" stroke="currentColor" strokeLinejoin="round" />
                {/* Bottom Layer */}
                <path d="M10 36L32 48L54 36" stroke="currentColor" strokeLinejoin="round" />
                {/* Subtle Cyan Geospatial Guides */}
                <path d="M22 36V45M42 36V45" stroke="#06b6d4" strokeWidth="1" strokeOpacity="0.3" strokeDasharray="2 2" />
                <path d="M22 45H42" stroke="#06b6d4" strokeWidth="1" strokeOpacity="0.25" />
              </svg>
            </div>
            <p className="text-sm font-semibold text-slate-200 tracking-wide font-sans">
              No remote-sensing image loaded
            </p>
            <p className="text-xs text-slate-400 mt-1.5 font-sans leading-relaxed">
              Upload a GeoTIFF or select a scenario from the top menu.
            </p>
          </div>
        ) : (
          <div 
            className="relative transition-transform duration-150 ease-out max-w-full max-h-full inline-grid place-items-center"
            style={{ transform: `scale(${zoom})` }}
          >
            {/* Base Image: rendered when activeLayer === 'base' OR no overlay exists OR when user fades overlay below 98% */}
            {showBase && (
              <img 
                src={baseImageUrl} 
                alt="Satellite Base" 
                style={{ gridArea: '1 / 1 / 2 / 2' }}
                className="max-w-full max-h-[480px] w-auto h-auto object-contain rounded-lg shadow-2xl border border-space-800 pointer-events-none select-none block"
              />
            )}

            {/* Evidence Overlay Mask */}
            {showOverlay && overlayUrl && (
              <img 
                src={overlayUrl} 
                alt="Evidence Overlay" 
                style={{ 
                  gridArea: '1 / 1 / 2 / 2',
                  opacity: overlayOpacity 
                }}
                className="max-w-full max-h-[480px] w-auto h-auto object-contain rounded-lg shadow-2xl border border-space-800 pointer-events-none select-none block transition-opacity duration-150"
              />
            )}

            {/* Vector Bounding Box Overlay */}
            {showBoundingBoxes && boundingBoxes && boundingBoxes.length > 0 && (
              <div 
                style={{ gridArea: '1 / 1 / 2 / 2' }}
                className="w-full h-full relative pointer-events-none overflow-hidden rounded-lg z-10"
              >
                {boundingBoxes.map((b: any, idx: number) => {
                  const rawBox: any = b?.box;
                  const [ymin, xmin, ymax, xmax] = Array.isArray(rawBox)
                    ? rawBox
                    : [rawBox?.ymin ?? 0, rawBox?.xmin ?? 0, rawBox?.ymax ?? 0, rawBox?.xmax ?? 0];
                  return (
                    <div
                      key={idx}
                      style={{
                        top: `${Math.max(0, ymin * 100)}%`,
                        left: `${Math.max(0, xmin * 100)}%`,
                        width: `${Math.min(100, (xmax - xmin) * 100)}%`,
                        height: `${Math.min(100, (ymax - ymin) * 100)}%`,
                      }}
                      className="absolute border-2 border-cyan-400 bg-cyan-400/15 transition-all shadow-lg rounded-sm"
                    >
                      <div className="absolute -top-5 left-0 px-1.5 py-0.5 rounded bg-cyan-500 text-black font-mono text-[9px] font-bold whitespace-nowrap shadow-md">
                        {b.label} {b.score ? `(${Math.round(b.score * 100)}%)` : ''}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* Crosshair Center HUD */}
        <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
          <div className="w-6 h-6 border-t border-l border-cyan-500/20"></div>
          <div className="w-6 h-6 border-t border-r border-cyan-500/20"></div>
          <div className="w-6 h-6 border-b border-l border-cyan-500/20"></div>
          <div className="w-6 h-6 border-b border-r border-cyan-500/20"></div>
        </div>
      </div>

      {/* Bottom Opacity Bar */}
      {overlayUrl && activeLayer === 'overlay' && baseImageUrl && (
        <div className="px-4 py-2 bg-space-900/90 border-t border-space-800 flex items-center justify-between text-xs z-20">
          <div className="flex items-center space-x-2 text-slate-400">
            <SlidersHorizontal className="w-3.5 h-3.5 text-cyan-400" />
            <span className="font-mono text-[11px]">Overlay Opacity</span>
          </div>
          <div className="flex items-center space-x-3 w-48">
            <input 
              type="range" 
              min="0.1" 
              max="1.0" 
              step="0.05"
              value={overlayOpacity}
              onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
              className="w-full accent-cyan-400 h-1 bg-space-800 rounded-lg appearance-none cursor-pointer"
            />
            <span className="font-mono text-[11px] text-slate-300 w-8">
              {Math.round(overlayOpacity * 100)}%
            </span>
          </div>
        </div>
      )}

    </div>
  );
};
