import React, { useState, useRef, useEffect } from 'react';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Maximize2,
  Minimize2,
  Eye,
  EyeOff,
  Image as ImageIcon,
  Compass,
  MapPin,
  Crosshair,
} from 'lucide-react';
import { BoundingBox, SpatialPolygon, SpatialPoint } from '../types';

interface SingleImageViewerProps {
  imageUrl: string;
  imageName?: string;
  resolution?: string;
  boundingBoxes?: BoundingBox[];
  polygons?: SpatialPolygon[];
  points?: SpatialPoint[];
  focusedEvidenceId?: string | null;
  onSelectEvidence?: (id: string) => void;
  geospatialMetadata?: any;
  geospatialEvidence?: any;
}

export const SingleImageViewer: React.FC<SingleImageViewerProps> = ({
  imageUrl,
  imageName = 'Uploaded Satellite Image',
  boundingBoxes = [],
  points = [],
  focusedEvidenceId,
  onSelectEvidence,
  geospatialMetadata,
  geospatialEvidence,
}) => {
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [showOverlay, setShowOverlay] = useState<boolean>(true);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [cursorHud, setCursorHud] = useState<{
    pixelX: number;
    pixelY: number;
    lat: number | null;
    lon: number | null;
    feature: string;
    evidenceSource: string;
  } | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const viewerWrapperRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const animFrameRef = useRef<number | null>(null);

  const handleImageCursorMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const clientX = e.clientX;
    const clientY = e.clientY;
    const currentTarget = e.currentTarget;

    if (animFrameRef.current !== null) {
      cancelAnimationFrame(animFrameRef.current);
    }

    animFrameRef.current = requestAnimationFrame(() => {
      const rect = currentTarget.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      const xPct = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
      const yPct = Math.max(0, Math.min(100, ((clientY - rect.top) / rect.height) * 100));

      let lat: number | null = null;
      let lon: number | null = null;
      const bounds = geospatialEvidence?.bounds || geospatialMetadata?.bounds;
      const isGeoAvailable = geospatialEvidence?.status === 'available' || geospatialMetadata?.isGeoreferenced;

      if (bounds && isGeoAvailable) {
        const north = bounds.north ?? bounds.max_lat;
        const south = bounds.south ?? bounds.min_lat;
        const east = bounds.east ?? bounds.max_lon;
        const west = bounds.west ?? bounds.min_lon;
        if (typeof north === 'number' && typeof south === 'number' && typeof east === 'number' && typeof west === 'number') {
          lat = north - (yPct / 100) * (north - south);
          lon = west + (xPct / 100) * (east - west);
        }
      }

      const hitBox = boundingBoxes.find(
        (b) => xPct >= b.x && xPct <= b.x + b.width && yPct >= b.y && yPct <= b.y + b.height
      );

      setCursorHud({
        pixelX: Math.round((xPct / 100) * (geospatialMetadata?.width || 1024)),
        pixelY: Math.round((yPct / 100) * (geospatialMetadata?.height || 1024)),
        lat,
        lon,
        feature: hitBox ? hitBox.label : 'Background / Unclassified',
        evidenceSource: hitBox ? 'Visual Model Grounding' : 'None',
      });
    });
  };

  const handleImageCursorLeave = () => {
    if (animFrameRef.current !== null) {
      cancelAnimationFrame(animFrameRef.current);
    }
    setCursorHud(null);
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY * -0.0015;
    setZoom((prev) => Math.min(Math.max(0.75, prev + delta), 4.5));
  };

  const resetTransform = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const toggleFullscreen = () => {
    if (!isFullscreen) {
      if (viewerWrapperRef.current?.requestFullscreen) {
        viewerWrapperRef.current.requestFullscreen().catch(() => {
          // Fallback if browser iframe policy blocks requestFullscreen
          setIsFullscreen(true);
        });
      } else {
        setIsFullscreen(true);
      }
    } else {
      if (document.fullscreenElement && document.exitFullscreen) {
        document.exitFullscreen().catch(() => {
          setIsFullscreen(false);
        });
      } else {
        setIsFullscreen(false);
      }
    }
  };

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isFullscreen) {
        setIsFullscreen(false);
      }
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isFullscreen]);

  const hasVisualEvidence = boundingBoxes.length > 0 || points.length > 0;

  return (
    <div
      ref={viewerWrapperRef}
      id="single-image-viewer"
      className={`rounded-2xl bg-[#080d1a] border border-slate-800 overflow-hidden shadow-2xl transition-all ${
        isFullscreen ? 'fixed inset-0 z-50 rounded-none border-none flex flex-col' : ''
      }`}
    >
      {/* Top Toolbar - Clean, Useful Controls Only */}
      <div className="px-4 py-3 bg-slate-900/90 border-b border-slate-800 flex items-center justify-between gap-3 text-xs">
        {/* Simple Image Label */}
        <div className="flex items-center gap-2 font-medium text-slate-200">
          <ImageIcon className="w-4 h-4 text-cyan-400 shrink-0" />
          <span className="truncate max-w-[200px] sm:max-w-md font-sans text-xs">
            {imageName}
          </span>
        </div>

        {/* Useful Controls: Zoom In, Zoom Out, Reset, Fullscreen */}
        <div className="flex items-center gap-2">
          {/* Visual evidence toggle if present */}
          {hasVisualEvidence && (
            <button
              onClick={() => setShowOverlay(!showOverlay)}
              className={`px-2.5 py-1.5 rounded-lg border text-xs font-mono transition flex items-center gap-1.5 ${
                showOverlay
                  ? 'bg-cyan-950/80 text-cyan-300 border-cyan-700/60'
                  : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
              }`}
              title="Toggle Visual Evidence Overlay"
            >
              {showOverlay ? <Eye className="w-3.5 h-3.5 text-cyan-400" /> : <EyeOff className="w-3.5 h-3.5" />}
              <span className="hidden sm:inline">{showOverlay ? 'Evidence: On' : 'Evidence: Off'}</span>
            </button>
          )}

          {/* Zoom In & Zoom Out */}
          <div className="flex items-center bg-slate-950 rounded-lg border border-slate-800 p-0.5">
            <button
              onClick={() => setZoom((z) => Math.max(0.75, z - 0.25))}
              className="p-1.5 hover:bg-slate-800 rounded text-slate-300 transition"
              title="Zoom out"
              aria-label="Zoom out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <span className="px-2 font-mono text-[11px] text-slate-400 min-w-[42px] text-center select-none">
              {Math.round(zoom * 100)}%
            </span>
            <button
              onClick={() => setZoom((z) => Math.min(4.5, z + 0.25))}
              className="p-1.5 hover:bg-slate-800 rounded text-slate-300 transition"
              title="Zoom in"
              aria-label="Zoom in"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
          </div>

          {/* Reset */}
          <button
            onClick={resetTransform}
            className="p-2 rounded-lg bg-slate-950 border border-slate-800 hover:bg-slate-800 text-slate-300 hover:text-white transition flex items-center gap-1"
            title="Reset"
            aria-label="Reset view"
          >
            <RotateCcw className="w-4 h-4" />
            <span className="text-[11px] font-sans hidden sm:inline">Reset</span>
          </button>

          {/* Fullscreen */}
          <button
            onClick={toggleFullscreen}
            className={`p-2 rounded-lg border transition flex items-center gap-1 ${
              isFullscreen
                ? 'bg-cyan-950 border-cyan-600 text-cyan-300'
                : 'bg-slate-950 border-slate-800 hover:bg-slate-800 text-slate-300 hover:text-white'
            }`}
            title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
            aria-label={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
          >
            {isFullscreen ? (
              <>
                <Minimize2 className="w-4 h-4 text-cyan-400" />
                <span className="text-[11px] font-sans hidden sm:inline">Exit</span>
              </>
            ) : (
              <>
                <Maximize2 className="w-4 h-4" />
                <span className="text-[11px] font-sans hidden sm:inline">Fullscreen</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main Large Image Viewer Area */}
      <div
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        className={`relative ${
          isFullscreen ? 'flex-1 h-full' : 'h-[440px] sm:h-[520px]'
        } w-full overflow-hidden bg-black flex items-center justify-center select-none ${
          isDragging ? 'cursor-grabbing' : 'cursor-grab'
        }`}
      >
        <div
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: 'center center',
            transition: isDragging ? 'none' : 'transform 0.1s ease-out',
          }}
          className="relative max-w-full max-h-full flex items-center justify-center"
        >
          {/* Main Uploaded Satellite Image */}
          <div
            onMouseMove={handleImageCursorMove}
            onMouseLeave={handleImageCursorLeave}
            className="relative pointer-events-auto cursor-crosshair"
          >
            <img
              src={imageUrl}
              alt={imageName}
              className={`${
                isFullscreen ? 'max-h-[85vh]' : 'max-h-[440px] sm:max-h-[520px]'
              } w-auto object-contain rounded shadow-2xl`}
              draggable={false}
            />

            {/* Visual Evidence Overlays */}
            {showOverlay && (
              <div className="absolute inset-0 pointer-events-none">
                {/* Bounding Boxes */}
                {boundingBoxes.map((b) => {
                  const isFocused = focusedEvidenceId === b.id;
                  const boxColor = b.color || '#22d3ee';
                  return (
                    <div
                      key={b.id}
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectEvidence?.(b.id);
                      }}
                      style={{
                        left: `${b.x}%`,
                        top: `${b.y}%`,
                        width: `${b.width}%`,
                        height: `${b.height}%`,
                        borderColor: boxColor,
                      }}
                      className={`absolute border-2 pointer-events-auto cursor-pointer transition-all duration-200 ${
                        isFocused
                          ? 'border-4 ring-4 ring-cyan-400/40 bg-cyan-500/25 z-30'
                          : 'hover:bg-cyan-500/15 z-20'
                      }`}
                    >
                      <div
                        style={{ backgroundColor: boxColor }}
                        className="absolute top-0 left-0 -translate-y-full px-1.5 py-0.5 text-[10px] font-mono font-bold text-black rounded-t flex items-center gap-1 shadow whitespace-nowrap"
                      >
                        <span>{b.label}</span>
                      </div>
                    </div>
                  );
                })}

                {/* Spatial Points */}
                {points.map((pt) => {
                  const isFocused = focusedEvidenceId === pt.id;
                  const ptX = pt.pixelX ?? 50;
                  const ptY = pt.pixelY ?? 50;
                  return (
                    <div
                      key={pt.id}
                      style={{ left: `${ptX}%`, top: `${ptY}%` }}
                      className={`absolute -translate-x-1/2 -translate-y-1/2 pointer-events-auto cursor-pointer z-20 transition-all ${
                        isFocused ? 'scale-150 z-30' : 'hover:scale-125'
                      }`}
                    >
                      <div className="relative flex items-center justify-center">
                        <span className="animate-ping absolute inline-flex h-5 w-5 rounded-full bg-emerald-400 opacity-75" />
                        <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-emerald-500 border-2 border-white shadow" />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Real-time Cursor HUD */}
        {cursorHud && (
          <div className="absolute top-3 left-3 z-30 bg-slate-950/90 backdrop-blur-md border border-cyan-500/40 rounded-xl px-4 py-2 text-xs font-mono shadow-2xl flex flex-wrap items-center gap-3.5 text-slate-200 pointer-events-none">
            <div className="flex items-center gap-1.5 text-cyan-400">
              <Crosshair className="w-3.5 h-3.5" />
              <span className="text-[10px] text-slate-400 uppercase">PIXEL:</span>
              <span className="font-bold">({cursorHud.pixelX}, {cursorHud.pixelY})</span>
            </div>
            <div className="flex items-center gap-1.5">
              <MapPin className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-[10px] text-slate-400 uppercase">LAT:</span>
              <span className={cursorHud.lat !== null ? "text-emerald-300 font-bold" : "text-amber-400 font-semibold"}>
                {cursorHud.lat !== null ? `${cursorHud.lat.toFixed(5)}° N` : 'Geolocation unavailable'}
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <Compass className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-[10px] text-slate-400 uppercase">LON:</span>
              <span className={cursorHud.lon !== null ? "text-emerald-300 font-bold" : "text-amber-400 font-semibold"}>
                {cursorHud.lon !== null ? `${cursorHud.lon.toFixed(5)}° E` : 'Geolocation unavailable'}
              </span>
            </div>
            <div className="flex items-center gap-1.5 border-l border-slate-800 pl-2.5">
              <span className="text-[10px] text-slate-400 uppercase">FEATURE:</span>
              <span className="text-cyan-300 font-bold">{cursorHud.feature}</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] text-slate-400 uppercase">SOURCE:</span>
              <span className="text-slate-300">{cursorHud.evidenceSource}</span>
            </div>
          </div>
        )}

        {/* Minimal Navigation Hint */}
        <div className="absolute bottom-3 left-3 bg-slate-950/80 backdrop-blur-sm border border-slate-800/80 rounded-lg px-3 py-1.5 text-[11px] font-sans text-slate-300 flex items-center gap-2 shadow-lg pointer-events-none">
          <span className="w-2 h-2 rounded-full bg-cyan-400" />
          <span>Click & drag to pan • Scroll to zoom</span>
        </div>

        {/* Spatial Evidence Status Hint */}
        {boundingBoxes.length === 0 && (
          <div className="absolute bottom-3 right-3 bg-slate-950/80 backdrop-blur-sm border border-slate-800/80 rounded-lg px-3 py-1.5 text-[11px] font-sans text-slate-400 pointer-events-none">
            No spatial evidence returned for this analysis.
          </div>
        )}
      </div>
    </div>
  );
};
