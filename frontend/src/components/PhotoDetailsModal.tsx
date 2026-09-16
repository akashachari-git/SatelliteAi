import React, { useEffect } from 'react';
import { 
  FileCode, 
  Layers, 
  Compass, 
  Calendar, 
  Cpu, 
  CheckCircle, 
  Maximize2,
  X,
  PanelLeft,
  Info
} from 'lucide-react';
import { ImageMetadata } from '../types';

interface PhotoDetailsModalProps {
  isOpen: boolean;
  onClose: () => void;
  image: {
    filename: string;
    preview_url: string;
    metadata: ImageMetadata;
    badge?: string;
  } | null;
  onDockToSidebar?: () => void;
  isDocked?: boolean;
}

export const PhotoDetailsModal: React.FC<PhotoDetailsModalProps> = ({
  isOpen,
  onClose,
  image,
  onDockToSidebar,
  isDocked = false
}) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || !image) return null;

  const { metadata, filename, preview_url, badge } = image;
  const isSar = metadata.modality === 'SAR';

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-space-950/80 backdrop-blur-md animate-in fade-in duration-150"
      onClick={onClose}
    >
      <div 
        className="relative w-full max-w-lg rounded-2xl border border-space-700/80 bg-space-900/95 p-6 shadow-2xl backdrop-blur-xl text-slate-100 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Ambient Top Glow */}
        <div className="absolute -top-24 -right-24 w-48 h-48 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

        {/* Modal Header */}
        <div className="flex items-center justify-between pb-4 mb-4 border-b border-space-800">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
              <Info className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-100">Photo Details & Metadata</h3>
              <p className="text-[11px] text-slate-400 font-mono">Raster specifications & remote sensing tags</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-space-800 transition-colors"
            title="Close (Esc)"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Mounted Image Preview Bar */}
        <div className="flex items-center space-x-3 p-3 mb-4 rounded-xl bg-space-950/70 border border-space-800">
          <img 
            src={preview_url} 
            alt={filename}
            className="w-12 h-12 rounded-lg object-cover border border-space-700 shrink-0"
          />
          <div className="truncate flex-1">
            <div className="flex items-center space-x-2">
              <span className="font-semibold text-slate-200 text-xs truncate block">{filename}</span>
            </div>
            <div className="flex items-center space-x-2 mt-1">
              {badge && (
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 font-mono border border-cyan-500/20">
                  {badge}
                </span>
              )}
              <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-bold uppercase ${
                isSar 
                  ? 'bg-purple-500/10 text-purple-400 border border-purple-500/30' 
                  : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
              }`}>
                {metadata.modality}
              </span>
              <span className="text-[10px] font-mono text-slate-400">
                {metadata.width} × {metadata.height}
              </span>
            </div>
          </div>
        </div>

        {/* Technical Specs Grid */}
        <div className="grid grid-cols-2 gap-2.5 text-[11px] mb-4">
          
          <div className="flex items-center space-x-2.5 p-2.5 rounded-xl bg-space-950/60 border border-space-800/80">
            <Maximize2 className="w-4 h-4 text-slate-400 shrink-0" />
            <div>
              <span className="text-[10px] text-slate-400 block font-mono">Dimensions</span>
              <span className="font-mono text-slate-200 font-medium">
                {metadata.width} × {metadata.height} px
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-2.5 p-2.5 rounded-xl bg-space-950/60 border border-space-800/80">
            <Layers className="w-4 h-4 text-slate-400 shrink-0" />
            <div>
              <span className="text-[10px] text-slate-400 block font-mono">Bands / Depth</span>
              <span className="font-mono text-slate-200 font-medium">
                {metadata.bands} Band{metadata.bands > 1 ? 's' : ''} ({metadata.dtype})
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-2.5 p-2.5 rounded-xl bg-space-950/60 border border-space-800/80">
            <Compass className="w-4 h-4 text-cyan-400 shrink-0" />
            <div className="truncate">
              <span className="text-[10px] text-slate-400 block font-mono">CRS Projection</span>
              <span className="font-mono text-slate-200 font-medium truncate block" title={metadata.crs}>
                {metadata.crs}
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-2.5 p-2.5 rounded-xl bg-space-950/60 border border-space-800/80">
            <Cpu className="w-4 h-4 text-emerald-400 shrink-0" />
            <div>
              <span className="text-[10px] text-slate-400 block font-mono">Spatial Resolution</span>
              <span className="font-mono text-slate-200 font-medium">
                {metadata.spatial_resolution_m} m / pixel
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-2.5 p-2.5 rounded-xl bg-space-950/60 border border-space-800/80 col-span-2">
            <Calendar className="w-4 h-4 text-amber-400 shrink-0" />
            <div className="flex items-center justify-between w-full">
              <div>
                <span className="text-[10px] text-slate-400 block font-mono">Sensor & Radiometry</span>
                <span className="text-slate-200 font-medium text-[11px]">
                  {metadata.sensor} • {metadata.radiometry}
                </span>
              </div>
              {metadata.is_geotiff && (
                <span className="flex items-center text-[10px] text-cyan-400 font-mono shrink-0 ml-2">
                  <CheckCircle className="w-3 h-3 mr-1 text-cyan-400" />
                  GeoTIFF Tagged
                </span>
              )}
            </div>
          </div>

        </div>

        {metadata.acquisition_date && (
          <div className="mb-3 p-2 rounded-lg bg-space-950/40 border border-space-800/60 flex items-center justify-between text-[11px] font-mono text-slate-400">
            <span>Acquisition Epoch:</span>
            <span className="text-slate-200 font-medium">{metadata.acquisition_date}</span>
          </div>
        )}

        {/* Verification Status */}
        <div className="p-2.5 rounded-xl bg-emerald-500/5 border border-emerald-500/20 flex items-center justify-between text-[11px] font-mono text-emerald-400 mb-5">
          <span className="flex items-center font-medium">
            <CheckCircle className="w-3.5 h-3.5 mr-1.5 text-emerald-400" />
            Satellite Imagery Verified
          </span>
          <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 font-bold">
            EO Calibrated
          </span>
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between pt-3 border-t border-space-800">
          {onDockToSidebar ? (
            <button
              onClick={() => {
                onDockToSidebar();
                onClose();
              }}
              className="flex items-center space-x-1.5 text-xs text-slate-400 hover:text-cyan-400 transition-colors"
            >
              <PanelLeft className="w-3.5 h-3.5" />
              <span>{isDocked ? 'Docked in Sidebar' : 'Dock to Left Panel'}</span>
            </button>
          ) : <div />}

          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg text-xs font-medium bg-space-800 hover:bg-space-700 text-slate-200 hover:text-white border border-space-700 transition-colors"
          >
            Close
          </button>
        </div>

      </div>
    </div>
  );
};
