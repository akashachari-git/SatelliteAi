import React from 'react';
import { 
  FileCode, 
  Layers, 
  Compass, 
  Calendar, 
  Cpu, 
  CheckCircle, 
  HardDrive,
  Maximize2,
  X
} from 'lucide-react';
import { ImageMetadata } from '../types';

interface MetadataCardProps {
  metadata: ImageMetadata;
  title?: string;
  badge?: string;
  onClose?: () => void;
}

export const MetadataCard: React.FC<MetadataCardProps> = ({ 
  metadata, 
  title = "Raster Metadata",
  badge,
  onClose
}) => {
  const isSar = metadata.modality === 'SAR';

  return (
    <div className="rounded-xl border border-space-700/80 bg-space-900/60 p-4 backdrop-blur-sm shadow-md animate-in fade-in duration-200">
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-space-800">
        <div className="flex items-center space-x-2 truncate">
          <FileCode className="w-4 h-4 text-cyan-400 shrink-0" />
          <span className="text-xs font-semibold text-slate-200 truncate">{title}</span>
        </div>
        <div className="flex items-center space-x-1.5 shrink-0">
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
          {onClose && (
            <button
              onClick={onClose}
              className="p-1 rounded-md text-slate-400 hover:text-white hover:bg-space-800 transition-colors ml-1"
              title="Close details"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2.5 text-[11px]">
        
        <div className="flex items-center space-x-2 p-2 rounded-lg bg-space-950/60 border border-space-800/80">
          <Maximize2 className="w-3.5 h-3.5 text-slate-400 shrink-0" />
          <div>
            <span className="text-[10px] text-slate-400 block font-mono">Dimensions</span>
            <span className="font-mono text-slate-200 font-medium">
              {metadata.width} × {metadata.height} px
            </span>
          </div>
        </div>

        <div className="flex items-center space-x-2 p-2 rounded-lg bg-space-950/60 border border-space-800/80">
          <Layers className="w-3.5 h-3.5 text-slate-400 shrink-0" />
          <div>
            <span className="text-[10px] text-slate-400 block font-mono">Bands / Depth</span>
            <span className="font-mono text-slate-200 font-medium">
              {metadata.bands} Band{metadata.bands > 1 ? 's' : ''} ({metadata.dtype})
            </span>
          </div>
        </div>

        <div className="flex items-center space-x-2 p-2 rounded-lg bg-space-950/60 border border-space-800/80">
          <Compass className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
          <div className="truncate">
            <span className="text-[10px] text-slate-400 block font-mono">CRS Projection</span>
            <span className="font-mono text-slate-200 font-medium truncate block" title={metadata.crs}>
              {metadata.crs}
            </span>
          </div>
        </div>

        <div className="flex items-center space-x-2 p-2 rounded-lg bg-space-950/60 border border-space-800/80">
          <Cpu className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          <div>
            <span className="text-[10px] text-slate-400 block font-mono">Spatial Resolution</span>
            <span className="font-mono text-slate-200 font-medium">
              {metadata.spatial_resolution_m} m / pixel
            </span>
          </div>
        </div>

        <div className="flex items-center space-x-2 p-2 rounded-lg bg-space-950/60 border border-space-800/80 col-span-2">
          <Calendar className="w-3.5 h-3.5 text-amber-400 shrink-0" />
          <div className="flex items-center justify-between w-full">
            <div>
              <span className="text-[10px] text-slate-400 block font-mono">Sensor & Radiometry</span>
              <span className="text-slate-200 font-medium">
                {metadata.sensor} • {metadata.radiometry}
              </span>
            </div>
            {metadata.is_geotiff && (
              <span className="flex items-center text-[10px] text-cyan-400 font-mono">
                <CheckCircle className="w-3 h-3 mr-1 text-cyan-400" />
                GeoTIFF Tagged
              </span>
            )}
          </div>
        </div>

      </div>

      {metadata.acquisition_date && (
        <div className="mt-2.5 pt-2 border-t border-space-800/60 flex items-center justify-between text-[11px] font-mono text-slate-400">
          <span>Acquisition Epoch:</span>
          <span className="text-slate-200">{metadata.acquisition_date}</span>
        </div>
      )}

      <div className="mt-2.5 pt-2 border-t border-space-800/60 flex items-center justify-between text-[11px] font-mono text-emerald-400">
        <span className="flex items-center">
          <CheckCircle className="w-3.5 h-3.5 mr-1.5 text-emerald-400" />
          Satellite Imagery Verified
        </span>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-300">
          EO Calibrated
        </span>
      </div>
    </div>
  );
};
