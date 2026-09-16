import React, { useState, useRef, useEffect } from 'react';
import { 
  GitCompare, 
  ArrowLeftRight, 
  Layers, 
  Activity, 
  Calendar, 
  Send, 
  Loader2, 
  TrendingUp, 
  PieChart, 
  MapPin,
  Sparkles,
  UploadCloud,
  ShieldCheck,
  Info,
  User,
  RotateCcw,
  FileText
} from 'lucide-react';
import { SwipeViewer } from '../components/SwipeViewer';
import { ReportModal } from '../components/ReportModal';
import { AnalysisResult } from '../types';
import { api } from '../services/api';

const FormattedMessage: React.FC<{ content: string }> = ({ content }) => {
  const renderInline = (text: string) => {
    const parts = text.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={i} className="text-white font-semibold">
            {part.slice(2, -2)}
          </strong>
        );
      }
      return part;
    });
  };

  const lines = content.split('\n');

  return (
    <div className="space-y-2 text-xs sm:text-sm text-slate-200 leading-relaxed font-sans">
      {lines.map((line, idx) => {
        const trimmed = line.trim();
        if (!trimmed) return <div key={idx} className="h-1" />;

        const isBullet = trimmed.startsWith('•') || trimmed.startsWith('- ') || trimmed.startsWith('* ');
        if (isBullet) {
          const bulletText = trimmed.replace(/^[•\-\*]\s*/, '');
          return (
            <div key={idx} className="flex items-start space-x-2 pl-1.5 py-0.5">
              <span className="text-amber-400 font-bold text-xs mt-0.5 shrink-0">•</span>
              <span className="text-slate-300 leading-relaxed">{renderInline(bulletText)}</span>
            </div>
          );
        }

        return (
          <p key={idx} className="text-slate-200 leading-relaxed">
            {renderInline(line)}
          </p>
        );
      })}
    </div>
  );
};

interface StudioChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  confidence?: number;
  stats?: any;
}

interface ChangeStudioPageProps {
  currentAnalysis: AnalysisResult | null;
  setCurrentAnalysis: (res: AnalysisResult | null) => void;
  activeImages: any[];
  setActiveImages: React.Dispatch<React.SetStateAction<any[]>>;
  onLoadChangeDemo: () => void;
}

export const ChangeStudioPage: React.FC<ChangeStudioPageProps> = ({
  currentAnalysis,
  setCurrentAnalysis,
  activeImages,
  setActiveImages,
  onLoadChangeDemo
}) => {
  const [query, setQuery] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<'swipe' | 'heatmap' | 'mask'>('swipe');
  const [messages, setMessages] = useState<StudioChatMessage[]>([]);
  const [isReportModalOpen, setIsReportModalOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Check if we have 2 images
  const hasTwoImages = activeImages.length >= 2;
  const beforeImg = activeImages[0];
  const afterImg = activeImages[1];

  // Synchronize initial or external analysis into chat stream
  useEffect(() => {
    if (currentAnalysis?.answer) {
      setMessages(prev => {
        const alreadyExists = prev.some(m => m.content === currentAnalysis.answer);
        if (alreadyExists) return prev;
        return [
          ...prev,
          {
            id: Date.now().toString(),
            role: 'assistant',
            content: currentAnalysis.answer,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            confidence: currentAnalysis.confidence?.percentage,
            stats: currentAnalysis.raw_result?.change_statistics || currentAnalysis.raw_result
          }
        ];
      });
    }
  }, [currentAnalysis]);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isProcessing]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setIsUploading(true);
      setUploadProgress(0);
      setUploadError(null);
      const res = await api.uploadImage(file, (percent) => {
        setUploadProgress(percent);
      });
      setActiveImages(prev => [...prev, {
        filename: res.filename,
        server_path: res.server_path,
        preview_url: res.preview_url,
        metadata: res.metadata
      }]);
    } catch (err: any) {
      setUploadError(err.message || 'File upload error');
    } finally {
      setIsUploading(false);
      setUploadProgress(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleRemoveImage = (index: number) => {
    setActiveImages(prev => prev.filter((_, i) => i !== index));
    setCurrentAnalysis(null);
    setMessages([]);
  };

  const handleRunChangeQuery = async (customQ?: string) => {
    if (!hasTwoImages) return;
    const q = (customQ || query).trim();
    if (!q || isProcessing) return;

    const userMsg: StudioChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: q,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMsg]);
    setQuery('');
    setIsProcessing(true);

    try {
      const paths = [beforeImg.server_path, afterImg.server_path];
      const metas = [beforeImg.metadata, afterImg.metadata];
      const res = await api.analyzeQuery(q, paths, metas);
      setCurrentAnalysis(res);

      const assistantMsg: StudioChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: res.answer,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        confidence: res.confidence?.percentage,
        stats: res.raw_result?.change_statistics || res.raw_result
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (err: any) {
      const errorMsg: StudioChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Error during change analysis: ${err.message || 'Server error'}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsProcessing(false);
    }
  };

  const changeStats = currentAnalysis?.raw_result?.change_statistics || currentAnalysis?.raw_result;
  const heatmapUrl = currentAnalysis?.evidence?.find(e => e.type === 'change_heatmap')?.url;
  const maskUrl = currentAnalysis?.evidence?.find(e => e.type === 'mask_overlay')?.url;

  const renderInputImageryCard = () => (
    <div className="rounded-2xl border border-space-700/80 bg-space-900/70 p-4 shadow-xl space-y-3">
      <div className="flex items-center justify-between pb-3 border-b border-space-800">
        <span className="text-xs font-bold text-white uppercase tracking-wider font-mono">
          Input Imagery
        </span>
        <span className="text-[10px] font-mono text-slate-400">
          GeoTIFF / PNG / TIFF
        </span>
      </div>

      {/* Upload Zone */}
      <div 
        onClick={() => !isUploading && fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-colors bg-space-950/60 group ${
          isUploading ? 'border-amber-500/60 cursor-wait' : 'border-space-700 hover:border-amber-500/50'
        }`}
      >
        <input 
          ref={fileInputRef} 
          type="file" 
          accept=".tif,.tiff,.geotiff,.png,.jpg,.jpeg" 
          onChange={handleFileUpload} 
          className="hidden" 
        />
        {isUploading ? (
          <div className="py-1 space-y-2.5 animate-in fade-in duration-150">
            <Loader2 className="w-7 h-7 mx-auto text-amber-400 animate-spin" />
            <p className="text-xs font-semibold text-amber-300">
              {uploadProgress !== null && uploadProgress < 100 
                ? `Uploading Image (${uploadProgress}%)...` 
                : 'Validating Satellite Raster...'}
            </p>
            <div className="w-3/4 mx-auto bg-space-800 rounded-full h-1.5 overflow-hidden">
              <div 
                className="bg-gradient-to-r from-amber-500 to-orange-500 h-1.5 rounded-full transition-all duration-150"
                style={{ width: `${uploadProgress ?? 100}%` }}
              />
            </div>
            <p className="text-[10px] text-slate-400 font-mono">
              High-speed client compression & raster ingestion
            </p>
          </div>
        ) : (
          <>
            <UploadCloud className="w-7 h-7 mx-auto text-slate-400 group-hover:text-amber-400 transition-colors mb-2" />
            <p className="text-xs font-medium text-slate-200">
              Upload Satellite Image
            </p>
            <p className="text-[10px] text-slate-400 mt-1 font-mono">
              GeoTIFF with CRS or standard RGB
            </p>
            <div className="mt-2.5 pt-2 border-t border-space-800/80 flex items-center justify-center space-x-1.5 text-[10.5px] font-mono text-amber-400">
              <ShieldCheck className="w-3.5 h-3.5 text-amber-400 shrink-0" />
              <span>Satellite Images Only (Non-EO Auto-Rejected)</span>
            </div>
          </>
        )}
      </div>

      {uploadError && (
        <div className="text-[11px] font-mono text-rose-400 bg-rose-500/10 border border-rose-500/20 p-2 rounded-lg">
          {uploadError}
        </div>
      )}

      {/* Active Image Thumbnails List */}
      {activeImages.length > 0 && (
        <div className="pt-2 border-t border-space-800/80 space-y-2">
          <span className="text-[10px] font-mono uppercase text-slate-400 block font-semibold">
            Mounted Rasters ({activeImages.length}/2):
          </span>
          {activeImages.map((img: any, idx: number) => {
            const badge = idx === 0 ? 'T1 (Reference)' : 'T2 (Monitoring)';
            return (
              <div key={idx} className="flex items-center justify-between p-2 rounded-lg bg-space-950 border border-space-800 text-xs">
                <div className="flex items-center space-x-2 truncate">
                  <img 
                    src={img.preview_url} 
                    alt="thumb" 
                    className="w-8 h-8 rounded object-cover border border-space-700 shrink-0" 
                  />
                  <div className="truncate">
                    <span className="font-medium text-slate-200 block truncate text-[11px]">
                      {img.filename}
                    </span>
                    <span className="text-[10px] font-mono text-amber-400">
                      {badge} • {img.metadata.width}×{img.metadata.height}
                    </span>
                  </div>
                </div>

                <button
                  onClick={() => handleRemoveImage(idx)}
                  className="text-slate-400 hover:text-rose-400 text-xs p-1 rounded hover:bg-rose-500/10 transition-colors ml-2 shrink-0"
                  title="Remove image"
                >
                  ×
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      
      {/* Studio Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 rounded-2xl bg-space-900/80 border border-space-800">
        <div>
          <div className="flex items-center space-x-2">
            <GitCompare className="w-5 h-5 text-amber-400" />
            <h2 className="text-sm font-bold text-white uppercase font-mono tracking-wider">
              Bi-Temporal Remote Sensing Change Studio
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Detect, quantify, and map landscape flux between two temporal satellite observations.
          </p>
        </div>

        <div className="flex items-center space-x-2.5">
          {currentAnalysis && (
            <button
              onClick={() => setIsReportModalOpen(true)}
              className="flex items-center space-x-1.5 px-3 py-2 rounded-lg bg-space-800 hover:bg-space-700 text-slate-200 border border-space-700 hover:border-amber-500/50 text-xs font-mono transition-all shadow-md active:scale-95"
              title="Generate Formal PDF Intelligence Dossier"
            >
              <FileText className="w-3.5 h-3.5 text-amber-400" />
              <span>PDF Dossier</span>
            </button>
          )}

          {!hasTwoImages && (
            <button
              onClick={onLoadChangeDemo}
              className="flex items-center space-x-2 px-3.5 py-2 rounded-lg bg-amber-500 hover:bg-amber-400 text-space-950 font-bold text-xs transition-colors shadow-lg shadow-amber-500/20"
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Load Curated Change Scenario</span>
            </button>
          )}
        </div>
      </div>

      {!hasTwoImages ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
          {renderInputImageryCard()}
          <div className="rounded-2xl border border-dashed border-space-700 bg-space-950 p-8 text-center flex flex-col items-center justify-center min-h-[260px]">
            <GitCompare className="w-10 h-10 text-amber-500/60 mx-auto mb-3" />
            <h3 className="text-sm font-semibold text-slate-200">Requires Two Temporal Images</h3>
            <p className="text-xs text-slate-400 max-w-sm mx-auto mt-2 mb-5 leading-relaxed">
              Bi-temporal analysis requires a T1 (earlier reference) and T2 (later monitoring) image. Upload two rasters on the left or load our pre-configured scenario.
            </p>
            <button
              onClick={onLoadChangeDemo}
              className="px-4 py-2.5 rounded-lg bg-amber-500 hover:bg-amber-400 text-space-950 text-xs font-bold transition-all shadow-lg shadow-amber-500/20 flex items-center space-x-2"
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Load Pre-Configured Bi-Temporal Demo</span>
            </button>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          
          {/* ========================================================= */}
          {/* LEFT SIDEBAR: Input Rasters & Change Statistics Breakdown */}
          {/* ========================================================= */}
          <div className="lg:col-span-4 space-y-4">
            {renderInputImageryCard()}

            {/* Change Statistics Breakdown Card */}
            {changeStats && (changeStats.change_percentage !== undefined || changeStats.change_pct !== undefined) && (
              <div className="rounded-2xl border border-space-700/80 bg-space-900/70 p-4 shadow-xl space-y-3">
                <div className="flex items-center justify-between pb-2 border-b border-space-800">
                  <span className="text-xs font-bold text-amber-400 font-mono flex items-center space-x-1.5">
                    <Activity className="w-3.5 h-3.5" />
                    <span>Spatial Change Metrics</span>
                  </span>
                  {currentAnalysis?.confidence && (
                    <span className="text-[10px] font-mono text-slate-400">
                      {currentAnalysis.confidence.percentage}% Verified
                    </span>
                  )}
                </div>

                <div className="space-y-2.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-400 font-mono">Changed Area Footprint:</span>
                    <div className="flex items-center space-x-1.5">
                      <span className="text-rose-400 font-bold font-mono text-sm">
                        {changeStats.change_percentage ?? changeStats.change_pct}%
                      </span>
                      {changeStats.changed_area_km2 !== undefined && (
                        <span className="text-slate-300 font-mono text-xs">
                          ({changeStats.changed_area_km2} km²)
                        </span>
                      )}
                    </div>
                  </div>

                  {changeStats.dominant_sector && (
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-slate-400 font-mono">Dominant Sector:</span>
                      <span className="text-amber-300 font-semibold font-mono">
                        {changeStats.dominant_sector} Quadrant
                      </span>
                    </div>
                  )}

                  {changeStats.quadrant_distribution && (
                    <div className="pt-2 border-t border-space-800/80 space-y-1.5">
                      <span className="text-[10px] font-mono uppercase text-slate-400 block font-semibold">
                        Quadrant Distribution:
                      </span>
                      <div className="grid grid-cols-4 gap-1 text-center font-mono text-[10.5px]">
                        {Object.entries(changeStats.quadrant_distribution).map(([q, pct]) => (
                          <div key={q} className="p-1.5 rounded-lg bg-space-950 border border-space-800/90">
                            <span className="text-slate-400 block text-[10px]">{q}</span>
                            <span className="text-slate-200 font-bold">{String(pct)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* ========================================================= */}
          {/* MAIN COLUMN: Image Viewer + Below Image: Unified Chat Box */}
          {/* ========================================================= */}
          <div className="lg:col-span-8 space-y-6">
            
            {/* View Selector Tabs */}
            <div className="flex items-center space-x-2 bg-space-900/80 p-1.5 rounded-xl border border-space-800 text-xs font-mono">
              <button
                onClick={() => setActiveView('swipe')}
                className={`px-3 py-1.5 rounded-lg flex items-center space-x-1.5 transition-all ${
                  activeView === 'swipe' ? 'bg-amber-500 text-space-950 font-bold shadow' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <ArrowLeftRight className="w-3.5 h-3.5" />
                <span>Interactive Swipe Curtain</span>
              </button>

              {heatmapUrl && (
                <button
                  onClick={() => setActiveView('heatmap')}
                  className={`px-3 py-1.5 rounded-lg flex items-center space-x-1.5 transition-all ${
                    activeView === 'heatmap' ? 'bg-amber-500 text-space-950 font-bold shadow' : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Activity className="w-3.5 h-3.5" />
                  <span>Difference Heatmap</span>
                </button>
              )}

              {maskUrl && (
                <button
                  onClick={() => setActiveView('mask')}
                  className={`px-3 py-1.5 rounded-lg flex items-center space-x-1.5 transition-all ${
                    activeView === 'mask' ? 'bg-amber-500 text-space-950 font-bold shadow' : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Layers className="w-3.5 h-3.5" />
                  <span>Segmented Change Mask</span>
                </button>
              )}
            </div>

            {/* Viewer Component */}
            {activeView === 'swipe' && (
              <SwipeViewer
                beforeImageUrl={beforeImg.preview_url}
                afterImageUrl={afterImg.preview_url}
                beforeLabel={`T1: ${beforeImg.metadata.acquisition_date || '2021 Reference'}`}
                afterLabel={`T2: ${afterImg.metadata.acquisition_date || '2023 Monitoring'}`}
                heightClass="h-[460px]"
              />
            )}

            {activeView === 'heatmap' && heatmapUrl && (
              <div className="rounded-2xl border border-space-700 bg-space-950 overflow-hidden shadow-2xl p-4 flex flex-col items-center justify-center min-h-[460px]">
                <img src={heatmapUrl} alt="Difference Heatmap" className="max-h-[420px] object-contain rounded-lg" />
                <div className="mt-3 flex items-center space-x-4 text-[11px] font-mono text-slate-400">
                  <span className="flex items-center"><span className="w-2.5 h-2.5 rounded-full bg-blue-500 mr-1.5"></span>Minimal Delta</span>
                  <span className="flex items-center"><span className="w-2.5 h-2.5 rounded-full bg-amber-400 mr-1.5"></span>Moderate Shift</span>
                  <span className="flex items-center"><span className="w-2.5 h-2.5 rounded-full bg-rose-500 mr-1.5"></span>Severe Transformation</span>
                </div>
              </div>
            )}

            {activeView === 'mask' && maskUrl && (
              <div className="rounded-2xl border border-space-700 bg-space-950 overflow-hidden shadow-2xl p-4 flex flex-col items-center justify-center min-h-[460px]">
                <img src={maskUrl} alt="Change Mask" className="max-h-[420px] object-contain rounded-lg" />
                <p className="text-[11px] font-mono text-rose-400 mt-2">
                  Otsu Adaptive Bimodal Threshold: Red segments indicate detected change footprint.
                </p>
              </div>
            )}

            {/* ========================================================= */}
            {/* BELOW THE IMAGE: Unified Chat Box (Query + Response as 1) */}
            {/* ========================================================= */}
            <div className="rounded-2xl border border-space-800/80 bg-space-900/70 backdrop-blur-xl shadow-xl overflow-hidden flex flex-col">
              
              {/* Chat Header */}
              <div className="flex items-center justify-between px-5 py-3.5 border-b border-space-800/60 bg-space-950/40">
                <div className="flex items-center space-x-2.5">
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-amber-500/20 via-orange-500/20 to-amber-600/20 border border-amber-500/30 flex items-center justify-center shrink-0">
                    <Sparkles className="w-4 h-4 text-amber-400" />
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs font-bold text-white tracking-wide font-mono uppercase">
                        Bi-Temporal Change Intelligence
                      </span>
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                      <span className="text-[10px] px-1.5 py-0.2 rounded bg-amber-500/10 text-amber-300 font-mono border border-amber-500/20">
                        CVA + Diff
                      </span>
                    </div>
                    <span className="text-[11px] text-slate-400 font-sans block">
                      Conversational Multi-Temporal Landscape Flux Analysis
                    </span>
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  {messages.length > 0 && (
                    <button
                      onClick={() => setMessages([])}
                      className="flex items-center space-x-1.5 py-1 px-2.5 rounded-lg bg-space-800/60 hover:bg-space-700 text-slate-300 text-[11px] font-mono transition-colors"
                      title="Clear Conversation"
                    >
                      <RotateCcw className="w-3 h-3 text-slate-400" />
                      <span className="hidden sm:inline">Clear Chat</span>
                    </button>
                  )}
                  {currentAnalysis && (
                    <button
                      onClick={() => setIsReportModalOpen(true)}
                      className="flex items-center space-x-1.5 py-1 px-2.5 rounded-lg bg-space-800/60 hover:bg-space-700 text-slate-200 text-[11px] font-mono transition-colors border border-space-700/60 hover:border-amber-500/40"
                      title="Generate PDF Dossier"
                    >
                      <FileText className="w-3.5 h-3.5 text-amber-400" />
                      <span className="hidden sm:inline">PDF Dossier</span>
                    </button>
                  )}
                  {currentAnalysis?.confidence && (
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-300 font-mono border border-amber-500/20">
                      {currentAnalysis.confidence.percentage}% Confidence
                    </span>
                  )}
                </div>
              </div>

              {/* Conversation Stream (End-to-End Chat Body) */}
              <div className="px-5 py-4 space-y-4 max-h-[380px] overflow-y-auto">
                {messages.length === 0 && !isProcessing && (
                  <div className="p-4 rounded-xl bg-space-950/40 border border-space-800/50 text-slate-400 text-xs flex items-center space-x-3">
                    <GitCompare className="w-5 h-5 text-amber-400/70 shrink-0" />
                    <p className="leading-relaxed">
                      Ask questions about landscape flux between T1 and T2 dates. Or select one of the suggested prompts below to inspect urban development, vegetation change, or spatial distribution.
                    </p>
                  </div>
                )}

                {messages.map((msg) => (
                  <div key={msg.id} className="space-y-3">
                    {msg.role === 'user' ? (
                      /* User Message Bubble */
                      <div className="flex justify-end items-start space-x-2.5">
                        <div className="p-3 px-4 rounded-2xl rounded-tr-xs bg-amber-500/15 border border-amber-500/25 text-slate-100 text-xs leading-relaxed max-w-[85%]">
                          <div className="flex items-center justify-between space-x-4 mb-1">
                            <span className="text-[11px] font-mono text-amber-400 font-semibold uppercase tracking-wider">
                              You
                            </span>
                            <span className="text-[10px] text-slate-400">
                              {msg.timestamp}
                            </span>
                          </div>
                          <p className="whitespace-pre-wrap font-sans text-[13px]">{msg.content}</p>
                        </div>
                        <div className="w-7 h-7 rounded-lg bg-space-800/80 flex items-center justify-center text-slate-300 shrink-0 mt-0.5">
                          <User className="w-4 h-4 text-amber-400" />
                        </div>
                      </div>
                    ) : (
                      /* Assistant Response Bubble */
                      <div className="flex items-start space-x-2.5">
                        <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-amber-500/20 to-orange-500/20 border border-amber-500/30 flex items-center justify-center shrink-0 mt-0.5">
                          <Sparkles className="w-4 h-4 text-amber-400" />
                        </div>

                        <div className="flex-1 bg-space-950/70 rounded-2xl rounded-tl-xs p-4 space-y-3 border border-space-800/80">
                          <div className="flex items-center justify-between pb-2 border-b border-space-800/60">
                            <div className="flex items-center space-x-2">
                              <span className="text-xs font-bold text-amber-400 tracking-wide font-mono">
                                Change Intelligence
                              </span>
                              {msg.confidence && (
                                <span className="text-[10px] px-2 py-0.5 rounded bg-amber-500/10 text-amber-300 font-mono border border-amber-500/20">
                                  {msg.confidence}% Confidence
                                </span>
                              )}
                            </div>
                            <span className="text-[10.5px] font-mono text-slate-400">{msg.timestamp}</span>
                          </div>

                          {/* Conversational Narrative Answer */}
                          <div className="pt-1">
                            <FormattedMessage content={msg.content} />
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}

                {/* Live Analyzing Pipeline Indicator */}
                {isProcessing && (
                  <div className="flex items-start space-x-2.5">
                    <div className="w-7 h-7 rounded-lg bg-amber-500/15 flex items-center justify-center shrink-0 mt-0.5">
                      <Sparkles className="w-3.5 h-3.5 text-amber-400 animate-pulse" />
                    </div>
                    <div className="p-3 rounded-2xl rounded-tl-xs bg-space-950/60 text-xs text-slate-300 flex items-center space-x-3">
                      <Loader2 className="w-4 h-4 animate-spin text-amber-400 shrink-0" />
                      <span className="font-mono text-amber-300 text-[11.5px]">
                        Computing change vector magnitude and Otsu thresholding...
                      </span>
                    </div>
                  </div>
                )}

                <div ref={chatBottomRef} />
              </div>

              {/* Chat Input Section (Integrated at bottom of chat box) */}
              <div className="p-4 pt-2 border-t border-space-800/60 bg-space-950/30 space-y-2.5">
                
                {/* Prompt Suggestion Chips */}
                <div className="flex flex-wrap gap-1.5">
                  {[
                    "What changed between these two dates?",
                    "Has the built-up area increased, decreased, or remained unchanged?",
                    "Where did the change occur?"
                  ].map((sq, i) => (
                    <button
                      key={i}
                      onClick={() => {
                        setQuery(sq);
                        handleRunChangeQuery(sq);
                      }}
                      className="text-[10.5px] px-2.5 py-1 rounded-lg bg-space-950 hover:bg-space-800 text-slate-300 border border-space-800/90 transition-colors font-mono hover:text-amber-300 hover:border-amber-500/30"
                    >
                      "{sq}"
                    </button>
                  ))}
                </div>

                {/* Chat Input Field */}
                <div className="relative flex items-center rounded-xl bg-space-950 border border-space-800 focus-within:border-amber-500/50 transition-all">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleRunChangeQuery();
                      }
                    }}
                    placeholder="Ask about landscape changes between T1 and T2..."
                    className="w-full bg-transparent text-xs text-slate-100 placeholder-slate-400 focus:outline-none font-mono py-2.5 pl-3.5 pr-12"
                  />
                  <button
                    onClick={() => handleRunChangeQuery()}
                    disabled={isProcessing || !query.trim()}
                    className="absolute right-1.5 p-1.5 rounded-lg bg-amber-500 hover:bg-amber-400 text-space-950 font-bold transition-all disabled:opacity-40 disabled:hover:bg-amber-500 shadow-md shadow-amber-500/20"
                    title="Send query"
                  >
                    {isProcessing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                  </button>
                </div>
              </div>

            </div>

          </div>

        </div>
      )}

      {/* Report Modal */}
      {currentAnalysis && (
        <ReportModal
          analysis={currentAnalysis}
          isOpen={isReportModalOpen}
          onClose={() => setIsReportModalOpen(false)}
        />
      )}

    </div>
  );
};
