import React, { useState, useRef, useEffect } from 'react';
import { Radio, Sparkles, Send, Loader2, Info, Eye, Layers, ShieldCheck, UploadCloud, User, RotateCcw, FileText } from 'lucide-react';
import { DualModalityViewer } from '../components/DualModalityViewer';
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
              <span className="text-purple-400 font-bold text-xs mt-0.5 shrink-0">•</span>
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
  insights?: string[];
}

interface OpticalSarPageProps {
  currentAnalysis: AnalysisResult | null;
  setCurrentAnalysis: (res: AnalysisResult | null) => void;
  activeImages: any[];
  setActiveImages: React.Dispatch<React.SetStateAction<any[]>>;
  onLoadOpticalSarDemo: () => void;
}

export const OpticalSarPage: React.FC<OpticalSarPageProps> = ({
  currentAnalysis,
  setCurrentAnalysis,
  activeImages,
  setActiveImages,
  onLoadOpticalSarDemo
}) => {
  const [query, setQuery] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [messages, setMessages] = useState<StudioChatMessage[]>([]);
  const [isReportModalOpen, setIsReportModalOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Check if we have an Optical and SAR image
  const optImg = activeImages.find(img => img.metadata.modality !== 'SAR') || activeImages[0];
  const sarImg = activeImages.find(img => img.metadata.modality === 'SAR') || activeImages[1];
  const hasBoth = optImg && sarImg && optImg.server_path !== sarImg.server_path;

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
            insights: currentAnalysis.raw_result?.complementary_insights
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

  const handleRunFusionQuery = async (customQ?: string) => {
    if (!hasBoth) return;
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
      const paths = [optImg.server_path, sarImg.server_path];
      const metas = [optImg.metadata, sarImg.metadata];
      const res = await api.analyzeQuery(q, paths, metas);
      setCurrentAnalysis(res);

      const assistantMsg: StudioChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: res.answer,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        confidence: res.confidence?.percentage,
        insights: res.raw_result?.complementary_insights
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (err: any) {
      const errorMsg: StudioChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Error during cross-modal fusion analysis: ${err.message || 'Server error'}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsProcessing(false);
    }
  };

  const sarMetrics = currentAnalysis?.raw_result?.sar_metrics;
  const optMetrics = currentAnalysis?.raw_result?.optical_metrics;
  const fusedUrl = currentAnalysis?.evidence?.find(e => e.type === 'optical_sar_fused')?.url;

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
          isUploading ? 'border-purple-500/60 cursor-wait' : 'border-space-700 hover:border-purple-500/50'
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
            <Loader2 className="w-7 h-7 mx-auto text-purple-400 animate-spin" />
            <p className="text-xs font-semibold text-purple-300">
              {uploadProgress !== null && uploadProgress < 100 
                ? `Uploading Image (${uploadProgress}%)...` 
                : 'Validating Satellite Raster...'}
            </p>
            <div className="w-3/4 mx-auto bg-space-800 rounded-full h-1.5 overflow-hidden">
              <div 
                className="bg-gradient-to-r from-purple-500 to-indigo-500 h-1.5 rounded-full transition-all duration-150"
                style={{ width: `${uploadProgress ?? 100}%` }}
              />
            </div>
            <p className="text-[10px] text-slate-400 font-mono">
              High-speed client compression & raster ingestion
            </p>
          </div>
        ) : (
          <>
            <UploadCloud className="w-7 h-7 mx-auto text-slate-400 group-hover:text-purple-400 transition-colors mb-2" />
            <p className="text-xs font-medium text-slate-200">
              Upload Satellite Image
            </p>
            <p className="text-[10px] text-slate-400 mt-1 font-mono">
              GeoTIFF with CRS or standard RGB
            </p>
            <div className="mt-2.5 pt-2 border-t border-space-800/80 flex items-center justify-center space-x-1.5 text-[10.5px] font-mono text-purple-400">
              <ShieldCheck className="w-3.5 h-3.5 text-purple-400 shrink-0" />
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
            const isSar = img.metadata?.modality === 'SAR' || img.filename?.toLowerCase().includes('sar');
            const badge = isSar ? 'SAR Microwave' : 'Optical Sensor';
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
                    <span className="text-[10px] font-mono text-purple-400">
                      {badge} • {img.metadata?.width || '1024'}×{img.metadata?.height || '1024'}
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
            <Radio className="w-5 h-5 text-purple-400" />
            <h2 className="text-sm font-bold text-white uppercase font-mono tracking-wider">
              Cross-Modal Optical + SAR Fusion Studio
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Synthesize multispectral optical reflectance with C-band SAR microwave backscatter.
          </p>
        </div>

        <div className="flex items-center space-x-2.5">
          {currentAnalysis && (
            <button
              onClick={() => setIsReportModalOpen(true)}
              className="flex items-center space-x-1.5 px-3 py-2 rounded-lg bg-space-800 hover:bg-space-700 text-slate-200 border border-space-700 hover:border-purple-500/50 text-xs font-mono transition-all shadow-md active:scale-95"
              title="Generate Formal PDF Intelligence Dossier"
            >
              <FileText className="w-3.5 h-3.5 text-purple-400" />
              <span>PDF Dossier</span>
            </button>
          )}

          {!hasBoth && (
            <button
              onClick={onLoadOpticalSarDemo}
              className="flex items-center space-x-2 px-3.5 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 text-white font-bold text-xs transition-colors shadow-lg shadow-purple-600/20"
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Load Optical + SAR Demo Pair</span>
            </button>
          )}
        </div>
      </div>

      {!hasBoth ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
          {renderInputImageryCard()}
          <div className="rounded-2xl border border-dashed border-space-700 bg-space-950 p-8 text-center flex flex-col items-center justify-center min-h-[260px]">
            <Radio className="w-10 h-10 text-purple-500/60 mx-auto mb-3" />
            <h3 className="text-sm font-semibold text-slate-200">Requires Optical & SAR Image Pair</h3>
            <p className="text-xs text-slate-400 max-w-sm mx-auto mt-2 mb-5 leading-relaxed">
              Upload both a Sentinel-2 Optical raster and a Sentinel-1 SAR backscatter raster on the left, or load our pre-configured scenario.
            </p>
            <button
              onClick={onLoadOpticalSarDemo}
              className="px-4 py-2.5 rounded-lg bg-purple-600 hover:bg-purple-500 text-white text-xs font-bold transition-all shadow-lg shadow-purple-600/20 flex items-center space-x-2"
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Load Pre-Configured Optical + SAR Scenario</span>
            </button>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          
          {/* ========================================================= */}
          {/* LEFT SIDEBAR: Input Imagery & Sensor Strengths Summary     */}
          {/* ========================================================= */}
          <div className="lg:col-span-4 space-y-4">
            {renderInputImageryCard()}

            {/* Complementary Sensor Notes */}
            <div className="rounded-2xl border border-space-700/80 bg-space-900/70 p-4 shadow-xl space-y-3">
              <span className="text-xs font-bold text-purple-400 font-mono uppercase tracking-wider block border-b border-space-800 pb-2">
                Sensor Modality Profiles
              </span>

              <div className="space-y-2.5 text-xs">
                <div className="p-3 rounded-xl bg-space-950/80 border border-space-800/80">
                  <span className="text-[10.5px] font-mono text-emerald-400 uppercase tracking-wider block mb-1 font-semibold">
                    Optical Reflectance
                  </span>
                  <p className="text-slate-300 leading-normal text-[11px]">
                    Discriminates vegetation pigment, water chlorophyll, soil mineral signatures, and visible multispectral reflectance.
                  </p>
                </div>

                <div className="p-3 rounded-xl bg-space-950/80 border border-space-800/80">
                  <span className="text-[10.5px] font-mono text-purple-400 uppercase tracking-wider block mb-1 font-semibold">
                    SAR Microwave Backscatter
                  </span>
                  <p className="text-slate-300 leading-normal text-[11px]">
                    All-weather cloud penetration, double-bounce detection of vertical building facets, and smooth water specular dampening.
                  </p>
                </div>
              </div>

              {currentAnalysis?.raw_result?.complementary_insights && (
                <div className="pt-2.5 border-t border-space-800/80 space-y-1.5">
                  <span className="text-[10px] font-mono uppercase text-slate-400 block font-semibold">
                    Fused Complementary Findings:
                  </span>
                  {currentAnalysis.raw_result.complementary_insights.map((ins: string, idx: number) => (
                    <div key={idx} className="flex items-start space-x-1.5 text-[11px] text-slate-300">
                      <span className="text-purple-400 font-bold shrink-0">•</span>
                      <span className="leading-snug">{ins}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* ========================================================= */}
          {/* MAIN COLUMN: Viewer + Below Image: Unified Chat Box        */}
          {/* ========================================================= */}
          <div className="lg:col-span-8 space-y-6">
            <DualModalityViewer
              opticalUrl={optImg.preview_url}
              sarUrl={sarImg.preview_url}
              fusedUrl={fusedUrl}
              sarMetrics={sarMetrics}
              opticalMetrics={optMetrics}
            />

            {/* ========================================================= */}
            {/* BELOW THE IMAGE: Unified Chat Box (Query + Response as 1) */}
            {/* ========================================================= */}
            <div className="rounded-2xl border border-space-800/80 bg-space-900/70 backdrop-blur-xl shadow-xl overflow-hidden flex flex-col">
              
              {/* Chat Header */}
              <div className="flex items-center justify-between px-5 py-3.5 border-b border-space-800/60 bg-space-950/40">
                <div className="flex items-center space-x-2.5">
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-purple-500/20 via-pink-500/20 to-purple-600/20 border border-purple-500/30 flex items-center justify-center shrink-0">
                    <Sparkles className="w-4 h-4 text-purple-400" />
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs font-bold text-white tracking-wide font-mono uppercase">
                        Joint Fusion Intelligence
                      </span>
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                      <span className="text-[10px] px-1.5 py-0.2 rounded bg-purple-500/10 text-purple-300 font-mono border border-purple-500/20">
                        Optical + SAR
                      </span>
                    </div>
                    <span className="text-[11px] text-slate-400 font-sans block">
                      Conversational Cross-Modal Remote Sensing Synthesis
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
                      className="flex items-center space-x-1.5 py-1 px-2.5 rounded-lg bg-space-800/60 hover:bg-space-700 text-slate-200 text-[11px] font-mono transition-colors border border-space-700/60 hover:border-purple-500/40"
                      title="Generate PDF Dossier"
                    >
                      <FileText className="w-3.5 h-3.5 text-purple-400" />
                      <span className="hidden sm:inline">PDF Dossier</span>
                    </button>
                  )}
                  {currentAnalysis?.confidence && (
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-300 font-mono border border-purple-500/20">
                      {currentAnalysis.confidence.percentage}% Confidence
                    </span>
                  )}
                </div>
              </div>

              {/* Conversation Stream (End-to-End Chat Body) */}
              <div className="px-5 py-4 space-y-4 max-h-[380px] overflow-y-auto">
                {messages.length === 0 && !isProcessing && (
                  <div className="p-4 rounded-xl bg-space-950/40 border border-space-800/50 text-slate-400 text-xs flex items-center space-x-3">
                    <Radio className="w-5 h-5 text-purple-400/70 shrink-0" />
                    <p className="leading-relaxed">
                      Ask questions combining optical reflectance and radar microwave backscatter. Or select one of the suggested prompts below to inspect buildings, water bodies, or terrain features.
                    </p>
                  </div>
                )}

                {messages.map((msg) => (
                  <div key={msg.id} className="space-y-3">
                    {msg.role === 'user' ? (
                      /* User Message Bubble */
                      <div className="flex justify-end items-start space-x-2.5">
                        <div className="p-3 px-4 rounded-2xl rounded-tr-xs bg-purple-500/15 border border-purple-500/25 text-slate-100 text-xs leading-relaxed max-w-[85%]">
                          <div className="flex items-center justify-between space-x-4 mb-1">
                            <span className="text-[11px] font-mono text-purple-400 font-semibold uppercase tracking-wider">
                              You
                            </span>
                            <span className="text-[10px] text-slate-400">
                              {msg.timestamp}
                            </span>
                          </div>
                          <p className="whitespace-pre-wrap font-sans text-[13px]">{msg.content}</p>
                        </div>
                        <div className="w-7 h-7 rounded-lg bg-space-800/80 flex items-center justify-center text-slate-300 shrink-0 mt-0.5">
                          <User className="w-4 h-4 text-purple-400" />
                        </div>
                      </div>
                    ) : (
                      /* Assistant Response Bubble */
                      <div className="flex items-start space-x-2.5">
                        <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-purple-500/20 to-pink-500/20 border border-purple-500/30 flex items-center justify-center shrink-0 mt-0.5">
                          <Sparkles className="w-4 h-4 text-purple-400" />
                        </div>

                        <div className="flex-1 bg-space-950/70 rounded-2xl rounded-tl-xs p-4 space-y-3 border border-space-800/80">
                          <div className="flex items-center justify-between pb-2 border-b border-space-800/60">
                            <div className="flex items-center space-x-2">
                              <span className="text-xs font-bold text-purple-400 tracking-wide font-mono">
                                Joint Fusion Intelligence
                              </span>
                              {msg.confidence && (
                                <span className="text-[10px] px-2 py-0.5 rounded bg-purple-500/10 text-emerald-400 font-mono border border-purple-500/20">
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
                    <div className="w-7 h-7 rounded-lg bg-purple-500/15 flex items-center justify-center shrink-0 mt-0.5">
                      <Sparkles className="w-3.5 h-3.5 text-purple-400 animate-pulse" />
                    </div>
                    <div className="p-3 rounded-2xl rounded-tl-xs bg-space-950/60 text-xs text-slate-300 flex items-center space-x-3">
                      <Loader2 className="w-4 h-4 animate-spin text-purple-400 shrink-0" />
                      <span className="font-mono text-purple-300 text-[11.5px]">
                        Synthesizing optical multispectral reflectance with C-SAR backscatter...
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
                    "Identify built-up and water-covered regions using both images.",
                    "Which regions are more clearly identified using SAR?",
                    "Use the optical and SAR images together to identify urban structures."
                  ].map((sq, i) => (
                    <button
                      key={i}
                      onClick={() => {
                        setQuery(sq);
                        handleRunFusionQuery(sq);
                      }}
                      className="text-[10.5px] px-2.5 py-1 rounded-lg bg-space-950 hover:bg-space-800 text-slate-300 border border-space-800/90 transition-colors font-mono hover:text-purple-300 hover:border-purple-500/30"
                    >
                      "{sq}"
                    </button>
                  ))}
                </div>

                {/* Chat Input Field */}
                <div className="relative flex items-center rounded-xl bg-space-950 border border-space-800 focus-within:border-purple-500/50 transition-all">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleRunFusionQuery();
                      }
                    }}
                    placeholder="Ask about cross-modal optical + SAR fusion..."
                    className="w-full bg-transparent text-xs text-slate-100 placeholder-slate-400 focus:outline-none font-mono py-2.5 pl-3.5 pr-12"
                  />
                  <button
                    onClick={() => handleRunFusionQuery()}
                    disabled={isProcessing || !query.trim()}
                    className="absolute right-1.5 p-1.5 rounded-lg bg-purple-600 hover:bg-purple-500 text-white font-bold transition-all disabled:opacity-40 disabled:hover:bg-purple-600 shadow-md shadow-purple-600/20"
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
