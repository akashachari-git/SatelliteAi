import React, { useState, useRef, useEffect } from 'react';
import { 
  UploadCloud, 
  Send, 
  Sparkles, 
  Loader2, 
  FileText, 
  Layers, 
  AlertCircle, 
  CheckCircle, 
  ExternalLink,
  ChevronRight,
  RefreshCw,
  Plus,
  ShieldCheck,
  ArrowUp,
  Globe2,
  User,
  Bot,
  RotateCcw,
  Info
} from 'lucide-react';
import { ImageMetadata, AnalysisResult } from '../types';
import { api } from '../services/api';
import { MetadataCard } from '../components/MetadataCard';
import { ImageViewer } from '../components/ImageViewer';
import { ReportModal } from '../components/ReportModal';
import { PhotoDetailsModal } from '../components/PhotoDetailsModal';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  task?: string;
  model_name?: string;
  method?: string;
  analysisResult?: AnalysisResult;
  timestamp: string;
}

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
        if (!trimmed) {
          return <div key={idx} className="h-1" />;
        }

        // Check if line is a bullet item
        const isBullet = trimmed.startsWith('•') || trimmed.startsWith('- ') || trimmed.startsWith('* ');
        if (isBullet) {
          const bulletText = trimmed.replace(/^[•\-\*]\s*/, '');
          return (
            <div key={idx} className="flex items-start space-x-2 pl-1.5 py-0.5">
              <span className="text-cyan-400 font-bold text-xs mt-0.5 shrink-0">•</span>
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

interface WorkspacePageProps {
  currentAnalysis: AnalysisResult | null;
  setCurrentAnalysis: (res: AnalysisResult | null) => void;
  activeImages: Array<{
    filename: string;
    server_path: string;
    preview_url: string;
    metadata: ImageMetadata;
  }>;
  setActiveImages: React.Dispatch<React.SetStateAction<Array<{
    filename: string;
    server_path: string;
    preview_url: string;
    metadata: ImageMetadata;
  }>>>;
  onViewChangeStudio: () => void;
  onViewOpticalSar: () => void;
}

export const WorkspacePage: React.FC<WorkspacePageProps> = ({
  currentAnalysis,
  setCurrentAnalysis,
  activeImages,
  setActiveImages,
  onViewChangeStudio,
  onViewOpticalSar
}) => {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isReportModalOpen, setIsReportModalOpen] = useState(false);
  const [selectedDetailImage, setSelectedDetailImage] = useState<{
    filename: string;
    preview_url: string;
    metadata: ImageMetadata;
    badge?: string;
  } | null>(null);
  const [dockedDetailImage, setDockedDetailImage] = useState<{
    filename: string;
    preview_url: string;
    metadata: ImageMetadata;
    badge?: string;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Chat messages state - starts empty for a true interactive chatbot experience without automatic scene description
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isAnalyzing]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setIsUploading(true);
      setUploadProgress(0);
      setErrorMsg(null);
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
      setErrorMsg(err.message || 'File upload error');
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

  const handleClearChat = () => {
    setMessages([]);
    setCurrentAnalysis(null);
  };

  const handleRunAnalysis = async (customQuery?: string) => {
    const q = (customQuery || query).trim();
    if (!q) {
      return;
    }

    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: q,
      timestamp: timeStr
    };
    setMessages(prev => [...prev, userMsg]);
    setQuery('');

    // If no images mounted, reply conversationally from the chatbot
    if (activeImages.length === 0) {
      const assistantMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "Please mount or upload a satellite image (GeoTIFF, TIFF, PNG) first using the Input Imagery panel so I can analyze the remote sensing features and answer your question.",
        task: "INPUT_REQUIRED",
        model_name: "SatQuery AI",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, assistantMsg]);
      return;
    }

    try {
      setIsAnalyzing(true);
      setErrorMsg(null);
      const paths = activeImages.map(img => img.server_path);
      const metas = activeImages.map(img => img.metadata);
      const result = await api.analyzeQuery(q, paths, metas);
      setCurrentAnalysis(result);

      const assistantMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: result.answer,
        task: result.task,
        model_name: result.model_name,
        method: result.method,
        analysisResult: result,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (err: any) {
      const errorChatMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `I encountered an issue processing your request: ${err.message || 'Analysis processing failed'}. Please try rephrasing or inspect image coverage.`,
        task: "ERROR",
        model_name: "SatQuery AI",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, errorChatMsg]);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const primaryImage = activeImages[0];
  const secondaryImage = activeImages[1];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      
      {/* Workspace Header Notice */}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-xl bg-space-900/60 border border-space-800">
        <div className="flex items-center space-x-2 text-xs font-mono text-slate-300">
          <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
          <span className="font-bold text-white">Active Session:</span>
          <span>{activeImages.length} Image(s) Mounted</span>
          {activeImages.length >= 2 && (
            <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 text-[10px]">
              Multi-Image Pair
            </span>
          )}
        </div>

        <div className="flex items-center space-x-3 text-xs">
          {activeImages.length >= 2 && (
            <>
              <button
                onClick={onViewChangeStudio}
                className="text-amber-400 hover:text-amber-300 font-mono text-[11px] flex items-center"
              >
                <span>Bi-Temporal Studio</span>
                <ChevronRight className="w-3.5 h-3.5 ml-0.5" />
              </button>
              <button
                onClick={onViewOpticalSar}
                className="text-purple-400 hover:text-purple-300 font-mono text-[11px] flex items-center"
              >
                <span>Optical+SAR Studio</span>
                <ChevronRight className="w-3.5 h-3.5 ml-0.5" />
              </button>
            </>
          )}
        </div>
      </div>

      {/* 2-Column Main Layout: Input/Metadata on Left, Satellite View + Query/Results on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* ========================================================= */}
        {/* LEFT PANEL: Input Configuration & Raster Metadata */}
        {/* ========================================================= */}
        <div className="lg:col-span-4 space-y-4">
          <div className="rounded-2xl border border-space-700/80 bg-space-900/70 p-4 shadow-xl">
            <div className="flex items-center justify-between pb-3 mb-3 border-b border-space-800">
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
                isUploading ? 'border-cyan-500/60 cursor-wait' : 'border-space-700 hover:border-cyan-500/50'
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
                  <Loader2 className="w-7 h-7 mx-auto text-cyan-400 animate-spin" />
                  <p className="text-xs font-semibold text-cyan-300">
                    {uploadProgress !== null && uploadProgress < 100 
                      ? `Uploading Image (${uploadProgress}%)...` 
                      : 'Analyzing & Validating Raster...'}
                  </p>
                  <div className="w-3/4 mx-auto bg-space-800 rounded-full h-1.5 overflow-hidden">
                    <div 
                      className="bg-gradient-to-r from-cyan-500 to-blue-500 h-1.5 rounded-full transition-all duration-150"
                      style={{ width: `${uploadProgress ?? 100}%` }}
                    />
                  </div>
                  <p className="text-[10px] text-slate-400 font-mono">
                    High-speed client compression & raster ingestion
                  </p>
                </div>
              ) : (
                <>
                  <UploadCloud className="w-7 h-7 mx-auto text-slate-400 group-hover:text-cyan-400 transition-colors mb-2" />
                  <p className="text-xs font-medium text-slate-200">
                    Upload Satellite Image
                  </p>
                  <p className="text-[10px] text-slate-400 mt-1 font-mono">
                    GeoTIFF with CRS or standard RGB
                  </p>
                  <div className="mt-2.5 pt-2 border-t border-space-800/80 flex items-center justify-center space-x-1.5 text-[10.5px] font-mono text-cyan-400">
                    <ShieldCheck className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                    <span>Satellite Images Only (Non-EO Auto-Rejected)</span>
                  </div>
                </>
              )}
            </div>

            {/* Upload Rejection Alert */}
            {errorMsg && (
              <div className="mt-3 p-3 rounded-xl bg-rose-500/15 border border-rose-500/30 text-rose-300 text-xs flex items-start space-x-2 animate-in fade-in duration-150">
                <AlertCircle className="w-4 h-4 shrink-0 text-rose-400 mt-0.5" />
                <span className="leading-relaxed">{errorMsg}</span>
              </div>
            )}

            {/* Active Image Thumbnails List */}
            {activeImages.length > 0 && (
              <div className="mt-4 space-y-2">
                <span className="text-[10px] font-mono uppercase text-slate-400 block font-semibold">
                  Mounted Rasters ({activeImages.length}):
                </span>
                {activeImages.map((img, idx) => {
                  const badge = idx === 0 ? 'Primary (T1)' : 'Secondary (T2 / SAR)';
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
                          <span className="text-[10px] font-mono text-cyan-400">
                            {img.metadata.modality} • {img.metadata.width}×{img.metadata.height}
                          </span>
                        </div>
                      </div>

                      <div className="flex items-center space-x-1.5 shrink-0 ml-2">
                        <button
                          onClick={() => setSelectedDetailImage({
                            filename: img.filename,
                            preview_url: img.preview_url,
                            metadata: img.metadata,
                            badge
                          })}
                          className="px-2 py-1 rounded-md text-[10.5px] font-medium bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 transition-all flex items-center space-x-1 shadow-sm hover:shadow-cyan-500/10 active:scale-95"
                          title="View photo details and metadata"
                        >
                          <Info className="w-3 h-3 text-cyan-400" />
                          <span>View Details</span>
                        </button>
                        <button
                          onClick={() => {
                            if (dockedDetailImage?.preview_url === img.preview_url) {
                              setDockedDetailImage(null);
                            }
                            if (selectedDetailImage?.preview_url === img.preview_url) {
                              setSelectedDetailImage(null);
                            }
                            handleRemoveImage(idx);
                          }}
                          className="text-slate-400 hover:text-rose-400 text-xs p-1 rounded hover:bg-rose-500/10 transition-colors"
                          title="Remove image"
                        >
                          ×
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Docked Metadata Inspector Card (Shown on demand when pinned/docked) */}
          {dockedDetailImage && (
            <MetadataCard 
              metadata={dockedDetailImage.metadata} 
              title={dockedDetailImage.filename}
              badge={dockedDetailImage.badge}
              onClose={() => setDockedDetailImage(null)}
            />
          )}

          {/* Co-Registration Validation Badge */}
          {currentAnalysis?.co_registration && (
            <div className="rounded-xl border border-space-700 bg-space-900/60 p-3.5 text-xs font-mono space-y-2">
              <div className="flex items-center justify-between text-[11px] font-bold">
                <span className="text-slate-300">Co-Registration Index:</span>
                <span className={currentAnalysis.co_registration.is_compatible ? 'text-emerald-400' : 'text-amber-400'}>
                  {currentAnalysis.co_registration.co_registration_score}% Compatible
                </span>
              </div>
              <div className="space-y-1">
                {currentAnalysis.co_registration.checks.map((chk, i) => (
                  <div key={i} className="flex items-center justify-between text-[10px] text-slate-400">
                    <span>{chk.name}:</span>
                    <span className={chk.status === 'PASS' ? 'text-emerald-400' : 'text-amber-400'}>
                      {chk.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ========================================================= */}
        {/* RIGHT PANEL: Satellite View + Query & Results Below */}
        {/* ========================================================= */}
        <div className="lg:col-span-8 space-y-6">
          
          {/* SATELLITE VIEW */}
          <div className="h-[520px]">
            <ImageViewer 
              baseImageUrl={primaryImage?.preview_url}
              evidenceItems={currentAnalysis?.evidence || []}
              boundingBoxes={currentAnalysis?.raw_result?.bounding_boxes || []}
              title="Satellite View"
              subtitle={currentAnalysis ? `${currentAnalysis.task}` : "Ready for Query"}
            />
          </div>

          {/* ========================================================= */}
          {/* BELOW SATELLITE VIEW: Unified Conversational Intelligence Box */}
          {/* ========================================================= */}
          <div className="rounded-2xl border border-space-800/80 bg-space-900/60 backdrop-blur-xl shadow-xl overflow-hidden flex flex-col">
            
            {/* Chat Box Header - Matches reference screenshot */}
            <div className="flex items-center justify-between px-5 py-3.5 bg-transparent">
              <div className="flex items-center space-x-2.5">
                <div className="w-8 h-8 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center shrink-0">
                  <Sparkles className="w-4 h-4 text-cyan-400" />
                </div>
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs font-bold text-white tracking-wide font-sans">
                      Natural Language Query
                    </span>
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 font-mono border border-cyan-500/20">
                      Vision Engine
                    </span>
                  </div>
                  <span className="text-[11px] text-slate-400 font-sans block mt-0.5">
                    Multimodal Remote Sensing Vision Q&A
                  </span>
                </div>
              </div>

              <div className="flex items-center space-x-2">
                {messages.length > 0 && (
                  <button
                    onClick={handleClearChat}
                    className="flex items-center space-x-1.5 py-1 px-2.5 rounded-lg bg-space-800/60 hover:bg-space-700 text-slate-300 text-[11px] font-mono transition-colors"
                    title="Clear Conversation"
                  >
                    <RotateCcw className="w-3 h-3 text-slate-400" />
                    <span className="hidden sm:inline">Clear Chat</span>
                  </button>
                )}
                {currentAnalysis && (
                  <>
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 font-mono">
                      {currentAnalysis.task}
                    </span>
                    <button
                      onClick={() => setIsReportModalOpen(true)}
                      className="flex items-center space-x-1.5 py-1 px-2.5 rounded-lg bg-space-800/60 hover:bg-space-700 text-slate-200 text-[11px] font-mono transition-colors"
                      title="Generate PDF Dossier"
                    >
                      <FileText className="w-3.5 h-3.5 text-cyan-400" />
                      <span className="hidden sm:inline">PDF Dossier</span>
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Conversation Stream (Rendered when messages exist) */}
            {messages.length > 0 && (
              <div className="px-5 py-2 space-y-4 max-h-[380px] overflow-y-auto border-t border-space-800/40">
                {messages.map((msg) => (
                  <div key={msg.id} className="space-y-3">
                    {msg.role === 'user' ? (
                      /* User Message Bubble */
                      <div className="flex justify-end items-start space-x-2.5">
                        <div className="p-3 px-4 rounded-2xl rounded-tr-xs bg-cyan-500/15 border border-cyan-500/25 text-slate-100 text-xs leading-relaxed max-w-[85%]">
                          <div className="flex items-center justify-between space-x-4 mb-1">
                            <span className="text-[11px] font-sans text-cyan-400 font-semibold uppercase tracking-wider">
                              You
                            </span>
                            <span className="text-[10px] text-slate-400">
                              {msg.timestamp}
                            </span>
                          </div>
                          <p className="whitespace-pre-wrap font-sans text-[13px]">{msg.content}</p>
                        </div>
                        <div className="w-7 h-7 rounded-lg bg-space-800/80 flex items-center justify-center text-slate-300 shrink-0 mt-0.5">
                          <User className="w-4 h-4 text-cyan-400" />
                        </div>
                      </div>
                    ) : (
                      /* Assistant Response Bubble - Natural Language Chatbot */
                      <div className="flex items-start space-x-2.5">
                        <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-cyan-500/20 to-blue-600/20 border border-cyan-500/30 flex items-center justify-center shrink-0 mt-0.5">
                          <Sparkles className="w-4 h-4 text-cyan-400" />
                        </div>

                        <div className="flex-1 bg-space-950/70 rounded-2xl rounded-tl-xs p-4 space-y-3 border border-space-800/80">
                          <div className="flex items-center justify-between pb-2 border-b border-space-800/60">
                            <div className="flex items-center space-x-2">
                              <span className="text-xs font-bold text-white tracking-wide font-sans">
                                SatQuery AI
                              </span>
                              {msg.task && (
                                <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 font-mono border border-cyan-500/20">
                                  {msg.task}
                                </span>
                              )}
                            </div>
                            <div className="flex items-center space-x-3 text-[10.5px] font-mono text-slate-400">
                              {msg.model_name && (
                                <span>Model: <span className="text-cyan-300">{msg.model_name}</span></span>
                              )}
                              <span>{msg.timestamp}</span>
                            </div>
                          </div>

                          {/* Conversational Narrative Answer in Pure Text */}
                          <div className="pt-1">
                            <FormattedMessage content={msg.content} />
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}

                {/* Analyzing Pipeline Live Indicator */}
                {isAnalyzing && (
                  <div className="flex items-start space-x-2.5">
                    <div className="w-7 h-7 rounded-lg bg-cyan-500/15 flex items-center justify-center shrink-0 mt-0.5">
                      <Sparkles className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
                    </div>
                    <div className="p-3 rounded-2xl rounded-tl-xs bg-space-950/60 text-xs text-slate-300 flex items-center space-x-3">
                      <Loader2 className="w-4 h-4 animate-spin text-cyan-400 shrink-0" />
                      <span className="font-mono text-cyan-300 text-[11.5px]">
                        SatQuery AI is processing remote sensing query...
                      </span>
                    </div>
                  </div>
                )}

                <div ref={chatBottomRef} />
              </div>
            )}

            {/* Bottom Section: Chat Input Area - Matches reference screenshot */}
            <div className="p-4 pt-1 bg-transparent space-y-2">

              {/* Error Alert */}
              {errorMsg && (
                <div className="p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs flex items-start space-x-2">
                  <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                  <span>{errorMsg}</span>
                </div>
              )}

              {/* Input Area with Arrow Mark in the Corner */}
              <div className="relative flex items-center rounded-xl bg-space-950/90 border border-space-800/80 focus-within:border-cyan-500/50 transition-all">
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleRunAnalysis();
                    }
                  }}
                  placeholder="Ask SatQuery AI... (e.g. detect each and every water body, ice, buildings...)"
                  rows={2}
                  className="w-full bg-transparent text-sm text-slate-100 placeholder-slate-400 focus:outline-none resize-none font-sans py-3 pl-4 pr-12 min-h-[54px]"
                />

                {/* Arrow Mark in the Corner - Matches reference screenshot */}
                <button
                  onClick={() => handleRunAnalysis()}
                  disabled={isAnalyzing || !query.trim()}
                  aria-label="Ask SatQuery AI"
                  title="Run Query (Enter)"
                  className="absolute right-2.5 bottom-2.5 w-8 h-8 rounded-lg bg-[#11232e] hover:bg-[#183446] border border-cyan-500/25 text-cyan-400 flex items-center justify-center transition-all active:scale-95 disabled:opacity-30 disabled:cursor-not-allowed group shadow-sm"
                >
                  {isAnalyzing ? (
                    <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
                  ) : (
                    <ArrowUp className="w-4 h-4 text-cyan-400 group-hover:-translate-y-0.5 transition-transform" />
                  )}
                </button>
              </div>

            </div>

          </div>

        </div>

      </div>

      {/* Report Modal */}
      {currentAnalysis && (
        <ReportModal
          analysis={currentAnalysis}
          isOpen={isReportModalOpen}
          onClose={() => setIsReportModalOpen(false)}
        />
      )}

      {/* Photo Details & Metadata Modal */}
      <PhotoDetailsModal
        isOpen={!!selectedDetailImage}
        onClose={() => setSelectedDetailImage(null)}
        image={selectedDetailImage}
        isDocked={dockedDetailImage?.preview_url === selectedDetailImage?.preview_url}
        onDockToSidebar={() => {
          if (selectedDetailImage) {
            setDockedDetailImage(selectedDetailImage);
          }
        }}
      />


    </div>
  );
};
