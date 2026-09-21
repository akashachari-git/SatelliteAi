import React, { useState } from 'react';
import { 
  FileDown, 
  FileText, 
  CheckCircle, 
  X, 
  Loader2, 
  ShieldCheck, 
  Layers, 
  ExternalLink 
} from 'lucide-react';
import { AnalysisResult } from '../types';
import { api } from '../services/api';

interface ReportModalProps {
  analysis: AnalysisResult;
  isOpen: boolean;
  onClose: () => void;
}

export const ReportModal: React.FC<ReportModalProps> = ({
  analysis,
  isOpen,
  onClose
}) => {
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false);
  const [isGeneratingJson, setIsGeneratingJson] = useState(false);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [jsonUrl, setJsonUrl] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleDownloadPdf = async () => {
    try {
      setIsGeneratingPdf(true);
      const res = await api.generatePdfReport(analysis);
      setPdfUrl(res.pdf_url);
      window.open(res.pdf_url, '_blank');
    } catch (err) {
      console.error('PDF error', err);
    } finally {
      setIsGeneratingPdf(false);
    }
  };

  const handleDownloadJson = async () => {
    try {
      setIsGeneratingJson(true);
      const res = await api.generateJsonReport(analysis);
      setJsonUrl(res.json_url);
      window.open(res.json_url, '_blank');
    } catch (err) {
      console.error('JSON error', err);
    } finally {
      setIsGeneratingJson(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-space-950/80 backdrop-blur-md animate-in fade-in duration-150">
      <div className="relative w-full max-w-2xl rounded-2xl border border-space-700 bg-space-900 shadow-2xl p-6 overflow-hidden">
        
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-space-800">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center">
              <FileText className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white uppercase font-mono tracking-wider">
                SatQuery AI // Intelligence Dossier
              </h3>
              <p className="text-xs text-slate-400">Formal Geospatial Analysis & Verification Report</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-space-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Summary */}
        <div className="py-4 space-y-3 text-xs">
          <div className="p-3 rounded-xl bg-space-950/70 border border-space-800 space-y-2">
            <div className="flex items-center justify-between font-mono text-[11px]">
              <span className="text-slate-400">Query:</span>
              <span className="text-cyan-300 font-semibold truncate max-w-sm">"{analysis.query}"</span>
            </div>
            <div className="flex items-center justify-between font-mono text-[11px]">
              <span className="text-slate-400">Task / Tool:</span>
              <span className="text-slate-200">{analysis.task} • {analysis.selected_tool}</span>
            </div>
            <div className="flex items-center justify-between font-mono text-[11px]">
              <span className="text-slate-400">Calibrated Confidence:</span>
              <span className="text-emerald-400 font-bold">{analysis.confidence.percentage}%</span>
            </div>
            <div className="flex items-center justify-between font-mono text-[11px]">
              <span className="text-slate-400">Observable Steps:</span>
              <span className="text-slate-300">{analysis.execution_trace.length} pipeline steps verified</span>
            </div>
          </div>

          <div className="p-3 rounded-xl bg-space-950/40 border border-space-800 text-slate-300 font-mono text-[11px] leading-relaxed">
            {analysis.answer}
          </div>
        </div>

        {/* Download Buttons */}
        <div className="pt-4 border-t border-space-800 flex items-center justify-end space-x-3">
          <button
            onClick={handleDownloadJson}
            disabled={isGeneratingJson}
            className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-space-800 hover:bg-space-700 text-slate-200 text-xs font-medium transition-colors disabled:opacity-50"
          >
            {isGeneratingJson ? <Loader2 className="w-4 h-4 animate-spin text-cyan-400" /> : <FileDown className="w-4 h-4 text-cyan-400" />}
            <span>Export Machine JSON</span>
          </button>

          <button
            onClick={handleDownloadPdf}
            disabled={isGeneratingPdf}
            className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white text-xs font-semibold shadow-lg shadow-cyan-500/20 transition-all active:scale-95 disabled:opacity-50"
          >
            {isGeneratingPdf ? <Loader2 className="w-4 h-4 animate-spin" /> : <FileDown className="w-4 h-4" />}
            <span>Download Dossier PDF</span>
          </button>
        </div>

      </div>
    </div>
  );
};
