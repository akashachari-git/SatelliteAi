import React, { useState, useEffect } from 'react';
import { Sparkles, Key, CheckCircle, AlertCircle, ExternalLink, X, Loader2 } from 'lucide-react';
import { api } from '../services/api';

interface GeminiKeyModalProps {
  isOpen: boolean;
  onClose: () => void;
  onKeySaved: (key: string) => void;
}

export const GeminiKeyModal: React.FC<GeminiKeyModalProps> = ({
  isOpen,
  onClose,
  onKeySaved
}) => {
  const [keyInput, setKeyInput] = useState('');
  const [isValidating, setIsValidating] = useState(false);
  const [statusMsg, setStatusMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  useEffect(() => {
    const saved = localStorage.getItem('satquery_gemini_api_key') || '';
    setKeyInput(saved);
    setStatusMsg(null);
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSave = async () => {
    const trimmed = keyInput.trim();
    if (!trimmed) {
      localStorage.removeItem('satquery_gemini_api_key');
      onKeySaved('');
      setStatusMsg({ type: 'success', text: 'API key removed. Running on local precision engine.' });
      return;
    }

    setIsValidating(true);
    setStatusMsg(null);
    try {
      const res = await api.validateGeminiKey(trimmed);
      if (res.valid) {
        localStorage.setItem('satquery_gemini_api_key', trimmed);
        onKeySaved(trimmed);
        setStatusMsg({ type: 'success', text: 'Gemini 1.5 Flash connected successfully!' });
        setTimeout(() => onClose(), 1200);
      } else {
        // Still allow saving in case of local network restriction
        localStorage.setItem('satquery_gemini_api_key', trimmed);
        onKeySaved(trimmed);
        setStatusMsg({ type: 'success', text: 'Key saved locally for queries!' });
        setTimeout(() => onClose(), 1200);
      }
    } catch (err: any) {
      localStorage.setItem('satquery_gemini_api_key', trimmed);
      onKeySaved(trimmed);
      setStatusMsg({ type: 'success', text: 'Key saved for requests!' });
      setTimeout(() => onClose(), 1200);
    } finally {
      setIsValidating(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-space-950/80 backdrop-blur-md animate-in fade-in duration-150">
      <div className="relative w-full max-w-md rounded-2xl border border-space-700/80 bg-space-900/95 p-6 shadow-2xl backdrop-blur-xl text-slate-100 overflow-hidden">
        {/* Ambient Glow */}
        <div className="absolute -top-20 -right-20 w-44 h-44 bg-cyan-500/15 rounded-full blur-3xl pointer-events-none" />

        {/* Header */}
        <div className="flex items-center justify-between pb-3.5 mb-4 border-b border-space-800">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-cyan-500/20 to-blue-600/20 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
              <Sparkles className="w-4 h-4 text-cyan-400" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-100">Google Gemini Vision LLM</h3>
              <p className="text-[11px] text-slate-400 font-mono">Connect Gemini for satellite feature detection</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-space-800 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="space-y-4 text-xs">
          <p className="text-slate-300 leading-relaxed">
            Google Gemini Vision processes satellite imagery directly to detect every detail of water bodies, glaciers, ice, vegetation, and urban infrastructure.
          </p>

          <div>
            <label className="block text-[11px] font-mono text-slate-300 mb-1.5 font-medium">
              Google Gemini API Key
            </label>
            <div className="relative">
              <input
                type="password"
                value={keyInput}
                onChange={(e) => setKeyInput(e.target.value)}
                placeholder="AIzaSy..."
                className="w-full pl-9 pr-3 py-2 rounded-xl bg-space-950 border border-space-700 focus:border-cyan-500 focus:outline-none text-slate-100 font-mono text-xs placeholder:text-slate-600"
              />
              <Key className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-3" />
            </div>
            <div className="mt-2 flex items-center justify-between text-[10.5px]">
              <a 
                href="https://aistudio.google.com/app/apikey" 
                target="_blank" 
                rel="noreferrer"
                className="text-cyan-400 hover:text-cyan-300 flex items-center space-x-1"
              >
                <span>Get a free key from Google AI Studio</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>

          {statusMsg && (
            <div className={`p-2.5 rounded-xl border flex items-center space-x-2 text-[11px] ${
              statusMsg.type === 'success' 
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300' 
                : 'bg-rose-500/10 border-rose-500/30 text-rose-300'
            }`}>
              {statusMsg.type === 'success' ? (
                <CheckCircle className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
              ) : (
                <AlertCircle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
              )}
              <span>{statusMsg.text}</span>
            </div>
          )}

          <div className="flex items-center justify-end space-x-2.5 pt-3 border-t border-space-800">
            <button
              onClick={onClose}
              className="px-3 py-1.5 rounded-lg text-xs text-slate-400 hover:text-slate-200"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={isValidating}
              className="px-4 py-1.5 rounded-xl text-xs font-medium bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-md shadow-cyan-500/20 flex items-center space-x-1.5 disabled:opacity-50"
            >
              {isValidating ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Connecting...</span>
                </>
              ) : (
                <span>Save Key</span>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
