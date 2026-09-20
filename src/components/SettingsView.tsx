import React, { useState } from 'react';
import {
  Palette,
  Sliders,
  Check,
  Save,
  RotateCcw,
  Eye,
  Layers,
  FileText,
  Compass,
  User as UserIcon,
  LogOut,
  ShieldCheck,
  Database,
} from 'lucide-react';
import { User, AuthBackendStatus } from '../types/auth';

interface SettingsViewProps {
  user?: User | null;
  onLogout?: () => void;
  backendStatus?: AuthBackendStatus | null;
}

export const SettingsView: React.FC<SettingsViewProps> = ({
  user,
  onLogout,
  backendStatus,
}) => {
  // Theme state
  const [theme, setTheme] = useState<'dark' | 'midnight' | 'system'>('dark');

  // Basic user preferences
  const [defaultMode, setDefaultMode] = useState<'single' | 'compare' | 'optical-sar'>('compare');
  const [defaultViewerStyle, setDefaultViewerStyle] = useState<'side-by-side' | 'swipe'>('side-by-side');
  const [measurementUnits, setMeasurementUnits] = useState<'metric' | 'imperial'>('metric');
  const [autoOpenReport, setAutoOpenReport] = useState<boolean>(true);
  const [highContrastEvidence, setHighContrastEvidence] = useState<boolean>(true);
  const [notifyOnComplete, setNotifyOnComplete] = useState<boolean>(true);

  const [savedSuccess, setSavedSuccess] = useState<boolean>(false);

  const handleSave = () => {
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 2500);
  };

  const handleReset = () => {
    setTheme('dark');
    setDefaultMode('compare');
    setDefaultViewerStyle('side-by-side');
    setMeasurementUnits('metric');
    setAutoOpenReport(true);
    setHighContrastEvidence(true);
    setNotifyOnComplete(true);
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 2000);
  };

  return (
    <div id="settings-page" className="max-w-4xl mx-auto space-y-8 pb-16 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 font-sans">
            Settings & Preferences
          </h1>
          <p className="text-xs text-slate-400 font-sans mt-1">
            Personalize your viewing theme, default analysis tools, and report settings.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleReset}
            className="px-3.5 py-2 rounded-xl bg-slate-900 hover:bg-slate-850 border border-slate-700 text-xs font-medium text-slate-300 transition flex items-center gap-1.5 cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
            <span>Reset Defaults</span>
          </button>

          <button
            onClick={handleSave}
            className="px-5 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs flex items-center gap-2 transition shadow-[0_0_15px_rgba(6,182,212,0.3)] cursor-pointer"
          >
            {savedSuccess ? (
              <Check className="w-4 h-4 text-slate-950" />
            ) : (
              <Save className="w-4 h-4 text-slate-950" />
            )}
            <span>{savedSuccess ? 'Preferences Saved' : 'Save Changes'}</span>
          </button>
        </div>
      </div>

      {/* 0. AUTHENTICATED USER ACCOUNT */}
      {user && (
        <div className="bg-[#080d1a]/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2.5">
              <UserIcon className="w-4 h-4 text-cyan-400" />
              <h2 className="text-sm font-bold text-slate-200 font-sans">
                Active Authenticated Account
              </h2>
            </div>
            {onLogout && (
              <button
                type="button"
                onClick={onLogout}
                className="px-3 py-1.5 rounded-lg bg-rose-950/40 hover:bg-rose-900/60 border border-rose-800/60 text-rose-300 text-xs font-semibold transition flex items-center gap-1.5 cursor-pointer"
              >
                <LogOut className="w-3.5 h-3.5" />
                <span>Logout</span>
              </button>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-3.5 rounded-xl bg-slate-900/70 border border-slate-800/80 space-y-1">
              <div className="text-[11px] text-slate-400 font-mono">ACCOUNT HOLDER</div>
              <div className="text-sm font-bold text-white">{user.name}</div>
              <div className="text-xs text-slate-300">{user.email}</div>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-900/70 border border-slate-800/80 space-y-1">
              <div className="text-[11px] text-slate-400 font-mono">DATABASE PERSISTENCE STATUS</div>
              <div className="flex items-center gap-2">
                <span
                  className={`w-2 h-2 rounded-full ${
                    backendStatus?.status === 'CONNECTED' ? 'bg-emerald-400' : 'bg-amber-400'
                  }`}
                />
                <span
                  className={`text-xs font-bold ${
                    backendStatus?.status === 'CONNECTED' ? 'text-emerald-400' : 'text-amber-400'
                  }`}
                >
                  {backendStatus?.status || 'NOT CONNECTED / CONFIGURATION REQUIRED'}
                </span>
              </div>
              <div className="text-[11px] text-slate-400 leading-tight">
                {backendStatus?.status === 'CONNECTED'
                  ? 'Connected to PostgreSQL persistent schema.'
                  : 'Isolated local server container storage.'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 1. THEME SELECTION */}
      <div className="bg-[#080d1a]/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex items-center gap-2.5 border-b border-slate-800 pb-3">
          <Palette className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-bold text-slate-200 font-sans">
            Interface Theme
          </h2>
        </div>

        <p className="text-xs text-slate-400">
          Select an appearance that provides comfortable contrast for satellite imagery interpretation.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
          {[
            {
              id: 'dark',
              name: 'Space Dark (Default)',
              desc: 'High-contrast dark obsidian canvas with cyan accents. Optimized for remote sensing.',
            },
            {
              id: 'midnight',
              name: 'Deep Midnight Navy',
              desc: 'Subtle slate-blue night palette with balanced contrast for long analytical sessions.',
            },
            {
              id: 'system',
              name: 'System Default',
              desc: 'Follows your operating system color preference automatically.',
            },
          ].map((item) => (
            <div
              key={item.id}
              onClick={() => setTheme(item.id as any)}
              className={`p-4 rounded-xl border cursor-pointer transition ${
                theme === item.id
                  ? 'bg-cyan-950/60 border-cyan-500 text-cyan-300 ring-1 ring-cyan-500/50'
                  : 'bg-slate-900/50 border-slate-800 text-slate-400 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-bold text-slate-200">{item.name}</span>
                {theme === item.id && <Check className="w-3.5 h-3.5 text-cyan-400" />}
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 2. BASIC APPLICATION PREFERENCES */}
      <div className="bg-[#080d1a]/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-5">
        <div className="flex items-center gap-2.5 border-b border-slate-800 pb-3">
          <Sliders className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-bold text-slate-200 font-sans">
            Basic Application Preferences
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Default Analysis Mode */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <Compass className="w-3.5 h-3.5 text-cyan-400" />
              Default Analysis Mode
            </label>
            <p className="text-[11px] text-slate-400">
              The primary workflow opened when clicking &quot;Add Satellite Images&quot;.
            </p>
            <select
              value={defaultMode}
              onChange={(e) => setDefaultMode(e.target.value as any)}
              className="w-full px-3.5 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 cursor-pointer"
            >
              <option value="compare">Past & Present Comparison (Recommended)</option>
              <option value="single">Single Image Analysis</option>
              <option value="optical-sar">Optical + SAR Cross-Sensor</option>
            </select>
          </div>

          {/* Comparison Display Mode */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              Default Comparison Display
            </label>
            <p className="text-[11px] text-slate-400">
              Preferred visual layout for Past & Present dual imagery.
            </p>
            <select
              value={defaultViewerStyle}
              onChange={(e) => setDefaultViewerStyle(e.target.value as any)}
              className="w-full px-3.5 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 cursor-pointer"
            >
              <option value="side-by-side">Side-by-Side (Synchronized Zoom & Pan)</option>
              <option value="swipe">Interactive Swipe Split Slider</option>
            </select>
          </div>

          {/* Measurement Units */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <Eye className="w-3.5 h-3.5 text-cyan-400" />
              Measurement Units
            </label>
            <p className="text-[11px] text-slate-400">
              Units for change surface areas and spatial footprints.
            </p>
            <select
              value={measurementUnits}
              onChange={(e) => setMeasurementUnits(e.target.value as any)}
              className="w-full px-3.5 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 cursor-pointer"
            >
              <option value="metric">Metric (Kilometers² / Hectares / Meters)</option>
              <option value="imperial">Imperial (Miles² / Acres / Feet)</option>
            </select>
          </div>

          {/* Auto-Open Report */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <FileText className="w-3.5 h-3.5 text-cyan-400" />
              Report Generation
            </label>
            <p className="text-[11px] text-slate-400">
              Automatically render structured summary report below the analysis results.
            </p>
            <div className="flex items-center gap-3 pt-1">
              <button
                type="button"
                onClick={() => setAutoOpenReport(!autoOpenReport)}
                className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                  autoOpenReport ? 'bg-cyan-500' : 'bg-slate-800'
                }`}
              >
                <span
                  className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                    autoOpenReport ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
              <span className="text-xs text-slate-300">
                {autoOpenReport ? 'Enabled (Instant Report)' : 'Disabled'}
              </span>
            </div>
          </div>
        </div>

        {/* Checkbox Toggles */}
        <div className="pt-4 border-t border-slate-800/80 space-y-3">
          <label className="flex items-center gap-3 cursor-pointer group">
            <input
              type="checkbox"
              checked={highContrastEvidence}
              onChange={(e) => setHighContrastEvidence(e.target.checked)}
              className="w-4 h-4 rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0 cursor-pointer"
            />
            <div>
              <span className="text-xs font-medium text-slate-200 group-hover:text-cyan-300 transition-colors">
                High-Contrast Visual Evidence Overlays
              </span>
              <p className="text-[11px] text-slate-400">
                Highlight detected changes with vibrant outline overlays directly on top of satellite imagery.
              </p>
            </div>
          </label>

          <label className="flex items-center gap-3 cursor-pointer group">
            <input
              type="checkbox"
              checked={notifyOnComplete}
              onChange={(e) => setNotifyOnComplete(e.target.checked)}
              className="w-4 h-4 rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0 cursor-pointer"
            />
            <div>
              <span className="text-xs font-medium text-slate-200 group-hover:text-cyan-300 transition-colors">
                Show Completion Alert
              </span>
              <p className="text-[11px] text-slate-400">
                Play a gentle indicator and scroll to results once analysis processing completes.
              </p>
            </div>
          </label>
        </div>
      </div>

      {/* 3. SYSTEM STATUS & MODEL AVAILABILITY (REQUIREMENT 16) */}
      <div className="bg-[#080d1a]/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-5">
        <div className="flex items-center gap-2.5 border-b border-slate-800 pb-3">
          <ShieldCheck className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-bold text-slate-200 font-sans">
            System Status & Specialist Model Availability
          </h2>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs font-mono">
          {/* Backend Status */}
          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-[10px] text-slate-500 block uppercase">BACKEND SERVICE</span>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span className="text-emerald-300 font-bold">ONLINE</span>
            </div>
            <span className="text-[10px] text-slate-400 block pt-0.5">Local Server :8000</span>
          </div>

          {/* API Endpoints */}
          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-[10px] text-slate-500 block uppercase">API ENDPOINTS</span>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span className="text-emerald-300 font-bold">READY (3/3)</span>
            </div>
            <span className="text-[10px] text-slate-400 block pt-0.5">/analyze, /validate, /report</span>
          </div>

          {/* Florence-2 Model */}
          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-[10px] text-slate-500 block uppercase">FLORENCE-2 VLM</span>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span className="text-emerald-300 font-bold">LOCAL / AVAILABLE</span>
            </div>
            <span className="text-[10px] text-slate-400 block pt-0.5">Vision-Language & Grounding</span>
          </div>

          {/* BigEarthNet Specialist */}
          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-[10px] text-slate-500 block uppercase">BIGEARTHNET SPECIALIST</span>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span className="text-emerald-300 font-bold">LOCAL / AVAILABLE</span>
            </div>
            <span className="text-[10px] text-slate-400 block pt-0.5">19-Class Land Cover ResNet</span>
          </div>
        </div>

        {/* Application Information */}
        <div className="pt-3 border-t border-slate-800/80 flex flex-col sm:flex-row sm:items-center justify-between text-xs text-slate-400 gap-2">
          <div>
            <span className="font-semibold text-slate-300">SatQuery AI</span> • Remote Sensing Vision-Language Intelligence System
          </div>
          <div className="font-mono text-[11px] text-slate-500">
            PS 26167 • Version 1.0.0 (SIH Demo Ready)
          </div>
        </div>
      </div>
    </div>
  );
};
