import React, { useState } from 'react';
import { 
  Globe2, 
  Layers, 
  GitCompare, 
  Radio, 
  BookOpen, 
  Sparkles, 
  ShieldCheck,
  LogOut,
  User as UserIcon
} from 'lucide-react';
import { DemoScenario, User } from '../types';

interface NavbarProps {
  currentTab: string;
  setCurrentTab: (tab: string) => void;
  demoScenarios: Record<string, DemoScenario>;
  onSelectScenario: (scenarioId: string) => void;
  isLoadingDemo?: boolean;
  isLanding?: boolean;
  currentUser?: User | null;
  onLogout?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentTab,
  setCurrentTab,
  demoScenarios,
  onSelectScenario,
  isLoadingDemo = false,
  isLanding = false,
  currentUser = null,
  onLogout
}) => {
  const [demoMenuOpen, setDemoMenuOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);

  const navItems = [
    { id: 'workspace', label: 'Single image VQA', icon: Layers, desc: 'Interactive Multi-Modal Analysis' },
    { id: 'change', label: 'Bi-Temporal Change', icon: GitCompare, desc: 'T1 vs T2 Change Detection' },
    { id: 'opt-sar', label: 'Optical + SAR Fusion', icon: Radio, desc: 'Dual-Sensor Structural Fusion' },
    { id: 'docs', label: 'Documentation', icon: BookOpen, desc: 'Architecture & Geospatial Specs' },
  ];

  return (
    <aside 
      className={`h-screen w-16 sm:w-20 flex flex-col justify-between items-center py-5 z-50 shrink-0 transition-all duration-300 ${
        isLanding
          ? 'fixed top-0 left-0 bg-transparent border-r-0 border-transparent shadow-none pointer-events-auto'
          : 'sticky top-0 bg-black/50 backdrop-blur-2xl border-r border-neutral-800/70 shadow-2xl'
      }`}
    >
      
      {/* Top: Brand Logo */}
      <div className="flex flex-col items-center space-y-6">
        <div 
          onClick={() => setCurrentTab('landing')}
          className="relative group flex items-center justify-center cursor-pointer"
        >
          <div className={`w-11 h-11 rounded-2xl p-0.5 shadow-lg group-hover:scale-105 transition-all duration-300 ${
            isLanding
              ? 'bg-black/40 border border-neutral-800/60 group-hover:border-neutral-600 backdrop-blur-xl'
              : 'bg-black border border-neutral-800 group-hover:border-neutral-600'
          }`}>
            <div className="w-full h-full bg-transparent rounded-[14px] flex items-center justify-center">
              <Globe2 className="w-5 h-5 text-white group-hover:rotate-12 transition-transform duration-300" />
            </div>
          </div>

          {/* Hover Tooltip with Smooth Transition */}
          <div className="absolute left-full ml-3.5 px-3 py-1.5 rounded-xl bg-black border border-neutral-800 text-white shadow-2xl backdrop-blur-md whitespace-nowrap opacity-0 -translate-x-2 pointer-events-none group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-200 ease-out z-50 flex items-center space-x-2">
            <span className="text-xs font-bold tracking-wider">SATQUERY</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-neutral-900 text-neutral-300 font-mono border border-neutral-800">Home</span>
            {/* Arrow */}
            <div className="absolute -left-1 top-1/2 -translate-y-1/2 w-2 h-2 bg-black border-l border-b border-neutral-800 rotate-45" />
          </div>
        </div>

        {/* Divider */}
        <div className={`w-8 h-[1px] ${isLanding ? 'bg-neutral-800/40' : 'bg-neutral-800'}`} />

        {/* Middle: Navigation Symbol List */}
        <nav className="flex flex-col items-center space-y-3">
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = currentTab === item.id;
            return (
              <div key={item.id} className="relative group flex items-center justify-center">
                <button
                  onClick={() => setCurrentTab(item.id)}
                  aria-label={item.label}
                  className={`relative flex items-center justify-center w-11 h-11 rounded-2xl transition-all duration-200 ${
                    active
                      ? 'bg-neutral-900/90 text-white border border-neutral-700 shadow-xl scale-105 backdrop-blur-md'
                      : isLanding
                        ? 'text-neutral-400 hover:text-white bg-black/40 hover:bg-black/75 border border-neutral-800/50 hover:border-neutral-700/80 backdrop-blur-md'
                        : 'text-neutral-400 hover:text-white hover:bg-neutral-900/60 hover:border hover:border-neutral-800'
                  }`}
                >
                  {/* Active Indicator Bar on left edge */}
                  {active && (
                    <span className="absolute -left-2.5 sm:-left-3.5 w-1 h-6 rounded-r-full bg-white shadow-sm" />
                  )}
                  <Icon className={`w-5 h-5 transition-transform duration-200 group-hover:scale-110 ${active ? 'text-white' : 'text-neutral-400'}`} />
                </button>

                {/* Hover Tooltip with Name and Smooth Transition */}
                <div className="absolute left-full ml-3.5 px-3 py-2 rounded-xl bg-black border border-neutral-800 shadow-2xl backdrop-blur-md whitespace-nowrap opacity-0 -translate-x-2 pointer-events-none group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-200 ease-out z-50 flex flex-col min-w-[140px]">
                  <div className="flex items-center justify-between space-x-2">
                    <span className="text-xs font-semibold text-white tracking-wide">{item.label}</span>
                    {active && (
                      <span className="text-[9px] px-1.5 py-0.2 rounded bg-neutral-900 text-neutral-300 font-mono border border-neutral-800">Active</span>
                    )}
                  </div>
                  <span className="text-[10px] text-slate-400 font-mono mt-0.5">{item.desc}</span>
                  {/* Tooltip Arrow */}
                  <div className="absolute -left-1 top-1/2 -translate-y-1/2 w-2 h-2 bg-black border-l border-b border-neutral-800 rotate-45" />
                </div>
              </div>
            );
          })}
        </nav>
      </div>

      {/* Bottom Actions: Demo Scenarios & Status Indicator */}
      <div className="flex flex-col items-center space-y-4">
        
        {/* Curated Demo Scenarios Button */}
        <div className="relative">
          <div className="relative group flex items-center justify-center">
            <button
              onClick={() => setDemoMenuOpen(!demoMenuOpen)}
              disabled={isLoadingDemo}
              aria-label="Load Curated Demo"
              className={`relative flex items-center justify-center w-11 h-11 rounded-2xl text-neutral-300 hover:text-white shadow-lg transition-all duration-200 active:scale-95 disabled:opacity-50 ${
                isLanding
                  ? 'bg-black/40 border border-neutral-800/60 hover:border-neutral-600 backdrop-blur-xl'
                  : 'bg-black border border-neutral-800 hover:border-neutral-600'
              } ${demoMenuOpen ? 'ring-2 ring-neutral-500 bg-neutral-900 text-white' : ''}`}
            >
              <Sparkles className="w-5 h-5" />
            </button>

            {/* Hover Tooltip for Demo Button */}
            {!demoMenuOpen && (
              <div className="absolute left-full ml-3.5 px-3 py-2 rounded-xl bg-black border border-neutral-800 shadow-2xl backdrop-blur-md whitespace-nowrap opacity-0 -translate-x-2 pointer-events-none group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-200 ease-out z-50 flex flex-col">
                <span className="text-xs font-semibold text-white">Curated EO Demos</span>
                <span className="text-[10px] text-slate-400 font-mono mt-0.5">Authentic GeoTIFF Scenarios</span>
                {/* Arrow */}
                <div className="absolute -left-1 top-1/2 -translate-y-1/2 w-2 h-2 bg-black border-l border-b border-neutral-800 rotate-45" />
              </div>
            )}
          </div>

          {/* Demo Flyout Menu (Anchored to right of the sidebar) */}
          {demoMenuOpen && (
            <div 
              className="absolute left-full bottom-0 ml-3.5 w-80 rounded-2xl bg-black border border-neutral-800 shadow-2xl p-2.5 z-50 backdrop-blur-xl animate-in fade-in slide-in-from-left-2 duration-150"
              onMouseLeave={() => setDemoMenuOpen(false)}
            >
              <div className="px-3 py-2 border-b border-neutral-800">
                <span className="text-[11px] font-mono text-neutral-300 uppercase tracking-wider font-semibold block">
                  Curated Remote Sensing Scenarios
                </span>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Real GeoTIFF rasters with authentic geospatial tags
                </p>
              </div>

              <div className="py-1.5 space-y-1 max-h-[380px] overflow-y-auto">
                {Object.values(demoScenarios).map((scenario) => (
                  <button
                    key={scenario.id}
                    onClick={() => {
                      onSelectScenario(scenario.id);
                      setDemoMenuOpen(false);
                    }}
                    className="w-full text-left px-3 py-2.5 rounded-xl hover:bg-neutral-900 text-xs transition-colors flex flex-col group border border-transparent hover:border-neutral-800"
                  >
                    <div className="flex items-center justify-between text-slate-200 group-hover:text-white font-medium">
                      <span>{scenario.title}</span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-neutral-900 text-slate-400 font-mono border border-neutral-800">
                        {scenario.task.replace('_', ' ')}
                      </span>
                    </div>
                    <span className="text-[11px] text-slate-400 mt-1 line-clamp-1">
                      "{scenario.default_query}"
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Divider */}
        <div className={`w-8 h-[1px] ${isLanding ? 'bg-neutral-800/40' : 'bg-neutral-800'}`} />

        {/* User Profile & Logout Menu */}
        {currentUser && (
          <div className="relative">
            <div className="relative group flex items-center justify-center">
              <button
                onClick={() => setUserMenuOpen(!userMenuOpen)}
                aria-label="User Account Menu"
                className={`relative flex items-center justify-center w-10 h-10 rounded-2xl overflow-hidden shadow-lg transition-all duration-200 active:scale-95 border ${
                  userMenuOpen 
                    ? 'ring-2 ring-cyan-400 border-cyan-400' 
                    : 'border-neutral-700 hover:border-neutral-500'
                } bg-neutral-900`}
              >
                {currentUser.picture ? (
                  <img 
                    src={currentUser.picture} 
                    alt={currentUser.name} 
                    className="w-full h-full object-cover" 
                    referrerPolicy="no-referrer"
                  />
                ) : (
                  <div className="w-full h-full bg-cyan-600/30 flex items-center justify-center text-cyan-300 font-bold text-xs font-mono">
                    {currentUser.name ? currentUser.name.charAt(0).toUpperCase() : 'U'}
                  </div>
                )}
              </button>

              {/* Hover Tooltip */}
              {!userMenuOpen && (
                <div className="absolute left-full ml-3.5 px-3 py-1.5 rounded-xl bg-black border border-neutral-800 text-white shadow-2xl backdrop-blur-md whitespace-nowrap opacity-0 -translate-x-2 pointer-events-none group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-200 ease-out z-50 flex flex-col">
                  <span className="text-xs font-semibold text-white">{currentUser.name}</span>
                  <span className="text-[10px] text-slate-400 font-mono">{currentUser.email}</span>
                  <div className="absolute -left-1 top-1/2 -translate-y-1/2 w-2 h-2 bg-black border-l border-b border-neutral-800 rotate-45" />
                </div>
              )}
            </div>

            {/* User Account Flyout Menu */}
            {userMenuOpen && (
              <div 
                className="absolute left-full bottom-0 ml-3.5 w-64 rounded-2xl bg-black border border-neutral-800 shadow-2xl p-3 z-50 backdrop-blur-xl animate-in fade-in slide-in-from-left-2 duration-150"
                onMouseLeave={() => setUserMenuOpen(false)}
              >
                {/* User Header */}
                <div className="flex items-center space-x-3 pb-3 mb-2 border-b border-neutral-800">
                  {currentUser.picture ? (
                    <img 
                      src={currentUser.picture} 
                      alt={currentUser.name} 
                      className="w-9 h-9 rounded-xl object-cover border border-neutral-700" 
                      referrerPolicy="no-referrer"
                    />
                  ) : (
                    <div className="w-9 h-9 rounded-xl bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center text-cyan-300 font-bold text-xs">
                      {currentUser.name.charAt(0).toUpperCase()}
                    </div>
                  )}
                  <div className="truncate">
                    <span className="text-xs font-bold text-white block truncate">
                      {currentUser.name}
                    </span>
                    <span className="text-[11px] text-slate-400 block truncate font-mono">
                      {currentUser.email}
                    </span>
                  </div>
                </div>

                {/* Account Status */}
                <div className="px-2 py-1.5 mb-2 rounded-lg bg-neutral-900/80 border border-neutral-800/80 flex items-center justify-between text-[10.5px] font-mono text-cyan-300">
                  <span className="flex items-center space-x-1.5">
                    <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                    <span>Google Verified</span>
                  </span>
                  <span className="text-[9px] text-neutral-400">OAuth 2.0</span>
                </div>

                {/* Logout Action Button */}
                <button
                  onClick={() => {
                    setUserMenuOpen(false);
                    if (onLogout) onLogout();
                  }}
                  className="w-full flex items-center space-x-2 px-3 py-2 rounded-xl text-xs font-medium text-rose-300 hover:text-white bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/20 hover:border-rose-500/40 transition-colors"
                >
                  <LogOut className="w-4 h-4 text-rose-400" />
                  <span>Log Out</span>
                </button>
              </div>
            )}
          </div>
        )}

        {/* Divider */}
        <div className={`w-8 h-[1px] ${isLanding ? 'bg-neutral-800/40' : 'bg-neutral-800'}`} />

        {/* Status Indicator Tooltip */}
        <div className="relative group flex items-center justify-center cursor-default">
          <div className={`w-8 h-8 rounded-xl flex items-center justify-center ${
            isLanding
              ? 'bg-black/40 border border-neutral-800/60 backdrop-blur-xl'
              : 'bg-black border border-neutral-800'
          }`}>
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-sm shadow-emerald-400 animate-pulse"></span>
          </div>

          <div className="absolute left-full ml-3.5 px-3 py-1.5 rounded-xl bg-black border border-neutral-800 text-emerald-400 text-[11px] font-mono shadow-2xl backdrop-blur-md whitespace-nowrap opacity-0 -translate-x-2 pointer-events-none group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-200 ease-out z-50 flex items-center space-x-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span>BigEarthNet-19 Ready</span>
            {/* Arrow */}
            <div className="absolute -left-1 top-1/2 -translate-y-1/2 w-2 h-2 bg-black border-l border-b border-neutral-800 rotate-45" />
          </div>
        </div>

      </div>

    </aside>
  );
};
