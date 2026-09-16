import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { LandingPage } from './pages/LandingPage';
import { WorkspacePage } from './pages/WorkspacePage';
import { ChangeStudioPage } from './pages/ChangeStudioPage';
import { OpticalSarPage } from './pages/OpticalSarPage';
import { DocsPage } from './pages/DocsPage';
import { AnalysisResult, DemoScenario, ImageMetadata } from './types';
import { api } from './services/api';
import earthBg from './assets/earth-horizon.png';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<string>('landing');
  const [demoScenarios, setDemoScenarios] = useState<Record<string, DemoScenario>>({});

  // Completely isolated state for Workspace Studio
  const [workspaceAnalysis, setWorkspaceAnalysis] = useState<AnalysisResult | null>(null);
  const [workspaceImages, setWorkspaceImages] = useState<Array<{
    filename: string;
    server_path: string;
    preview_url: string;
    metadata: ImageMetadata;
  }>>([]);

  // Completely isolated state for Bi-Temporal Change Studio
  const [changeAnalysis, setChangeAnalysis] = useState<AnalysisResult | null>(null);
  const [changeImages, setChangeImages] = useState<Array<{
    filename: string;
    server_path: string;
    preview_url: string;
    metadata: ImageMetadata;
  }>>([]);

  // Completely isolated state for Optical + SAR Fusion Studio
  const [optSarAnalysis, setOptSarAnalysis] = useState<AnalysisResult | null>(null);
  const [optSarImages, setOptSarImages] = useState<Array<{
    filename: string;
    server_path: string;
    preview_url: string;
    metadata: ImageMetadata;
  }>>([]);

  const [isLoadingDemo, setIsLoadingDemo] = useState(false);

  useEffect(() => {
    // Fetch demo scenarios on initial load
    api.getDemoScenarios().then(res => {
      setDemoScenarios(res.scenarios || {});
    }).catch(console.error);
  }, []);

  const handleSelectScenario = async (scenarioId: string) => {
    try {
      setIsLoadingDemo(true);
      const res = await api.loadDemoScenario(scenarioId);
      const loaded = res.loaded_images || [];
      // Mount raster images cleanly without triggering automatic scene description
      if (res.task === 'CHANGE_ANALYSIS' || res.task === 'CHANGE_VQA') {
        setChangeImages(loaded);
        setChangeAnalysis(null);
        setCurrentTab('change');
      } else if (res.task === 'OPTICAL_SAR_ANALYSIS') {
        setOptSarImages(loaded);
        setOptSarAnalysis(null);
        setCurrentTab('opt-sar');
      } else {
        setWorkspaceImages(loaded);
        setWorkspaceAnalysis(null);
        setCurrentTab('workspace');
      }
    } catch (err) {
      console.error('Failed to load demo scenario', err);
    } finally {
      setIsLoadingDemo(false);
    }
  };

  return (
    <div className="h-screen w-screen bg-black text-slate-100 flex flex-row antialiased overflow-hidden relative selection:bg-cyan-500/30">
      
      {/* ============================================================ */}
      {/* CINEMATIC STATIONARY EARTH ORBIT BACKGROUND LAYER            */}
      {/* Persistent across all views: Landing, Workspace, Studios    */}
      {/* ============================================================ */}
      <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden select-none bg-black">
        {/* Authentic Earth Horizon Orbit Background */}
        <img 
          src={earthBg} 
          alt="Earth Horizon from Space" 
          className="absolute inset-0 w-full h-full object-cover object-center opacity-75 scale-105 filter brightness-95 contrast-110"
        />

        {/* Seamless Smooth Dark Gradient Vignettes (Atmospheric depth) */}
        <div className="absolute inset-0 bg-gradient-to-b from-black/80 via-black/35 to-black/90" />
        <div className="absolute inset-0 bg-gradient-to-r from-black/75 via-transparent to-black/75" />

        {/* Stationary Geospatial Coordinate Grid */}
        <div className="absolute inset-0 bg-geo-grid opacity-20" />

        {/* Technical HUD Coordinate Stamps */}
        <div className="absolute top-5 right-8 text-[9px] font-mono text-neutral-400/50 flex items-center space-x-2">
          <span>EO-REF: 12°58'N 77°35'E</span>
          <span>•</span>
          <span>WGS 84 / UTM 43N</span>
        </div>
      </div>

      <Navbar
        currentTab={currentTab}
        setCurrentTab={setCurrentTab}
        demoScenarios={demoScenarios}
        onSelectScenario={handleSelectScenario}
        isLoadingDemo={isLoadingDemo}
        isLanding={currentTab === 'landing'}
      />

      <main className="flex-1 min-w-0 h-screen overflow-y-auto overflow-x-hidden relative bg-transparent scroll-smooth z-10">
        {currentTab === 'landing' && (
          <LandingPage
            onStartAnalysis={() => setCurrentTab('workspace')}
            onSelectScenario={handleSelectScenario}
            demoScenarios={demoScenarios}
          />
        )}

        {currentTab === 'workspace' && (
          <WorkspacePage
            currentAnalysis={workspaceAnalysis}
            setCurrentAnalysis={setWorkspaceAnalysis}
            activeImages={workspaceImages}
            setActiveImages={setWorkspaceImages}
            onViewChangeStudio={() => setCurrentTab('change')}
            onViewOpticalSar={() => setCurrentTab('opt-sar')}
          />
        )}

        {currentTab === 'change' && (
          <ChangeStudioPage
            currentAnalysis={changeAnalysis}
            setCurrentAnalysis={setChangeAnalysis}
            activeImages={changeImages}
            setActiveImages={setChangeImages}
            onLoadChangeDemo={() => handleSelectScenario('scenario_3_change')}
          />
        )}

        {currentTab === 'opt-sar' && (
          <OpticalSarPage
            currentAnalysis={optSarAnalysis}
            setCurrentAnalysis={setOptSarAnalysis}
            activeImages={optSarImages}
            setActiveImages={setOptSarImages}
            onLoadOpticalSarDemo={() => handleSelectScenario('scenario_4_optical_sar')}
          />
        )}

        {currentTab === 'docs' && (
          <DocsPage />
        )}
      </main>
    </div>
  );
};

export default App;
