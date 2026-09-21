import React, { useState, useRef, useCallback } from 'react';
import { Sliders, Calendar, ArrowLeftRight, Activity } from 'lucide-react';

interface SwipeViewerProps {
  beforeImageUrl: string;
  afterImageUrl: string;
  beforeLabel?: string;
  afterLabel?: string;
  heightClass?: string;
}

export const SwipeViewer: React.FC<SwipeViewerProps> = ({
  beforeImageUrl,
  afterImageUrl,
  beforeLabel = "T1 (Earlier Reference)",
  afterLabel = "T2 (Later Monitoring)",
  heightClass = "h-[440px]"
}) => {
  const [sliderPosition, setSliderPosition] = useState<number>(50); // percentage
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const width = rect.width;
    const percentage = Math.max(0, Math.min(100, (x / width) * 100));
    setSliderPosition(percentage);
  }, []);

  const handleMouseDown = () => setIsDragging(true);
  const handleMouseUp = () => setIsDragging(false);

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      handleMove(e.clientX);
    }
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (e.touches.length > 0) {
      handleMove(e.touches[0].clientX);
    }
  };

  return (
    <div 
      className="relative rounded-2xl border border-space-700/80 bg-space-950 overflow-hidden shadow-2xl select-none"
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Top Header Labels */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-space-900/90 border-b border-space-800 text-xs font-mono">
        <div className="flex items-center space-x-1.5 text-cyan-400">
          <Calendar className="w-3.5 h-3.5" />
          <span>{beforeLabel}</span>
        </div>
        <div className="flex items-center space-x-1.5 text-slate-400">
          <ArrowLeftRight className="w-3.5 h-3.5" />
          <span>Swipe Curtain: {Math.round(sliderPosition)}%</span>
        </div>
        <div className="flex items-center space-x-1.5 text-amber-400">
          <Calendar className="w-3.5 h-3.5" />
          <span>{afterLabel}</span>
        </div>
      </div>

      {/* Swipe Canvas Container */}
      <div 
        ref={containerRef}
        onMouseMove={handleMouseMove}
        onTouchMove={handleTouchMove}
        className={`relative w-full ${heightClass} bg-space-950 flex items-center justify-center overflow-hidden cursor-ew-resize`}
      >
        {/* Underneath Layer: AFTER IMAGE (T2) */}
        <img 
          src={afterImageUrl} 
          alt="After Capture" 
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />

        {/* Top Layer (Clipped): BEFORE IMAGE (T1) */}
        <div 
          className="absolute inset-0 overflow-hidden pointer-events-none"
          style={{ width: `${sliderPosition}%` }}
        >
          <img 
            src={beforeImageUrl} 
            alt="Before Capture" 
            className="absolute inset-0 max-w-none h-full object-contain pointer-events-none"
            style={{ 
              width: containerRef.current ? `${containerRef.current.clientWidth}px` : '100%' 
            }}
          />
        </div>

        {/* Draggable Divider Line & Knob */}
        <div 
          className="absolute top-0 bottom-0 w-1 bg-cyan-400 shadow-[0_0_12px_rgba(6,182,212,0.8)] z-30 flex items-center justify-center pointer-events-auto"
          style={{ left: `${sliderPosition}%` }}
          onMouseDown={handleMouseDown}
          onTouchStart={() => setIsDragging(true)}
        >
          <div className="w-8 h-8 -ml-3.5 rounded-full bg-cyan-500 border-2 border-white shadow-xl flex items-center justify-center text-white cursor-ew-resize hover:scale-110 active:scale-95 transition-transform">
            <ArrowLeftRight className="w-4 h-4" />
          </div>
        </div>

        {/* Corner Badges */}
        <div className="absolute bottom-3 left-3 px-2 py-1 rounded bg-space-950/80 border border-space-800 text-[10px] font-mono text-cyan-400 pointer-events-none">
          BEFORE (T1)
        </div>
        <div className="absolute bottom-3 right-3 px-2 py-1 rounded bg-space-950/80 border border-space-800 text-[10px] font-mono text-amber-400 pointer-events-none">
          AFTER (T2)
        </div>

      </div>
    </div>
  );
};
