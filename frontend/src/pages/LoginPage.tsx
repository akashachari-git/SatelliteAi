import React, { useState, useEffect, useRef } from 'react';
import { Globe2, ShieldCheck, AlertCircle, Loader2, KeyRound, ExternalLink } from 'lucide-react';
import { api } from '../services/api';
import { User } from '../types';
import earthBg from '../assets/earth-horizon.png';

interface LoginPageProps {
  onLoginSuccess: (user: User) => void;
}

declare global {
  interface Window {
    google?: {
      accounts: {
        id: {
          initialize: (config: any) => void;
          renderButton: (parent: HTMLElement, options: any) => void;
          prompt: (momentListener?: (notification: any) => void) => void;
          cancel: () => void;
        };
      };
    };
  }
}

export const LoginPage: React.FC<LoginPageProps> = ({ onLoginSuccess }) => {
  const [clientId, setClientId] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isInitializing, setIsInitializing] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isGsiLoaded, setIsGsiLoaded] = useState<boolean>(false);
  const [showConfigModal, setShowConfigModal] = useState<boolean>(false);
  const [manualClientIdInput, setManualClientIdInput] = useState<string>('');

  const googleButtonContainerRef = useRef<HTMLDivElement>(null);

  // 1. Fetch server-configured Google Client ID
  useEffect(() => {
    let mounted = true;
    api.getAuthConfig()
      .then((cfg) => {
        if (!mounted) return;
        const envClientId = (import.meta as any).env?.VITE_GOOGLE_CLIENT_ID || '';
        const effectiveId = cfg.client_id || envClientId || localStorage.getItem('satquery_google_client_id') || '';
        setClientId(effectiveId);
        setManualClientIdInput(effectiveId);
      })
      .catch((err) => {
        console.error('Failed to load auth config:', err);
      })
      .finally(() => {
        if (mounted) setIsInitializing(false);
      });

    return () => {
      mounted = false;
    };
  }, []);

  // 2. Poll for Google Identity Services script readiness
  useEffect(() => {
    let interval: any = null;
    let attempts = 0;

    const checkGsi = () => {
      if (window.google?.accounts?.id) {
        setIsGsiLoaded(true);
        if (interval) clearInterval(interval);
      } else {
        attempts++;
        if (attempts > 40 && interval) {
          clearInterval(interval);
        }
      }
    };

    checkGsi();
    if (!window.google?.accounts?.id) {
      interval = setInterval(checkGsi, 200);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, []);

  // 3. Callback when genuine Google Identity Services returns credential
  const handleGoogleCredentialResponse = async (response: any) => {
    if (!response || !response.credential) {
      setErrorMessage('Google sign-in failed. Please try again.');
      return;
    }

    try {
      setIsLoading(true);
      setErrorMessage(null);
      // Verify Google ID token cryptographically on the SatQuery backend
      const authRes = await api.loginWithGoogle(response.credential);
      if (authRes.success && authRes.user) {
        onLoginSuccess(authRes.user);
      } else {
        setErrorMessage('Google sign-in failed. Please try again.');
      }
    } catch (err: any) {
      console.error('Google verification failed:', err);
      setErrorMessage(err.message || 'Google sign-in failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // 4. Initialize Google Identity Services when both GSI and clientId are ready
  useEffect(() => {
    if (!isGsiLoaded || !clientId || !window.google?.accounts?.id) {
      return;
    }

    try {
      window.google.accounts.id.initialize({
        client_id: clientId,
        callback: handleGoogleCredentialResponse,
        auto_select: false,
        cancel_on_tap_outside: true,
      });

      // Render the official Google-hosted sign in button inside hidden/anchor element
      if (googleButtonContainerRef.current) {
        googleButtonContainerRef.current.innerHTML = '';
        window.google.accounts.id.renderButton(googleButtonContainerRef.current, {
          type: 'standard',
          theme: 'filled_black',
          size: 'large',
          text: 'continue_with',
          shape: 'rectangular',
          logo_alignment: 'left',
          width: 320,
        });
      }
    } catch (err) {
      console.error('Failed to initialize Google Identity Services:', err);
    }
  }, [isGsiLoaded, clientId]);

  // 5. Trigger Google-hosted Authentication Flow
  const handleContinueWithGoogle = () => {
    setErrorMessage(null);

    // If client ID is missing, open configuration dialog
    if (!clientId) {
      setShowConfigModal(true);
      return;
    }

    if (!isGsiLoaded || !window.google?.accounts?.id) {
      setErrorMessage('Google sign-in failed. Please try again.');
      return;
    }

    try {
      setIsLoading(true);
      // Attempt to trigger the genuine Google Identity modal / prompt
      window.google.accounts.id.prompt((notification: any) => {
        if (notification.isNotDisplayed() || notification.isSkippedMoment()) {
          // If One Tap is blocked or dismissed, fallback to invoking the rendered official GIS button
          const btn = googleButtonContainerRef.current?.querySelector('div[role="button"]') as HTMLElement | null;
          if (btn) {
            btn.click();
          } else {
            setErrorMessage('Google sign-in failed. Please try again.');
            setIsLoading(false);
          }
        }
      });
    } catch (err) {
      console.error('Error invoking Google Identity Services:', err);
      setErrorMessage('Google sign-in failed. Please try again.');
      setIsLoading(false);
    }
  };

  const handleSaveClientId = (e: React.FormEvent) => {
    e.preventDefault();
    const clean = manualClientIdInput.trim();
    if (!clean) {
      setErrorMessage('Please provide a valid Google Cloud OAuth Web Client ID.');
      return;
    }
    localStorage.setItem('satquery_google_client_id', clean);
    setClientId(clean);
    setShowConfigModal(false);
    setErrorMessage(null);
  };

  return (
    <div className="min-h-screen w-screen bg-black text-slate-100 flex flex-col justify-center items-center relative overflow-hidden selection:bg-cyan-500/30">
      
      {/* Background: Earth Horizon Orbit */}
      <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden select-none bg-black">
        <img 
          src={earthBg} 
          alt="Earth Horizon from Space" 
          className="absolute inset-0 w-full h-full object-cover object-center opacity-65 scale-105 filter brightness-90 contrast-115"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-black/85 via-black/45 to-black/95" />
        <div className="absolute inset-0 bg-geo-grid opacity-25" />
        
        {/* Technical HUD Coordinate Stamp */}
        <div className="absolute top-6 right-8 text-[10px] font-mono text-neutral-400/50 flex items-center space-x-2">
          <span>SECURE GATEWAY</span>
          <span>•</span>
          <span>OAUTH 2.0 / GIS</span>
        </div>
      </div>

      {/* Main Login Card */}
      <div className="relative z-10 w-full max-w-md px-6 sm:px-8 py-10 my-8 mx-auto bg-black/75 backdrop-blur-2xl border border-neutral-800 rounded-3xl shadow-2xl flex flex-col items-center text-center">
        
        {/* Logo */}
        <div className="w-14 h-14 rounded-2xl bg-neutral-900 border border-neutral-700 shadow-xl flex items-center justify-center mb-6 transition-transform hover:scale-105 duration-300">
          <Globe2 className="w-7 h-7 text-white" />
        </div>

        {/* SatQuery AI Brand Name */}
        <div className="text-[11px] font-mono tracking-widest text-cyan-400 uppercase font-semibold mb-2">
          SatQuery AI
        </div>

        {/* Heading */}
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white mb-2 font-sans">
          Welcome to SatQuery AI
        </h1>

        {/* Subtitle */}
        <p className="text-sm text-slate-300 mb-8 max-w-xs font-sans leading-relaxed">
          Interactive Multimodal Remote Sensing Assistant
        </p>

        {/* Error Alert Box */}
        {errorMessage && (
          <div className="w-full mb-6 p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center space-x-2.5 text-left animate-in fade-in duration-200">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
            <span className="font-medium">{errorMessage}</span>
          </div>
        )}

        {/* Primary Action Button: "Continue with Google" */}
        <div className="w-full space-y-4">
          
          {/* Custom Stylized Button (Triggers Real Google Identity Flow) */}
          <button
            onClick={handleContinueWithGoogle}
            disabled={isLoading || isInitializing}
            className="w-full h-12 rounded-xl bg-white hover:bg-neutral-100 text-neutral-900 font-medium text-sm transition-all duration-200 shadow-lg hover:shadow-xl flex items-center justify-center space-x-3 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed group border border-neutral-200"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-neutral-900" />
                <span className="font-sans">Connecting to Google...</span>
              </>
            ) : (
              <>
                {/* Official Google G SVG Icon */}
                <svg className="w-5 h-5 shrink-0" viewBox="0 0 24 24" aria-hidden="true">
                  <path
                    fill="#4285F4"
                    d="M23.745 12.27c0-.7-.06-1.4-.19-2.07H12v4.51h6.6c-.29 1.52-1.14 2.82-2.4 3.68v3.05h3.88c2.27-2.09 3.665-5.17 3.665-9.17z"
                  />
                  <path
                    fill="#34A853"
                    d="M12 24c3.24 0 5.95-1.08 7.93-2.91l-3.88-3.05c-1.08.72-2.45 1.16-4.05 1.16-3.12 0-5.77-2.1-6.72-4.93H1.25v3.15C3.26 21.36 7.33 24 12 24z"
                  />
                  <path
                    fill="#FBBC05"
                    d="M5.28 14.27c-.25-.72-.38-1.49-.38-2.27s.13-1.55.38-2.27V6.58H1.25C.45 8.18 0 9.99 0 12s.45 3.82 1.25 5.42l4.03-3.15z"
                  />
                  <path
                    fill="#EA4335"
                    d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.42-3.42C17.95 1.19 15.24 0 12 0 7.33 0 3.26 2.64 1.25 6.58l4.03 3.15c.95-2.83 3.6-4.98 6.72-4.98z"
                  />
                </svg>
                <span className="font-sans font-medium text-neutral-900 tracking-tight">
                  Continue with Google
                </span>
              </>
            )}
          </button>

          {/* Hidden Container for Google Identity Services Button Element */}
          <div 
            ref={googleButtonContainerRef} 
            className="flex justify-center overflow-hidden h-0 opacity-0 pointer-events-none"
            aria-hidden="true"
          />

          {/* Small Text */}
          <p className="text-xs text-slate-400 font-sans">
            Sign in securely with your Google account
          </p>
        </div>

        {/* Security / Technology Notice */}
        <div className="mt-8 pt-6 border-t border-neutral-800/80 w-full flex items-center justify-center space-x-2 text-[11px] text-slate-400 font-mono">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
          <span>Google Identity Services • Verified OAuth 2.0</span>
        </div>

        {/* Client ID Configuration Helper (If needed) */}
        <div className="mt-4">
          <button
            onClick={() => setShowConfigModal(true)}
            className="text-[10.5px] font-mono text-neutral-400 hover:text-cyan-400 transition-colors flex items-center space-x-1"
          >
            <KeyRound className="w-3 h-3" />
            <span>{clientId ? 'Google Client ID Configured' : 'Configure Google Client ID'}</span>
          </button>
        </div>

      </div>

      {/* Google Client ID Setup Modal */}
      {showConfigModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4 animate-in fade-in duration-150">
          <div className="w-full max-w-md bg-neutral-900 border border-neutral-800 rounded-2xl p-6 shadow-2xl">
            <div className="flex items-center space-x-2 text-white font-bold text-base mb-2">
              <KeyRound className="w-5 h-5 text-cyan-400" />
              <span>Google OAuth Web Client ID</span>
            </div>
            
            <p className="text-xs text-slate-300 mb-4 leading-relaxed">
              Google Sign-In requires a Google Cloud OAuth 2.0 Web Client ID authorized for this domain (<span className="font-mono text-cyan-300">http://localhost:5173</span>).
              SatQuery AI verifies tokens with Google and never handles passwords.
            </p>

            <form onSubmit={handleSaveClientId} className="space-y-4">
              <div>
                <label className="block text-[11px] font-mono text-slate-400 mb-1">
                  Google Client ID (.apps.googleusercontent.com)
                </label>
                <input
                  type="text"
                  value={manualClientIdInput}
                  onChange={(e) => setManualClientIdInput(e.target.value)}
                  placeholder="e.g. 123456789-abc.apps.googleusercontent.com"
                  className="w-full px-3.5 py-2.5 rounded-xl bg-black border border-neutral-700 text-xs text-white placeholder-slate-400 focus:outline-none focus:border-cyan-500 font-mono"
                />
              </div>

              <div className="flex items-center justify-between text-[11px] text-slate-400">
                <a
                  href="https://console.cloud.google.com/apis/credentials"
                  target="_blank"
                  rel="noreferrer"
                  className="text-cyan-400 hover:underline flex items-center space-x-1"
                >
                  <span>Google Cloud Console</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
                <span>Can also set in .env</span>
              </div>

              <div className="flex items-center justify-end space-x-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowConfigModal(false)}
                  className="px-4 py-2 rounded-xl bg-neutral-800 hover:bg-neutral-700 text-xs text-slate-300 transition-colors"
                >
                  Close
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-xs font-semibold text-black transition-colors"
                >
                  Save & Apply
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Footer copyright */}
      <footer className="relative z-10 text-[11px] text-slate-400 font-mono mt-2">
        SatQuery AI • Earth Observation Intelligence Platform
      </footer>

    </div>
  );
};
