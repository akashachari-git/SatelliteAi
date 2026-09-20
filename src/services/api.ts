/**
 * SatQuery AI - Production API Client Service.
 * Connects the React frontend directly to the real FastAPI backend.
 * Zero mock results, zero fabricated metrics, zero hardcoded coordinates.
 */
import {
  AnalysisMode,
  AnalysisResult,
  FileMetadata,
  HistoryItem,
} from '../types';
import {
  AnalyzeRequest,
  AnalyzeResponse,
  ChangeAnalysisRequest,
  OpticalSARAnalysisRequest,
  ValidateImageRequest,
  GeoTIFFMetadata,
  ClassifyTaskRequest,
  ClassifyTaskResponse,
  CursorFeatureRequest,
  CursorFeatureResponse,
  HealthResponse,
  ReportResponse,
  ModelInfoSchema,
} from '../types/api';

// Environment-configurable backend base URL (defaults to http://127.0.0.1:8000)
export const API_BASE_URL: string = (
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE_URL) ||
  'http://127.0.0.1:8000'
).replace(/\/$/, '');

export interface ApiConfig {
  baseUrl: string;
  isBackendConnected: boolean;
  activeEnvironment: string;
  endpoints: {
    health: string;
    validateImage: string;
    classifyTask: string;
    analyze: string;
    changeAnalysis: string;
    opticalSarAnalysis: string;
    models: string;
    history: string;
    cursorFeature: string;
    generateReport: string;
  };
}

export const API_CONFIG: ApiConfig = {
  baseUrl: API_BASE_URL,
  isBackendConnected: true,
  activeEnvironment: 'SatQuery AI Production Backend (FastAPI + PyTorch/ONNX)',
  endpoints: {
    health: `${API_BASE_URL}/api/health`,
    validateImage: `${API_BASE_URL}/api/validate-image`,
    classifyTask: `${API_BASE_URL}/api/classify-task`,
    analyze: `${API_BASE_URL}/api/analyze`,
    changeAnalysis: `${API_BASE_URL}/api/change-analysis`,
    opticalSarAnalysis: `${API_BASE_URL}/api/optical-sar-analysis`,
    models: `${API_BASE_URL}/api/models`,
    history: `${API_BASE_URL}/api/history`,
    cursorFeature: `${API_BASE_URL}/api/geospatial/cursor-feature`,
    generateReport: `${API_BASE_URL}/api/report/generate`,
  },
};

export interface FileValidationResult {
  valid: boolean;
  status: 'VALID' | 'INVALID' | 'UNVERIFIED';
  filename: string;
  fileSize: string;
  errorTitle?: string;
  errorMessage?: string;
  reason?: string;
  validationBadge?: string;
  dimensions?: string;
  detectedModality?: string;
  crs?: string;
  gsd?: string;
  sensor?: string;
  bands?: number;
  datatype?: string;
  errors?: string[];
  warnings?: string[];
  isOrdinaryPhoto?: boolean;
}

export interface TaskClassificationResult {
  taskType: string;
  primaryCategory: string;
  selectedModel: string;
  confidenceScore: number;
  reasoning: string;
}

export interface RunAnalysisParams {
  query: string;
  mode: AnalysisMode;
  files: {
    single?: FileMetadata | null;
    optical?: FileMetadata | null;
    sar?: FileMetadata | null;
    before?: FileMetadata | null;
    after?: FileMetadata | null;
  };
  enableDemoSimulation?: boolean;
  onProgress?: (step: number, title: string) => void;
}

/**
 * 1. Health Check
 */
export async function checkBackendHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/api/health`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  if (!res.ok) {
    throw new Error(`Health check failed with status ${res.status}`);
  }
  return res.json();
}

/**
 * 2. Real Image Validation Flow
 */
export async function validateSatelliteImage(
  file: File | { name: string; size?: number; sizeBytes?: number; fileDataUri?: string },
  mode: string = 'single',
  role: string = 'single'
): Promise<FileValidationResult> {
  const filename = file.name;
  const fileSizeBytes = 'size' in file && typeof file.size === 'number' ? file.size : (file.sizeBytes || 0);
  const fileDataUri = 'fileDataUri' in file ? file.fileDataUri : undefined;

  const fileSizeFormatted = fileSizeBytes > 1024 * 1024
    ? `${(fileSizeBytes / (1024 * 1024)).toFixed(1)} MB`
    : `${(fileSizeBytes / 1024).toFixed(1)} KB`;

  try {
    const res = await fetch(`${API_BASE_URL}/api/validate-image`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        filename,
        fileSizeBytes,
        mode,
        role,
        fileDataUri,
      }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      return {
        valid: false,
        status: 'INVALID',
        filename,
        fileSize: fileSizeFormatted,
        errorTitle: 'Validation Rejected',
        errorMessage: errData.detail || `Validation failed with status ${res.status}`,
        reason: errData.detail || 'Incompatible raster format or missing headers.',
        validationBadge: '✗ Rejected',
        errors: [errData.detail || 'Server rejected file.'],
      };
    }

    const meta: GeoTIFFMetadata = await res.json();

    return {
      valid: meta.isValid,
      status: meta.isValid ? 'VALID' : 'INVALID',
      filename: meta.filename,
      fileSize: fileSizeFormatted,
      dimensions: `${meta.width} × ${meta.height} px`,
      detectedModality: meta.modality,
      crs: meta.crs || 'Local Pixel Space (Unprojected)',
      gsd: meta.resolution || 'Unspecified GSD',
      sensor: meta.sensor,
      bands: meta.bands,
      datatype: meta.datatype,
      validationBadge: meta.isValid ? '✓ Valid Satellite Raster' : '✗ Invalid Format',
      errorMessage: meta.isValid ? undefined : meta.validationMessage,
      reason: meta.validationMessage,
    };
  } catch (err: any) {
    return {
      valid: false,
      status: 'UNVERIFIED',
      filename,
      fileSize: fileSizeFormatted,
      errorTitle: 'Backend Unavailable',
      errorMessage: `Could not connect to SatQuery AI backend at ${API_BASE_URL}. Ensure the FastAPI server is running.`,
      reason: err?.message || 'Network connection failed.',
      validationBadge: '⚠ Connection Error',
      errors: [err?.message || 'Connection refused.'],
    };
  }
}

/**
 * 3. Task Classification Endpoint
 */
export async function classifyTask(
  params: ClassifyTaskRequest
): Promise<ClassifyTaskResponse> {
  const res = await fetch(`${API_BASE_URL}/api/classify-task`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Classification failed with status ${res.status}`);
  }
  return res.json();
}

/**
 * 4. Central Real Analysis Execution
 * Connects Single, Bi-Temporal, and Optical+SAR modes to the real backend.
 */
export async function executeRemoteSensingAnalysis(
  params: RunAnalysisParams
): Promise<AnalysisResult> {
  const { query, mode, files, onProgress } = params;

  if (onProgress) onProgress(1, 'Agent Input Ingestion & Validation');

  let endpoint = `${API_BASE_URL}/api/analyze`;
  let requestBody: any = {};

  if (mode === 'bi-temporal') {
    endpoint = `${API_BASE_URL}/api/change-analysis`;
    const beforeF = files.before;
    const afterF = files.after;

    if (!beforeF || !afterF) {
      throw new Error('Bi-temporal analysis requires both "Past" (T1) and "Present" (T2) satellite images.');
    }

    requestBody = {
      query,
      beforeImage: {
        filename: beforeF.name,
        fileDataUri: beforeF.fileDataUri,
        fileSizeBytes: beforeF.sizeBytes,
        previewUrl: beforeF.previewUrl,
      },
      afterImage: {
        filename: afterF.name,
        fileDataUri: afterF.fileDataUri,
        fileSizeBytes: afterF.sizeBytes,
        previewUrl: afterF.previewUrl,
      },
    };
  } else if (mode === 'optical-sar') {
    endpoint = `${API_BASE_URL}/api/optical-sar-analysis`;
    const optF = files.optical;
    const sarF = files.sar;

    if (!optF || !sarF) {
      throw new Error('Optical + SAR fusion requires both an Optical raster and a SAR Radar observation.');
    }

    requestBody = {
      query,
      opticalImage: {
        filename: optF.name,
        fileDataUri: optF.fileDataUri,
        fileSizeBytes: optF.sizeBytes,
        previewUrl: optF.previewUrl,
      },
      sarImage: {
        filename: sarF.name,
        fileDataUri: sarF.fileDataUri,
        fileSizeBytes: sarF.sizeBytes,
        previewUrl: sarF.previewUrl,
      },
    };
  } else {
    // Single image mode
    endpoint = `${API_BASE_URL}/api/analyze`;
    const singleF = files.single;

    if (!singleF) {
      throw new Error('Single image analysis requires an uploaded satellite observation.');
    }

    requestBody = {
      query,
      mode: 'single',
      images: {
        single: {
          filename: singleF.name,
          fileDataUri: singleF.fileDataUri,
          fileSizeBytes: singleF.sizeBytes,
          previewUrl: singleF.previewUrl,
        },
      },
    };
  }

  if (onProgress) onProgress(3, 'Agent Planning & Specialist Selection');

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minute timeout for CPU inference

  let rawRes: Response;
  try {
    rawRes = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
      signal: controller.signal,
    });
  } catch (netErr: any) {
    clearTimeout(timeoutId);
    if (netErr.name === 'AbortError') {
      throw new Error('Analysis timed out after 120 seconds. Please try again with a smaller raster.');
    }
    throw new Error(
      `Cannot connect to SatQuery AI backend at ${API_BASE_URL}. Ensure the backend server is running.`
    );
  }
  clearTimeout(timeoutId);

  if (!rawRes.ok) {
    const errDetail = await rawRes.json().catch(() => ({}));
    throw new Error(errDetail.detail || `Analysis request failed with status ${rawRes.status}`);
  }

  if (onProgress) onProgress(6, 'Model Execution & Evidence Synthesis');

  const backendData: AnalyzeResponse = await rawRes.json();

  if (onProgress) onProgress(8, 'Geospatial Grounding & Report Ready');

  // Convert AnalyzeResponse to AnalysisResult structure for frontend
  const result: AnalysisResult = {
    query: backendData.query,
    mode: backendData.mode as AnalysisMode,
    taskType: backendData.taskType as any,
    selectedModel: backendData.selectedModel,
    answer: backendData.answer,
    confidence: backendData.confidence ?? null, // Strictly null if uncomputed
    evidence: backendData.evidence || [],
    boundingBoxes: backendData.boundingBoxes?.map((b) => ({
      ...b,
      color: b.color || '#06b6d4',
    })) || [],
    changedRegions: backendData.changedRegions || undefined,
    changeMetric: backendData.changeMetric || undefined,
    crossModalEvidence: backendData.crossModalEvidence as any,
    executionSteps: backendData.executionSteps || [],
    imageryMetadata: backendData.imageryMetadata || {},
    imageOverlayType: backendData.imageOverlayType as any,
    isSimulation: backendData.isSimulation ?? false,
    timestamp: backendData.timestamp,
    inputInformation: backendData.inputInformation,
    agentPlan: backendData.agentPlan || undefined,
    evidenceHierarchy: backendData.evidenceHierarchy || undefined,
    geospatialEvidence: backendData.geospatialEvidence || undefined,
    hasReliableResult: true,
  };

  return result;
}

/**
 * 5. Interactive Cursor Feature Query
 */
export async function queryCursorFeature(
  x: number,
  y: number,
  metadata?: Record<string, any> | null,
  features?: Record<string, any>[] | null
): Promise<CursorFeatureResponse> {
  const res = await fetch(`${API_BASE_URL}/api/geospatial/cursor-feature`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ x, y, metadata, features }),
  });
  if (!res.ok) {
    throw new Error(`Cursor query failed with status ${res.status}`);
  }
  return res.json();
}

/**
 * 6. Report Generation
 */
export async function generateReport(
  result: AnalysisResult | AnalyzeResponse
): Promise<ReportResponse> {
  const res = await fetch(`${API_BASE_URL}/api/report/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ result }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Report generation failed with status ${res.status}`);
  }
  return res.json();
}

/**
 * 7. Model Technical Registry
 */
export async function getModelRegistry(): Promise<ModelInfoSchema[]> {
  const res = await fetch(`${API_BASE_URL}/api/models`);
  if (!res.ok) {
    throw new Error(`Failed to load models: ${res.status}`);
  }
  return res.json();
}

/**
 * 8. Historical Analysis Records & Telemetry Helpers
 */
export interface BackendHealthResponse {
  ok: boolean;
  status: string;
  app?: string;
  version?: string;
  accelerator?: string;
  architecture?: string;
  activeModelsCount?: number;
  timestamp?: string;
  models?: Record<string, any>;
  raw?: HealthResponse;
}

export async function getBackendHealth(): Promise<BackendHealthResponse> {
  try {
    const raw = await checkBackendHealth();
    const isHealthy = raw.status === 'healthy' || raw.status === 'ok';
    return {
      ok: isHealthy,
      status: isHealthy ? 'Online' : 'Offline',
      app: raw.app,
      version: raw.version,
      accelerator: raw.accelerator,
      architecture: raw.architecture,
      activeModelsCount: raw.activeModelsCount,
      timestamp: raw.timestamp,
      raw,
    };
  } catch {
    return {
      ok: false,
      status: 'Offline',
    };
  }
}

export async function getAnalysisHistory(): Promise<HistoryItem[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/history`);
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data) && data.length > 0) {
        return data;
      }
    }
  } catch {
    // Fall back to local storage
  }

  try {
    const stored = localStorage.getItem('satquery_analysis_history');
    if (stored) {
      return JSON.parse(stored);
    }
  } catch {}

  return [];
}

export const fetchHistory = getAnalysisHistory;

export async function addHistoryItemApi(item: HistoryItem): Promise<boolean> {
  try {
    const existing = await getAnalysisHistory();
    const updated = [item, ...existing.filter((i) => i.id !== item.id)];
    localStorage.setItem('satquery_analysis_history', JSON.stringify(updated.slice(0, 50)));
    return true;
  } catch {
    return false;
  }
}

export async function deleteJobApi(id: string): Promise<boolean> {
  try {
    const existing = await getAnalysisHistory();
    const updated = existing.filter((i) => i.id !== id);
    localStorage.setItem('satquery_analysis_history', JSON.stringify(updated));
    return true;
  } catch {
    return false;
  }
}
