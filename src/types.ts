export type AnalysisMode = 'single' | 'optical-sar' | 'bi-temporal';

export type TaskType =
  | 'vqa'
  | 'captioning'
  | 'scene-captioning'
  | 'grounding'
  | 'text-guided-grounding'
  | 'change-analysis'
  | 'change-based-vqa'
  | 'optical-sar'
  | 'optical-sar-analysis';

export interface GeoBounds {
  north: number; // Max Latitude (deg)
  south: number; // Min Latitude (deg)
  east: number;  // Max Longitude (deg)
  west: number;  // Min Longitude (deg)
}

export interface RasterBandInfo {
  index: number;
  name: string;
  wavelength?: string;
  description: string;
  minVal?: number;
  maxVal?: number;
}

export interface GeospatialMetadata {
  isGeoreferenced: boolean;
  crs?: string;
  crsName?: string;
  epsgCode?: number;
  bounds?: GeoBounds;
  centroid?: { lat: number; lng: number };
  pixelResolution?: string;
  pixelScaleMeters?: { x: number; y: number };
  dimensions?: { width: number; height: number };
  bandCount?: number;
  bands?: RasterBandInfo[];
  nodataValue?: number;
  affineTransform?: number[]; // [xOrigin, pixelWidth, 0, yOrigin, 0, -pixelHeight]
}

export interface FileMetadata {
  id: string;
  name: string;
  size: string;
  sizeBytes?: number;
  modality: string;
  dimensions: string;
  gsd: string;
  acquisitionDate: string;
  crs: string;
  sensor: string;
  previewUrl: string;
  geospatialMetadata?: GeospatialMetadata;
  fileDataUri?: string;
}

export interface BoundingBox {
  id: string;
  label: string;
  x: number; // percentage 0-100
  y: number; // percentage 0-100
  width: number; // percentage 0-100
  height: number; // percentage 0-100
  color: string;
  confidence: number | null;
  description: string;
  geoCoordinates?: string;
  geoBounds?: GeoBounds;
  areaHectares?: number;
}

export interface SpatialPoint {
  id: string;
  label: string;
  lat: number;
  lng: number;
  pixelX: number; // percentage 0-100
  pixelY: number; // percentage 0-100
  confidence: number | null;
  description?: string;
  color?: string;
  featureClass?: string;
}

export interface SpatialPolygon {
  id: string;
  label: string;
  coordinates: [number, number][]; // [[lng, lat], ...] WGS84
  pixelPoints: [number, number][]; // [[x%, y%], ...]
  areaHectares?: number;
  confidence: number | null;
  description: string;
  category?: string;
  color?: string;
}

export type SpatialEvidenceType =
  | 'Bounding Box'
  | 'Segmentation Mask'
  | 'Change Mask'
  | 'Point'
  | 'Polygon'
  | 'Textual Evidence';

export interface SpatialEvidenceItem {
  id: string;
  type: SpatialEvidenceType;
  label: string;
  category?: string;
  geoCoordinates?: string;
  geoBounds?: GeoBounds;
  center?: { lat: number; lng: number };
  pixelBounds?: { x: number; y: number; width: number; height: number };
  confidence: number | null;
  associatedQuery: string;
  specialist: string;
  model: string;
  sourceModality: string;
  evidenceMetadata?: Record<string, string | number | boolean>;
  explanation: string;
  color?: string;
  isValidated: boolean;
}

export interface ExecutionTraceStep {
  id: string;
  stepNumber: number;
  title: string;
  status: 'completed' | 'running' | 'pending' | 'failed' | 'skipped';
  durationMs?: number;
  timestamp?: string;
  summary: string;
  details?: {
    label: string;
    value: string;
  }[];
}

export interface ChangeMetric {
  increasedAreaKm2: number;
  decreasedAreaKm2: number;
  netChangePercentage: number;
  primaryClass: string;
  changeRegionsCount: number;
}

export type ChangeDirection =
  | 'Increased'
  | 'Decreased'
  | 'Newly appeared'
  | 'Disappeared'
  | 'No significant change'
  | 'Uncertain';

export type ChangeCategory =
  | 'Urban Expansion & Infrastructure'
  | 'Deforestation / Vegetation Loss'
  | 'Water Body Dynamics & Infill'
  | 'Agricultural Land Turnover'
  | 'Barren Land Transition'
  | 'Industrial Construction'
  | 'No Significant Transition'
  | 'Uncertain';

export interface ChangedRegion {
  id: string;
  label: string;
  category: string;
  direction: ChangeDirection;
  coordinates: string;
  areaKm2: number;
  x: number; // percentage 0-100
  y: number; // percentage 0-100
  width: number; // percentage 0-100
  height: number; // percentage 0-100
  confidence: number | null;
  spectralShift: string;
  ndviDelta?: number;
  ndbiDelta?: number;
  backscatterDeltaDb?: number;
}

export interface TemporalMetadata {
  t1Date: string;
  t2Date: string;
  intervalDays: number;
  sensorT1: string;
  sensorT2: string;
  crs: string;
  resolutionGsd: string;
  spatialOverlapPct: number;
  coRegistrationRmsePixels: number;
}

export interface CrossModalEvidence {
  opticalEvidence: string[];
  sarEvidence: string[];
  fusedEvidence: string[];
  corroboratingFeatures: string[];
  sensorComplementarityNotes: string;
  cloudPenetrationVerified?: boolean;
  opticalIndexMetrics?: {
    ndviMean?: number;
    ndbiMean?: number;
    ndwiMean?: number;
  };
  sarBackscatterMetrics?: {
    vvMeanDb?: number;
    vhMeanDb?: number;
    crossPolRatioDb?: number;
  };
}

export interface MultimodalRegion {
  id: string;
  label: string;
  category: 'built-up' | 'water' | 'vegetation' | 'infrastructure' | 'cloud-covered';
  modalitySupport: 'Optical + SAR' | 'SAR Dominant' | 'Optical Dominant';
  opticalSignature: string;
  sarBackscatterDb: string;
  coordinates: string;
  x: number; // percentage 0-100
  y: number; // percentage 0-100
  width: number; // percentage 0-100
  height: number; // percentage 0-100
  confidence: number | null;
  description: string;
}

export interface OpticalSarCompatibilityResult {
  compatible: boolean;
  errors: string[];
  warnings: string[];
  spatialCorrespondence: boolean;
  crsMatch: boolean;
  resolutionMatch: boolean;
  dimensionsMatch: boolean;
  opticalModalityVerified: boolean;
  sarModalityVerified: boolean;
  boundsOverlapPct: number;
  opticalSensor: string;
  sarSensor: string;
  acquisitionMetadataMatch: boolean;
}

export interface BiTemporalCompatibilityResult {
  compatible: boolean;
  errors: string[];
  warnings: string[];
  coRegistered: boolean;
  crsMatch: boolean;
  resolutionMatch: boolean;
  dimensionsMatch: boolean;
  modalityMatch: boolean;
  temporalDeltaValid: boolean;
  overlapPercentage: number;
}

export interface ChangeMaskData {
  width: number;
  height: number;
  clustersCount: number;
  changedPixelsRatio: number;
  maskUrl?: string;
  labels?: string[];
}

export interface DetectedFeatureItem {
  name: string;
  status: 'Detected' | 'Not Present in Scene';
  extent?: string;
  coverage?: string;
  description: string;
  confidence?: number | null;
  iconName?: string;
}

export interface AnalysisResult {
  query: string;
  mode: AnalysisMode;
  taskType: TaskType;
  selectedModel: string;
  answer: string;
  hasReliableResult?: boolean;
  whyThisAnswer?: string;
  confidence: number | null;
  evidence: string[];
  geospatialMetadata?: GeospatialMetadata;
  spatialEvidenceItems?: SpatialEvidenceItem[];
  points?: SpatialPoint[];
  polygons?: SpatialPolygon[];
  boundingBoxes?: BoundingBox[];
  changeMetric?: ChangeMetric;
  changeSummary?: string;
  changeDirection?: ChangeDirection;
  changeCategories?: string[];
  changedRegions?: ChangedRegion[];
  temporalMetadata?: TemporalMetadata;
  changeMapUrl?: string;
  changeMaskData?: ChangeMaskData;
  segmentationMaskUrl?: string;
  crossModalEvidence?: CrossModalEvidence;
  multimodalRegions?: MultimodalRegion[];
  opticalSarCompatibility?: OpticalSarCompatibilityResult;
  detectedFeatures?: DetectedFeatureItem[];
  executionSteps: ExecutionTraceStep[];
  imageryMetadata: {
    coordinates: string;
    resolution: string;
    dimensions: string;
    modality: string;
    sensor: string;
    cloudCover?: string;
    bounds?: GeoBounds;
    crs?: string;
    bands?: RasterBandInfo[];
  };
  imageOverlayType?: 'grounding' | 'change' | 'fusion' | 'none';
  groundingStatus?: 'available' | 'unavailable' | 'not-requested';
  isSimulation?: boolean;
  inputInformation?: string;
  timestamp?: string;
  spatialEvidenceAvailable?: boolean;
  evidenceNote?: string;
  modelStatus?: ModelDeploymentStatus;
  confidenceLabel?: string;
  selectionReason?: string;
  validationStatus?: 'PASSED' | 'FAILED' | 'UNCERTAIN' | 'WARNING';
  validationNotes?: string[];
  isMultiStep?: boolean;
  multiStepSequence?: string[];
  auditId?: string;
}

export interface HistoryItem {
  id: string;
  query: string;
  analysisType: string;
  date: string;
  input: string;
  status: 'Completed' | 'Processing' | 'Flagged';
  confidence: number | null;
  isSimulation?: boolean;
  result: AnalysisResult;
}

export type SystemConnectionStatus =
  | 'Online'
  | 'Offline'
  | 'Connecting'
  | 'Warning'
  | 'Unavailable';

export interface SystemStatusData {
  backend: {
    status: SystemConnectionStatus;
    host: string;
    port: number;
    version: string;
    uptime: string;
    runtime: string;
    pid?: number;
  };
  database: {
    status: SystemConnectionStatus;
    type: string;
    recordsCount: number;
    integrity: string;
    healthy: boolean;
  };
  modelService: {
    status: 'Ready' | 'Demo' | 'Loading' | 'Unavailable';
    activeModelsCount: number;
    accelerator: string;
    checkpoints: { name: string; status: string; loaded: boolean }[];
    lastHealthCheck: string;
  };
  geospatial: {
    status: SystemConnectionStatus;
    engine: string;
    supportedCrs: string[];
    maxDimensions: string;
    maxFileSizeMb: number;
    metadataPreservation: string;
  };
  gpu: {
    available: boolean;
    detectedDevice: string;
    status: string;
    note: string;
  };
  apiConnectivity: {
    status: SystemConnectionStatus;
    httpCode: number;
    endpointLatencyMs: number;
  };
}

export type ModelDeploymentStatus =
  | 'Integrated'
  | 'Ready'
  | 'Loading'
  | 'Demo'
  | 'Training Required'
  | 'Unavailable'
  | 'Error'
  | 'Available'
  | 'Planned';

export interface ModelInfo {
  id: string;
  name: string;
  category: string;
  architecture: string;
  modalities: string[];
  gsdRange: string;
  parameters: string;
  inputResolution: string;
  description: string;
  status: ModelDeploymentStatus;
  supportedTasks: string[];
  benchmarkStatus?: string;
  version?: string;
  availability?: string;
  inputRequirements?: string;
  checkpoint?: string;
  evidenceCapability?: string;
}

export interface BigEarthNetPipelineSpec {
  datasetRoot: string;
  splitDir: string;
  checkpointDir: string;
  modality: 'S2' | 'S1' | 'MM';
  batchSize: number;
  imageSize: number;
  learningRate: number;
  weightDecay: number;
  loraRank: number;
  clcClassesCount: number;
  isMounted: boolean;
}

export interface BenchmarkSpec {
  benchmark: string;
  targetModalities: string[];
  prescribedMetrics: string[];
  evaluationStatus: string;
  sampleCount?: number;
  overallAccuracy?: number;
  averageAccuracy?: number;
  categoryAccuracies?: Record<string, number>;
}

// ---------------------------------------------------------------------------
// Persistence & Job Management Types
// ---------------------------------------------------------------------------
export type JobStatus =
  | 'QUEUED'
  | 'VALIDATING'
  | 'PROCESSING'
  | 'COMPLETED'
  | 'FAILED'
  | 'CANCELLED';

export interface AnalysisJob {
  id: string; // Analysis ID (sq-job-xxxx)
  userId: string;
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  durationMs: number;
  status: JobStatus;
  progress: number;
  currentStep: string;
  query: string;
  inputType: 'single' | 'bi-temporal' | 'optical-sar';
  inputSummary: string;
  uploadedFileIds: string[];
  uploadedFilesMetadata: any[];
  detectedModality: string;
  selectedTask: string;
  selectedModel: string;
  response: {
    answer: string;
    whyThisAnswer?: string;
    isSimulation: boolean;
  };
  confidence: number | null;
  evidenceReferences: {
    id: string;
    type: string;
    label: string;
    coordinates?: string;
    confidence: number | null;
  }[];
  executionTrace: ExecutionTraceStep[];
  errorInfo: { code: string; message: string; phase: string } | null;
  reportDossier?: {
    dossierId: string;
    format: string;
    generatedAt: string;
    reproducibleSeed: string;
  };
  resultPayload?: any;
}

export interface PersistenceHealth {
  status: string;
  timestamp: string;
  postgres: {
    status: 'CONNECTED' | 'NOT CONNECTED / CONFIGURATION REQUIRED' | 'ERROR';
    dialect: string;
    connectionConfigured: boolean;
    activeStorage: string;
    host: string;
    database: string;
    poolSize: number;
    latencyMs: number;
    tablesReady: boolean;
    tables: {
      analyses: boolean;
      uploaded_files: boolean;
      agent_executions: boolean;
      evidence_records: boolean;
      reports: boolean;
    };
    message: string;
  };
  storageVault: {
    status: string;
    isolationLevel: string;
    vaultDirectory: string;
    totalFilesStored: number;
    totalSizeFormatted: string;
    maxPerFileLimitMb: number;
    supportedFormats: string[];
    cleanupPolicy: string;
  };
  jobQueue: {
    status: string;
    activeWorkerThreads: number;
    queueCapacity: number;
    totalPersistedAnalyses: number;
    distribution: Record<string, number>;
    stateMachineStages: string[];
  };
}

export * from './types/evaluation';

