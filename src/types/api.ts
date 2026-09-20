/**
 * SatQuery AI - Production API TypeScript Interfaces.
 * Strictly mirrors backend schemas from backend/schemas.py and backend/orchestrator/.
 */

export interface BoundingBox {
  id: string;
  label: string;
  confidence?: number | null;
  x: number;
  y: number;
  width: number;
  height: number;
  description: string;
  projectedBbox?: Record<string, number> | null;
  geographicBbox?: {
    min_lon: number;
    min_lat: number;
    max_lon: number;
    max_lat: number;
  } | null;
  centerLatLon?: {
    latitude: number;
    longitude: number;
  } | null;
  color?: string;
}

export interface GeospatialEvidence {
  status: 'available' | 'partial' | 'unavailable';
  crs?: string | null;
  bounds?: Record<string, number> | null;
  resolution?: string | null;
  pixelCoordinates?: Record<string, number> | null;
  geographicCoordinates?: {
    latitude: number;
    longitude: number;
  } | null;
  geographicBounds?: {
    north: number;
    south: number;
    east: number;
    west: number;
  } | null;
  area?: {
    area_m2: number;
    area_km2: number;
  } | null;
  alignmentStatus?: 'geospatially aligned' | 'pixel-aligned' | 'alignment unavailable' | null;
  limitations: string[];
}

export interface AgentPlan {
  intents: string[];
  selected_tools: string[];
  is_multi_specialist: boolean;
  limitations: string[];
  execution_graph: Record<string, any>;
}

export interface EvidenceHierarchy {
  direct_evidence: string[];
  supporting_evidence: string[];
  limitations: string[];
  disagreements: string[];
  synthesized_answer: string;
}

export interface ExecutionTraceStep {
  id: string;
  stepNumber: number;
  title: string;
  status: 'completed' | 'running' | 'pending' | 'failed' | 'skipped';
  durationMs: number;
  summary: string;
  details?: {
    label: string;
    value: string;
  }[] | null;
}

export interface ChangeMetric {
  increasedAreaKm2: number;
  decreasedAreaKm2: number;
  netChangePercentage: number;
  primaryClass: string;
  changeRegionsCount: number;
}

export interface GeoTIFFMetadata {
  filename: string;
  format: string;
  width: number;
  height: number;
  bands: number;
  crs?: string | null;
  geotransform?: number[] | null;
  resolution?: string | null;
  bounds?: Record<string, number> | null;
  datatype: string;
  modality: string;
  sensor: string;
  isValid: boolean;
  validationMessage: string;
}

export interface ValidateImageRequest {
  filename: string;
  fileSizeBytes?: number;
  mode?: string;
  role?: string;
  fileDataUri?: string;
}

export interface ClassifyTaskRequest {
  query: string;
  mode: string;
  imageCount?: number;
  modalities?: string[];
}

export interface ClassifyTaskResponse {
  taskType: string;
  primaryCapability: string;
  recommendedModel: string;
  confidence: number;
  reasoning: string;
}

export interface AnalyzeRequest {
  query: string;
  mode?: string;
  images: Record<string, any>;
}

export interface ChangedRegion {
  id: string;
  label: string;
  direction?: 'increase' | 'decrease' | 'modified';
  x: number;
  y: number;
  width: number;
  height: number;
  delta?: number;
  areaKm2?: number;
}

export interface AnalyzeResponse {
  query: string;
  mode: string;
  taskType: string;
  selectedModel: string;
  answer: string;
  confidence?: number | null;
  evidence: string[];
  boundingBoxes?: BoundingBox[] | null;
  changedRegions?: ChangedRegion[] | null;
  changeMetric?: ChangeMetric | null;
  crossModalEvidence?: Record<string, any> | null;
  multimodalRegions?: Record<string, any>[] | null;
  executionSteps: ExecutionTraceStep[];
  imageryMetadata: Record<string, any>;
  imageOverlayType: string;
  isSimulation: boolean;
  timestamp: string;
  inputInformation: string;
  agentPlan?: AgentPlan | null;
  evidenceHierarchy?: EvidenceHierarchy | null;
  geospatialEvidence?: GeospatialEvidence | null;
}

export interface ChangeAnalysisRequest {
  query: string;
  beforeImage: Record<string, any>;
  afterImage: Record<string, any>;
}

export interface OpticalSARAnalysisRequest {
  query: string;
  opticalImage: Record<string, any>;
  sarImage: Record<string, any>;
}

export interface CursorFeatureRequest {
  x: number;
  y: number;
  metadata?: Record<string, any> | null;
  features?: Record<string, any>[] | null;
}

export interface CursorFeatureResponse {
  pixel: { col: number; row: number };
  geographic: { latitude: number; longitude: number } | null;
  projected: { x: number; y: number } | null;
  crs: string | null;
  geospatial_status: 'available' | 'unavailable';
  evidence_feature: Record<string, any> | null;
}

export interface ModelInfoSchema {
  id: string;
  name: string;
  category: string;
  architecture: string;
  modalities: string[];
  gsdRange: string;
  parameters: string;
  inputResolution: string;
  description: string;
  status: string;
  supportedTasks: string[];
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  environment: string;
  accelerator: string;
  architecture: string;
  activeModelsCount: number;
  version: string;
}

export interface ReportResponse {
  reportId: string;
  timestamp: string;
  markdown: string;
  json: Record<string, any>;
  filename: string;
}
