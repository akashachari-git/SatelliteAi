export interface ImageMetadata {
  filename: string;
  file_size_bytes?: number;
  file_size_mb: number;
  format: string;
  is_geotiff: boolean;
  width: number;
  height: number;
  bands: number;
  dtype: string;
  crs: string;
  spatial_resolution_m: number;
  bounds?: {
    min_x: number;
    min_y: number;
    max_x: number;
    max_y: number;
  } | null;
  modality: 'Optical' | 'Multispectral' | 'SAR';
  acquisition_date?: string | null;
  sensor: string;
  radiometry: string;
  preview_url: string;
  server_path: string;
}

export interface CompatibilityCheck {
  name: string;
  status: 'PASS' | 'WARN' | 'FAIL';
  details: string;
}

export interface CoRegistrationResult {
  is_compatible: boolean;
  co_registration_score: number;
  suggested_mode: string;
  checks: CompatibilityCheck[];
  warnings: string[];
  summary: string;
}

export interface TraceStep {
  stage_id: string;
  stage_name: string;
  description: string;
  status: 'COMPLETED' | 'WARNING' | 'FAILED' | 'ERROR';
  duration_ms: number;
  details?: any;
}

export interface ConfidenceProfile {
  score: number;
  percentage: number;
  label: string;
  factors: {
    model_agreement?: string;
    evidence_strength?: string;
    input_compatibility?: string;
    spectral_congruence?: string;
  };
  calibrated?: boolean;
}

export interface EvidenceItem {
  type: string;
  url: string;
}

export interface AnalysisResult {
  analysis_id?: string;
  success: boolean;
  query: string;
  task: string;
  method?: string;
  selected_tool: string;
  model_name: string;
  answer: string;
  confidence: ConfidenceProfile;
  evidence: EvidenceItem[];
  evidence_object?: any;
  raw_result: any;
  limitations?: string[];
  parsed_intent?: any;
  images_metadata: ImageMetadata[];
  co_registration?: CoRegistrationResult | null;
  execution_trace: TraceStep[];
  total_duration_seconds: number;
  created_at: string;
}

export interface DemoScenario {
  id: string;
  title: string;
  description: string;
  default_query: string;
  suggested_queries: string[];
  image_count: number;
  task: string;
  images: Array<{
    filename: string;
    modality: string;
    description: string;
  }>;
}

export interface BenchmarkItem {
  key: string;
  name: string;
  task: string;
  metrics: string[];
  configured: boolean;
  dataset_path: string;
  description: string;
}

export interface User {
  id: string;
  sub: string;
  email: string;
  name: string;
  picture?: string;
  given_name?: string;
  family_name?: string;
  auth_provider: 'google';
  verified_at: number;
}

export interface AuthConfig {
  client_id: string;
  auth_enabled: boolean;
}

export interface AuthResponse {
  success: boolean;
  session_token: string;
  user: User;
}

