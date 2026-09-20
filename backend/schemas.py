"""
SatQuery AI - Pydantic Request & Response Schemas
Provides strict type-safety for remote-sensing inputs, geospatial metadata,
and intelligence outputs.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    id: str
    label: str
    confidence: Optional[float] = None
    x: float
    y: float
    width: float
    height: float
    description: str
    projectedBbox: Optional[Dict[str, float]] = None
    geographicBbox: Optional[Dict[str, float]] = None
    centerLatLon: Optional[Dict[str, float]] = None

class GeospatialEvidence(BaseModel):
    status: str = "unavailable"  # "available" | "partial" | "unavailable"
    crs: Optional[str] = None
    bounds: Optional[Dict[str, float]] = None
    resolution: Optional[str] = None
    pixelCoordinates: Optional[Dict[str, float]] = None
    geographicCoordinates: Optional[Dict[str, float]] = None
    geographicBounds: Optional[Dict[str, float]] = None
    area: Optional[Dict[str, float]] = None
    alignmentStatus: Optional[str] = None  # "geospatially aligned" | "pixel-aligned" | "alignment unavailable"
    limitations: List[str] = Field(default_factory=list)

class ChangeMetric(BaseModel):
    increasedAreaKm2: float
    decreasedAreaKm2: float
    netChangePercentage: float
    primaryClass: str
    changeRegionsCount: int

class ExecutionTraceStep(BaseModel):
    id: str
    stepNumber: int
    title: str
    status: str = "completed"
    durationMs: int = 0
    summary: str
    details: Optional[List[Dict[str, str]]] = None

class GeoTIFFMetadata(BaseModel):
    filename: str
    format: str
    width: int
    height: int
    bands: int
    crs: Optional[str] = None
    geotransform: Optional[List[float]] = None
    resolution: Optional[str] = None
    bounds: Optional[Dict[str, float]] = None
    datatype: str
    modality: str
    sensor: str
    isValid: bool
    validationMessage: str

class ValidateImageRequest(BaseModel):
    filename: str
    fileSizeBytes: Optional[int] = 0
    mode: str = "single"
    role: Optional[str] = "single"
    fileDataUri: Optional[str] = None

class ClassifyTaskRequest(BaseModel):
    query: str
    mode: str
    imageCount: int = 1
    modalities: List[str] = Field(default_factory=lambda: ["Optical RGB"])

class ClassifyTaskResponse(BaseModel):
    taskType: str
    primaryCapability: str
    recommendedModel: str
    confidence: float
    reasoning: str

class AnalyzeRequest(BaseModel):
    query: str
    mode: str = "single"
    images: Dict[str, Optional[Dict[str, Any]]] = Field(default_factory=dict)

class AnalyzeResponse(BaseModel):
    query: str
    mode: str
    taskType: str
    selectedModel: str
    answer: str
    confidence: Optional[float] = None
    evidence: List[str]
    boundingBoxes: Optional[List[BoundingBox]] = None
    changeMetric: Optional[ChangeMetric] = None
    crossModalEvidence: Optional[Dict[str, Any]] = None
    multimodalRegions: Optional[List[Dict[str, Any]]] = None
    executionSteps: List[ExecutionTraceStep]
    imageryMetadata: Dict[str, Any]
    imageOverlayType: str = "grounding"
    isSimulation: bool = False
    timestamp: str
    inputInformation: str
    agentPlan: Optional[Dict[str, Any]] = None
    evidenceHierarchy: Optional[Dict[str, Any]] = None
    geospatialEvidence: Optional[Dict[str, Any]] = None

class ChangeAnalysisRequest(BaseModel):
    query: str
    beforeImage: Dict[str, Any]
    afterImage: Dict[str, Any]

class OpticalSARAnalysisRequest(BaseModel):
    query: str
    opticalImage: Dict[str, Any]
    sarImage: Dict[str, Any]

class CursorFeatureRequest(BaseModel):
    x: float
    y: float
    metadata: Optional[Dict[str, Any]] = None
    features: Optional[List[Dict[str, Any]]] = None

class ModelInfoSchema(BaseModel):
    id: str
    name: str
    category: str
    architecture: str
    modalities: List[str]
    gsdRange: str
    parameters: str
    inputResolution: str
    description: str
    status: str
    supportedTasks: List[str]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    environment: str
    accelerator: str
    architecture: str
    activeModelsCount: int
    version: str
