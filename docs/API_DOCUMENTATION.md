# SatQuery AI — REST API Reference

The backend exposes a high-performance REST API developed with FastAPI.

### Base URL: `http://localhost:8000/api`

---

### Endpoints

#### 1. Ingestion & Validation
- **`POST /api/upload`**: Multipart file upload. Extracts GeoTIFF tags, dimensions, CRS, resolution, sensor modality, and creates an optimized PNG preview.
- **`POST /api/validate`**: Validates pair compatibility (dimensions, resolution ratio, CRS alignment, spatial overlap percentage).

#### 2. Agentic Analysis
- **`POST /api/analyze`**: Main orchestration endpoint.
  - **Payload**: `{"query": string, "image_paths": string[], "user_metadata": object[]}`
  - **Returns**: Structured analysis object containing:
    - `task`: Classified task (`SINGLE_VQA`, `GROUNDING`, `CHANGE_ANALYSIS`, `CHANGE_VQA`, `OPTICAL_SAR_ANALYSIS`, `CAPTIONING`)
    - `selected_tool`: Tool identifier
    - `answer`: Domain-aware grounded response
    - `confidence`: Calibrated multi-factor confidence profile
    - `evidence`: URLs to masks, heatmaps, bounding boxes, or false-color composites
    - `execution_trace`: Observable step-by-step pipeline stages with durations

#### 3. Intelligence Reports
- **`POST /api/report/generate-pdf`**: Generates formal PDF intelligence dossier with embedded evidence and trace.
- **`POST /api/report/generate-json`**: Exports machine-readable JSON dossier.

#### 4. Demos & Benchmarks
- **`GET /api/demo-scenarios`**: Returns curated demonstration scenarios.
- **`POST /api/demo-scenarios/load/{id}`**: Mounts genuine GeoTIFF demonstration rasters.
- **`GET /api/benchmarks`**: Lists benchmark configurations.
- **`POST /api/benchmarks/run/{key}`**: Runs evaluation against benchmark test split.
