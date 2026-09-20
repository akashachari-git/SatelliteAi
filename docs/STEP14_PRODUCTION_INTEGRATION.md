# SatQuery AI — Step 14: Production API + Frontend Integration Documentation

## 1. Overview & Architecture

Step 14 bridges the SatQuery AI production React/Vite frontend to the real FastAPI backend pipeline established in Steps 8–13.

### Complete Flow
```
UPLOAD (GeoTIFF / PNG / JPEG)
   ↓
VALIDATE (/api/validate-image: CRS, affine, bands, GSD, modality)
   ↓
QUERY (Natural Language Remote Sensing Question)
   ↓
AGENT PLAN (SatQueryAgent query classifier & tool selection)
   ↓
SPECIALIST MODELS (Florence-2 VLM, BigEarthNet ResNet-18, Bi-Temporal, Optical+SAR)
   ↓
EVIDENCE COMBINATION (EvidenceCombiner synthesis & disagreement detection)
   ↓
GEOSPATIAL EVIDENCE (GeoreferenceEngine pixel-to-geographic bounds)
   ↓
RESULT (Direct & supporting evidence, disagreements, execution trace, overlays)
   ↓
REPORT (/api/report/generate structured Markdown & JSON intelligence reports)
```

The frontend displays genuine backend results exclusively. All mock data, hardcoded Rotterdam/Bengaluru figures, fake confidence numbers, and sample labels have been excised from the production execution path.

---

## 2. API Endpoints Used

| Method | Endpoint | Description | Production Schema |
|---|---|---|---|
| `GET` | `/api/health` | System telemetry, active model count, runtime device | `HealthResponse` |
| `POST` | `/api/validate-image` | Spaceborne telemetry and GeoTIFF header validation | `GeoTIFFMetadata` |
| `POST` | `/api/classify-task` | Query intent classification & specialist recommendation | `ClassifyTaskResponse` |
| `POST` | `/api/analyze` | Central 8-stage agent pipeline for Single Image analysis | `AnalyzeResponse` |
| `POST` | `/api/change-analysis` | Bi-temporal change detection between T1 and T2 | `AnalyzeResponse` |
| `POST` | `/api/optical-sar-analysis` | Cross-modal Optical + SAR radar fusion | `AnalyzeResponse` |
| `GET` | `/api/models` | Technical model registry with capabilities and constraints | `List[ModelInfoSchema]` |
| `GET` | `/api/history` | Historical mission intelligence records | `List[Dict[str, Any]]` |
| `POST` | `/api/geospatial/cursor-feature` | Real-time pixel-to-coordinate query for viewer HUD | `CursorFeatureResponse` |
| `POST` | `/api/report/generate` | Structured Markdown & JSON mission intelligence report | `ReportResponse` |

---

## 3. Environment Variables

| Variable | Default Value | Description |
|---|---|---|
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Backend base URL for frontend API client |
| `ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Allowed CORS origins in backend `main.py` |

---

## 4. Supported Modes & Specialist Routing

### 1. Single Image Mode
- **Endpoints**: `/api/analyze`
- **Specialists**:
  - `microsoft/Florence-2-base`: Remote-sensing captioning, VQA, and phrase grounding.
  - `BigEarthNet ResNet-18 (10-Band ONNX)`: 43-class CORINE multi-label classification (requires 10-band Sentinel-2 GeoTIFF).
- **Evidence Output**:
  - Direct evidence (primary model findings)
  - Supporting evidence (corroborating spectral features)
  - Disagreements (e.g., when Florence-2 and BigEarthNet diverge)
  - Pixel bounding boxes with projected and geographic coordinates when available.

### 2. Past & Present (Bi-Temporal) Mode
- **Endpoints**: `/api/change-analysis`
- **Specialists**: `BiTemporalChangeService`
- **Evidence Output**:
  - Distinguishes candidate changes from confirmed semantic changes.
  - Detects CRS and spatial extent mismatches.
  - Computes change boundaries and per-region labels without fabricated percentages.

### 3. Optical + SAR Mode
- **Endpoints**: `/api/optical-sar-analysis`
- **Specialists**: `OpticalSARFusionService`
- **Evidence Output**:
  - Optical evidence (multispectral indices NDVI, NDWI, NDBI)
  - SAR evidence (VV/VH backscatter, cross-ratio in dB)
  - Cross-sensor corroboration summary isolating permanent water bodies and dense urban structures.

---

## 5. Geospatial Information Handling

- **Available Georeferencing**: Displays EPSG code, projected CRS, bounding coordinates (WGS84 lat/lon), pixel resolution, and geographic bounds for grounded features.
- **Unavailable Georeferencing**: Displays an explicit disclaimer:
  > *"Geospatial coordinates unavailable because the source raster does not contain sufficient georeferencing metadata."*
- **No Guessing**: The frontend never invents or hardcodes coordinates, EPSG codes, or centroids.

---

## 6. Interactive Image Viewer HUD

The viewer includes an interactive telemetry HUD on cursor hover:
- `PIXEL`: `(X, Y)` relative to raster dimensions.
- `LATITUDE / LONGITUDE`: Populated from genuine geotransform when available, or displays `"Geolocation unavailable"`.
- `FEATURE`: Active visual evidence under cursor.
- `EVIDENCE SOURCE`: Specialist model attribution.

---

## 7. Report Generation

Connected to `POST /api/report/generate`:
- Produces downloadable `.md` and `.json` reports containing:
  - Query & selected specialists
  - Verified raster metadata
  - Direct and supporting evidence
  - Disagreements & limitations
  - Bounding boxes and geographic coordinates
  - Execution trace provenance

---

## 8. Development & Verification Commands

### Start Backend:
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### Run Backend Tests:
```bash
python -m unittest discover -s backend/tests -v
```

### Run Frontend Integration Tests:
```bash
cmd.exe /c npx tsx src/tests/productionIntegration.test.ts
```

### Build Frontend:
```bash
cmd.exe /c npm run build
```

---

## 9. Known Limitations

1. **BigEarthNet 10-Band Requirement**: BigEarthNet ResNet-18 requires a 10-band Sentinel-2 GeoTIFF. When an ordinary 3-band RGB image is provided, Florence-2 handles captioning/VQA and an honest limitation note is appended explaining that BigEarthNet is unavailable for 3-band imagery.
2. **CPU Inference Latency**: When running Florence-2 on CPU-only machines, inference may take 5–15 seconds per query. The frontend displays progressive loading stages (`Uploading & validating...`, `Planning analysis...`, `Running specialist models...`, `Combining evidence...`).
