# SatQuery AI
**An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries**


## Overview
SatQuery AI enables non-expert users and GIS professionals to analyze complex Earth Observation (EO) imagery through plain natural language queries. By replacing static VLM wrappers with an autonomous **13-Stage Agentic Controller** and **Specialist Model Registry**, SatQuery AI dynamically routes queries to dedicated remote-sensing tools adapted on **BigEarthNet-19 (CORINE)** and **Sentinel-1/2** datasets.

---

## Core Capabilities
- **Single-Image VQA & Captioning**: Natural-language inquiries grounded in BigEarthNet spectral-spatial classifications (NDVI, NDWI, NDBI).
- **Text-Guided Region Grounding**: Interactive bounding-box localization and segmented mask overlays (water bodies, built-up areas, vegetation).
- **Bi-Temporal Change Detection & Change VQA**: Automated difference heatmaps, Otsu segmentation masks, quadrant flux aggregation, and swipe-curtain before/after comparison.
- **Optical + SAR Cross-Modal Fusion**: Joint synthesis of optical surface reflectance with Sentinel-1 C-SAR microwave backscatter double-bounce reflections.
- **Authentic GeoTIFF Ingestion**: Automated extraction of ModelTiepoint, ModelPixelScale, CRS (UTM/WGS84), GSD resolution, and sensor modality.
- **Auditable Execution Trace**: Observable step-by-step pipeline stages with millisecond durations and parameter disclosures.
- **PDF Intelligence Dossier**: Instant generation of formal, downloadable PDF reports with embedded evidence visualizations and JSON export.
- **Curated Demonstration Mode**: One-click instant loaders with authentic GeoTIFF imagery across 4 benchmark scenarios.

---

## Quick Start Guide

### 1. Prerequisites
- **Python**: 3.10+ (Tested on Python 3.12)
- **Node.js**: 18+ (Tested on Node v24)
- **Package Manager**: npm

### 2. Backend Setup
```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
The backend will launch at `http://127.0.0.1:8000`.
Swagger API documentation is accessible at `http://127.0.0.1:8000/docs`.

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
The web application will open at `http://localhost:5173`.

### 4. Running Automated Tests
```bash
python -m unittest discover -s backend/tests -p "test_*.py"
```

---

## Demonstration Scenarios
Open the web app and click **"Load Curated Demo"** in the top navigation bar:
1. **Scenario 1 (Urban VQA)**: *"Describe the major land-cover types visible in this image."*
2. **Scenario 2 (Water Grounding)**: *"Highlight the water body referred to in the query."*
3. **Scenario 3 (Bi-Temporal Change)**: *"What changed between these two dates, and where did the change occur?"*
4. **Scenario 4 (Optical + SAR)**: *"Identify built-up and water-covered regions using both images."*

---

## Documentation Links
- [System Architecture](docs/ARCHITECTURE.md)
- [Model Documentation & Adaptation](docs/MODEL_DOCUMENTATION.md)
- [Dataset & Benchmark Specifications](docs/DATASET_DOCUMENTATION.md)
- [REST API Reference](docs/API_DOCUMENTATION.md)
- [Platform Presentation Guide](docs/DEMO_GUIDE.md)
- [Evaluation Protocol](docs/EVALUATION.md)
