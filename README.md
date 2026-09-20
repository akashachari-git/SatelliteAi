# SatQuery AI — Remote Sensing Vision-Language Assistant

**Smart India Hackathon (SIH 2024)** • **Problem Statement 26167**  
*An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries*

---

## 1. Overview

SatQuery AI is an evidence-grounded remote sensing intelligence system designed to answer natural-language queries about satellite imagery. Rather than relying on a single black-box model, SatQuery AI implements an **agentic multi-specialist orchestrator** that classifies user intent, routes tasks to local vision-language and remote-sensing specialist models, extracts geospatial metadata, and synthesizes an auditable, evidence-backed answer.

### Core Capabilities
- **Single-Image Remote Sensing Analysis**: Visual Question Answering (VQA), detailed scene captioning, and text-guided spatial grounding (bounding boxes).
- **Bi-Temporal Analysis (Past & Present)**: Co-registered change detection between dual acquisitions, candidate change region segmentation, surface area estimation, and synchronized comparison viewers.
- **Optical + SAR Cross-Modal Analysis**: Complementary physical corroboration combining multispectral optical reflectance (NDVI, NDWI, NDBI) with microwave SAR radar backscatter (Sentinel-1 VV/VH).
- **Geospatial Grounding**: Automatic extraction of Coordinate Reference Systems (CRS), affine geotransforms, and real-time cursor coordinate tracking (Latitude, Longitude, Pixel X/Y) or honest `"Geolocation unavailable"` reporting.
- **Auditable Execution Trace & Reporting**: Complete step-by-step provenance for every query, with downloadable intelligence dossiers in Markdown and JSON.

---

## 2. Architecture

SatQuery AI couples a lightweight, responsive React 18 single-page application with a persistent Python/FastAPI backend hosting local neural models:

```
┌─────────────────────────────────────────────────────────┐
│                    REACT 18 FRONTEND                    │
│  - 6-Stage Demo Flow: Upload → Validate → Ask →         │
│    Analyze → Evidence → Report                          │
│  - "How SatQuery analyzed this" Agent Transparency Card │
│  - Real-time Cursor HUD (requestAnimationFrame)         │
│  - Synchronized Dual-Viewer (Side-by-Side, Overlay)     │
└────────────────────────────┬────────────────────────────┘
                             │ REST API (JSON)
                             ▼
┌─────────────────────────────────────────────────────────┐
│                 FASTAPI PYTHON BACKEND                  │
│  - SatQueryAgent & AgentPlanner (Intent Routing)        │
│  - Local Specialist Models:                             │
│      • Microsoft Florence-2-base (VLM / Grounding)      │
│      • BigEarthNet ResNet-18 (10-Band RS Classifier)    │
│      • Bi-Temporal Classical Differencing Engine        │
│      • Optical + SAR Physical Corroborator              │
│  - GeoreferenceEngine (rasterio + pyproj)               │
│  - EvidenceCombiner & Execution Tracer                  │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Scientific Honesty & Model Scope

In strict accordance with scientific integrity and PS 26167 requirements:
- **Florence-2-base** (`backend/models/checkpoints/florence2-base`): Integrated locally for general vision-language understanding, captioning, and bounding box grounding. It is **not** fine-tuned on specialized remote-sensing corpora.
- **BigEarthNet ResNet-18** (`backend/models/checkpoints/bigearthnet/`): A specialized local ONNX model evaluating 10-band Sentinel-2 rasters across 19 Corine Land Cover classes. It is an **image classifier**, not a VLM.
- **Bi-Temporal Differencing**: Operates via classical radiometric, structural (SSIM), and spectral differencing. Outputs are classified as **"candidate change regions"** rather than "confirmed change".
- **Optical + SAR Fusion**: Operates via rule-based physical corroboration (spectral indices + radar backscatter), not a learned multimodal cross-attention transformer.
- **Zero Hallucinated Coordinates**: Unprojected imagery (plain PNG/JPEG) explicitly reports `"Geolocation unavailable (Pixel Coordinate Space)"`.

---

## 4. Quick Start & Setup

### Prerequisites
- **Node.js**: v18 or later
- **Python**: 3.10, 3.11, or 3.12
- **Hardware**: Standard x86_64 CPU (4+ cores recommended, 8GB+ RAM). Dedicated GPU is optional.

### 1. Clone & Install Frontend
```bash
git clone https://github.com/ajayreddy347/satquery_ai.git
cd satquery_ai
npm install
```

### 2. Set Up Backend Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r backend/requirements.txt
```

### 3. Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 4. Start Services
**Terminal 1 (Backend):**
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
*Health endpoint will be available at `http://localhost:8000/api/health`.*

**Terminal 2 (Frontend):**
```bash
npm run dev
```
*Frontend will be running at `http://localhost:5173`.*

---

## 5. Running Automated Tests

### Backend Unit & Integration Tests (144 Tests)
```bash
python -m unittest discover -s backend/tests -v
```

### Frontend Production & Demo Tests (16 Tests)
```bash
npx tsx src/tests/productionIntegration.test.ts
```

### Production Build Verification
```bash
npm run build
```

---

## 6. Demonstration Guide

Refer to [docs/STEP17_SIH_DEMO_SCRIPT.md](docs/STEP17_SIH_DEMO_SCRIPT.md) for the deterministic 3-demo jury presentation script:
- **Demo 1**: Single Image VQA, Grounding, and Geospatial Calibration.
- **Demo 2**: Past & Present Bi-Temporal Change Detection.
- **Demo 3**: Optical + SAR Cross-Modal Sensor Corroboration.

For system readiness and CPU latencies, see [docs/STEP17_DEMO_READINESS.md](docs/STEP17_DEMO_READINESS.md).  
For the detailed PS 26167 compliance audit, see [docs/STEP18_FINAL_PS26167_AUDIT.md](docs/STEP18_FINAL_PS26167_AUDIT.md).

---

## 7. Deployment

See [docs/STEP18_DEPLOYMENT.md](docs/STEP18_DEPLOYMENT.md) for full deployment instructions:
- **Frontend**: Deployable as a static single-page application on Vercel, Netlify, or Cloudflare Pages.
- **Backend**: Requires a persistent container or VM (Render, Railway, Fly.io, or AWS EC2) to support resident Florence-2 model weights and PyTorch CPU inference.

---

## 8. License & Acknowledgments

Developed for the **Smart India Hackathon (SIH 2024)** under **Problem Statement 26167**.  
Built using Microsoft Florence-2, TU Berlin BigEarthNet, FastAPI, PyTorch, rasterio, React, and Vite.
