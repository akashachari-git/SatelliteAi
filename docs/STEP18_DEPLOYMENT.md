# SatQuery AI — Production Deployment Architecture & Guide
**System**: SatQuery AI (SIH Problem Statement 26167)  
**Target**: Production Deployment (Vercel Static Frontend + Persistent Python/FastAPI Backend)  
**Date**: September 2024  

---

## 1. Architecture Overview

SatQuery AI uses a decoupled, hybrid architecture to balance interactive responsiveness with heavy remote sensing inference:

```
┌─────────────────────────────────────────────────────────┐
│              FRONTEND (Vercel / Cloudflare)             │
│  - React 18 + TypeScript + Vite                         │
│  - Static SPA with synchronized imagery viewers         │
│  - Real-time Cursor HUD (Client-side requestAnimFrame)  │
│  - Communicates via VITE_API_BASE_URL                   │
└────────────────────────────┬────────────────────────────┘
                             │ HTTPS / REST
                             ▼
┌─────────────────────────────────────────────────────────┐
│        BACKEND (Persistent Container / VM Service)      │
│  - FastAPI (Python 3.10+)                               │
│  - Local Specialist Models:                             │
│      • Florence-2 VLM (PyTorch, 0.9 GB Checkpoint)      │
│      • BigEarthNet Classifier (ONNX Runtime, 43 MB)     │
│      • Bi-Temporal & Optical-SAR Differencing Engines   │
│  - Geospatial Engine (rasterio, pyproj)                 │
└─────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Why Serverless / Edge Functions Cannot Host the Backend**:
> `microsoft/Florence-2-base` requires approximately 1.5–2.5 GB of resident memory during inference and execution latencies of ~8–15 seconds on CPU. Standard serverless functions (AWS Lambda, Vercel Serverless) have cold-start timeouts and bundle size limits (typically 50–250 MB) that are incompatible with deep learning weights and PyTorch dependencies.
> Therefore, the backend **must be deployed on a persistent container or VM** (e.g., Render Web Service, Railway, Fly.io, AWS EC2, or a dedicated server).

---

## 2. Frontend Deployment (Vercel / Cloudflare Pages / Netlify)

The frontend is a standard Vite single-page application and can be deployed to any static host.

### Build Configuration
- **Framework Preset**: Vite
- **Build Command**: `npm run build`
- **Output Directory**: `dist`
- **Install Command**: `npm install`

### Environment Variables
Configure the following in the Vercel/host dashboard:

| Variable | Description | Example Value |
| :--- | :--- | :--- |
| `VITE_API_BASE_URL` | URL of the deployed FastAPI backend | `https://api.satquery.ai` or `https://satquery-backend.onrender.com` |

---

## 3. Backend Deployment (Docker / Persistent VM / Render / Railway)

### Option A: Docker Deployment
A production `backend/Dockerfile` is provided:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for GDAL and rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and checkpoints
COPY . .

ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t satquery-backend -f backend/Dockerfile backend
docker run -p 8000:8000 -e ALLOWED_ORIGINS="https://your-frontend.vercel.app" satquery-backend
```

### Option B: Direct Python Service (Render / Railway / VM)
Start command:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

### Backend Environment Variables

| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `PORT` | Optional | `8000` | Port for the HTTP server |
| `ALLOWED_ORIGINS` | Recommended | `*` | Comma-separated CORS origins (e.g., `https://satquery.vercel.app`) |
| `FLORENCE2_CHECKPOINT_DIR` | Optional | `backend/models/checkpoints/florence2-base` | Local path to Florence-2 weights |
| `BIGEARTHNET_MODEL_PATH` | Optional | `backend/models/checkpoints/bigearthnet/bigearthnet_resnet18_10band.onnx` | Local path to ONNX classifier |
| `BIGEARTHNET_ROOT` | Optional | `""` | Local root directory for BigEarthNet evaluation dataset |
| `RSVQA_ROOT` | Optional | `""` | Local root directory for RSVQA evaluation dataset |

---

## 4. Security & Configuration Best Practices

1. **No Hardcoded Secrets**: SatQuery AI relies entirely on local models and does not require external third-party API keys (e.g. OpenAI or Gemini) for core inference.
2. **CORS Hardening**: In production, set `ALLOWED_ORIGINS` to the exact frontend domain (e.g., `ALLOWED_ORIGINS="https://satquery.vercel.app"`).
3. **Storage Sanitization**: Uploaded rasters are processed in memory and temporary cache without storing user satellite imagery permanently unless persisted to local history.
4. **Health Endpoint**: Load balancers should monitor `GET /api/health` which returns `HTTP 200` with runtime device status.
