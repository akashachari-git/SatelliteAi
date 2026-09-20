# SatQuery AI - Live Deployment Guide
**Smart India Hackathon PS 26167: Multimodal Remote-Sensing VLM Assistant**

---

## 1. System Architecture Overview

SatQuery AI is architected as a decoupled web application:
- **Frontend**: React 19 + Vite Single Page Application (SPA), deployable to any global static edge network (e.g., **Vercel**, Netlify, AWS CloudFront).
- **Backend**: Python 3.10+ FastAPI microservice running PyTorch, ONNX Runtime, and geospatial raster engines (`rasterio`, `shapely`, `numpy`), deployable to a **persistent container or VM** (e.g., **Render, Railway, AWS EC2, GCP Cloud Run with min-instances, or DigitalOcean App Platform**).

```
User Browser
    │
    ▼
┌──────────────────────────────────────────────┐
│ Vercel (Edge CDN)                            │
│ React 19 + Vite SPA (SatQuery AI Dashboard)  │
└──────────────────────┬───────────────────────┘
                       │ HTTPS / REST API calls
                       ▼
┌──────────────────────────────────────────────┐
│ Persistent Container / VM (Render/AWS/GCP)   │
│ FastAPI Backend (Uvicorn on Port 8000)       │
│  ├─ Geospatial Validator & Raster Engine     │
│  ├─ Agentic Task Classifier & Planner        │
│  ├─ Florence-2-base Local VLM (PyTorch CPU)  │
│  ├─ BigEarthNet-19 Multi-Label (ONNX)        │
│  ├─ Bi-temporal Differencing & SSIM Engine   │
│  └─ Optical-SAR Wavelet Fusion Engine        │
└──────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Persistent Backend Requirement**:
> Florence-2-base (`~0.9 GB` parameters) and the BigEarthNet ResNet-18 model execute locally on CPU/GPU. They cannot run inside standard ephemeral serverless frontend functions (like Vercel Serverless Functions or AWS Lambda with 250MB package limits). The backend requires a persistent environment with at least **4 GB RAM** and **2+ vCPUs** for responsive inference.

---

## 2. Backend Deployment (Persistent Container / VM)

### Option A: Deploy via Docker (Recommended)

1. **Build Container Image**:
   ```bash
   docker build -t satquery-backend -f backend/Dockerfile .
   ```

2. **Run Container Locally or on Server**:
   ```bash
   docker run -d \
     -p 8000:8000 \
     -e PORT=8000 \
     -e ALLOWED_ORIGINS="https://your-satquery-frontend.vercel.app" \
     --name satquery-backend \
     satquery-backend
   ```

### Option B: Deploy to Render / Railway / Cloud Provider

1. **Root Directory**: `backend` (or repository root with working directory set to `backend`).
2. **Build Command**:
   ```bash
   pip install --no-cache-dir -r requirements.txt
   ```
3. **Start Command**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
   ```
4. **Environment Variables**:
   | Variable | Example Value | Description |
   | :--- | :--- | :--- |
   | `PORT` | `8000` | Port assigned by host platform |
   | `ALLOWED_ORIGINS` | `https://your-frontend.vercel.app` | Comma-separated list of allowed frontend domains for CORS |
   | `FLORENCE2_CHECKPOINT_DIR` | `backend/models/checkpoints/florence2-base` | Path to local Florence-2 weights |
   | `BIGEARTHNET_MODEL_PATH` | `backend/models/checkpoints/bigearthnet/bigearthnet_resnet18_10band.onnx` | Path to ONNX classifier |

---

## 3. Frontend Deployment (Vercel)

The frontend is a Vite + React SPA that compiles to static assets.

### Deploying to Vercel

1. **Connect GitHub Repository**:
   - Go to [Vercel Dashboard](https://vercel.com/dashboard) and click **"Add New Project"**.
   - Select the `SatelliteAi` repository.

2. **Project Settings**:
   - **Framework Preset**: `Vite`
   - **Root Directory**: `./`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`

3. **Environment Variables on Vercel**:
   | Variable | Value |
   | :--- | :--- |
   | `VITE_API_BASE_URL` | `https://your-backend-service.onrender.com` (your live backend URL) |

4. **SPA Routing**:
   - The repository includes [`vercel.json`](file:///c:/Users/Admin/OneDrive/Desktop/satquery%20ai/vercel.json) at root:
     ```json
     {
       "version": 2,
       "rewrites": [
         { "source": "/(.*)", "destination": "/index.html" }
       ]
     }
     ```
   - This ensures all client-side navigation routes reload cleanly without 404 errors.

---

## 4. CORS Configuration

To allow the frontend on Vercel to securely communicate with the backend:
1. In the backend hosting platform's environment variables, set:
   ```env
   ALLOWED_ORIGINS=https://your-frontend.vercel.app
   ```
2. For local testing alongside production, you can supply multiple comma-separated origins:
   ```env
   ALLOWED_ORIGINS=https://your-frontend.vercel.app,http://localhost:5173,http://127.0.0.1:5173
   ```
3. The backend [`backend/main.py`](file:///c:/Users/Admin/OneDrive/Desktop/satquery%20ai/backend/main.py) parses this into a strict whitelist for FastAPI's `CORSMiddleware`, eliminating insecure `"*"` wildcards in production.

---

## 5. Health-Check Verification

Once the backend is deployed, verify its status using `curl` or a web browser:

```bash
curl -X GET "https://your-backend-domain.com/api/health"
```

Expected JSON Response:
```json
{
  "status": "healthy",
  "timestamp": "2026-09-20 15:30:00 UTC",
  "environment": "SatQuery AI Remote Sensing Backend",
  "accelerator": "cpu",
  "architecture": "FastAPI + PyTorch/ONNX Modular Engine",
  "activeModelsCount": 5,
  "version": "1.0.0"
}
```

Verify available registered models:
```bash
curl -X GET "https://your-backend-domain.com/api/models"
```

---

## 6. Connecting Frontend to Backend

1. When deploying the frontend, verify `VITE_API_BASE_URL` matches your deployed backend URL:
   ```env
   VITE_API_BASE_URL=https://your-backend-domain.com
   ```
2. The frontend [`src/services/api.ts`](file:///c:/Users/Admin/OneDrive/Desktop/satquery%20ai/src/services/api.ts) automatically strips trailing slashes and routes all requests (`/api/health`, `/api/validate-image`, `/api/analyze`, `/api/change-analysis`, `/api/optical-sar-analysis`, `/api/report`) to this base URL.

---

## 7. Testing the Live Application

Perform the following verification steps on the live deployment:

1. **System Health Badge**:
   - Open the live frontend URL in a browser.
   - Verify the top navigation bar shows **"API Connected"** (green indicator).
2. **Single-Image Analysis Flow**:
   - Upload a sample remote sensing image (RGB GeoTIFF or PNG/JPEG).
   - Enter a query: `"Describe this remote sensing image"` or `"Detect water bodies and vegetation"`.
   - Verify: Analysis completes, radar/land-cover/evidence cards render, visual bounding boxes or segmentation masks appear, and audit trace displays tool execution.
3. **Bi-Temporal Change Detection Flow**:
   - Switch to **"Past & Present"** mode.
   - Upload pre-event (T1) and post-event (T2) images.
   - Enter: `"Analyze changes between the two dates"`.
   - Verify: Change mask, difference heatmap, and changed area statistics render.
4. **Optical + SAR Multi-Sensor Fusion Flow**:
   - Switch to **"Optical + SAR"** mode.
   - Upload optical RGB and SAR VV/VH images.
   - Enter: `"Perform multi-sensor fusion and identify features"`.
   - Verify: Wavelet-fused imagery and cross-sensor evidence render.
5. **PDF Report Export**:
   - Click **"Export Report"**.
   - Verify: Clean PDF/HTML report is generated containing queries, results, and execution traces.
