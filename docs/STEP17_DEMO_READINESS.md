# SatQuery AI — Step 17 Demo Readiness & Live-Jury Guide
**System**: SatQuery AI  
**Target Event**: Smart India Hackathon (SIH) Evaluation  
**Problem Statement**: PS 26167 (Remote Sensing Vision-Language Question Answering)  
**Date**: September 2024  

---

## 1. System Summary & Working Demo Flows

SatQuery AI is an evidence-grounded, multi-specialist remote sensing analysis system. It connects user natural-language queries to specialized local AI models and classical geospatial algorithms via an agentic orchestrator.

### Fully Working Demo Flows

| Demo Flow | Trigger / Mode | Models / Algorithms Used | Output Artifacts | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1. Single-Image VQA & Captioning** | Mode: `single`<br>Query: *"What does this image show?"* | Florence-2 (VLM) + BigEarthNet (ResNet) | Natural-language scene summary, land-cover classes, confidence score | **READY** |
| **2. Text-Guided Grounding** | Mode: `single`<br>Query: *"Where are the buildings?"* / *"Where is the vegetation?"* | Florence-2 Grounding (`<CAPTION_TO_PHRASE_GROUNDING>`) | Pixel/world bounding boxes, feature labels, cursor HUD mapping | **READY** |
| **3. Bi-Temporal Change Detection** | Mode: `bi-temporal`<br>Query: *"What changed between these two images?"* | Classical Difference (SSIM + Morphological Filtering) + Semantic Classification | Candidate change regions, change area (km²), synchronized dual viewer | **READY** |
| **4. Optical + SAR Cross-Modal Analysis** | Mode: `optical-sar`<br>Query: *"What features are supported by both optical and SAR evidence?"* | Spectral Indices (NDVI, NDWI, NDBI) + Microwave Radar Backscatter Thresholding | Optical evidence, SAR evidence, joint physical corroboration | **READY** |
| **5. Geospatial Coordinate Mapping** | Any GeoTIFF with CRS/Affine headers | `rasterio` + `pyproj` affine transform | Real-time cursor HUD: Latitude, Longitude, Pixel (X,Y) | **READY** |
| **6. Automated Report Generation** | Step 6 Report button | Structured Markdown / JSON report exporter | Complete audit trail with inputs, model plan, evidence, limitations | **READY** |

---

## 2. Partially Available / Unavailable Flows

To maintain strict scientific honesty and integrity during jury evaluation:

| Flow / Capability | Current Status | Why It Is Limited / Planned |
| :--- | :--- | :--- |
| **Full Public Benchmark Evaluation (Multi-Dataset)** | **Partially Available** | Evaluation harnesses for BigEarthNet, RSVQA, and xView are implemented (`backend/benchmark/`), but full multi-gigabyte benchmark archives are not downloaded locally due to storage constraints. Skips and discovery are handled cleanly without fake scores. |
| **Deep Learned Optical-SAR Fusion Transformer** | **Not Implemented** (Classical Rule-Based Active) | The current implementation uses classical spectral indices (NDVI/NDWI) and SAR backscatter thresholding. It is **not** a learned end-to-end multimodal neural network; it is presented honestly as evidence-based physical corroboration. |
| **Sub-Meter Unprojected Georeferencing** | **Unavailable for plain PNG/JPG** | Plain PNG/JPEG files lack geospatial tags; the system explicitly reports *"Geolocation unavailable (Pixel Coordinate Space)"* rather than fabricating coordinates. |

---

## 3. Required Sample Data

The following verified sample datasets are bundled or referenced for demonstration:

1. **Local GeoTIFF Rasters (`sample_rasters/` or `data/`)**:
   - `berlin_sample.tif` / `sample_geotiff.tif`: High-resolution WorldView / Sentinel-2 scenes with genuine EPSG:32633 CRS and affine geotransforms.
2. **Local Test Dataset / Demo Presets**:
   - **Single Image**: Dubai Waterfront (WorldView-3, 0.3m GSD, WGS 84 / UTM 40N).
   - **Past & Present**: Dubai Urban Expansion (2021 Baseline vs 2023 Monitoring).
   - **Optical + SAR**: Munich Airport (Sentinel-2 Optical Multispectral + Sentinel-1 SAR Radar).
3. **Synthetic / Simulation Fallbacks**:
   - Clearly labeled as *"Demo Preset — Synthetic Sample"* to prevent any confusion with live satellite downlinks.

---

## 4. Expected Inference Times (CPU Hardware)

This evaluation workstation operates in **CPU-only mode** (no dedicated NVIDIA CUDA GPU). Expected execution latencies:

| Processing Stage | Expected Latency | Notes |
| :--- | :--- | :--- |
| **Input Raster Ingestion & Validation** | ~200–500 ms | Instantaneous header inspection and metadata verification |
| **Agentic Query Classification & Planning** | ~50–100 ms | Rule-based and embedding intent classification |
| **BigEarthNet Land-Cover Tiling** | ~1.5–3.0 s | 19-class ResNet inference over raster patches |
| **Florence-2 Vision-Language Grounding** | ~8–15 s | CPU inference for sequence-to-sequence autoregressive decoding |
| **Bi-Temporal Classical Change Segmentation** | ~1.0–2.5 s | Fast OpenCV-based structural similarity and contour extraction |
| **Optical + SAR Cross-Modal Corroboration** | ~1.0–2.0 s | NumPy spectral index computation and radar backscatter thresholding |
| **Total End-to-End Pipeline** | **~10–18 s** | Honest progress feedback (*"Running vision-language analysis..."*) is displayed |

---

## 5. Known Limitations & Governance

1. **CPU Execution Throughput**: Processing multiple concurrent requests will queue. For live presentation, execute one query at a time.
2. **Atmospheric & Solar Angle Variance**: Bi-temporal difference algorithms can flag cloud shadows or seasonal sun elevation changes as candidate change regions. The UI transparently presents these as *"Candidate change regions"* subject to semantic confirmation.
3. **No Decorative Hallucinations**: When a query cannot be answered from the raster (e.g. asking for vehicle driver identity or indoor temperatures), SatQuery AI returns *"No reliable result available"* rather than guessing.

---

## 6. Live-Demo Checklist

Before presenting to SIH judges:
- [ ] Ensure backend is running: `uvicorn backend.app.main:app --port 8000` (or `python -m backend.app.main`).
- [ ] Ensure frontend is running: `npm run dev` (serving at `http://localhost:5173`).
- [ ] Verify Settings page shows **Backend: Online**, **API Endpoints: Ready (3/3)**, **Models: Local / Available**.
- [ ] Test **Reset Analysis** button to ensure clean slate between demonstration runs.
- [ ] Keep `docs/STEP17_SIH_DEMO_SCRIPT.md` open for reference during the presentation.
