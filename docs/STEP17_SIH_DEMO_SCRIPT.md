# SatQuery AI — SIH Live Demonstration Script
**Target**: Smart India Hackathon (SIH) • Problem Statement 26167  
**System**: SatQuery AI (Evidence-Grounded Remote Sensing Multi-Specialist Agent)  
**Date**: September 2024 / Live Presentation  

---

## Overview

This document provides a deterministic, step-by-step walkthrough for demonstrating SatQuery AI to the SIH evaluation jury. 

> [!IMPORTANT]
> **Live Demo Ground Rules**:
> 1. **No Fake Results**: SatQuery AI never invents coordinates, confidence, bounding boxes, or detection results. Every output is computed dynamically by local specialist models and classical remote sensing algorithms.
> 2. **Hardware Environment**: This system runs on standard laptop CPU hardware without NVIDIA CUDA. Florence-2 inference takes ~8–15 seconds per query; progress states ("Running vision-language analysis...") communicate processing honestly without fake progress bars.
> 3. **Data Availability**: Real GeoTIFF sample rasters are located in `sample_rasters/` or `data/`. If specific paired sensor data is unavailable, use the clearly labeled local test imagery or verified demo samples.

---

## Demo 1 — Single Image Remote Sensing Analysis

### Objectives
Demonstrate single-raster scene understanding, text-guided grounding (bounding boxes), land-cover analysis (BigEarthNet), and geospatial calibration.

### Procedure
1. **Navigate to Analysis**:
   - From the Home page, click **"Start Analysis"** (or click "Analyze" in the navigation bar).
   - Ensure the mode is set to **"Single Image"**.
2. **Upload / Select Image**:
   - Drag & drop a real satellite raster (e.g., `sample_rasters/berlin_sample.tif` or click **"Load Sample: Dubai Waterfront (WorldView-3)"**).
3. **Validate**:
   - Observe **STEP 2: VALIDATE**.
   - Note the validation badge: sensor telemetry (`WorldView-3`), spatial resolution (`0.3m GSD`), projection (`WGS 84 / UTM zone 40N`), and format verification.
   - Point out that ordinary non-satellite images (selfies, wallpapers) are rejected immediately with clear actionable feedback.
4. **Ask Scene Question**:
   - In **STEP 3: ASK**, select or type:
     - `"What does this image show?"` (Scene description & land cover)
     - or `"Where are the buildings?"` (Text-guided grounding)
5. **Run Analysis**:
   - Click **"Analyze Satellite Image"**.
   - Observe **STEP 4: ANALYZE** with active processing indicators:
     - *"Running vision-language analysis..."*
     - *"Generating evidence..."*
     - *"Preparing result..."*
6. **Show Answer**:
   - Observe **ANSWER** section displaying synthesized findings grounded in model evidence.
7. **Expand Agent Transparency ("How SatQuery analyzed this")**:
   - Open the clean expandable card:
     - **Query**: `"Where are the buildings?"`
     - **Agent Interpretation**: `Vision-Language Remote Sensing / Grounding`
     - **Selected Specialist**: `Florence-2 Vision-Language Specialist`
     - **Model / Tool**: `florence-2-grounding, bigearthnet-classifier`
     - **Geospatial Grounding**: `Available (WGS 84 / UTM zone 40N)` or `Pixel Coordinate Space`
8. **Show Evidence Hierarchy**:
   - **Primary Direct Evidence**: Explicit bounding boxes and detected features.
   - **Secondary Supporting Evidence**: Land-cover class distributions from BigEarthNet.
9. **Show Spatial Evidence**:
   - In the Viewer, highlight the bounding boxes overlaid on the high-resolution imagery.
   - Hover over the image to show the **Real-Time Cursor HUD**:
     - `PIXEL`: `(x, y)`
     - `LAT / LON`: Real geographic coordinates (or `"Geolocation unavailable"` if unprojected)
     - `FEATURE`: Detected feature label
     - `SOURCE`: Model attribution
10. **Open Execution Trace**:
    - Click **"View Multi-Stage Execution Trace"** at the bottom to inspect execution timings (e.g., raster preprocessing, model inference, evidence synthesis).
11. **Download Report**:
    - Scroll to **STEP 6: REPORT** and click **"Download Markdown"** or **"Download JSON"** to demonstrate automated auditing.

---

## Demo 2 — Past & Present (Bi-Temporal Change Detection)

### Objectives
Demonstrate temporal change detection between two acquisitions (T1 vs T2), candidate change segmentation, and limitation governance.

### Procedure
1. **Switch Mode**:
   - Click **"Past & Present"** tab.
2. **Upload T1 & T2**:
   - Upload baseline past image (T1) and monitoring present image (T2), or click **"Load Sample: Dubai Urban Expansion (2021 vs 2023)"**.
3. **Validate Alignment**:
   - Verify that both slots display **"✓ Valid Satellite Image"**.
   - Note the pair compatibility check: disjoint geographic areas (e.g., Mumbai vs Bengaluru) trigger an immediate incompatibility alert: *"Geographic Extent Disparity"*.
4. **Ask Change Question**:
   - Select or enter:
     - `"What changed between these two images?"`
     - or `"Where are the major changed regions?"`
5. **Run Analysis**:
   - Click **"Analyze Changes"**.
6. **Show Answer & Candidate Change Regions**:
   - Observe the honest terminology:
     - Output is classified as **"Candidate change region"** rather than "confirmed change", respecting the absence of ground-truth field surveys.
     - Note the computed change surface area (e.g., in km² or hectares).
7. **Inspect Side-by-Side Synchronized Viewer**:
   - Drag or zoom either image to demonstrate synchronized pan and zoom.
   - Switch between:
     - **Side-by-Side**: Adjacent panels for visual comparison.
     - **Change View**: Amber highlighted candidate change zones.
     - **Overlay Blend**: Slider to dissolve between past (0%) and present (100%).
8. **Explain Limitations**:
   - Point out the **Model Limitations & Uncertainty Governance** section:
     - Mentions sensor angle disparity, seasonal phenology, and lack of sub-surface ground truth.

---

## Demo 3 — Optical + SAR (Cross-Modal Sensor Fusion)

### Objectives
Demonstrate complementary information extraction from Optical (multispectral visual) and SAR (microwave radar backscatter) imagery.

### Procedure
1. **Switch Mode**:
   - Click **"Optical + SAR"** tab.
2. **Upload Optical & SAR**:
   - Upload optical multispectral raster and SAR radar raster, or click **"Load Sample: Munich Airport (Sentinel-2 + Sentinel-1)"**.
3. **Validate Modalities**:
   - Note that both slots verify different physical modalities:
     - Optical: RGB / Multispectral reflectance.
     - SAR: C-band / L-band microwave backscatter.
   - Note that uploading two SAR images triggers a modality mismatch alert: *"Both slots contain SAR radar rasters. Optical + SAR fusion requires one Optical and one SAR image."*
4. **Ask Cross-Modal Question**:
   - Select:
     - `"What features are supported by both optical and SAR evidence?"`
5. **Run Analysis**:
   - Click **"Analyze Optical + SAR"**.
6. **Show Corroborated Evidence**:
   - Observe the distinct evidence sections:
     - **Optical Spectral Indices**: NDVI (vegetation), NDWI (water), NDBI (built-up).
     - **SAR Radar Microwave Backscatter**: High double-bounce returns from vertical structures, low specular reflection from flat water/runways.
     - **Joint Physical Corroboration**: Features confirmed by both spectral color and geometric radar roughness.
7. **Explain Fusion Nature**:
   - Transparently clarify to the jury: this is a **classical rule-based / index-corroborated evidence fusion pipeline**, not a black-box deep learned multimodal transformer.

---

## Demo Reset Workflow

To demonstrate system clean-up between judge questions:
1. Click **"Reset Analysis"** or **"New Analysis"** in the top bar.
2. Verify that:
   - Uploaded files are cleared.
   - Current result and overlays disappear.
   - Slot validation resets to idle.
   - Query reverts to default.
   - **Persistent history is preserved** in the History tab.

---

## Summary Checklist for Judges

| Evaluation Dimension | Where Demonstrated |
| :--- | :--- |
| **Agentic Tool Selection** | "How SatQuery analyzed this" card & Execution Trace |
| **Specialist Execution** | Florence-2 grounding & BigEarthNet classification |
| **Bi-Temporal Analysis** | Past & Present synchronized comparison & candidate change masks |
| **Optical + SAR Fusion** | Cross-modal spectral indices + SAR backscatter corroboration |
| **Geospatial Grounding** | Real-time Cursor HUD (Lat/Lon/Pixel) & CRS calibration |
| **Uncertainty & Honesty** | "Candidate change", "Geolocation unavailable", Limitations section |
| **Auditability** | Structured PDF/Markdown/JSON downloadable reports |
