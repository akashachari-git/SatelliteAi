# SatQuery AI — Platform Demonstration Guide

This guide describes how to present the complete capabilities of SatQuery AI for technical review and evaluation.

---

## 1. Quick One-Click Demonstrations
Click **"Load Curated Demo"** in the top navigation bar to instantly mount authentic GeoTIFF imagery:

### Scenario 1: Urban Land-Cover VQA
- **Rasters**: `urban_satellite_eo.tif` (GeoTIFF with UTM 43N tags, 10m GSD)
- **Prompt**: *"Describe the major land-cover types visible in this image."*
- **What to show judges**:
  - The system detects the GeoTIFF metadata (512x512, 10m resolution, CRS EPSG:32643).
  - The agent classifies the query into `SINGLE_VQA`.
  - The model calculates NDVI, NDWI, and NDBI and classifies the scene using BigEarthNet 19-class taxonomy.
  - The confidence score is calibrated based on spectral margin.

### Scenario 2: Text-Guided Water Grounding
- **Rasters**: `water_reservoir_eo.tif` (Curved river and reservoir body)
- **Prompt**: *"Highlight the water body referred to in the query."*
- **What to show judges**:
  - The agent selects `GroundingModel`.
  - Notice the interactive image viewer displaying both a **cyan segmentation mask overlay** (NDWI thresholded) and **labeled bounding boxes** with coordinates.
  - The opacity slider allows seamless visual verification against the base imagery.

### Scenario 3: Bi-Temporal Urban Expansion
- **Rasters**: `temporal_t1_2021.tif` (Earlier) and `temporal_t2_2023.tif` (Later)
- **Prompt**: *"What changed between these two dates, and where did the change occur?"*
- **What to show judges**:
  - The system checks co-registration and spatial overlap between T1 and T2.
  - The agent routes to `ChangeDetectionModel` and `ChangeVQAModel`.
  - Open the **Bi-Temporal Studio** tab: show the **interactive swipe curtain**, the **difference heatmap**, and the **quadrant breakdown** (confirming changes concentrated in the Eastern quadrant).

### Scenario 4: Optical + SAR Cross-Modal Fusion
- **Rasters**: `crossmodal_optical.tif` (Sentinel-2 Optical) and `crossmodal_sar_s1.tif` (Sentinel-1 C-SAR backscatter)
- **Prompt**: *"Identify built-up and water-covered regions using both images."*
- **What to show judges**:
  - The system detects different modalities (Optical vs SAR).
  - The agent selects `OpticalSARFusion`.
  - Open the **Optical+SAR Studio** to toggle side-by-side comparison and false-color fusion (Red=Albedo, Green=SAR backscatter double-bounce, Blue=Water absorption).

---

## 2. Incompatible Query Rejection Demonstration
Try asking:
*"What changed between these two dates?"* with only **ONE** image uploaded.
- **Outcome**: The agent controller catches the precondition failure in Stage 4 and explains clearly:
  *"This task requires two temporally corresponding images. Please upload an earlier (T1) and later (T2) image to perform change detection."*
- Proves the system possesses real agentic reasoning rather than a hardcoded chatbot.

---

## 3. Auditable Execution Trace & PDF Report
1. Open the **Agent Trace** tab to view the visual DAG showing every observable pipeline stage with millisecond latencies.
2. Click **"Intelligence Dossier (PDF)"** to download the formal analysis report generated via ReportLab with embedded evidence visualizations.
