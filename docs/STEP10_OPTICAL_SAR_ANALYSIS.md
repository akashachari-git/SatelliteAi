# STEP 10: Optical + SAR Multimodal Analysis Pipeline

## 1. Executive Summary & Audit Findings

### Prior Architecture & Audit
Before Step 10, an audit of `backend/models/optical_sar_fusion.py` and associated modules revealed:
1. **Fictitious Deep Learning Claims**: The prior implementation claimed to run a `"620M parameter CrossSens-Fusion Transformer"` with `"CrossSens-ViT-L/14"` weights, reporting a hardcoded confidence score of `0.968` (96.8%) and a fake sub-pixel co-registration RMSE of `0.3 px`.
2. **Template Mock**: The inference function completely ignored the pixel values of the uploaded optical and SAR rasters, returning static pre-written text strings and mock bounding boxes.
3. **No Local SAR / Multimodal Weights**: Inspection of `backend/models/checkpoints/` and the workspace confirmed that **no genuine learned SAR (Sentinel-1) or Optical-SAR multimodal fusion model weights exist locally**.
4. **Spatial Registration**: The prior code did not perform genuine CRS or geotransform verification, nor did it check spatial compatibility.

### Step 10 Remediation & Guiding Principles
- **Complete Elimination of Fabricated Metrics**: Removed all references to fictitious 620M models, fake 96.8% confidence scores, and fake sub-pixel alignment claims. Confidence is strictly reported as `None`.
- **Scientifically Grounded Physical Fusion**: Implemented an evidence-based physical corroboration pipeline (`backend/app/models/optical_sar_inference.py`) combining multispectral optical indices with calibrated radar backscatter physics.
- **Genuine Optical & SAR Processing**: Directly processes uploaded raster arrays, extracts valid spectral and polarimetric signatures, and detects physical surface phenomena.
- **Strict Spatial & Modality Integrity**: Rejects invalid inputs (e.g. optical-only, SAR-only, optical RGB disguised as SAR, conflicting CRS projections) and explicitly discloses when pixel-space coordinates are used due to unprojected imagery.

---

## 2. Technical Architecture

```
                       User Query & Uploaded Observations
                       (1 Optical Raster + 1 SAR Raster)
                                       │
                                       ▼
                     SatQueryAgent Autonomous Router
                (classify_task: 'optical-sar-analysis')
                                       │
                                       ▼
                       GeoTIFFValidator Gatekeeper
                 (Modality check: 1 Optical & 1 SAR)
                                       │
                                       ▼
                   OpticalSARFusionService (Physical Pipeline)
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        ▼                                                             ▼
 [Optical Evidence Engine]                                   [SAR Radar Engine]
 - 10-Band Sentinel-2 Check                                  - Reject RGB photo without SAR meta
   ├── NDVI (B08 - B04)/(B08 + B04)                          - dB Calibration: 10*log10(σ⁰)
   ├── NDWI (B03 - B08)/(B03 + B08)                          - Dual-Pol VV/VH ratio (VV_dB - VH_dB)
   └── NDBI (B11 - B08)/(B11 + B08)                          - Specular extinction: σ⁰ < -20.0 dB
 - 3-Band RGB Fallback:                                      - Double-bounce corner: σ⁰ > -6.0 dB
   └── Visible albedo & VARI                                 - Rough canopy volume: [-15, -8] dB
        │                                                             │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                                       ▼
                    [Cross-Modal Corroboration Engine]
     ├── Water: Optical Absorption (NDWI / dark albedo) ∩ SAR Specular Extinction (< -20 dB)
     └── Built-Up: Optical Albedo/NDBI ∩ SAR Double-Bounce Reflections (> -6 dB)
                                       │
                                       ▼
                         [Spatial Region Extractor]
     ├── Connected component labeling (scipy.ndimage)
     ├── Bounding boxes with pixel extents [x, y, w, h]
     └── Honest metadata (confidence=None, spatial alignment notes)
```

---

## 3. Modality Processing Details

### Optical Processing
1. **Sentinel-2 10-Band Multispectral Input**:
   - Bands: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12.
   - Spectral Indices:
     - $\text{NDVI} = \frac{\text{B08} - \text{B04}}{\text{B08} + \text{B04}}$ (Vegetation health)
     - $\text{NDWI} = \frac{\text{B03} - \text{B08}}{\text{B03} + \text{B08}}$ (Water detection)
     - $\text{NDBI} = \frac{\text{B11} - \text{B08}}{\text{B11} + \text{B08}}$ (Built-up / impervious surfaces)
2. **Standard 3-Band RGB Input**:
   - Computes visible surface albedo: $\text{mean}(R, G, B)$.
   - Computes Visible Atmospherically Resistant Index: $\text{VARI} = \frac{G - R}{G + R - B}$.
   - Explicitly notes absence of NIR/SWIR bands for spectral index derivation.

### SAR Radar Processing
1. **Polarimetric Representations Supported**:
   - Single-polarization (VV or VH amplitude / dB)
   - Dual-polarization (VV + VH)
   - Cross-polarization ratio: $\text{VV} / \text{VH}$ in dB: $\sigma^0_{\text{VV}} - \sigma^0_{\text{VH}}$
2. **Backscatter Calibration**:
   - Converts linear backscatter amplitudes to decibels ($\text{dB}$) via $10 \cdot \log_{10}(\max(\text{DN}, 10^{-6}))$.
3. **Physical Radar Scattering Signatures**:
   - **Specular Extinction** ($\sigma^0 < -20.0\text{ dB}$): Smooth dielectric surfaces (calm water bodies, smooth asphalt) mirror radar pulses away from the antenna.
   - **Double-Bounce Reflections** ($\sigma^0 > -6.0\text{ dB}$): Vertical man-made structures, buildings, and corner reflectors produce strong dihedral scattering back to the sensor.
   - **Volume Scattering** ($-15.0\text{ dB} \le \sigma^0 \le -8.0\text{ dB}$): Tree canopies and rough vegetation produce multiple depolarizing bounces.
4. **Masquerading Input Rejection**:
   - 3-channel images without radar metadata (`modality: "SAR"`, `sensor: "Sentinel-1"`, or polarization band names) are strictly rejected as optical imagery disguised as SAR.

---

## 4. Spatial Registration & Compatibility

- **CRS Verification**: If both inputs provide projected Coordinate Reference Systems (e.g. `EPSG:32643` vs `EPSG:32644`), mismatched CRSes trigger an explicit `ValueError: CRS Incompatibility`.
- **Unprojected / Local Pixel Space**: If one or both images are unprojected (e.g. PNG or unprojected TIFF), analysis is executed in pixel space, and the response explicitly notes:
  > *"Spatial registration is unavailable; pixel-level fused spatial interpretation is limited."*
- **Aspect Ratio Validation**: Observations whose aspect ratios differ by more than 40% are rejected to prevent nonsensical spatial pairing.
- **Resampling Disclosure**: If spatial resolutions differ between Optical and SAR, SAR is resampled to the Optical grid, with transparent disclosure that approximate resampling was performed and sub-pixel orthorectification was not applied.

---

## 5. Query Routing & Task Classification

The autonomous router in `SatQueryAgent.classify_task` prioritizes cross-sensor Optical+SAR intent before general bi-temporal comparison:
- Queries matching:
  - `"Analyze optical and SAR together"`
  - `"Compare optical and radar imagery"`
  - `"What does SAR reveal that optical does not?"`
  - `"Analyze this optical SAR pair"`
  - `"Combine optical and SAR evidence"`
- Seamlessly route to `taskType = "optical-sar-analysis"`.
- Uses `selectedModel = "Optical + SAR Multimodal Fusion Specialist"`.

---

## 6. Verification & Test Suite

Focused test suite: `backend/tests/test_optical_sar_pipeline.py` (13 tests):
1. `test_01_valid_optical_sar_pair_processing`: Valid Optical + SAR pair produces genuine cross-modal evidence.
2. `test_02_optical_only_input_rejected`: Optical-only input is rejected for multimodal route.
3. `test_03_sar_only_input_rejected`: SAR-only input is rejected.
4. `test_04_incompatible_crs_rejected`: Conflicting projected CRSes are rejected.
5. `test_05_incompatible_aspect_ratio_rejected`: Severely mismatched aspect ratios are rejected.
6. `test_06_invalid_sar_rgb_masquerading_rejected`: 3-channel optical image without SAR metadata is rejected.
7. `test_07_missing_georeferencing_uses_pixel_space`: Pixel space coordinates preserved without fake geographic coordinates.
8. `test_08_genuine_vv_vh_and_db_handling`: Dual-pol VV/VH and cross-ratio evaluated.
9. `test_09_multispectral_indices_computed_when_available`: 10-band Sentinel-2 inputs trigger NDVI, NDWI, NDBI.
10. `test_10_cross_modal_fused_evidence_generation`: Cross-sensor corroboration isolates water and built-up structures.
11. `test_11_optical_sar_query_routing`: Natural language optical+SAR queries route to `optical-sar-analysis`.
12. `test_12_no_fabricated_confidence_or_neural_claims`: Confidence is None; no 620M parameter claims.
13. `test_13_agent_end_to_end_optical_sar_execution`: End-to-end execution through `SatQueryAgent.execute_pipeline`.

---

## 7. Limitations & Future Options

### Current Limitations
1. **No Learned Multimodal Weights**: Due to absence of local pretrained weights, fusion is physical and rule-based, not a learned neural embedding.
2. **Sub-Pixel Orthorectification**: In the absence of digital elevation models (DEM) and orbital ephemeris data, geometric distortion from radar layover/foreshortening cannot be terrain-corrected locally.

### Future Option for a Pretrained Remote-Sensing Multimodal Model
When local compute and storage allow, verified checkpoints such as:
- **Prithvi-100M-multimodal** (IBM / NASA) or
- **RemoteCLIP / CSPC** (Optical-SAR cross-modal contrastive models)
can be integrated by loading genuine PyTorch/ONNX checkpoints into `backend/models/checkpoints/` and registering them through `OpticalSARFusionService`.
