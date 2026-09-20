# STEP 9 — Bi-Temporal Change-Analysis Pipeline Audit & Upgrade

**Status:** Completed  
**Date:** September 20, 2026  
**Environment:** Windows, Python 3.13.3, NumPy 2.4.3, SciPy 1.17.1, PyTorch 2.14.0+cpu, ONNX Runtime 1.24.3  

---

## 1. Executive Summary

In Step 9, we audited the existing bi-temporal change analysis implementation in SatQuery AI and replaced the previous mock/template system with a scientifically honest, evidence-grounded difference analysis engine.

Key outcomes:
1. **Audit Completed**: Confirmed that no learned change-detection model weights (such as ChangeFormer or BIT) existed locally, and that prior outputs were static string templates with fabricated metrics (e.g. `+2.85 km²`, `32.4%`, `0.28 px RMSE`).
2. **Dedicated Classical Difference Engine**: Created [bitemporal_inference.py](file:///c:/Users/Admin/OneDrive/Desktop/satquery%20ai/backend/app/models/bitemporal_inference.py) implementing consistent reference scaling, normalized Euclidean spectral differencing, adaptive thresholding, and morphological connected-component region extraction.
3. **Dual-State BigEarthNet Land-Cover Prior**: When both temporal observations contain authentic 10-band Sentinel-2 data, the engine independently runs the verified BigEarthNet ResNet-18 model on each date and evaluates 19-class land-cover probability deltas (e.g. urban expansion vs. vegetation decline).
4. **Strict Scientific Honesty**:
   - Differentiates raw spectral difference vs. candidate change vs. confirmed semantic change.
   - Bounding boxes are strictly in image pixel coordinates unless genuine georeferencing is available.
   - Eliminates fabricated confidence scores (`confidence = None`).
   - Rejects incompatible CRS projections and mismatched aspect ratios.
5. **Zero Regressions**: All **65/65 backend tests pass** (41 BigEarthNet + 12 Florence-2 + 12 Bi-Temporal).

---

## 2. Pre-Upgrade Audit Findings

| Audit Dimension | Pre-Upgrade State | Finding & Critique |
| :--- | :--- | :--- |
| **Model Type** | Mock Template | Neither classical CV nor learned deep learning. Ignored raster arrays completely. |
| **Checkpoints** | None | The path `"checkpoints/changeformer_v2_levir_bitemp_weights.pt"` in `agent.py` was fictitious; no change weights exist on disk. |
| **Input Support** | None | Headers were checked by `GeoTIFFValidator`, but raster arrays were never ingested or processed. |
| **Co-Registration** | None | Images were neither co-registered nor resized. The trace `"Co-Registration RMSE: 0.28 px"` was fabricated. |
| **Difference Mask** | None | No difference tensor, pixel mask, or thresholding was calculated. |
| **Statistics** | Hardcoded | Constants (`+2.85 km²`, `+32.4%`, `18 regions`) were emitted based purely on keyword matching. |
| **Descriptions** | Static Templates | Text was generated from hardcoded strings rather than image evidence. |
| **Nuisance vs Change** | Cannot distinguish | Because no pixels were examined, illumination, phenology, and registration errors could not be separated from real changes. |

---

## 3. Upgraded Architecture & Algorithms

### A. Raster Ingestion & Preprocessing
1. **Unpacking**: Ingests arrays, file paths, base64 URIs, and PIL images. Detects 10-band Sentinel-2 data via `extract_raster_bands()`.
2. **Spatial & CRS Validation**:
   - Rejects mismatched CRS projections (e.g. `EPSG:32643` vs `EPSG:32644`).
   - Checks aspect ratios; rejects gross mismatches (>40% variance).
   - If dimensions differ slightly, resamples T2 to match T1 using bilinear interpolation and explicitly records: *"Spatial Alignment: Resampled T2 to match T1. Note: Resampling is approximate; sub-pixel orthorectification was not applied."*
3. **Reference Scaling**:
   - Normalizes both images to `[0.0, 1.0]` using consistent reference scales (e.g. `/ 255.0` for uint8, `/ 10000.0` for Sentinel-2 surface reflectance) rather than independent percentile stretching to prevent background inversion.

### B. Pixel-Level Differencing & Adaptive Thresholding
- **Euclidean Spectral Distance**:
  $$\Delta(x, y) = \sqrt{\frac{1}{C}\sum_{c=1}^{C} \left(I_{T2}(x, y, c) - I_{T1}(x, y, c)\right)^2}$$
- **Adaptive Thresholding**:
  $$\tau = \min\left(0.70, \max\left(0.12, \mu_\Delta + 1.5 \cdot \sigma_\Delta\right)\right)$$
- **Candidate Change Mask**:
  $$M_{\text{raw}}(x, y) = \begin{cases} 1 & \text{if } \Delta(x, y) > \tau \\ 0 & \text{otherwise} \end{cases}$$

### C. Morphological Noise Cleanup & Region Extraction
- **Binary Opening**: $M_{\text{clean}} = M_{\text{raw}} \circ S_{3\times3}$ using `scipy.ndimage.binary_opening` to remove single-pixel noise.
- **Connected Component Labeling**: Groups contiguous changed pixels using `scipy.ndimage.label`.
- **Region Filtering**: Discards clusters with area $< 16$ pixels.
- **Bounding Boxes**: Computes $[x, y, \text{width}, \text{height}]$ in image pixel coordinates.

### D. Dual-State BigEarthNet Land-Cover Prior
When authentic 10-band Sentinel-2 data is provided for both dates:
1. `bigearthnet_service.predict()` runs independently on $T1$ and $T2$.
2. Probability deltas are computed across all 19 CORINE classes:
   $$\Delta P_c = P_{T2}(c) - P_{T1}(c)$$
3. Significant shifts ($|\Delta P_c| \ge 10\%$) are extracted and cited as supporting evidence (e.g. *"Continuous urban fabric (Increased +25.0%)"*).
4. If inputs are RGB-only, the system explicitly states that BigEarthNet 10-band classification is unavailable.

---

## 4. Query Routing & Task Classification

In [agent.py](file:///c:/Users/Admin/OneDrive/Desktop/satquery%20ai/backend/orchestrator/agent.py), temporal intent classification handles:
- **Change Analysis (`change-analysis`)**:
  - Queries: *"What changed?"*, *"Compare these images"*, *"What is different between past and present?"*
- **Change-Based VQA (`change-based-vqa`)**:
  - Queries: *"Has vegetation changed?"*, *"Has water increased or decreased?"*, *"Where has construction occurred?"*
- Both tasks select `bitemporal-diff-net` ([tools.py](file:///c:/Users/Admin/OneDrive/Desktop/satquery%20ai/backend/registry/tools.py)).

---

## 5. Test Suite Verification

### Dedicated Bi-Temporal Tests (`backend/tests/test_bitemporal_pipeline.py`)
```
Ran 12 tests in 0.043s
OK
```
1. `test_01_identical_images_yield_zero_change`: **Passed** (0 changed pixels, 0.0% coverage, 0 boxes).
2. `test_02_known_synthetic_difference_detected`: **Passed** (accurately localized 30x30 change patch).
3. `test_03_dimension_mismatch_resampled_with_warning`: **Passed** (honest resampling warning recorded).
4. `test_04_gross_aspect_ratio_mismatch_rejected`: **Passed** (`ValueError` raised).
5. `test_05_incompatible_crs_rejected`: **Passed** (`ValueError` raised).
6. `test_06_missing_georeferencing_uses_pixel_space`: **Passed** (pixel coordinates without guessed lat/lon).
7. `test_07_bigearthnet_dual_state_prior_when_10band_present`: **Passed** (evaluated 19-class shifts).
8. `test_08_rgb_inputs_avoid_fake_bigearthnet_classification`: **Passed** (BigEarthNet bypassed honestly).
9. `test_09_no_fabricated_confidence_scores`: **Passed** (`confidence = None`).
10. `test_10_temporal_query_routing`: **Passed** (routed to `change-analysis` and `change-based-vqa`).
11. `test_11_invalid_input_rejection_no_silent_fallback`: **Passed** (`ValueError` raised).
12. `test_12_agent_end_to_end_bitemporal_execution`: **Passed** (end-to-end pipeline execution).

### Complete Backend Test Suite
```
Ran 65 tests in 19.063s
OK
```
All 41 existing BigEarthNet tests, 12 Florence-2 tests, and 12 bi-temporal tests pass with zero errors and zero regressions.

---

## 6. Known Limitations & Scientific Disclaimers

1. **Candidate Change vs. Confirmed Semantic Change**:
   - Pixel-level differences reflect radiance/reflectance shifts. Without dense semantic segmentation or ground truth, simple differencing cannot definitively prove anthropogenic construction versus seasonal phenology or soil moisture changes.
2. **Co-Registration Sensitivity**:
   - The engine assumes the two rasters are roughly co-registered. Uncorrected spatial misalignment or parallax can produce edge artifacts. Resampling is approximate and does not replace sub-pixel orthorectification.
3. **Illumination & Atmospheric Shifts**:
   - While joint reference normalization mitigates linear radiometric differences, severe cloud cover, haze, or shadow variations will register as candidate change regions.

---

## 7. Future Roadmap for Learned Change Detection

If a genuine remote-sensing change-detection model (e.g. ChangeFormer-V2 or BIT trained on LEVIR-CD or OSCD) is integrated in the future:
1. Verify model weights on CPU/GPU with deterministic input tensors.
2. Ensure input dimensions match the model's receptive field (e.g. $256\times256$ or $512\times512$).
3. Compare learned change predictions with the classical baseline and BigEarthNet dual-state prior.
