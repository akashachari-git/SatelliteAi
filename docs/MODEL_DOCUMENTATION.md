# SatQuery AI — Specialist Models & Remote Sensing Adaptation

## 1. Remote Sensing Adaptation Mandate
General-purpose Vision-Language Models (e.g. GPT-4V, standard CLIP) suffer from documented failure modes when evaluating satellite imagery:
1. **Lack of radiometric calibration**: Confusion between surface reflectance, cloud shadows, and seasonal phenology.
2. **Failure on microwave physics**: Treating SAR backscatter as optical grayscale, failing to recognize dihedral double-bounce reflections or specular water absorption.
3. **Misalignment with standardized GIS taxonomies**: Conflating complex cultivation patterns with natural grasslands.

SatQuery AI explicitly adapts remote-sensing representations based on the **BigEarthNet-S2 (19-Class CORINE)** benchmark.

---

## 2. BigEarthNet-19 Nomenclature & Spectral Signatures
The adaptation layer models land cover through the standardized 19-class aggregated CORINE Land Cover (CLC) schema:

1. **Urban fabric** (NDBI: +0.42, Albedo: 0.48)
2. **Industrial or commercial units** (NDBI: +0.65, Albedo: 0.62)
3. **Arable land** (NDVI: +0.35, Albedo: 0.38)
4. **Permanent crops** (NDVI: +0.52, NDBI: -0.18)
5. **Pastures** (NDVI: +0.60, Albedo: 0.30)
6. **Complex cultivation patterns** (NDVI: +0.48, Albedo: 0.34)
7. **Land principally occupied by agriculture with natural vegetation** (NDVI: +0.55)
8. **Agro-forestry areas** (NDVI: +0.68)
9. **Broad-leaved forest** (NDVI: +0.78, NDWI: 0.02)
10. **Coniferous forest** (NDVI: +0.72, Albedo: 0.19)
11. **Mixed forest** (NDVI: +0.75, Albedo: 0.20)
12. **Natural grassland and sparsely vegetated areas** (NDVI: +0.40)
13. **Moors, heathland and sclerophyllous vegetation** (NDVI: +0.45)
14. **Transitional woodland, shrub** (NDVI: +0.58)
15. **Beaches, dunes, sands** (Albedo: 0.78, NDVI: -0.15)
16. **Bare rock and sparsely vegetated areas** (NDBI: 0.30, Albedo: 0.50)
17. **Inland wetlands** (NDWI: +0.45, NDVI: 0.30)
18. **Coastal wetlands** (NDWI: +0.55, Albedo: 0.15)
19. **Inland / Marine waters** (NDWI: +0.82, NDVI: -0.45, Albedo: 0.10)

---

## 3. Specialist Model Registry Details

### A. RemoteSensingVQA
- **Task**: `SINGLE_VQA`
- **Identifier**: `satquery-vqa-v1-bigearthnet`
- **Description**: Evaluates multi-spectral signatures (NDVI, NDWI, NDBI) and derives land-cover class distribution.

### B. RemoteSensingCaptioner
- **Task**: `CAPTIONING`
- **Identifier**: `satquery-caption-vrsbench`
- **Description**: Synthesizes structured descriptive paragraphs covering dominant terrain, spatial texture, and infrastructure.

### C. GroundingModel
- **Task**: `GROUNDING`
- **Identifier**: `satquery-grounding-v1`
- **Description**: Detects entity referred to in query (water, built-up, vegetation), calculates bounding boxes, and generates semi-transparent segmentation overlays.

### D. ChangeDetectionModel & ChangeVQAModel
- **Task**: `CHANGE_ANALYSIS` & `CHANGE_VQA`
- **Identifier**: `satquery-cd-logratio-v1` & `satquery-cdvqa-v1`
- **Description**: Bi-temporal difference modeling (log-ratio for SAR, Euclidean distance for optical), Otsu thresholding, quadrant flux aggregation.

### E. OpticalSARFusion
- **Task**: `OPTICAL_SAR_ANALYSIS`
- **Identifier**: `satquery-opt-sar-fusion-v1`
- **Description**: Dual-stream cross-modal synthesis combining Sentinel-2 optical reflectance with Sentinel-1 C-band SAR backscatter.
