# Step 12: Real Geospatial + Evidence Layer

## Overview

Step 12 upgrades SatQuery AI from raw pixel-coordinate evidence to genuine geospatially grounded evidence whenever an uploaded raster contains valid georeferencing metadata (CRS and affine transform).

### Core Scientific Principles
- **Zero Coordinate Fabrication**: SatQuery AI never invents or guesses coordinates, CRS, bounding boxes, or areas. Hardcoded defaults (such as `EPSG:32643` or Mumbai bounding extents) have been eliminated.
- **Explicit Geolocation Status**: Rasters without georeferencing metadata explicitly report `geospatial_status = "unavailable"`, accompanied by clear disclaimers:
  `"Geospatial coordinates unavailable because the source raster does not contain sufficient georeferencing metadata."`
- **Preservation of Model Predictions**: Specialist neural outputs (Florence-2, BigEarthNet, Bi-Temporal differencing, Optical+SAR fusion) remain unchanged, but spatial artifacts are augmented with WGS84 geographic bounding boxes and genuine ground area estimations when supported.

---

## 1. Geospatial Core Engine (`backend/geospatial/georeference.py`)

The reusable geospatial module `GeoreferenceEngine` encapsulates coordinate transformations and spatial math:

### Supported Metadata Extraction
- **CRS**: Projected (e.g. UTM, Web Mercator) or Geographic (EPSG:4326).
- **Affine Transform**: GDAL 6-tuple `[c, a, b, f, d, e]` mapping `(col, row)` to `(x_proj, y_proj)`:
  $$\begin{aligned}
  X_{\text{proj}} &= c + \text{col} \cdot a + \text{row} \cdot b \\
  Y_{\text{proj}} &= f + \text{col} \cdot d + \text{row} \cdot e
  \end{aligned}$$
- **Ground Sampling Distance (GSD)**: Calculated from transform pixel scale:
  $$\text{GSD} = \frac{|a| + |e|}{2}$$
- **Geographic Bounds (WGS84)**: Projected corners transformed to $(\text{latitude}, \text{longitude})$ via `pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)`.

---

## 2. Real Ground Area Calculation

Area is **only** calculated when valid georeferencing exists. Never outputs fabricated $\text{km}^2$.

### Projected CRS (e.g. UTM Zones)
Ground area is computed directly from pixel dimensions:
$$\text{Area}_{\text{m}^2} = N_{\text{pixels}} \times |a \cdot e - b \cdot d|$$
$$\text{Area}_{\text{km}^2} = \frac{\text{Area}_{\text{m}^2}}{1{,}000{,}000}$$

### Geographic CRS (e.g. EPSG:4326 Degrees)
Degrees are never treated as meters. At reference latitude $\phi_0$:
$$\begin{aligned}
m_{\text{lat}} &\approx 111{,}132.92 - 559.82 \cos(2\phi_0) + 1.175 \cos(4\phi_0) \\
m_{\text{lon}} &\approx 111{,}412.84 \cos(\phi_0) - 93.5 \cos(3\phi_0) \\
\text{Pixel Area}_{\text{m}^2} &= (|a| \cdot m_{\text{lon}}) \times (|e| \cdot m_{\text{lat}})
\end{aligned}$$

---

## 3. Spatial & Multimodal Alignment Validation

### Bi-Temporal Alignment
Distinguishes three genuine alignment states:
1. **`"geospatially aligned"`**: Both T1 and T2 possess matching CRS and co-registered georeferenced spatial footprints.
2. **`"pixel-aligned"`**: Dimensions match, but georeferencing is absent (unprojected local coordinates).
3. **`"alignment unavailable"`**: Dimensions differ and georeferencing is absent; approximate resampling applied.
- CRS mismatches (e.g. `EPSG:32643` vs `EPSG:32644`) are detected and rejected.

### Optical + SAR Cross-Sensor Validation
- Verifies projection consistency between Optical and SAR rasters.
- Records any spatial resampling in the execution trace and limitations.
- Preserves native co-registration when dimensions and projections match without claiming unmeasured sub-pixel accuracy.

---

## 4. Evidence Provenance & Hierarchy

`EvidenceCombiner` maintains strict evidence separation:
- **DIRECT**: Actual model inferences, raster-derived spectral indices, detected pixel bounding boxes, and genuine geospatial coordinates/transforms.
- **SUPPORTING**: Secondary model confirmations (e.g., BigEarthNet Corine Land Cover priors supporting change detection or grounding).
- **LIMITATIONS**: Missing CRS, unprojected local coordinates, approximate resampling, and sensor-specific caveats.

---

## 5. Cursor / Feature Query API

Exposes backend utility for coordinate querying:
`POST /api/geospatial/cursor-feature`

**Request**:
```json
{
  "x": 120.0,
  "y": 85.0,
  "metadata": {
    "crs": "EPSG:32643",
    "geotransform": [281000.0, 10.0, 0.0, 2102000.0, 0.0, -10.0]
  }
}
```

**Response (Georeferenced)**:
```json
{
  "pixel": {"x": 120.0, "y": 85.0},
  "coordinates": {
    "latitude": 19.001245,
    "longitude": 72.923412
  },
  "feature": "Candidate Change Cluster 1",
  "evidence_source": "Spatial cluster of 45 changed pixels",
  "geospatial_status": "available"
}
```

**Response (Unprojected)**:
```json
{
  "pixel": {"x": 120.0, "y": 85.0},
  "coordinates": null,
  "feature": null,
  "evidence_source": "Pixel Grid",
  "geospatial_status": "unavailable",
  "message": "Geolocation unavailable for this raster"
}
```
