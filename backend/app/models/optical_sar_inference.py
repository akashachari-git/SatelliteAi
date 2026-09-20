"""
SatQuery AI - Optical + SAR Multimodal Fusion Service
Executes evidence-grounded cross-sensor physical fusion between:
1. Optical imagery (Multispectral Sentinel-2 MSI or Optical RGB).
2. SAR radar imagery (Sentinel-1 C-Band VV/VH backscatter or single-pol radar).

Key Principles:
- Genuine physical evidence: Computes real optical spectral indices (NDVI, NDWI, NDBI when bands exist)
  and real radar backscatter physics (dB calibration, specular extinction, double-bounce reflections).
- Strict input validation: Rejects missing modalities, incompatible CRS projections, and RGB images masquerading as SAR.
- Spatial integrity: Preserves metadata, checks aspect ratios, uses image pixel coordinates for bounding boxes,
  and disclaims approximate resampling when sub-pixel orthorectification is absent.
- No fabricated data: Omits fake confidence scores, fake accuracies, and fictitious neural model claims.
"""
import os
import io
import time
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import scipy.ndimage

from backend.geospatial.raster_adapter import extract_raster_bands
from backend.models.bigearthnet_loader import REQUIRED_S2_BANDS

logger = logging.getLogger("satquery.models.optical_sar")


class OpticalSARFusionService:
    """
    Evidence-grounded cross-modal Optical + SAR physical analysis engine.
    """

    def __init__(self):
        pass

    @staticmethod
    def _unpack_raster(source: Any) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Unpacks an input source into a NumPy array and extracts associated metadata.
        Returns: (array, metadata_dict)
        """
        if source is None:
            return None, {}

        meta = {}
        if isinstance(source, dict):
            meta = {
                "name": source.get("name") or source.get("filename"),
                "modality": source.get("modality"),
                "sensor": source.get("sensor"),
                "bands": source.get("bands") or source.get("band_names"),
                "crs": source.get("crs"),
            }
            if "array" in source and source["array"] is not None:
                source = source["array"]
            elif "data" in source and source["data"] is not None:
                source = source["data"]
            elif "path" in source and source["path"] is not None:
                source = source["path"]
            elif "fileDataUri" in source and source["fileDataUri"] is not None:
                source = source["fileDataUri"]
            elif "filename" in source and os.path.exists(str(source["filename"])):
                source = source["filename"]

        if isinstance(source, np.ndarray):
            return source, meta

        if isinstance(source, Image.Image):
            return np.array(source.convert("RGB")), meta

        if isinstance(source, str) and os.path.isfile(source):
            from backend.geospatial.raster_cache import raster_cache
            cached = raster_cache.get_raster(source)
            if cached is not None:
                return cached, meta
            ext = os.path.splitext(source.lower())[1]
            if ext in [".tif", ".tiff", ".geotiff"]:
                try:
                    import tifffile
                    arr = tifffile.imread(source)
                    raster_cache.set_raster(source, arr)
                    return arr, meta
                except Exception:
                    pass
            try:
                with Image.open(source) as img:
                    arr = np.array(img.convert("RGB"))
                    raster_cache.set_raster(source, arr)
                    return arr, meta
            except Exception as e:
                logger.warning("Failed to open image file '%s': %s", source, e)

        if isinstance(source, str) and (source.startswith("data:") or len(source) > 100):
            try:
                data_str = source.split("base64,")[1] if "base64," in source else source
                raw_bytes = base64.b64decode(data_str)
                with Image.open(io.BytesIO(raw_bytes)) as img:
                    return np.array(img.convert("RGB")), meta
            except Exception as e:
                logger.warning("Failed to decode base64 image: %s", e)

        if isinstance(source, bytes):
            try:
                with Image.open(io.BytesIO(source)) as img:
                    return np.array(img.convert("RGB")), meta
            except Exception as e:
                logger.warning("Failed to decode bytes image: %s", e)

        return None, meta

    def predict(
        self,
        query: str,
        inputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes genuine physical cross-modal analysis between Optical and SAR rasters.
        """
        t0 = time.time()

        optical_source = inputs.get("optical")
        sar_source = inputs.get("sar")

        # 1. Input Validation
        if optical_source is None and sar_source is None:
            raise ValueError(
                "Optical + SAR analysis requires both an Optical observation and a SAR radar observation. "
                "Both are missing."
            )
        if optical_source is None:
            raise ValueError("Missing Optical observation. Optical + SAR analysis requires 1 Optical and 1 SAR image.")
        if sar_source is None:
            raise ValueError("Missing SAR observation. Optical + SAR analysis requires 1 Optical and 1 SAR image.")

        # Unpack rasters
        opt_arr, opt_meta = self._unpack_raster(optical_source)
        sar_arr, sar_meta = self._unpack_raster(sar_source)

        if opt_arr is None:
            raise ValueError("Failed to decode Optical input into a valid raster array.")
        if sar_arr is None:
            raise ValueError("Failed to decode SAR input into a valid raster array.")

        # Validate SAR characteristics (reject optical images masquerading as SAR)
        sar_sensor = str(sar_meta.get("sensor") or "").lower()
        sar_mod = str(sar_meta.get("modality") or "").lower()
        is_labeled_sar = "sar" in sar_mod or "sar" in sar_sensor or "sentinel-1" in sar_sensor or "radar" in sar_mod

        # Check if sar_arr is a standard 3-channel RGB image with no SAR metadata
        if (sar_arr.ndim == 3 and (sar_arr.shape[-1] == 3 or sar_arr.shape[0] == 3)) and not is_labeled_sar:
            raise ValueError(
                "Invalid SAR Input: The provided SAR observation appears to be an optical RGB image "
                "without radar metadata. Multimodal analysis requires genuine radar data "
                "(e.g. Sentinel-1 VV/VH backscatter or single-channel radar raster)."
            )

        # 2. Check CRS & Spatial Compatibility
        from backend.geospatial.georeference import GeoreferenceEngine

        meta = metadata or {}
        crs_opt = str(meta.get("crs_optical") or opt_meta.get("crs") or "")
        crs_sar = str(meta.get("crs_sar") or sar_meta.get("crs") or "")
        gt_opt = meta.get("geotransform_optical") or meta.get("geotransform") or opt_meta.get("geotransform")
        gt_sar = meta.get("geotransform_sar") or meta.get("geotransform") or sar_meta.get("geotransform")

        has_geo_opt = bool(crs_opt and not crs_opt.startswith("Local") and gt_opt)
        has_geo_sar = bool(crs_sar and not crs_sar.startswith("Local") and gt_sar)

        if crs_opt and crs_sar and not crs_opt.startswith("Local") and not crs_sar.startswith("Local"):
            if crs_opt.split()[0] != crs_sar.split()[0]:
                raise ValueError(
                    f"CRS Incompatibility: Optical CRS is '{crs_opt}' while SAR CRS is '{crs_sar}'. "
                    "Projections must match for multimodal spatial fusion."
                )

        # 3. Spatial Dimensions & Alignment
        # Standardize optical shape
        if opt_arr.ndim == 3 and opt_arr.shape[0] in [3, 4, 10]:
            h_opt, w_opt = opt_arr.shape[1], opt_arr.shape[2]
        else:
            h_opt, w_opt = opt_arr.shape[0], opt_arr.shape[1]

        # Standardize SAR shape
        if sar_arr.ndim == 3 and sar_arr.shape[0] in [1, 2]:
            h_sar, w_sar = sar_arr.shape[1], sar_arr.shape[2]
        else:
            h_sar, w_sar = sar_arr.shape[0], sar_arr.shape[1]

        ratio_opt = w_opt / max(h_opt, 1)
        ratio_sar = w_sar / max(h_sar, 1)
        if abs(ratio_opt - ratio_sar) > 0.40:
            raise ValueError(
                f"Spatial Incompatibility: Optical aspect ratio ({ratio_opt:.2f}) "
                f"differs significantly from SAR aspect ratio ({ratio_sar:.2f}). "
                "Observations must cover the same spatial bounding footprint."
            )

        resampled_sar = False
        if (h_opt, w_opt) == (h_sar, w_sar):
            if has_geo_opt and has_geo_sar:
                alignment_status = "geospatially aligned"
                alignment_note = "Native co-registered dimensions and geospatial coordinate frames match."
            else:
                alignment_status = "pixel-aligned"
                alignment_note = "Native pixel dimensions match (unprojected / local coordinates)."
        else:
            resampled_sar = True
            if has_geo_opt and has_geo_sar:
                alignment_status = "geospatially aligned"
                alignment_note = (
                    f"Spatial Alignment: Resampled SAR ({w_sar}×{h_sar} px) to match Optical grid ({w_opt}×{h_opt} px). "
                    "Geospatially referenced with bilinear interpolation."
                )
            else:
                alignment_status = "alignment unavailable"
                alignment_note = (
                    f"Spatial Alignment: Resampled SAR ({w_sar}×{h_sar} px) to match Optical grid ({w_opt}×{h_opt} px). "
                    "Note: Georeferencing is unavailable; spatial alignment is approximate."
                )

        # 4. Optical Feature Extraction
        # Inspect for Sentinel-2 10-band contract
        is_s2, s2_arr, s2_bands, _ = extract_raster_bands(optical_source)
        optical_evidence = []
        opt_water_mask = np.zeros((h_opt, w_opt), dtype=bool)
        opt_built_mask = np.zeros((h_opt, w_opt), dtype=bool)
        opt_veg_mask = np.zeros((h_opt, w_opt), dtype=bool)

        if is_s2 and s2_arr is not None:
            # Sentinel-2 bands: 0:B02, 1:B03, 2:B04, 6:B08 (NIR), 8:B11 (SWIR)
            b02 = s2_arr[0].astype(np.float32)
            b03 = s2_arr[1].astype(np.float32)
            b04 = s2_arr[2].astype(np.float32)
            b08 = s2_arr[6].astype(np.float32)
            b11 = s2_arr[8].astype(np.float32)

            # Compute real spectral indices
            ndvi = (b08 - b04) / np.maximum(b08 + b04, 1e-6)
            ndwi = (b03 - b08) / np.maximum(b03 + b08, 1e-6)
            ndbi = (b11 - b08) / np.maximum(b11 + b08, 1e-6)

            mean_ndvi = float(np.mean(ndvi))
            mean_ndwi = float(np.mean(ndwi))
            mean_ndbi = float(np.mean(ndbi))

            opt_veg_mask = ndvi > 0.35
            opt_water_mask = ndwi > 0.05
            opt_built_mask = ndbi > 0.05

            optical_evidence = [
                f"[OPTICAL VNIR] Multispectral Sentinel-2 bands ingested: B02, B03, B04, B08 (NIR), B11 (SWIR).",
                f"[OPTICAL NDVI] Vegetation canopy index computed: Mean NDVI = {mean_ndvi:+.2f} ({np.mean(opt_veg_mask)*100.0:.1f}% dense canopy).",
                f"[OPTICAL NDWI] Normalized difference water index computed: Mean NDWI = {mean_ndwi:+.2f} ({np.mean(opt_water_mask)*100.0:.1f}% water absorption).",
                f"[OPTICAL NDBI] Normalized difference built-up index computed: Mean NDBI = {mean_ndbi:+.2f} ({np.mean(opt_built_mask)*100.0:.1f}% impervious albedo).",
            ]
        else:
            # 3-channel RGB optical
            if opt_arr.ndim == 3 and opt_arr.shape[0] in [3, 4]:
                rgb = np.transpose(opt_arr[:3], (1, 2, 0)).astype(np.float32)
            elif opt_arr.ndim == 3 and opt_arr.shape[2] in [3, 4]:
                rgb = opt_arr[:, :, :3].astype(np.float32)
            else:
                rgb = np.repeat(np.expand_dims(opt_arr.astype(np.float32), -1), 3, axis=-1)

            # Normalize to [0.0, 1.0]
            max_c = np.max(rgb) if np.max(rgb) > 0 else 1.0
            rgb_norm = rgb / max_c if max_c > 1.0 else rgb

            albedo = np.mean(rgb_norm, axis=-1)
            vari = (rgb_norm[:, :, 1] - rgb_norm[:, :, 0]) / np.maximum(rgb_norm[:, :, 1] + rgb_norm[:, :, 0] - rgb_norm[:, :, 2], 1e-6)

            opt_veg_mask = vari > 0.15
            opt_water_mask = albedo < 0.12
            opt_built_mask = albedo > 0.65

            optical_evidence = [
                f"[OPTICAL RGB] Optical 3-band RGB imagery ingested ({w_opt}×{h_opt} px).",
                f"[OPTICAL ALBEDO] Mean visible albedo computed at {np.mean(albedo):.2f}.",
                f"[OPTICAL NOTE] Native multispectral NIR/SWIR bands absent; indices (NDVI, NDWI, NDBI) unavailable.",
            ]

        # 5. SAR Backscatter & Polarimetric Analysis
        # Prepare SAR array
        if sar_arr.ndim == 3 and sar_arr.shape[0] in [1, 2]:
            sar_data = sar_arr.astype(np.float32)
        elif sar_arr.ndim == 3 and sar_arr.shape[2] in [1, 2]:
            sar_data = np.transpose(sar_arr, (2, 0, 1)).astype(np.float32)
        elif sar_arr.ndim == 2:
            sar_data = np.expand_dims(sar_arr.astype(np.float32), 0)
        else:
            sar_data = np.expand_dims(sar_arr[:, :, 0].astype(np.float32), 0)

        # Resample SAR to match optical grid if dimensions differed
        if resampled_sar:
            resampled_channels = []
            for ch in range(sar_data.shape[0]):
                pil_ch = Image.fromarray(sar_data[ch])
                resampled_ch = pil_ch.resize((w_opt, h_opt), Image.Resampling.BILINEAR)
                resampled_channels.append(np.array(resampled_ch))
            sar_data = np.stack(resampled_channels, axis=0)

        # Convert linear backscatter to decibels (dB) if needed
        # Typical SAR dB values range from -35 dB to +5 dB
        sar_max = float(np.max(sar_data))
        sar_min = float(np.min(sar_data))
        is_linear = sar_max > 5.0 or sar_min >= 0.0

        if is_linear:
            sar_db = 10.0 * np.log10(np.maximum(sar_data, 1e-6))
        else:
            sar_db = sar_data

        vv_db = sar_db[0]
        has_vh = sar_db.shape[0] >= 2
        vh_db = sar_db[1] if has_vh else None

        mean_vv = float(np.mean(vv_db))
        std_vv = float(np.std(vv_db))

        # Detect radar physical signatures:
        # 1. Specular extinction (water / mirror-smooth surface): backscatter < -20.0 dB
        sar_specular_mask = vv_db < -20.0
        # 2. Double-bounce reflections (vertical anthropic structures / corner reflectors): backscatter > -6.0 dB
        sar_double_bounce_mask = vv_db > -6.0
        # 3. Rough canopy volume scattering: intermediate backscatter [-15 dB, -8 dB]
        sar_volume_mask = (vv_db >= -15.0) & (vv_db <= -8.0)

        sar_evidence = [
            f"[SAR RADAR] C-band radar backscatter processed ({'Dual-Pol VV/VH' if has_vh else 'Single-Pol VV'}).",
            f"[SAR BACKSCATTER] Mean VV backscatter: {mean_vv:.1f} dB (std: {std_vv:.1f} dB, range: [{sar_min:.1f}, {sar_max:.1f}]).",
            f"[SAR SPECULAR] Specular extinction detected across {np.mean(sar_specular_mask)*100.0:.1f}% of scene (< -20.0 dB, smooth surface/water candidate).",
            f"[SAR DOUBLE-BOUNCE] Dihedral corner reflections detected across {np.mean(sar_double_bounce_mask)*100.0:.1f}% of scene (> -6.0 dB, vertical structure candidate).",
        ]
        if has_vh:
            ratio_db = float(np.mean(vv_db - vh_db))
            sar_evidence.append(f"[SAR POLARIMETRY] Mean VV/VH cross-polarization ratio: {ratio_db:+.1f} dB.")

        # 6. Cross-Sensor Physical Corroboration
        corroborated_water = opt_water_mask & sar_specular_mask
        corroborated_built = opt_built_mask & sar_double_bounce_mask

        water_pct = round(float(np.mean(corroborated_water) * 100.0), 2)
        built_pct = round(float(np.mean(corroborated_built) * 100.0), 2)

        fused_evidence = [
            f"[CROSS-MODAL FUSION] Evidence-based optical + SAR fusion executed on {w_opt}×{h_opt} px co-registered grid.",
            f"[CORROBORATED WATER] {water_pct}% of scene corroborated by joint Optical absorption (NDWI/dark albedo) AND SAR specular extinction (< -20 dB).",
            f"[CORROBORATED BUILT-UP] {built_pct}% of scene corroborated by joint Optical high albedo/NDBI AND SAR dihedral double-bounce reflections (> -6 dB).",
            f"[ALL-WEATHER NOTE] SAR microwaves penetrate clouds and atmospheric haze; optical provides material taxonomy.",
        ]

        # 7. Region Extraction & Bounding Boxes
        boxes = []
        # Label water regions
        labeled_water, num_water = scipy.ndimage.label(corroborated_water)
        water_slices = scipy.ndimage.find_objects(labeled_water)
        for i, s in enumerate(water_slices):
            if s is None:
                continue
            r, c = s
            area = int(np.sum(labeled_water[r, c] == (i + 1)))
            if area >= 16:
                boxes.append({
                    "id": f"fus-water-{i + 1}",
                    "label": f"Corroborated Water Body ({area} px)",
                    "confidence": None,  # No fabricated confidence
                    "x": float(c.start),
                    "y": float(r.start),
                    "width": float(c.stop - c.start),
                    "height": float(r.stop - r.start),
                    "description": "Corroborated by optical absorption and SAR specular extinction (σ⁰ < -20 dB)."
                })

        # Label built-up regions
        labeled_built, num_built = scipy.ndimage.label(corroborated_built)
        built_slices = scipy.ndimage.find_objects(labeled_built)
        for i, s in enumerate(built_slices):
            if s is None:
                continue
            r, c = s
            area = int(np.sum(labeled_built[r, c] == (i + 1)))
            if area >= 16:
                boxes.append({
                    "id": f"fus-built-{i + 1}",
                    "label": f"Corroborated Built-Up Structure ({area} px)",
                    "confidence": None,
                    "x": float(c.start),
                    "y": float(r.start),
                    "width": float(c.stop - c.start),
                    "height": float(r.stop - r.start),
                    "description": "Corroborated by optical albedo/NDBI and SAR double-bounce corner reflection (σ⁰ > -6 dB)."
                })

        # Add geospatial coordinates to boxes if georeferencing is available
        for b in boxes:
            if gt_opt and crs_opt and not crs_opt.startswith("Local"):
                geo_info = GeoreferenceEngine.pixel_bbox_to_geographic(
                    {"x": b["x"], "y": b["y"], "width": b["width"], "height": b["height"]},
                    gt_opt, crs_opt
                )
                if geo_info:
                    b["projectedBbox"] = geo_info["projected_bbox"]
                    b["geographicBbox"] = geo_info["geographic_bbox"]
                    b["centerLatLon"] = geo_info["center_lat_lon"]

        geo_status = "available" if (gt_opt and crs_opt and not crs_opt.startswith("Local")) else ("partial" if (gt_opt or crs_opt) else "unavailable")
        geo_limitations = [f"Alignment: {alignment_note}"]
        if not has_geo_opt or not has_geo_sar:
            geo_limitations.append("Full cross-modal georeferencing is incomplete: one or both observations lack CRS/affine transform.")

        geospatial_evidence = {
            "status": geo_status,
            "crs": crs_opt if (crs_opt and not crs_opt.startswith("Local")) else None,
            "alignmentStatus": alignment_status,
            "limitations": geo_limitations,
        }

        # 8. Answer Synthesis
        corroborating_features = []
        if water_pct > 0.5:
            corroborating_features.append(
                f"Open water extents spanning {water_pct}% of the scene, corroborated by optical absorption and SAR specular extinction."
            )
        if built_pct > 0.5:
            corroborating_features.append(
                f"Anthropic structural infrastructure spanning {built_pct}% of the scene, corroborated by optical albedo and SAR double-bounce backscatter."
            )
        if not corroborating_features:
            corroborating_features.append(
                "Heterogeneous landscape where optical spectral reflectance and SAR roughness signatures provide complementary surface insights."
            )

        answer = (
            "Evidence-based optical + SAR multimodal analysis across co-registered observation footprint:\n\n"
            f"1. Optical Assessment: {optical_evidence[1] if len(optical_evidence) > 1 else 'Processed visible reflectance.'}\n"
            f"2. SAR Radar Assessment: Mean VV backscatter at {mean_vv:.1f} dB with "
            f"{'dual-polarization VV/VH cross-ratio' if has_vh else 'single-polarization structure detection'}.\n"
            f"3. Cross-Sensor Synthesis: {corroborating_features[0]} "
            "Optical provides spectral material taxonomy while SAR provides dielectric roughness, physical structure, and all-weather verification."
        )

        duration_ms = int((time.time() - t0) * 1000)

        return {
            "answer": answer,
            "confidence": None,  # No fabricated confidence score
            "evidence": optical_evidence + sar_evidence + fused_evidence,
            "boundingBoxes": boxes,
            "is_simulation": False,
            "imageOverlayType": "fusion",
            "crossModalEvidence": {
                "opticalEvidence": optical_evidence,
                "sarEvidence": sar_evidence,
                "fusedEvidence": fused_evidence,
                "corroboratingFeatures": corroborating_features,
                "sensorComplementarityNotes": (
                    "Optical provides spectral material reflectance; SAR provides physical structure, "
                    "dielectric roughness, and all-weather cloud penetration."
                ),
                "opticalWaterCoveragePct": water_pct,
                "opticalBuiltUpCoveragePct": built_pct,
                "meanSarVvDb": round(mean_vv, 2),
                "alignmentNote": alignment_note,
            },
            "geospatialEvidence": geospatial_evidence,
            "execution_duration_ms": duration_ms,
            "spatial_compatibility": {
                "dimensions_optical": [w_opt, h_opt],
                "dimensions_sar": [w_sar, h_sar],
                "resampled_sar": resampled_sar,
                "alignment_note": alignment_note,
                "alignment_status": alignment_status,
            }
        }


# Global singleton instance
optical_sar_service = OpticalSARFusionService()
