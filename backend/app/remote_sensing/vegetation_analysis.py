import numpy as np
from typing import Dict, Any, List, Optional
from scipy.ndimage import label, find_objects, binary_opening, binary_closing
from .image_processor import ImageProcessor
from .raster_service import RasterMetadataService
from .spectral_indices import SpectralIndicesCalculator

class VegetationAnalysisEngine:
    """
    Dedicated Remote-Sensing Vegetation & Canopy Vitality Engine.
    - Multispectral NDVI = (NIR - Red) / (NIR + Red) when NIR is available.
    - Visible Green Canopy Excess & Colorimetry for standard optical RGB imagery.
    - Categorizes canopy and determines spatial distribution across natural quadrants:
      'upper-left', 'upper-right', 'center', 'lower-left', 'lower-right'.
    - Directly answers questions like 'Is there vegetation?' and 'Which area has the most vegetation?'.
    """

    @classmethod
    def analyze_vegetation(cls, image_path: str, user_query: str = "") -> Dict[str, Any]:
        arr = ImageProcessor.load_as_array(image_path)
        meta = RasterMetadataService.inspect_raster(image_path)
        h, w = arr.shape[:2]
        total_pixels = h * w
        pixel_area_km2 = meta.get("pixel_area_km2", 0.0001)

        limitations = []
        is_multispectral = meta.get("is_geotiff", False) and (arr.ndim == 3 and arr.shape[2] >= 4 and meta.get("number_of_bands", 1) >= 4)

        if is_multispectral:
            ndvi = SpectralIndicesCalculator.calculate_ndvi(arr, red_band=2, nir_band=3)
            if ndvi is not None:
                ndvi_clipped = np.clip(ndvi, -1.0, 1.0)
                veg_mask = ndvi_clipped > 0.20
                dense_mask = ndvi_clipped > 0.40
                mod_mask = (ndvi_clipped >= 0.20) & (ndvi_clipped <= 0.40)
                sparse_mask = (ndvi_clipped > 0.05) & (ndvi_clipped < 0.20)

                method = "NDVI (Normalized Difference Vegetation Index: NIR - Red / NIR + Red)"
                confidence = 0.95
                spectral_bands = ["Red (665nm)", "NIR (842nm)"]
                mean_index_val = float(np.mean(ndvi_clipped[veg_mask])) if np.any(veg_mask) else 0.0
            else:
                is_multispectral = False

        if not is_multispectral:
            limitations.append(
                "Near-Infrared (NIR) 842nm band unavailable. Calculated visible optical green canopy contrast; "
                "cellular chlorophyll NIR scattering was not measured directly."
            )
            r = arr[:, :, 0].astype(np.float32)
            g = arr[:, :, 1].astype(np.float32)
            b = arr[:, :, 2].astype(np.float32)
            brightness = (r + g + b) / 3.0

            # Exclude obvious water
            is_water = ((b > r * 1.15) & (brightness < 120)) | ((brightness < 35.0) & (r < 28.0))

            # Optical green vegetation: green peak over red and blue
            veg_mask = (g > r * 1.10) & (g > b * 1.05) & (g > 30) & (~is_water)
            dense_mask = (g > r * 1.25) & (g > b * 1.18) & (brightness > 35) & (~is_water)
            mod_mask = veg_mask & (~dense_mask)
            sparse_mask = (g > r * 1.04) & (~veg_mask) & (~is_water)

            method = "Visible Optical Green Canopy Reflectance (Colorimetric Green Excess)"
            confidence = 0.86
            spectral_bands = ["Red (Visible)", "Green (Visible)", "Blue (Visible)"]
            mean_index_val = 0.0

        # Morphological clean up
        opened = binary_opening(veg_mask, structure=np.ones((3, 3)))
        clean_veg_mask = binary_closing(opened, structure=np.ones((4, 4)))

        veg_pixels = int(np.sum(clean_veg_mask))
        veg_pct = round((veg_pixels / total_pixels) * 100, 2)
        veg_area_km2 = round(veg_pixels * pixel_area_km2, 4)
        if veg_area_km2 == 0.0 and veg_pixels > 0:
            veg_area_km2 = round((veg_pixels * 1.0) / 1e6, 4)

        dense_pct = round((int(np.sum(dense_mask & clean_veg_mask)) / total_pixels) * 100, 2)
        mod_pct = round((int(np.sum(mod_mask & clean_veg_mask)) / total_pixels) * 100, 2)
        sparse_pct = round((int(np.sum(sparse_mask)) / total_pixels) * 100, 2)

        # -------------------------------------------------------------
        # Quadrant Distribution Analysis
        # -------------------------------------------------------------
        mid_y = h // 2
        mid_x = w // 2

        # 5 natural zones: upper-left, upper-right, center, lower-left, lower-right
        center_y_start, center_y_end = int(h * 0.25), int(h * 0.75)
        center_x_start, center_x_end = int(w * 0.25), int(w * 0.75)

        q_ul = clean_veg_mask[:mid_y, :mid_x]
        q_ur = clean_veg_mask[:mid_y, mid_x:]
        q_ll = clean_veg_mask[mid_y:, :mid_x]
        q_lr = clean_veg_mask[mid_y:, mid_x:]
        center_box = clean_veg_mask[center_y_start:center_y_end, center_x_start:center_x_end]

        quad_counts = {
            "upper-left": int(np.sum(q_ul)),
            "upper-right": int(np.sum(q_ur)),
            "lower-left": int(np.sum(q_ll)),
            "lower-right": int(np.sum(q_lr)),
            "center": int(np.sum(center_box))
        }

        # Determine dominant quadrant
        sorted_quads = sorted(quad_counts.items(), key=lambda item: item[1], reverse=True)
        dominant_quad = sorted_quads[0][0] if veg_pixels > 0 else "none"

        has_vegetation = veg_pixels > max(100, int(total_pixels * 0.001))

        # Build natural-language direct answer
        q_lower = user_query.lower()
        is_which_area = any(w in q_lower for w in ["which area", "where", "most vegetation", "locate"])

        if has_vegetation:
            if is_which_area:
                direct_ans = f"The {dominant_quad} of the image has the highest concentration of vegetation."
            else:
                direct_ans = f"Yes, vegetation is present across approximately {veg_pct}% of the image, concentrated most prominently in the {dominant_quad}."
        else:
            direct_ans = "No significant vegetation was detected in this image; the scene consists primarily of other land-cover types."

        # Generate emerald green mask overlay
        mask_overlay_url = ImageProcessor.create_mask_overlay(
            base_file_path=image_path,
            mask=clean_veg_mask,
            color=(16, 185, 129),
            alpha=0.45
        )

        return {
            "task": "vegetation_analysis",
            "has_vegetation": has_vegetation,
            "dominant_quadrant": dominant_quad,
            "direct_answer": direct_ans,
            "result": {
                "vegetation_detected": has_vegetation,
                "vegetation_percentage": veg_pct,
                "vegetation_area_km2": veg_area_km2,
                "vegetation_pixels": veg_pixels,
                "total_pixels": total_pixels,
                "dominant_area": dominant_quad,
                "quadrant_distribution": quad_counts,
                "canopy_breakdown": {
                    "dense_canopy_pct": dense_pct,
                    "moderate_canopy_pct": mod_pct,
                    "sparse_canopy_pct": sparse_pct
                },
                "mean_index_value": round(mean_index_val, 3)
            },
            "confidence": confidence,
            "method": method,
            "evidence": {
                "pixels_analyzed": total_pixels,
                "vegetation_pixels": veg_pixels,
                "resolution_m": meta.get("spatial_resolution_m", 10.0),
                "crs": meta.get("crs", "Unknown CRS"),
                "spectral_bands_used": spectral_bands,
                "is_multispectral": is_multispectral
            },
            "overlay": mask_overlay_url,
            "limitations": limitations
        }
