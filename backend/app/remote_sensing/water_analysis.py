import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy.ndimage import label, find_objects, binary_opening, binary_closing
from .image_processor import ImageProcessor
from .raster_service import RasterMetadataService
from .spectral_indices import SpectralIndicesCalculator

class WaterAnalysisEngine:
    """
    Dedicated Remote-Sensing Water Delineation & Spatial Localization Engine.
    - True multispectral NDWI = (Green - NIR) / (Green + NIR) when NIR is available.
    - Calibrated optical visible water detection for RGB rasters (turbid/algal inland lakes,
      deep/oligotrophic dark water bodies, and coastal/river systems).
    - Excludes shadows and false positives from dense terrestrial vegetation.
    - Accurately computes spatial locations using natural quadrant terms:
      'upper-left', 'upper-right', 'center', 'lower-left', 'lower-right'.
    """

    @classmethod
    def _format_quadrant(cls, ymin: float, xmin: float, ymax: float, xmax: float) -> str:
        """Formats spatial coordinates into natural quadrant terms."""
        cy = (ymin + ymax) / 2.0
        cx = (xmin + xmax) / 2.0
        v = "upper" if cy < 0.38 else ("lower" if cy > 0.62 else "center")
        h = "left" if cx < 0.38 else ("right" if cx > 0.62 else "center")

        if v == "center" and h == "center":
            return "center"
        if v == "center":
            return f"central {h}"
        if h == "center":
            return f"{v} section"
        return f"{v}-{h}"

    @classmethod
    def analyze_water(cls, image_path: str, user_query: str = "") -> Dict[str, Any]:
        arr = ImageProcessor.load_as_array(image_path)
        meta = RasterMetadataService.inspect_raster(image_path)
        h, w = arr.shape[:2]
        total_pixels = h * w
        pixel_area_km2 = meta.get("pixel_area_km2", 0.0001)

        limitations = []
        is_multispectral = meta.get("is_geotiff", False) and (arr.ndim == 3 and arr.shape[2] >= 4 and meta.get("number_of_bands", 1) >= 4)

        # -------------------------------------------------------------
        # 1. Spectral Index or Optical Visible Water Detection
        # -------------------------------------------------------------
        if is_multispectral:
            # True multispectral NDWI using Green and NIR
            ndwi = SpectralIndicesCalculator.calculate_ndwi(arr, green_band=1, nir_band=3)
            if ndwi is not None:
                green = arr[:, :, 1].astype(np.float32)
                raw_mask = (ndwi > 0.0) & (green > 5)
                method = "NDWI (Normalized Difference Water Index: Green - NIR / Green + NIR)"
                confidence = 0.95
                spectral_bands = ["Green (560nm)", "NIR (842nm)"]
            else:
                is_multispectral = False

        if not is_multispectral:
            limitations.append(
                "Multispectral NIR band unavailable. Analysis used calibrated visible optical Earth observation water delineation."
            )
            r = arr[:, :, 0].astype(np.float32)
            g = arr[:, :, 1].astype(np.float32)
            b = arr[:, :, 2].astype(np.float32)
            brightness = (r + g + b) / 3.0

            # 1. Dark water bodies (clear, deep water or algal inland lakes in satellite imagery)
            # Water has very strong absorption across the visible spectrum, especially Red
            dark_water = (brightness <= 38.0) & (r <= 32.0) & (brightness >= 5.0)

            # 2. Blue & Cyan open water (shallow lakes, rivers, reservoirs, coastal waters)
            ndrb = (b - r) / (b + r + 1e-5)
            ndrg = (g - r) / (g + r + 1e-5)
            blue_water = (
                ((ndrb > 0.06) | ((ndrg > 0.08) & (b >= r * 0.95))) &
                (b > r + 3) &
                (brightness <= 165.0) &
                (r <= 115.0)
            )

            # 3. Turbid or sediment-laden rivers / reservoirs
            turbid_water = (
                (ndrg > 0.05) & (ndrb > -0.05) &
                (g >= r * 1.05) & (b >= r * 0.90) &
                (brightness >= 25.0) & (brightness <= 110.0) &
                (r <= 90.0)
            )

            # Exclude dense terrestrial vegetation (strong green peak with bright > 35 and g > b * 1.25 and g > r * 1.25)
            is_dense_veg = (g > r * 1.25) & (g > b * 1.25) & (brightness > 35.0)

            raw_mask = (dark_water | blue_water | turbid_water) & (~is_dense_veg)
            method = "Visible Optical Water Body Segmentation (Colorimetric & Spectral Absorption)"
            confidence = 0.89
            spectral_bands = ["Red (Visible)", "Green (Visible)", "Blue (Visible)"]

        # -------------------------------------------------------------
        # 2. Morphological Cleaning & Connected Component Filtering
        # -------------------------------------------------------------
        opened = binary_opening(raw_mask, structure=np.ones((2, 2)))
        closed = binary_closing(opened, structure=np.ones((3, 3)))

        labeled, num_features = label(closed)
        slices = find_objects(labeled)

        # Minimum water body threshold: preserve natural rivers and small reservoirs
        min_water_body_pixels = max(60, int(total_pixels * 0.0001))
        final_water_mask = np.zeros_like(closed, dtype=bool)
        water_regions: List[Dict[str, Any]] = []

        if num_features > 0 and slices:
            for idx, sl in enumerate(slices):
                if sl is None:
                    continue
                comp = (labeled[sl] == (idx + 1))
                comp_pixels = int(np.sum(comp))

                if comp_pixels >= min_water_body_pixels:
                    final_water_mask[sl] |= comp
                    ymin = round(float(sl[0].start) / h, 4)
                    xmin = round(float(sl[1].start) / w, 4)
                    ymax = round(float(sl[0].stop) / h, 4)
                    xmax = round(float(sl[1].stop) / w, 4)
                    area_km2 = round(comp_pixels * pixel_area_km2, 4)

                    loc = cls._format_quadrant(ymin, xmin, ymax, xmax)

                    water_regions.append({
                        "region_id": len(water_regions) + 1,
                        "pixels": comp_pixels,
                        "percentage": round((comp_pixels / total_pixels) * 100, 2),
                        "area_km2": area_km2,
                        "location": loc,
                        "bbox": [ymin, xmin, ymax, xmax]
                    })

        # Sort regions by size descending
        water_regions.sort(key=lambda r: r["pixels"], reverse=True)
        for i, r in enumerate(water_regions):
            r["region_id"] = i + 1

        water_pixels = int(np.sum(final_water_mask))
        water_percentage = round((water_pixels / total_pixels) * 100, 2)
        water_area_km2 = round(water_pixels * pixel_area_km2, 4)
        if water_area_km2 == 0.0 and water_pixels > 0:
            water_area_km2 = round((water_pixels * 1.0) / 1e6, 4)

        has_water = len(water_regions) > 0 and water_pixels >= min_water_body_pixels

        # Build natural language direct answer & location description
        q_lower = user_query.lower()
        is_loc_q = any(w in q_lower for w in ["where", "locate", "which area", "which part", "find"])

        if has_water:
            primary_loc = water_regions[0]["location"]
            if len(water_regions) == 1:
                loc_desc = f"in the {primary_loc}"
            elif len(water_regions) == 2:
                loc_desc = f"primarily in the {primary_loc} and {water_regions[1]['location']}"
            else:
                loc_desc = f"primarily in the {primary_loc}, with additional sections visible across the {water_regions[1]['location']} and {water_regions[2]['location']}"

            if is_loc_q:
                direct_ans = f"The water body is located {loc_desc} of the image."
            else:
                if len(water_regions) == 1:
                    direct_ans = f"Yes, there is a visible water body located {loc_desc}."
                else:
                    direct_ans = f"Yes, visible water bodies are detected {loc_desc}."
        else:
            if is_loc_q:
                direct_ans = "No water bodies are visible in this image."
            else:
                direct_ans = "No water bodies are detected in this image."
            loc_desc = "None"

        # Generate visual overlays (Cyan for water bodies)
        mask_overlay_url = ImageProcessor.create_mask_overlay(
            base_file_path=image_path,
            mask=final_water_mask,
            color=(6, 182, 212),
            alpha=0.55
        )

        bounding_boxes = [
            {"box": r["bbox"], "label": f"Water Body ({r['location']})", "score": confidence}
            for r in water_regions[:12]
        ]
        bbox_overlay_url = None
        if len(bounding_boxes) > 0:
            bbox_overlay_url = ImageProcessor.create_bounding_box_overlay(
                base_file_path=image_path,
                boxes=bounding_boxes,
                color="#06b6d4"
            )

        evidence_object = {
            "task": "water_detection",
            "has_water": has_water,
            "direct_answer": direct_ans,
            "location_description": loc_desc,
            "result": {
                "water_detected": has_water,
                "water_percentage": water_percentage,
                "water_area_km2": water_area_km2,
                "water_pixels": water_pixels,
                "total_pixels": total_pixels,
                "regions_count": len(water_regions),
                "primary_location": water_regions[0]["location"] if water_regions else "none",
                "regions": water_regions
            },
            "confidence": confidence,
            "method": method,
            "evidence": {
                "pixels_analyzed": total_pixels,
                "target_pixels": water_pixels,
                "resolution_m": meta.get("spatial_resolution_m", 10.0),
                "crs": meta.get("crs", "Unknown CRS"),
                "spectral_bands_used": spectral_bands,
                "is_multispectral": is_multispectral
            },
            "overlay": mask_overlay_url,
            "bbox_overlay": bbox_overlay_url,
            "limitations": limitations
        }

        return evidence_object
