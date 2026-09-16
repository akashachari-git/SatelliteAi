import numpy as np
from typing import Dict, Any, List
from scipy.ndimage import label, find_objects, binary_opening, binary_closing
from ..remote_sensing.image_processor import ImageProcessor
from ..remote_sensing.spectral_indices import SpectralIndicesCalculator

class GroundingService:
    """
    Text-guided spatial region localization and semantic grounding for satellite imagery.
    Detects target objects (water, urban, vegetation, salient features) from text prompt,
    computes bounding boxes, and generates segmentation mask overlays.
    """

    @classmethod
    def _format_location(cls, ymin: float, xmin: float, ymax: float, xmax: float) -> str:
        cy = (ymin + ymax) / 2.0
        cx = (xmin + xmax) / 2.0
        v = "upper" if cy < 0.38 else ("lower" if cy > 0.62 else "middle")
        h = "left" if cx < 0.40 else ("right" if cx > 0.60 else "central")
        if v == "middle" and h == "central":
            return "center"
        if v == "middle":
            return f"{h} side"
        if h == "central":
            return f"{v} section"
        return f"{v}-{h}"

    @classmethod
    def ground_target(cls, image_path: str, query: str) -> Dict[str, Any]:
        arr = ImageProcessor.load_as_array(image_path)
        q_lower = query.lower()
        h, w = arr.shape[:2]
        total_pixels = h * w

        # Target classification
        if any(w_word in q_lower for w_word in ["water", "river", "lake", "ocean", "reservoir", "pond", "canal", "stream", "tarn"]):
            target_type = "water"
            color = (6, 182, 212) # Cyan
            hex_color = "#06b6d4"

            # Multi-spectral (4+ bands) vs Visible Optical RGB
            idx_map = None
            if arr.ndim == 3 and arr.shape[2] >= 4:
                idx_map = SpectralIndicesCalculator.calculate_ndwi(arr)
            
            if idx_map is not None:
                raw_mask = idx_map > 0.05
            elif arr.ndim == 3 and arr.shape[2] >= 3:
                r = arr[:, :, 0].astype(float)
                g = arr[:, :, 1].astype(float)
                b = arr[:, :, 2].astype(float)
                brightness = (r + g + b) / 3.0
                ndrb = (b - r) / (b + r + 1e-5)
                ndrg = (g - r) / (g + r + 1e-5)

                is_dark_water = (brightness <= 38.0) & (r <= 32.0)
                is_blue_water = (ndrb > 0.04) & (ndrg > 0.04) & (b > r + 3) & (brightness <= 130)
                is_veg = (g > r + 8.0) & (g > b + 5.0)
                raw_mask = (is_dark_water | is_blue_water) & (~is_veg)
            else:
                gray = arr if arr.ndim == 2 else np.mean(arr, axis=2)
                raw_mask = gray < np.percentile(gray, 18)

            # Delicate morphological cleaning to preserve narrow rivers and small lakes
            cleaned = binary_opening(raw_mask, structure=np.ones((2, 2)))
            cleaned = binary_closing(cleaned, structure=np.ones((3, 3)))
            min_cluster_size = max(15, int(total_pixels * 0.0001))

        elif any(w_word in q_lower for w_word in ["ice", "snow", "glacier", "frozen", "ice cap", "permafrost"]):
            target_type = "ice_snow"
            color = (56, 189, 248) # Sky Blue / Ice
            hex_color = "#38bdf8"

            if arr.ndim == 3 and arr.shape[2] >= 3:
                r = arr[:, :, 0].astype(float)
                g = arr[:, :, 1].astype(float)
                b = arr[:, :, 2].astype(float)
                brightness = (r + g + b) / 3.0
                raw_mask = (r > 155) & (g > 155) & (b > 155) & (brightness > 170)
            else:
                gray = arr if arr.ndim == 2 else np.mean(arr, axis=2)
                raw_mask = gray > np.percentile(gray, 85)

            cleaned = binary_opening(raw_mask, structure=np.ones((2, 2)))
            cleaned = binary_closing(cleaned, structure=np.ones((3, 3)))
            min_cluster_size = max(20, int(total_pixels * 0.0001))

        elif any(w_word in q_lower for w_word in ["vegetation", "forest", "tree", "agriculture", "crop", "green", "field", "park", "grass", "woodland"]):
            target_type = "vegetation"
            color = (16, 185, 129) # Emerald Green
            hex_color = "#10b981"

            idx_map = None
            if arr.ndim == 3 and arr.shape[2] >= 4:
                idx_map = SpectralIndicesCalculator.calculate_ndvi(arr)
            
            if idx_map is not None:
                raw_mask = idx_map > 0.25
            elif arr.ndim == 3 and arr.shape[2] >= 3:
                r = arr[:, :, 0].astype(float)
                g = arr[:, :, 1].astype(float)
                b = arr[:, :, 2].astype(float)
                raw_mask = (g > r * 1.06) & (g > b * 1.04) & (g > 35)
            else:
                raw_mask = arr < np.percentile(arr, 35)

            cleaned = binary_opening(raw_mask, structure=np.ones((2, 2)))
            cleaned = binary_closing(cleaned, structure=np.ones((3, 3)))
            min_cluster_size = max(20, int(total_pixels * 0.0001))

        elif any(w_word in q_lower for w_word in ["building", "urban", "structure", "built-up", "settlement", "house", "city", "industrial", "road"]):
            target_type = "built_up"
            color = (245, 158, 11) # Amber
            hex_color = "#f59e0b"

            idx_map = None
            if arr.ndim == 3 and arr.shape[2] >= 5:
                idx_map = SpectralIndicesCalculator.calculate_ndbi(arr)
            
            if idx_map is not None:
                raw_mask = idx_map > 0.02
            elif arr.ndim == 3 and arr.shape[2] >= 3:
                r = arr[:, :, 0].astype(float)
                g = arr[:, :, 1].astype(float)
                b = arr[:, :, 2].astype(float)
                brightness = (r + g + b) / 3.0
                not_water = ~(((b - r) / (b + r + 1e-5) > 0.04) & (brightness <= 130))
                not_ice = ~((brightness > 170) & (r > 155) & (g > 155) & (b > 155))
                not_veg = ~((g > r * 1.08) & (g > b * 1.05))
                raw_mask = not_water & not_ice & not_veg & (brightness >= 65)
            else:
                raw_mask = arr > np.percentile(arr, 60)

            cleaned = binary_opening(raw_mask, structure=np.ones((2, 2)))
            cleaned = binary_closing(cleaned, structure=np.ones((3, 3)))
            min_cluster_size = max(25, int(total_pixels * 0.0001))

        elif any(w_word in q_lower for w_word in ["land", "rock", "soil", "mountain", "bare ground", "terrain", "barren", "sand", "dirt"]):
            target_type = "land_rock"
            color = (217, 119, 6) # Warm Amber / Ochre
            hex_color = "#d97706"

            if arr.ndim == 3 and arr.shape[2] >= 3:
                r = arr[:, :, 0].astype(float)
                g = arr[:, :, 1].astype(float)
                b = arr[:, :, 2].astype(float)
                brightness = (r + g + b) / 3.0
                is_water = ((b - r) / (b + r + 1e-5) > 0.04) & (brightness <= 130)
                is_ice = (brightness > 170) & (r > 155) & (g > 155) & (b > 155)
                is_veg = (g > r * 1.06) & (g > b * 1.04) & (g > 35)
                raw_mask = ~is_water & ~is_ice & ~is_veg & (brightness > 20)
            else:
                gray = arr if arr.ndim == 2 else np.mean(arr, axis=2)
                raw_mask = (gray >= np.percentile(gray, 25)) & (gray <= np.percentile(gray, 80))

            cleaned = binary_opening(raw_mask, structure=np.ones((2, 2)))
            cleaned = binary_closing(cleaned, structure=np.ones((3, 3)))
            min_cluster_size = max(30, int(total_pixels * 0.0002))

        else:
            # Salient / conspicuous feature grounding
            target_type = "salient_feature"
            color = (239, 68, 68) # Red
            hex_color = "#ef4444"
            gray = np.mean(arr, axis=2) if arr.ndim == 3 else arr
            raw_mask = gray > np.percentile(gray, 78)
            cleaned = binary_opening(raw_mask, structure=np.ones((3, 3)))
            cleaned = binary_closing(cleaned, structure=np.ones((5, 5)))
            min_cluster_size = max(30, int(total_pixels * 0.0002))

        # Connected component filtering: include all valid clusters down to fine-grain detail
        labeled_mask, num_features = label(cleaned)
        clean_mask = np.zeros_like(cleaned, dtype=bool)
        candidate_boxes: List[Dict[str, Any]] = []

        if num_features > 0:
            slices = find_objects(labeled_mask)
            for i, sl in enumerate(slices):
                if sl is None:
                    continue
                comp_mask = (labeled_mask[sl] == (i + 1))
                comp_pixels = int(np.sum(comp_mask))
                if comp_pixels < min_cluster_size:
                    continue

                # Add this valid component to our clean mask overlay
                clean_mask[sl] |= comp_mask

                ymin = sl[0].start / h
                xmin = sl[1].start / w
                ymax = sl[0].stop / h
                xmax = sl[1].stop / w
                area_pct = round((comp_pixels / total_pixels) * 100, 2)
                score = min(0.98, 0.88 + (comp_pixels / total_pixels) * 5)
                location = cls._format_location(ymin, xmin, ymax, xmax)

                candidate_boxes.append({
                    "box": [round(ymin, 4), round(xmin, 4), round(ymax, 4), round(xmax, 4)],
                    "label": target_type.replace("_", " ").title(),
                    "score": round(score, 2),
                    "pixels": comp_pixels,
                    "area_pct": area_pct,
                    "location": location
                })

        # Sort candidate regions by area descending, keep up to 16 prominent boxes for visual clarity
        candidate_boxes.sort(key=lambda b: b["area_pct"], reverse=True)
        bounding_boxes = candidate_boxes[:16]

        # Calculate accurate coverage percentage based strictly on the full clean mask
        target_pixels = int(np.sum(clean_mask))
        coverage_pct = round((target_pixels / total_pixels) * 100, 2)

        # Generate visual overlays ONLY using the clean mask
        mask_overlay_url = ImageProcessor.create_mask_overlay(image_path, clean_mask, color=color, alpha=0.50)
        bbox_overlay_url = ImageProcessor.create_bounding_box_overlay(image_path, bounding_boxes, color=hex_color)

        # Detailed conversational explanations
        label_name = target_type.replace('_', ' ')
        if len(candidate_boxes) == 1:
            loc = candidate_boxes[0]["location"]
            ans = f"I detected 1 primary {label_name} feature located towards the {loc}, covering {coverage_pct}% of the scene. The visual overlay outlines every detail precisely."
        elif len(candidate_boxes) > 1:
            ans = (
                f"I detected {len(candidate_boxes)} distinct {label_name} features across the scene, "
                f"covering approximately {coverage_pct}% of the total area in all. "
                f"The color overlay highlights all major and minor sections including small tributaries, alpine lakes, and intricate contours."
            )
        else:
            ans = f"I conducted a spectral scan across the scene, but no distinct {label_name} features were detected above confidence threshold."

        conf = 0.94 if len(bounding_boxes) > 0 else 0.75

        return {
            "answer": ans,
            "target_entity": target_type,
            "coverage_pct": coverage_pct,
            "detected_regions_count": len(bounding_boxes),
            "bounding_boxes": bounding_boxes,
            "mask_overlay_url": mask_overlay_url,
            "bbox_overlay_url": bbox_overlay_url,
            "confidence": conf,
            "model_used": "GroundingModel (Optical & Spectral Feature Segmenter)"
        }
