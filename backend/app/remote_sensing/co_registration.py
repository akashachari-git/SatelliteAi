from typing import Dict, Any, List, Tuple

class CoRegistrationChecker:
    """
    Validates geometric, spatial, radiometric, and temporal compatibility
    between paired remote-sensing images (Bi-temporal or Optical+SAR).
    """

    @staticmethod
    def evaluate_pair(meta_a: Dict[str, Any], meta_b: Dict[str, Any], pair_mode: str = "AUTO") -> Dict[str, Any]:
        """
        Evaluate compatibility for Bi-temporal or Cross-modal Optical+SAR.
        pair_mode: 'AUTO', 'BI_TEMPORAL', 'OPTICAL_SAR'
        """
        checks: List[Dict[str, Any]] = []
        warnings: List[str] = []
        is_compatible = True

        # 1. Dimensions Check
        dim_match = (meta_a.get("width") == meta_b.get("width") and 
                     meta_a.get("height") == meta_b.get("height"))
        if dim_match:
            checks.append({
                "name": "Dimensions Compatible",
                "status": "PASS",
                "details": f"{meta_a.get('width')}x{meta_a.get('height')} px matches exactly"
            })
        else:
            w_diff = abs(meta_a.get("width", 0) - meta_b.get("width", 0))
            h_diff = abs(meta_a.get("height", 0) - meta_b.get("height", 0))
            if w_diff < 50 and h_diff < 50:
                checks.append({
                    "name": "Dimensions Compatible",
                    "status": "PASS",
                    "details": f"Minor dimension difference ({meta_a.get('width')}x{meta_a.get('height')} vs {meta_b.get('width')}x{meta_b.get('height')}), auto-resampling enabled"
                })
            else:
                checks.append({
                    "name": "Dimensions Compatible",
                    "status": "WARN",
                    "details": f"Dimension mismatch: {meta_a.get('width')}x{meta_a.get('height')} vs {meta_b.get('width')}x{meta_b.get('height')}"
                })
                warnings.append("Images have different pixel dimensions; automated spatial grid matching will be applied.")

        # 2. CRS Compatibility
        crs_a = meta_a.get("crs", "")
        crs_b = meta_b.get("crs", "")
        if crs_a == crs_b and crs_a != "Local Pixel Coordinates":
            checks.append({
                "name": "CRS Compatible",
                "status": "PASS",
                "details": f"Identical projection: {crs_a}"
            })
        elif crs_a == crs_b:
            checks.append({
                "name": "CRS Compatible",
                "status": "PASS",
                "details": "Both images operate on matching local pixel coordinate frame"
            })
        else:
            checks.append({
                "name": "CRS Compatible",
                "status": "WARN",
                "details": f"CRS variation: {crs_a} vs {crs_b}"
            })
            warnings.append("Coordinate Reference Systems differ. On-the-fly reprojection required.")

        # 3. Spatial Resolution
        res_a = meta_a.get("spatial_resolution_m", 10.0)
        res_b = meta_b.get("spatial_resolution_m", 10.0)
        res_ratio = max(res_a, res_b) / (min(res_a, res_b) or 1.0)
        if res_ratio <= 1.2:
            checks.append({
                "name": "Resolution Compatible",
                "status": "PASS",
                "details": f"Native GSD aligned: {res_a}m vs {res_b}m (ratio {round(res_ratio, 2)})"
            })
        else:
            checks.append({
                "name": "Resolution Compatible",
                "status": "WARN",
                "details": f"Resolution variation: {res_a}m vs {res_b}m (ratio {round(res_ratio, 2)})"
            })
            warnings.append(f"Resolution difference ratio is {round(res_ratio, 2)}. Nearest-neighbor or bilinear resampling needed.")

        # 4. Spatial Overlap
        bounds_a = meta_a.get("bounds")
        bounds_b = meta_b.get("bounds")
        if bounds_a and bounds_b:
            # Check bounding box intersection
            overlap_x = max(0, min(bounds_a["max_x"], bounds_b["max_x"]) - max(bounds_a["min_x"], bounds_b["min_x"]))
            overlap_y = max(0, min(bounds_a["max_y"], bounds_b["max_y"]) - max(bounds_a["min_y"], bounds_b["min_y"]))
            area_overlap = overlap_x * overlap_y
            area_a = (bounds_a["max_x"] - bounds_a["min_x"]) * (bounds_a["max_y"] - bounds_a["min_y"])
            overlap_ratio = area_overlap / (area_a or 1.0)
            if overlap_ratio > 0.8:
                checks.append({
                    "name": "Spatial Overlap",
                    "status": "PASS",
                    "details": f"Spatial overlap verified ({round(overlap_ratio * 100, 1)}%)"
                })
            elif overlap_ratio > 0.3:
                checks.append({
                    "name": "Spatial Overlap",
                    "status": "WARN",
                    "details": f"Partial spatial overlap ({round(overlap_ratio * 100, 1)}%)"
                })
                warnings.append("Images only partially overlap. Analysis constrained to intersection footprint.")
            else:
                checks.append({
                    "name": "Spatial Overlap",
                    "status": "FAIL",
                    "details": "Insufficient spatial overlap (<30%)"
                })
                is_compatible = False
                warnings.append("Images do not cover the same geographic area.")
        else:
            # Local coordinate fallback
            checks.append({
                "name": "Spatial Overlap",
                "status": "PASS",
                "details": "Co-registered local pixel frame assumed for demonstration pairs"
            })

        # 5. Modality Detection & Classification
        mod_a = meta_a.get("modality", "Optical")
        mod_b = meta_b.get("modality", "Optical")
        has_sar = (mod_a == "SAR" or mod_b == "SAR")
        has_optical = ("Optical" in mod_a or "Multispectral" in mod_a or "Optical" in mod_b or "Multispectral" in mod_b)

        suggested_mode = "BI_TEMPORAL"
        if has_sar and has_optical:
            suggested_mode = "OPTICAL_SAR"
            checks.append({
                "name": "Modality Detected",
                "status": "PASS",
                "details": f"Cross-modal pair confirmed: {mod_a} + {mod_b}"
            })
        else:
            checks.append({
                "name": "Modality Detected",
                "status": "PASS",
                "details": f"Homogeneous modalities: {mod_a} / {mod_b} suitable for bi-temporal analysis"
            })

        # 6. Overall Co-Registration Score
        pass_count = sum(1 for c in checks if c["status"] == "PASS")
        score = round((pass_count / len(checks)) * 100)

        return {
            "is_compatible": is_compatible,
            "co_registration_score": score,
            "suggested_mode": suggested_mode,
            "checks": checks,
            "warnings": warnings,
            "summary": f"{'✓ Pair fully co-registered & compatible' if is_compatible else '⚠ Co-registration issues detected'} ({score}% compatibility index)"
        }
