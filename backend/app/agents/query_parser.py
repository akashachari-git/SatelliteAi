import re
from typing import Dict, Any, List, Optional

class QueryParser:
    """
    Translates natural-language remote sensing queries into a structured JSON intent.
    Accurately discerns domain, target feature, question type (existence, location,
    identification, dominance, summary), and multi-image routing.
    """

    @classmethod
    def parse(cls, query: str, image_count: int = 1) -> Dict[str, Any]:
        q = query.strip().lower()

        # Check requested units
        units_requested = []
        if any(w in q for w in ["km2", "square kilometer", "square km", "sq km", "area"]):
            units_requested.append("km2")
        if any(w in q for w in ["percent", "percentage", "%", "coverage", "how much"]):
            units_requested.append("percentage")
        if any(w in q for w in ["count", "how many", "number of", "each"]):
            units_requested.append("count")

        # Location query flag
        is_location_query = any(w in q for w in [
            "where", "which area", "which part", "which quadrant", "locate",
            "find the", "find all", "show me", "spot the", "where is", "where are"
        ])

        # -------------------------------------------------------------
        # 1. CROSS-IMAGE / MULTI-IMAGE ROUTING PRECEDENCE
        # -------------------------------------------------------------
        cross_modal_keywords = [
            "sar", "radar", "microwave", "cross-modal", "crossmodal",
            "cross modal", "fusion", "dielectric", "backscatter", "sentinel-1"
        ]
        if (image_count >= 2 and any(w in q for w in cross_modal_keywords)) or (
            any(w in q for w in ["optical and sar", "combine optical and sar", "using both images", "both optical and sar"])
        ):
            return {
                "intent": "joint_intelligence",
                "operation": "fuse",
                "target": "cross_modal_synthesis",
                "outputs": ["fused_composite", "sar_metrics", "optical_metrics", "complementary_insights"],
                "requires_multispectral": False,
                "preferred_index": "Optical-SAR Dual-Stream Cross-Attention",
                "required_image_count": 2,
                "is_location_query": is_location_query,
                "units_requested": units_requested or ["sar_metrics", "optical_metrics"]
            }

        change_keywords = [
            "what changed", "difference between", "difference map", "changes between",
            "detect change", "temporal difference", "how has the area changed",
            "before and after", "compare", "between the two", "between these two",
            "increase", "decrease", "has the built-up increased", "has water increased",
            "has vegetation decreased", "temporal", "flux"
        ]
        if image_count >= 2 or any(w in q for w in ["what changed", "difference between", "changes between"]):
            if any(w in q for w in change_keywords) or any(w in q for w in ["changed", "difference"]):
                return {
                    "intent": "change_detection",
                    "operation": "compare",
                    "target": "temporal_flux",
                    "outputs": ["change_mask", "statistics", "heatmap", "overlay", "area"],
                    "requires_multispectral": False,
                    "preferred_index": "Change Vector Analysis (CVA) / Adaptive Otsu",
                    "required_image_count": 2,
                    "is_location_query": is_location_query,
                    "units_requested": units_requested or ["percentage", "area", "statistics"]
                }

        # -------------------------------------------------------------
        # 2. MULTI-TARGET & COMPOSITE FEATURE DETECTION
        # -------------------------------------------------------------
        # Detect multi-feature requests such as:
        # "detect each and every water body, ice, buildings..."
        detected_features = []
        if any(w in q for w in ["water", "water body", "water bodies", "lake", "lakes", "river", "rivers", "reservoir"]):
            detected_features.append("water")
        if any(w in q for w in ["ice", "snow", "glacier", "frozen"]):
            detected_features.append("ice")
        if any(w in q for w in ["building", "buildings", "built-up", "built up", "urban", "house", "houses", "structure", "structures", "roofs"]):
            detected_features.append("buildings")
        if any(w in q for w in ["vegetation", "forest", "greenery", "trees", "canopy", "crop", "crops", "agriculture", "farm"]):
            detected_features.append("vegetation")

        if len(detected_features) >= 2 or (
            any(w in q for w in ["detect each", "detect all", "detect each and every", "detect every", "find each", "identify each"])
            and len(detected_features) >= 1
        ):
            return {
                "intent": "multi_target_detection",
                "operation": "detect_multi",
                "target": "multiple_features",
                "features": detected_features,
                "outputs": ["bounding_boxes", "mask", "overlay", "direct_answer"],
                "requires_multispectral": False,
                "preferred_index": "Multimodal Feature Extractor & Optical Spectral Contrast",
                "required_image_count": 1,
                "is_location_query": is_location_query,
                "units_requested": units_requested or ["count", "location"]
            }

        # Standalone ice / snow query
        if any(w in q for w in ["ice", "snow", "glacier", "frozen"]):
            return {
                "intent": "ice_detection",
                "operation": "locate" if is_location_query else "detect",
                "target": "ice_snow",
                "outputs": ["mask", "percentage", "location", "direct_answer"],
                "requires_multispectral": False,
                "preferred_index": "Optical High-Albedo / Snow-Ice Contrast",
                "required_image_count": 1,
                "is_location_query": is_location_query,
                "units_requested": units_requested or ["percentage", "location"]
            }

        # Composite land-cover query: "Identify the water, vegetation and built-up areas"
        if ("water" in q or "lake" in q) and ("vegetation" in q or "forest" in q or "green" in q) and (
            "built-up" in q or "urban" in q or "buildings" in q or "built up" in q
        ):
            return {
                "intent": "land_cover_analysis",
                "sub_intent": "composite_identification",
                "operation": "classify_multi",
                "target": "land_cover",
                "outputs": ["breakdown", "locations", "overlay"],
                "requires_multispectral": False,
                "preferred_index": "Multi-Spectral Land Cover",
                "required_image_count": 1,
                "is_location_query": is_location_query,
                "units_requested": units_requested or ["percentage", "locations"]
            }

        # Dominant land-cover: "Which land-cover type is dominant?"
        if any(w in q for w in ["dominant", "dominates", "dominate", "most common land", "primary land cover"]):
            return {
                "intent": "land_cover_analysis",
                "sub_intent": "dominant_land_cover",
                "operation": "find_dominant",
                "target": "land_cover",
                "outputs": ["dominant_class", "percentage", "location"],
                "requires_multispectral": False,
                "preferred_index": "Land Cover Dominance",
                "required_image_count": 1,
                "is_location_query": is_location_query,
                "units_requested": units_requested or ["percentage", "class"]
            }

        # Major land-cover types: "What are the major land-cover types?"
        if any(w in q for w in ["major land-cover", "major land cover", "land-cover types", "land cover types", "types of land"]):
            return {
                "intent": "land_cover_analysis",
                "sub_intent": "major_types",
                "operation": "inventory",
                "target": "land_cover",
                "outputs": ["classes", "percentages", "locations"],
                "requires_multispectral": False,
                "preferred_index": "Land Cover Categorization",
                "required_image_count": 1,
                "is_location_query": is_location_query,
                "units_requested": units_requested or ["classes", "percentages"]
            }

        # -------------------------------------------------------------
        # 3. AGRICULTURE / CROPLAND QUERIES
        # -------------------------------------------------------------
        agri_keywords = [
            "agriculture", "agricultural", "farm", "farms", "farming", "crop", "crops",
            "cropland", "cultivated", "cultivation", "arable", "paddy", "orchard", "fields"
        ]
        if any(w in q for w in agri_keywords):
            return {
                "intent": "agriculture_detection",
                "operation": "detect",
                "target": "agriculture",
                "outputs": ["presence", "percentage", "location", "overlay"],
                "requires_multispectral": False,
                "preferred_index": "Cropland Geometry & Spectral Reflection",
                "required_image_count": 1,
                "is_location_query": is_location_query,
                "units_requested": units_requested or ["presence", "location"]
            }

        # -------------------------------------------------------------
        # 4. MAJOR OBJECTS IDENTIFICATION
        # -------------------------------------------------------------
        objects_keywords = [
            "major objects", "what objects", "visible objects", "objects are visible",
            "man-made objects", "prominent features", "prominent objects"
        ]
        if any(w in q for w in objects_keywords):
            return {
                "intent": "land_cover_analysis",
                "sub_intent": "visible_objects",
                "operation": "identify_objects",
                "target": "objects",
                "outputs": ["object_list", "locations"],
                "requires_multispectral": False,
                "preferred_index": "High-Frequency Optical Object Identification",
                "required_image_count": 1,
                "is_location_query": is_location_query,
                "units_requested": ["objects"]
            }

        # -------------------------------------------------------------
        # 5. WATER DETECTION & WATER LOCATION
        # -------------------------------------------------------------
        water_keywords = [
            "water", "lake", "lakes", "river", "rivers", "reservoir", "reservoirs",
            "pond", "ponds", "ocean", "sea", "canal", "canals", "stream", "streams",
            "water body", "water bodies", "water feature", "water features", "bay"
        ]
        if any(w in q for w in water_keywords):
            outputs = ["mask", "overlay", "percentage", "location", "direct_answer"]
            if "count" in units_requested or any(w in q for w in ["count", "how many", "distinct"]):
                outputs.append("count")
            return {
                "intent": "water_detection",
                "operation": "locate" if is_location_query else "detect",
                "target": "water",
                "outputs": outputs,
                "requires_multispectral": False,
                "preferred_index": "NDWI / Visible Optical Water Delineation",
                "required_image_count": 1,
                "is_location_query": is_location_query,
                "units_requested": units_requested or ["percentage", "location"]
            }

        # -------------------------------------------------------------
        # 6. VEGETATION DETECTION & CONCENTRATION
        # -------------------------------------------------------------
        vegetation_keywords = [
            "vegetation", "forest", "forests", "tree", "trees", "canopy",
            "greenery", "grass", "green area", "woods", "woodland"
        ]
        if any(w in q for w in vegetation_keywords):
            return {
                "intent": "vegetation_analysis",
                "operation": "locate" if is_location_query else "detect",
                "target": "vegetation",
                "outputs": ["mask", "percentage", "dominant_area", "overlay"],
                "requires_multispectral": False,
                "preferred_index": "NDVI / Visible Green Canopy Excess",
                "required_image_count": 1,
                "is_location_query": is_location_query,
                "units_requested": units_requested or ["percentage", "area"]
            }

        # -------------------------------------------------------------
        # 7. BUILT-UP / URBAN / BUILDINGS & ROADS
        # -------------------------------------------------------------
        urban_keywords = [
            "urban", "built-up", "built up", "city", "town", "developed",
            "settlement", "infrastructure", "developed areas", "built-up regions"
        ]
        building_road_keywords = [
            "building", "buildings", "road", "roads", "structure", "structures",
            "house", "houses", "roof", "roofs", "highway", "highways", "streets"
        ]
        if any(w in q for w in urban_keywords) or any(w in q for w in building_road_keywords):
            return {
                "intent": "building_detection",
                "operation": "locate" if is_location_query else "detect",
                "target": "urban_built_up",
                "outputs": ["count", "is_urban", "has_roads", "location", "overlay"],
                "requires_multispectral": False,
                "preferred_index": "NDBI / Structural Edge Energy",
                "required_image_count": 1,
                "is_location_query": is_location_query,
                "units_requested": units_requested or ["count", "location"]
            }

        # -------------------------------------------------------------
        # 8. GENERAL SCENE DESCRIPTION / SUMMARY
        # -------------------------------------------------------------
        scene_keywords = [
            "describe", "caption", "tell me about this image", "tell me about",
            "overview", "summary", "summarize", "what does this satellite image show",
            "what does this image show", "give me a summary"
        ]
        if any(w in q for w in scene_keywords):
            return {
                "intent": "scene_description",
                "operation": "describe",
                "target": "scene",
                "outputs": ["narrative", "dominant_class", "land_cover_breakdown"],
                "requires_multispectral": False,
                "preferred_index": "Optical Remote Sensing Synthesis",
                "required_image_count": 1,
                "is_location_query": is_location_query,
                "units_requested": ["description"]
            }

        # -------------------------------------------------------------
        # 9. FALLBACK LAND COVER INSPECTION
        # -------------------------------------------------------------
        return {
            "intent": "land_cover_analysis",
            "sub_intent": "general_inspection",
            "operation": "vqa",
            "target": "scene_features",
            "outputs": ["answer", "breakdown"],
            "requires_multispectral": False,
            "preferred_index": "Optical Remote Sensing Inspection",
            "required_image_count": 1,
            "is_location_query": is_location_query,
            "units_requested": units_requested
        }
