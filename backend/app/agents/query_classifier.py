from typing import Dict, Any, List, Optional
import re

class QueryClassifier:
    """
    Deterministic & rule-assisted remote-sensing query classifier.
    Maps natural-language user queries to specialist task definitions.
    """

    TASKS = [
        "SINGLE_VQA",
        "CAPTIONING",
        "GROUNDING",
        "CHANGE_ANALYSIS",
        "CHANGE_VQA",
        "OPTICAL_SAR_ANALYSIS",
        "UNSUPPORTED"
    ]

    @classmethod
    def classify(cls, query: str, image_count: int, modalities: List[str]) -> Dict[str, Any]:
        q = query.strip().lower()
        has_sar = any("sar" in m.lower() for m in modalities)
        has_optical = any("optical" in m.lower() or "multi" in m.lower() for m in modalities)

        # 1. Check for Grounding / Localization / Highlighting
        grounding_keywords = [
            "highlight", "localize", "locate", "segment", "bounding box", "find the", "find all",
            "where is the water", "where are the buildings", "delineate", "outline", "show me the",
            "detect the water", "detect water", "detect the lake", "detect lake", "detect the river",
            "water body", "water bodies", "detect each", "all the water", "detect all", "find water",
            "detect ice", "detect snow", "glacier", "glaciers", "frozen",
            "detect land", "detect rock", "soil", "terrain", "mountain", "barren",
            "detect buildings", "detect the buildings", "detect trees", "detect vegetation",
            "spot the", "identify the water", "identify water", "point out", "map the", "extract mask"
        ]
        if any(kw in q for kw in grounding_keywords):
            return {
                "task": "GROUNDING",
                "confidence": 0.94,
                "reasoning": "Query requests spatial localization or visual highlighting of specific geographic features.",
                "required_image_count": 1
            }

        # 2. Check for Captioning / General Scene Description
        captioning_keywords = [
            "describe", "caption", "tell me about this image", "scene summary",
            "summarize this image", "overview of this image", "what does this satellite image show",
            "describe this satellite"
        ]
        if any(kw in q for kw in captioning_keywords):
            return {
                "task": "CAPTIONING",
                "confidence": 0.93,
                "reasoning": "Query solicits an end-to-end descriptive summary of satellite scene context.",
                "required_image_count": 1
            }

        # 3. Check for Optical + SAR Cross-Modal Fusion
        opt_sar_keywords = [
            "sar", "optical and sar", "radar", "backscatter", "dielectric",
            "using both images", "combine optical and sar", "cross-modal"
        ]
        if any(kw in q for kw in opt_sar_keywords) or (image_count == 2 and has_sar and has_optical):
            # Check if query also specifically mentions change
            if not any(kw in q for kw in ["change", "before and after", "temporal", "difference between dates"]):
                return {
                    "task": "OPTICAL_SAR_ANALYSIS",
                    "confidence": 0.95,
                    "reasoning": "Query requires joint cross-sensor optical reflectance and microwave SAR backscatter fusion.",
                    "required_image_count": 2
                }

        # 4. Check for Bi-Temporal Change Analysis / Change VQA
        change_analysis_keywords = [
            "what changed", "difference between", "difference map", "changes between",
            "detect change", "temporal difference", "how has the area changed"
        ]
        if any(kw in q for kw in change_analysis_keywords):
            return {
                "task": "CHANGE_ANALYSIS",
                "confidence": 0.96,
                "reasoning": "Query explicitly requests comparative temporal difference mapping and change quantification.",
                "required_image_count": 2
            }

        change_vqa_keywords = [
            "increase", "decrease", "unchanged", "has the built-up", "has urban",
            "where did the change occur", "has water increased", "has vegetation decreased",
            "is the forest area declining", "has the area grown"
        ]
        if any(kw in q for kw in change_vqa_keywords):
            return {
                "task": "CHANGE_VQA",
                "confidence": 0.94,
                "reasoning": "Query asks a directional or semantic question about temporal change between two dates.",
                "required_image_count": 2
            }

        # 5. Default to Single-Image VQA for questions
        if any(kw in q for kw in ["what", "is there", "are there", "how many", "which", "does this", "can you see"]):
            # If 2 images are provided and query has change connotations
            if image_count == 2 and ("change" in q or "differ" in q):
                return {
                    "task": "CHANGE_VQA",
                    "confidence": 0.91,
                    "reasoning": "Two images provided with temporal inquiry; routed to Change VQA.",
                    "required_image_count": 2
                }
            return {
                "task": "SINGLE_VQA",
                "confidence": 0.92,
                "reasoning": "Inquiry regarding land cover, objects, or spectral composition in satellite imagery.",
                "required_image_count": 1
            }

        # 6. Fallback
        if len(q.split()) > 1:
            return {
                "task": "SINGLE_VQA" if image_count == 1 else ("CHANGE_ANALYSIS" if not has_sar else "OPTICAL_SAR_ANALYSIS"),
                "confidence": 0.75,
                "reasoning": "Heuristic fallback based on available image configuration.",
                "required_image_count": image_count
            }

        return {
            "task": "UNSUPPORTED",
            "confidence": 0.30,
            "reasoning": "Query too short or ambiguous to reliably determine remote sensing task intent.",
            "required_image_count": 1
        }
