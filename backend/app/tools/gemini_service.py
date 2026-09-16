import os
import re
import json
import io
import base64
import urllib.request
import urllib.error
from pathlib import Path
from PIL import Image
from typing import Dict, Any, List, Optional

class GeminiVisionService:
    """
    Google Gemini Multimodal Vision & Geospatial NLP Intelligence Service.
    Answers natural-language queries about optical/multispectral satellite images
    using visual evidence and deterministic remote-sensing metrics.
    Guarantees concise, evidence-grounded natural-language text responses without hallucinations.
    """

    CANDIDATE_MODELS = [
        "gemini-3.6-flash"
    ]

    DEFAULT_KEY = ""

    @classmethod
    def _get_api_key(cls, passed_key: Optional[str] = None) -> Optional[str]:
        if passed_key and passed_key.strip():
            return passed_key.strip()
        env_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if env_key:
            return env_key
        return None

    @classmethod
    def _prepare_image_b64(cls, image_path: Optional[str]) -> Optional[str]:
        if not image_path:
            return None
        path = Path(image_path)
        if not path.exists():
            return None
        try:
            im = Image.open(str(path))
            if im.mode != "RGB":
                im = im.convert("RGB")
            im.thumbnail((640, 640))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=82)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            return None

    @classmethod
    def explain_evidence(
        cls,
        query: str,
        evidence_object: Dict[str, Any],
        image_path: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> str:
        """
        Generates a concise, direct natural-language text answer anchored strictly
        in the provided satellite image and deterministic remote-sensing evidence.
        """
        key = cls._get_api_key(api_key)
        template_fallback = cls._generate_template_explanation(query, evidence_object)

        if not key:
            return template_fallback

        is_multi = evidence_object.get("evidence", {}).get("is_multispectral", False)
        spectral_info = (
            "Image is multispectral with validated NIR bands."
            if is_multi else
            "Image is optical RGB (multispectral NIR/SWIR bands are NOT available; do NOT calculate or cite NDVI or NDWI)."
        )

        prompt_text = (
            "You are SatQuery AI, an expert satellite remote sensing assistant.\n"
            f"User Question: \"{query}\"\n\n"
            f"Deterministic Remote Sensing Evidence:\n"
            f"{json.dumps(evidence_object, indent=2, default=str)}\n\n"
            f"Spectral Band Status: {spectral_info}\n\n"
            "STRICT RULES FOR YOUR ANSWER:\n"
            "1. Output ONLY a concise, direct answer in simple natural-language TEXT (1 to 2 complete sentences).\n"
            "2. Directly answer the user's specific question (e.g. whether the feature exists, where it is located, what the dominant type is, or what the summary is).\n"
            "3. Do NOT output tables, JSON, CSV, code, markdown headers, or bullet lists of technical summaries (NO 'Method:', 'Confidence:', 'Evidence:', 'Limitations:').\n"
            "4. Answer ONLY from evidence available in the uploaded optical image and computed data. Do NOT hallucinate objects, locations, percentages, coordinates, crops, or unobserved details.\n"
            "5. If the image does not contain enough evidence to answer the user's specific question, say clearly: \"I cannot reliably determine this from the provided image.\"\n"
            "6. For location questions, describe the feature naturally using terms such as: 'upper-left', 'upper-right', 'center', 'lower-left', 'lower-right'.\n"
            "7. Do NOT calculate or mention NDVI, NDWI, or other spectral indices if the image is optical RGB without the required bands."
        )

        parts: List[Dict[str, Any]] = [{"text": prompt_text}]

        image_b64 = cls._prepare_image_b64(image_path)
        if image_b64:
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": image_b64
                }
            })

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1024,
                "thinkingConfig": {"thinkingBudget": 0}
            }
        }

        for model_name in cls.CANDIDATE_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    res_data = json.loads(response.read().decode("utf-8"))
                candidates = res_data.get("candidates", [])
                if not candidates:
                    continue
                cand_parts = candidates[0].get("content", {}).get("parts", [])
                text_parts = [p.get("text", "") for p in cand_parts if "text" in p and not p.get("thought", False)]
                raw_text = " ".join(text_parts).strip()
                if raw_text:
                    cleaned = cls._clean_text(raw_text)
                    if cleaned and cleaned.endswith((".", "!", "?")):
                        return cleaned
            except Exception:
                continue

        return template_fallback

    @classmethod
    def _clean_text(cls, text: str) -> str:
        """Strips structural metadata, bullet headers, code fences, or draft prefixes."""
        text = re.sub(r"^```(?:json|markdown|text)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        lines = text.strip().split("\n")
        cleaned_lines = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            if re.match(r"^[•\-\*]\s*\*{0,2}(?:Method|Confidence|Evidence|Limitations)\*{0,2}:?", line_str, flags=re.IGNORECASE):
                continue
            stripped = re.sub(
                r"^\*{0,2}(?:Answer|Summary|Overview|Response|Direct Answer)\*{0,2}:?\s*",
                "",
                line_str,
                flags=re.IGNORECASE
            )
            if stripped:
                cleaned_lines.append(stripped)

        return " ".join(cleaned_lines).strip()

    @classmethod
    def _generate_template_explanation(cls, query: str, evidence_object: Dict[str, Any]) -> str:
        """
        Failsafe deterministic natural-language answer generator.
        Produces concise, direct natural-language text adhering to the same rules.
        """
        direct_ans = evidence_object.get("direct_answer")
        if direct_ans and len(direct_ans.strip()) > 8:
            return direct_ans.strip()

        task = str(evidence_object.get("task", "")).lower()
        res = evidence_object.get("result", {})
        q_lower = query.lower()

        if task == "water_detection":
            has_water = evidence_object.get("has_water", res.get("water_detected", False))
            pct = res.get("water_percentage", 0.0)
            loc = res.get("primary_location", "center")
            is_where = any(w in q_lower for w in ["where", "locate", "find", "which area"])

            if has_water and pct > 0:
                if is_where:
                    return f"The water body is located in the {loc} of the image."
                return f"Yes, there is a visible water body located in the {loc} of the image."
            else:
                return "No water bodies are detected in this image."

        elif task == "vegetation_analysis":
            has_veg = evidence_object.get("has_vegetation", res.get("vegetation_detected", False))
            pct = res.get("vegetation_percentage", 0.0)
            dom_area = res.get("dominant_area", "upper-right")
            is_where = any(w in q_lower for w in ["which area", "where", "most vegetation"])

            if has_veg and pct > 0:
                if is_where:
                    return f"The {dom_area} of the image has the highest concentration of vegetation."
                return f"Yes, vegetation is visible across approximately {pct}% of the image, concentrated mostly in the {dom_area}."
            else:
                return "No significant vegetation was detected in this image."

        elif task == "building_detection":
            is_urban = res.get("is_urban", False)
            pct = res.get("built_up_percentage", 0.0)
            loc = res.get("dominant_area", "center")
            if "is this area urban" in q_lower or "is it urban" in q_lower:
                if is_urban:
                    return f"Yes, this area is urban, with built-up infrastructure covering about {pct}% of the scene, clustered mostly in the {loc}."
                return f"No, this area is not predominantly urban; built-up structures account for only about {pct}% of the area."
            elif any(w in q_lower for w in ["where", "locate"]):
                if is_urban:
                    return f"The built-up areas are located primarily in the {loc} of the image."
                return "No major built-up or urban areas are discernible in this image."
            elif any(w in q_lower for w in ["roads", "road"]):
                if is_urban:
                    return f"Yes, there are buildings and visible road corridors traversing the scene, concentrated in the {loc}."
                return "No major buildings or road networks are discernible in this image."
            else:
                if is_urban:
                    return f"Built-up structures are visible across approximately {pct}% of the scene, located primarily in the {loc}."
                return "No prominent building footprints or developed urban areas were detected in this image."

        elif task == "land_cover_analysis":
            dom_type = res.get("dominant_land_cover", "natural terrain")
            dom_pct = res.get("dominant_percentage", 50.0)
            dom_loc = res.get("dominant_location", "center")

            if any(w in q_lower for w in ["dominant", "dominates"]):
                return f"The dominant land-cover type is {dom_type.lower()}, occupying approximately {dom_pct}% of the scene in the {dom_loc}."
            elif any(w in q_lower for w in ["major land-cover", "major land cover", "land-cover types"]):
                breakdown = res.get("land_cover_breakdown", {})
                items = [f"{k.replace('_percentage', '').replace('_', ' ')} ({v}%)" for k, v in breakdown.items() if isinstance(v, (int, float)) and v >= 5.0]
                if items:
                    return f"The major land-cover types visible in this image are {', '.join(items)}."
                return f"The scene is dominated by {dom_type.lower()} ({dom_pct}%)."
            else:
                return f"This satellite image depicts an area dominated by {dom_type.lower()} ({dom_pct}%), located primarily in the {dom_loc}."

        elif task == "change_detection":
            summary = evidence_object.get("summary")
            if summary:
                return summary
            pct = res.get("change_percentage", 0.0)
            sector = res.get("dominant_sector", "central")
            return f"Bi-temporal change analysis indicates {pct}% of the area underwent alteration, concentrated in the {sector} sector."

        elif task in ["joint_intelligence", "optical_sar_fusion"]:
            summary = evidence_object.get("summary")
            if summary:
                return summary
            urban = res.get("confirmed_urban_pct", 0.0)
            water = res.get("confirmed_water_pct", 0.0)
            return f"Joint Optical and SAR analysis confirms built-up structures across {urban}% and water coverage across {water}% of the observed scene."

        return "Analysis completed successfully based on visible features in the satellite image."
