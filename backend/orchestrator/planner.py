"""
SatQuery AI - Explicit Agent Planning & Multi-Specialist Controller.
Defines the internal planning representation, intent analysis, input-aware
validation, and dependency ordering for remote-sensing specialist execution.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import re


@dataclass
class AgentPlan:
    """
    Internal planning representation for autonomous remote-sensing analysis.
    Explicitly tracks intents, input requirements, selected tools,
    execution dependencies, expected evidence types, and operational limitations.
    """
    query: str
    intents: List[str]
    primary_intent: str
    required_modalities: List[str]
    required_images: int
    selected_tools: List[str]
    execution_order: List[str]
    dependencies: Dict[str, List[str]]
    expected_evidence: List[str]
    limitations: List[str] = field(default_factory=list)
    is_multi_specialist: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgentPlanner:
    """
    Deterministic, testable agent planner that performs:
    1. Structured linguistic intent analysis
    2. Input feasibility inspection (count, bands, dimensions, CRS)
    3. Multi-specialist selection and dependency graph construction
    4. Input-aware constraint identification
    """

    @staticmethod
    def analyze_intents(query: str, mode: str) -> List[str]:
        """
        Extracts all canonical intents from user query and explicit mode.
        Supported canonical intents:
        - 'optical-sar': Cross-sensor optical and radar fusion
        - 'bi-temporal': Temporal difference & change detection
        - 'change-vqa': Temporal question answering
        - 'land-cover': Land-cover classification (Corine CLC / BigEarthNet)
        - 'grounding': Spatial localization / bounding box detection
        - 'caption': Synoptic scene description / overview
        - 'vqa': Visual question answering
        """
        q_lower = query.lower()
        intents: List[str] = []

        # 1. Optical-SAR Multimodal Intent
        if mode == "optical-sar" or any(term in q_lower for term in [
            "optical and sar", "optical and radar", "what does sar reveal",
            "optical sar pair", "combine optical and sar", "sar vs optical",
            "optical vs sar", "cross-sensor", "cross-modal", "optical + sar",
            "radar and optical", "radar imagery", "optical radar"
        ]):
            intents.append("optical-sar")

        # 2. Bi-Temporal Change Intent
        if mode == "bi-temporal" or any(term in q_lower for term in [
            "change", "changed", "compare", "past and present", "different between",
            "construction occurred", "vegetation changed", "water increased", "water decreased",
            "t1", "before", "temporal", "increase", "decrease"
        ]):
            if any(term in q_lower for term in ["is", "has", "did", "how much", "increase", "decrease", "where has"]) and not any(term in q_lower for term in ["what changed", "compare"]):
                intents.append("change-vqa")
            else:
                intents.append("bi-temporal")

        # 3. Land Cover Classification Intent
        if any(term in q_lower for term in [
            "land cover", "land-cover", "vegetation class", "corine", "clc",
            "bigearthnet", "scene classification", "classify", "classes are present",
            "land-cover classes", "land cover classes", "identify the land-cover",
            "identify the land cover"
        ]):
            intents.append("land-cover")

        # 4. Spatial Grounding Intent
        if any(term in q_lower for term in [
            "highlight", "locate", "where is", "where are", "find", "show me",
            "bounding", "ground", "box", "detect the"
        ]):
            intents.append("grounding")

        # 5. Scene Captioning Intent
        if any(term in q_lower for term in [
            "describe", "caption", "overview", "synopsis", "summarize", "inventory",
            "what is shown", "scene description", "what does this satellite image contain",
            "what does this image contain", "what does the image show"
        ]):
            intents.append("caption")

        # 6. General VQA (if no other single-image intent detected, or explicitly asks a question)
        if not intents and mode == "single":
            intents.append("vqa")
        elif "vqa" not in intents and any(term in q_lower for term in ["what", "how many", "is there", "are there", "why", "which"]) and not any(i in ["caption", "grounding", "land-cover", "bi-temporal", "optical-sar"] for i in intents):
            intents.append("vqa")

        return intents if intents else ["vqa"]

    @classmethod
    def create_plan(
        cls,
        query: str,
        mode: str,
        image_count: int,
        images_dict: Dict[str, Any],
        is_s2_10band: bool = False,
        s2_msg: str = "",
        metadata_map: Optional[Dict[str, Any]] = None
    ) -> AgentPlan:
        """
        Creates an explicit, validated execution plan.
        Raises ValueError if the requested intent is physically impossible
        given the available image inputs.
        """
        intents = cls.analyze_intents(query, mode)
        limitations: List[str] = []
        expected_evidence: List[str] = []
        selected_tools: List[str] = []
        execution_order: List[str] = []
        dependencies: Dict[str, List[str]] = {}

        # 1. Physical Feasibility & Input Validation Gatekeeping
        if "optical-sar" in intents:
            if image_count < 2:
                raise ValueError(
                    "Input Validation Error: Optical + SAR analysis requires 2 observations "
                    "(1 Optical multispectral and 1 SAR radar observation). "
                    f"Only {image_count} image was provided."
                )
            opt_present = images_dict.get("optical") is not None
            sar_present = images_dict.get("sar") is not None
            if not (opt_present and sar_present) and mode != "optical-sar":
                # Check if 2 images exist in other keys
                keys = list(images_dict.keys())
                if len(keys) < 2:
                    raise ValueError(
                        "Input Validation Error: Optical + SAR analysis requires 1 Optical and 1 SAR radar observation."
                    )

            primary_intent = "optical-sar"
            required_modalities = ["Optical RGB / Multispectral", "SAR C-Band VV/VH"]
            required_images = 2
            selected_tools = ["optical-sar-fusion-net"]
            execution_order = ["optical-sar-fusion-net"]
            dependencies["optical-sar-fusion-net"] = []
            expected_evidence = [
                "optical_spectral_indices",
                "sar_calibrated_backscatter_db",
                "corroborated_water_mask",
                "corroborated_built_up_mask"
            ]
            return AgentPlan(
                query=query,
                intents=intents,
                primary_intent=primary_intent,
                required_modalities=required_modalities,
                required_images=required_images,
                selected_tools=selected_tools,
                execution_order=execution_order,
                dependencies=dependencies,
                expected_evidence=expected_evidence,
                limitations=limitations,
                is_multi_specialist=False
            )

        if "bi-temporal" in intents or "change-vqa" in intents:
            if image_count < 2:
                raise ValueError(
                    "Input Validation Error: Bi-temporal change analysis requires 2 observations "
                    "(T1 baseline and T2 monitoring). "
                    f"Only {image_count} image was provided."
                )
            primary_intent = "change-vqa" if "change-vqa" in intents else "bi-temporal"
            required_modalities = ["Bi-Temporal Optical / Multispectral"]
            required_images = 2
            selected_tools = ["bitemporal-diff-net"]
            execution_order = ["bitemporal-diff-net"]
            dependencies["bitemporal-diff-net"] = []
            expected_evidence = [
                "spectral_difference_map",
                "change_percentage_metric",
                "labeled_change_bounding_regions"
            ]
            if is_s2_10band:
                expected_evidence.append("bigearthnet_dual_state_land_cover_shifts")
            else:
                limitations.append("Native 10-band Sentinel-2 data absent; BigEarthNet land-cover prior unavailable.")

            return AgentPlan(
                query=query,
                intents=intents,
                primary_intent=primary_intent,
                required_modalities=required_modalities,
                required_images=required_images,
                selected_tools=selected_tools,
                execution_order=execution_order,
                dependencies=dependencies,
                expected_evidence=expected_evidence,
                limitations=limitations,
                is_multi_specialist=False
            )

        # 2. Single-Image / Multi-Specialist Intent Routing
        required_images = 1
        required_modalities = ["Optical RGB"]

        has_caption = "caption" in intents
        has_land_cover = "land-cover" in intents
        has_grounding = "grounding" in intents
        has_vqa = "vqa" in intents

        # Multi-Specialist Case 1: Caption + Land Cover
        if has_caption and has_land_cover:
            primary_intent = "scene-captioning"
            selected_tools = ["florence2-vlm", "bigearthnet-classifier"]
            execution_order = ["florence2-vlm", "bigearthnet-classifier"]
            dependencies["florence2-vlm"] = []
            dependencies["bigearthnet-classifier"] = []
            expected_evidence = ["florence2_scene_caption", "corine_clc_19_class_probabilities"]
            if not is_s2_10band:
                limitations.append(
                    "Native multispectral NIR/SWIR bands absent; BigEarthNet 10-band neural classification "
                    "unavailable for 3-channel RGB (using classical statistical baseline)."
                )

            return AgentPlan(
                query=query,
                intents=intents,
                primary_intent=primary_intent,
                required_modalities=required_modalities,
                required_images=required_images,
                selected_tools=selected_tools,
                execution_order=execution_order,
                dependencies=dependencies,
                expected_evidence=expected_evidence,
                limitations=limitations,
                is_multi_specialist=True
            )

        # Multi-Specialist Case 2: Grounding with 10-Band Land-Cover Support
        if has_grounding and is_s2_10band:
            primary_intent = "text-guided-grounding"
            selected_tools = ["florence2-vlm", "bigearthnet-classifier"]
            execution_order = ["florence2-vlm", "bigearthnet-classifier"]
            dependencies["florence2-vlm"] = []
            dependencies["bigearthnet-classifier"] = []
            expected_evidence = ["grounded_bounding_boxes", "supporting_land_cover_probabilities"]

            return AgentPlan(
                query=query,
                intents=intents,
                primary_intent=primary_intent,
                required_modalities=required_modalities,
                required_images=required_images,
                selected_tools=selected_tools,
                execution_order=execution_order,
                dependencies=dependencies,
                expected_evidence=expected_evidence,
                limitations=limitations,
                is_multi_specialist=True
            )

        # Single Specialist: Land Cover
        if has_land_cover:
            primary_intent = "land-cover-classification"
            selected_tools = ["bigearthnet-classifier"]
            execution_order = ["bigearthnet-classifier"]
            dependencies["bigearthnet-classifier"] = []
            expected_evidence = ["corine_clc_19_class_probabilities"]
            if not is_s2_10band:
                limitations.append(
                    f"Incompatible raster for BigEarthNet ResNet-18: {s2_msg or 'Requires 10-band Sentinel-2'}."
                )
            return AgentPlan(
                query=query,
                intents=intents,
                primary_intent=primary_intent,
                required_modalities=required_modalities,
                required_images=required_images,
                selected_tools=selected_tools,
                execution_order=execution_order,
                dependencies=dependencies,
                expected_evidence=expected_evidence,
                limitations=limitations,
                is_multi_specialist=False
            )

        # Single Specialist: Grounding
        if has_grounding:
            primary_intent = "text-guided-grounding"
            selected_tools = ["florence2-vlm"]
            execution_order = ["florence2-vlm"]
            dependencies["florence2-vlm"] = []
            expected_evidence = ["grounded_bounding_boxes"]
            return AgentPlan(
                query=query,
                intents=intents,
                primary_intent=primary_intent,
                required_modalities=required_modalities,
                required_images=required_images,
                selected_tools=selected_tools,
                execution_order=execution_order,
                dependencies=dependencies,
                expected_evidence=expected_evidence,
                limitations=limitations,
                is_multi_specialist=False
            )

        # Single Specialist: Caption
        if has_caption:
            primary_intent = "scene-captioning"
            selected_tools = ["florence2-vlm"]
            execution_order = ["florence2-vlm"]
            dependencies["florence2-vlm"] = []
            expected_evidence = ["florence2_scene_caption"]
            return AgentPlan(
                query=query,
                intents=intents,
                primary_intent=primary_intent,
                required_modalities=required_modalities,
                required_images=required_images,
                selected_tools=selected_tools,
                execution_order=execution_order,
                dependencies=dependencies,
                expected_evidence=expected_evidence,
                limitations=limitations,
                is_multi_specialist=False
            )

        # Default Single Specialist: VQA
        primary_intent = "vqa"
        selected_tools = ["florence2-vlm"]
        execution_order = ["florence2-vlm"]
        dependencies["florence2-vlm"] = []
        expected_evidence = ["florence2_vqa_answer"]

        # Check for unanswerable / non-geospatial query intent
        non_rs_indicators = ["stock price", "who is the president", "who is the mayor", "lottery", "crypto", "tomorrow weather"]
        if any(term in query.lower() for term in non_rs_indicators):
            limitations.append(
                "Query contains out-of-domain non-geospatial terms that cannot be resolved from Earth observation imagery."
            )

        return AgentPlan(
            query=query,
            intents=intents,
            primary_intent=primary_intent,
            required_modalities=required_modalities,
            required_images=required_images,
            selected_tools=selected_tools,
            execution_order=execution_order,
            dependencies=dependencies,
            expected_evidence=expected_evidence,
            limitations=limitations,
            is_multi_specialist=False
        )
