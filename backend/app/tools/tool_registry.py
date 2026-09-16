from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class ToolDefinition:
    tool_id: str
    name: str
    description: str
    supported_modalities: List[str]
    required_image_count: int
    supported_task: str
    input_requirements: Dict[str, Any]
    output_type: str
    confidence_factors: List[str]
    model_identifier: str
    allowed_parameters: Dict[str, Any]
    is_active: bool = True

class ToolRegistry:
    """
    Central Registry for all specialized remote-sensing AI models and tools.
    Supports dynamic agent discovery and routing.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        self.register(ToolDefinition(
            tool_id="RemoteSensingVQA",
            name="Remote Sensing Visual Question Answering",
            description="Answers natural language queries about single optical or multispectral satellite images using BigEarthNet land-cover reasoning.",
            supported_modalities=["Optical", "Multispectral", "SAR"],
            required_image_count=1,
            supported_task="SINGLE_VQA",
            input_requirements={"images": 1, "query": "str"},
            output_type="answer_with_spectral_metrics",
            confidence_factors=["spectral_congruence", "lexical_alignment", "class_margin"],
            model_identifier="satquery-vqa-v1-bigearthnet",
            allowed_parameters={"confidence_threshold": 0.5, "return_evidence": True}
        ))

        self.register(ToolDefinition(
            tool_id="RemoteSensingCaptioner",
            name="Remote Sensing Scene Captioner",
            description="Generates comprehensive, domain-aware descriptive captions of land-use, terrain, and structural elements.",
            supported_modalities=["Optical", "Multispectral", "SAR"],
            required_image_count=1,
            supported_task="CAPTIONING",
            input_requirements={"images": 1},
            output_type="structured_scene_description",
            confidence_factors=["scene_homogeneity", "multilabel_evidence_mass"],
            model_identifier="satquery-caption-vrsbench",
            allowed_parameters={"detail_level": "detailed", "include_metrics": True}
        ))

        self.register(ToolDefinition(
            tool_id="GroundingModel",
            name="Text-Guided Remote Sensing Spatial Grounding",
            description="Localizes and highlights specific spatial objects (water bodies, built-up areas, vegetation, roads) specified in text queries.",
            supported_modalities=["Optical", "Multispectral", "SAR"],
            required_image_count=1,
            supported_task="GROUNDING",
            input_requirements={"images": 1, "query": "str with target entity"},
            output_type="bounding_boxes_and_segmentation_mask",
            confidence_factors=["detection_overlap_iou", "spectral_purity_score"],
            model_identifier="satquery-grounding-v1",
            allowed_parameters={"mask_opacity": 0.45, "box_color": "#06b6d4"}
        ))

        self.register(ToolDefinition(
            tool_id="ChangeDetectionModel",
            name="Bi-Temporal Change Detection Engine",
            description="Computes spatial difference heatmaps, binary change masks, and quantitative change statistics between T1 and T2 images.",
            supported_modalities=["Optical", "Multispectral", "SAR"],
            required_image_count=2,
            supported_task="CHANGE_ANALYSIS",
            input_requirements={"images": 2, "temporal_order": "T1_earlier, T2_later"},
            output_type="change_heatmap_and_mask_with_statistics",
            confidence_factors=["spatial_coregistration_score", "snr_ratio", "otsu_bimodality"],
            model_identifier="satquery-cd-logratio-v1",
            allowed_parameters={"sensitivity": "medium", "filter_speckle": True}
        ))

        self.register(ToolDefinition(
            tool_id="ChangeVQAModel",
            name="Bi-Temporal Change Question Answering",
            description="Answers directional and semantic questions about land-cover transitions between two temporal satellite observations.",
            supported_modalities=["Optical", "Multispectral", "SAR"],
            required_image_count=2,
            supported_task="CHANGE_VQA",
            input_requirements={"images": 2, "query": "str regarding change"},
            output_type="directional_change_answer",
            confidence_factors=["temporal_delta_strength", "transition_consistency"],
            model_identifier="satquery-cdvqa-v1",
            allowed_parameters={"quantify_area": True}
        ))

        self.register(ToolDefinition(
            tool_id="OpticalSARFusion",
            name="Cross-Modal Optical + SAR Joint Analysis Engine",
            description="Fuses optical spectral reflectance with SAR microwave backscatter to detect structural vs dielectric properties.",
            supported_modalities=["Optical", "Multispectral", "SAR"],
            required_image_count=2,
            supported_task="OPTICAL_SAR_ANALYSIS",
            input_requirements={"images": 2, "modalities": ["Optical/MS", "SAR"]},
            output_type="cross_modal_composite_and_insights",
            confidence_factors=["cross_modal_alignment", "backscatter_optical_agreement"],
            model_identifier="satquery-opt-sar-fusion-v1",
            allowed_parameters={"composite_type": "false_color_structural"}
        ))

        self.register(ToolDefinition(
            tool_id="MetadataValidator",
            name="Geospatial Metadata Inspector",
            description="Inspects GeoTIFF tags, dimensions, CRS, bounds, radiometric profiles, and band configurations.",
            supported_modalities=["Optical", "Multispectral", "SAR"],
            required_image_count=1,
            supported_task="METADATA_INSPECTION",
            input_requirements={"images": 1},
            output_type="geospatial_metadata_profile",
            confidence_factors=["tag_completeness", "header_integrity"],
            model_identifier="internal-geotiff-inspector",
            allowed_parameters={}
        ))

        self.register(ToolDefinition(
            tool_id="ImageCompatibilityChecker",
            name="Co-Registration & Pair Compatibility Checker",
            description="Evaluates spatial overlap, CRS projection alignment, native GSD resolution ratio, and modality suitability for multi-image tasks.",
            supported_modalities=["Optical", "Multispectral", "SAR"],
            required_image_count=2,
            supported_task="COMPATIBILITY_CHECK",
            input_requirements={"images": 2},
            output_type="compatibility_scorecard",
            confidence_factors=["geometric_overlap_pct", "resolution_discrepancy"],
            model_identifier="internal-coreg-checker",
            allowed_parameters={}
        ))

    def register(self, tool: ToolDefinition):
        self._tools[tool.tool_id] = tool

    def get_tool(self, tool_id: str) -> Optional[ToolDefinition]:
        return self._tools.get(tool_id)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [asdict(t) for t in self._tools.values()]

    def find_tool_for_task(self, task: str, image_count: int, modalities: List[str]) -> Optional[ToolDefinition]:
        for tool in self._tools.values():
            if tool.supported_task == task and tool.required_image_count <= image_count:
                return tool
        return None

# Singleton instance
tool_registry = ToolRegistry()
