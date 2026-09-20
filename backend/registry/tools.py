"""
SatQuery AI - Specialist Model and Tool Registry
Maintains technical capabilities, supported modalities, parameter counts,
and deployment availability for all remote-sensing neural architectures.
"""
from typing import Dict, List, Any, Optional
from ..schemas import ModelInfoSchema

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "rs-vqa-transformer": {
        "id": "rs-vqa-transformer",
        "name": "RS-VQA Dual-Encoder",
        "category": "Visual Question Answering",
        "architecture": "Cross-Attention Swin-Transformer + DeBERTa-v3 with BigEarthNet Prior",
        "modalities": ["Optical RGB", "Multispectral (12-Band)"],
        "supportedInputCount": 1,
        "supportedFormats": ["GeoTIFF", "TIFF", "PNG", "JPEG"],
        "gsdRange": "0.3m – 30.0m",
        "parameters": "380M",
        "inputResolution": "512 × 512 / 1024 × 1024 px",
        "description": "Specialist geospatial VQA model trained on remote-sensing benchmark datasets (RSVQA, EarthVQA). Answers queries regarding infrastructure counts, land use, and spectral characteristics.",
        "status": "Connected",
        "supportedTasks": ["vqa", "land-cover-classification", "feature-verification"],
        "parametersConfig": {
            "temperature": 0.2,
            "maxTokens": 256,
            "confidenceThreshold": 0.65,
        }
    },
    "rs-captioning-engine": {
        "id": "rs-captioning-engine",
        "name": "GeoCaption-Net",
        "category": "Vision-Language Generation",
        "architecture": "CLIP-ViT-Large/14 + Autoregressive Remote Sensing Decoder",
        "modalities": ["Optical RGB", "NIR Multispectral"],
        "supportedInputCount": 1,
        "supportedFormats": ["GeoTIFF", "TIFF", "PNG", "JPEG"],
        "gsdRange": "0.5m – 15.0m",
        "parameters": "450M",
        "inputResolution": "768 × 768 px",
        "description": "Dense scene-level captioning engine generating comprehensive geographic descriptions including land-cover proportions, spatial topography, and visible anthropic structures.",
        "status": "Available",
        "supportedTasks": ["scene-captioning", "synoptic-description", "land-cover-inventory"],
        "parametersConfig": {
            "beamSize": 4,
            "lengthPenalty": 1.2,
            "topP": 0.9,
        }
    },
    "spatial-grounding-det": {
        "id": "spatial-grounding-det",
        "name": "SatGround-DETR",
        "category": "Spatial Localization",
        "architecture": "Deformable DETR with Linguistic Feature Grounding",
        "modalities": ["Optical RGB", "False Color NIR"],
        "supportedInputCount": 1,
        "supportedFormats": ["GeoTIFF", "TIFF", "PNG", "JPEG"],
        "gsdRange": "0.1m – 5.0m",
        "parameters": "285M",
        "inputResolution": "1024 × 1024 px",
        "description": "Text-guided bounding box and polygon detector mapping natural language phrases directly to geographic bounding coordinates and CRS spatial features.",
        "status": "Connected",
        "supportedTasks": ["text-guided-grounding", "target-detection", "spatial-anchoring"],
        "parametersConfig": {
            "iouThreshold": 0.45,
            "nmsThreshold": 0.5,
            "scoreThreshold": 0.70,
        }
    },
    "bitemporal-diff-net": {
        "id": "bitemporal-diff-net",
        "name": "Bi-Temporal Difference Specialist",
        "category": "Bi-Temporal Difference Modeling",
        "architecture": "Normalized Spectral Differencing + Adaptive Thresholding + BigEarthNet Dual-State Prior",
        "modalities": ["Bi-Temporal Optical RGB", "Bi-Temporal Sentinel-2 (10-Band)"],
        "supportedInputCount": 2,
        "supportedFormats": ["GeoTIFF", "TIFF", "PNG", "JPEG"],
        "gsdRange": "0.1m – 30.0m",
        "parameters": "Classical CV + 11.2M (BigEarthNet Prior)",
        "inputResolution": "Native / Multi-scale",
        "description": "Evidence-grounded bi-temporal change analysis executing normalized spectral differencing, adaptive thresholding, morphological region extraction, and optional dual-state BigEarthNet land-cover shift analysis on local CPU.",
        "status": "Connected",
        "supportedTasks": ["change-analysis", "change-based-vqa", "urban-sprawl-monitoring"],
        "parametersConfig": {
            "minRegionSizePx": 16,
            "morphologicalClean": True,
        }
    },
    "optical-sar-fusion-net": {
        "id": "optical-sar-fusion-net",
        "name": "Optical + SAR Multimodal Fusion Specialist",
        "category": "Cross-Modal Physical Fusion",
        "architecture": "Cross-Modal Physical Corroboration (Multispectral Indices + Polarimetric Radar Backscatter Physics)",
        "modalities": ["Optical RGB / Sentinel-2", "SAR (Sentinel-1 C-Band VV/VH)"],
        "supportedInputCount": 2,
        "supportedFormats": ["GeoTIFF", "TIFF", "PNG", "JPEG"],
        "gsdRange": "10.0m – 20.0m",
        "parameters": "Classical Multimodal Physics (No Learned Checkpoint)",
        "inputResolution": "Native / Multi-scale",
        "description": "Evidence-grounded cross-modal analysis executing optical spectral index derivation (NDVI, NDWI, NDBI), calibrated SAR backscatter thresholding (VV/VH, dB), and cross-sensor spatial corroboration.",
        "status": "Connected",
        "supportedTasks": ["optical-sar-analysis", "cloud-penetrating-mapping", "flood-extent-mapping"],
        "parametersConfig": {
            "sarSpecularThresholdDb": -20.0,
            "sarDoubleBounceThresholdDb": -6.0,
        }
    },
    "rs-foundational-vlm": {
        "id": "rs-foundational-vlm",
        "name": "EarthVLM-7B",
        "category": "Foundational Multimodal Model",
        "architecture": "Autoregressive 7B Remote Sensing Foundation Model (LoRA Adapted)",
        "modalities": ["Optical", "SAR", "Multispectral", "Digital Elevation Model"],
        "supportedInputCount": 2,
        "supportedFormats": ["GeoTIFF", "TIFF", "PNG", "JPEG"],
        "gsdRange": "0.1m – 60.0m",
        "parameters": "7.2B",
        "inputResolution": "Multi-scale Tiled 1024 × 1024 px",
        "description": "End-to-end vision-language foundation model designed for zero-shot conversational remote sensing analytics and complex multi-step reasoning.",
        "status": "Planned",
        "supportedTasks": ["vqa", "scene-captioning", "change-analysis", "optical-sar-analysis"],
        "parametersConfig": {
            "quantization": "4-bit",
            "contextWindow": 4096,
        }
    },
    "bigearthnet-classifier": {
        "id": "bigearthnet-classifier",
        "name": "BigEarthNet ResNet-18 Land-Cover Specialist",
        "category": "Land-Cover & Vegetation Classification",
        "architecture": "ResNet-18 (10-Band Sentinel-2 ONNX)",
        "modalities": ["Sentinel-2 Multispectral (10-Band: B02-B12)"],
        "supportedInputCount": 1,
        "supportedFormats": ["GeoTIFF", "TIFF"],
        "gsdRange": "10.0m – 20.0m",
        "parameters": "11.2M",
        "inputResolution": "120 × 120 px",
        "description": "Official BigEarthNet v2.0 multi-label land-cover classification model trained on 10-band Sentinel-2 imagery across 19 Corine Land Cover classes.",
        "status": "Connected",
        "supportedTasks": ["land-cover-classification", "vegetation-classification", "corine-clc-mapping"],
        "parametersConfig": {
            "bands": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
            "classesCount": 19,
            "threshold": 0.5,
        }
    },
    "florence2-vlm": {
        "id": "florence2-vlm",
        "name": "Florence-2 Vision-Language Specialist",
        "category": "Vision-Language Foundation Model",
        "architecture": "DaViT + BART-style Seq2Seq (microsoft/Florence-2-base)",
        "modalities": ["Optical RGB", "Sentinel-2 RGB Composite"],
        "supportedInputCount": 1,
        "supportedFormats": ["PNG", "JPEG", "TIFF", "GeoTIFF"],
        "gsdRange": "0.1m – 30.0m",
        "parameters": "231M",
        "inputResolution": "Dynamic / Multi-scale",
        "description": "Locally hosted Florence-2-base vision-language model executing genuine image captioning, visual question answering, and text-guided phrase grounding on CPU.",
        "status": "Connected",
        "supportedTasks": [
            "image_captioning",
            "visual_question_answering",
            "phrase_grounding",
            "scene-captioning",
            "vqa",
            "text-guided-grounding",
        ],
        "parametersConfig": {
            "device": "cpu",
            "precision": "float32",
            "checkpoint": "backend/models/checkpoints/florence2-base",
        }
    }
}

class ToolRegistry:
    @staticmethod
    def get_all_tools() -> List[ModelInfoSchema]:
        return [
            ModelInfoSchema(
                id=t["id"],
                name=t["name"],
                category=t["category"],
                architecture=t["architecture"],
                modalities=t["modalities"],
                gsdRange=t["gsdRange"],
                parameters=t["parameters"],
                inputResolution=t["inputResolution"],
                description=t["description"],
                status=t["status"],
                supportedTasks=t["supportedTasks"],
            )
            for t in TOOL_REGISTRY.values()
        ]

    @staticmethod
    def get_tool(tool_id: str) -> Optional[Dict[str, Any]]:
        return TOOL_REGISTRY.get(tool_id)

    @staticmethod
    def select_tool_for_task(task_type: str, mode: str) -> Dict[str, Any]:
        if mode == "bi-temporal" or task_type in ["change-analysis", "change-based-vqa"]:
            return TOOL_REGISTRY["bitemporal-diff-net"]
        elif mode == "optical-sar" or task_type == "optical-sar-analysis":
            return TOOL_REGISTRY["optical-sar-fusion-net"]
        elif task_type in ["land-cover-classification", "vegetation-classification", "corine-clc-mapping"]:
            return TOOL_REGISTRY["bigearthnet-classifier"]
        elif task_type in ["text-guided-grounding", "phrase_grounding"]:
            return TOOL_REGISTRY["florence2-vlm"]
        elif task_type in ["scene-captioning", "image_captioning"]:
            return TOOL_REGISTRY["florence2-vlm"]
        elif task_type in ["vqa", "visual_question_answering"]:
            return TOOL_REGISTRY["florence2-vlm"]
        else:
            return TOOL_REGISTRY["florence2-vlm"]
