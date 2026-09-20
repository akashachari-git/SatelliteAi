"""
SatQuery AI - Remote Sensing Land Cover Adaptation & Multi-label Inference Module.
Connects BigEarthNet ResNet-18 model loading with the public land-cover prediction interface.
Preserves deterministic heuristic fallback analysis when model is unavailable or when
imagery is incompatible (e.g. 3-band RGB or unverified band sequences).
"""
import os
import sys
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

# Resolve paths
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from backend.models.bigearthnet_loader import (
    BIGEARTHNET_19_CLASSES,
    REQUIRED_S2_BANDS,
    ModelStatus,
    bigearthnet_loader,
    validate_s2_10band_input,
)

logger = logging.getLogger("satquery.adaptation")


def get_bigearthnet_status() -> Dict[str, Any]:
    """
    Returns current BigEarthNet model availability and operational details.
    Does not disguise unavailable or unmounted models.
    """
    return {
        "status": bigearthnet_loader.status.value,
        "is_available": bigearthnet_loader.is_available,
        "engine": getattr(bigearthnet_loader.engine, "value", str(bigearthnet_loader.engine)),
        "onnx_path": getattr(bigearthnet_loader, "onnx_path", None),
        "checkpoint_path": bigearthnet_loader.checkpoint_path,
        "error_detail": bigearthnet_loader.error_detail,
        "classes_count": len(BIGEARTHNET_19_CLASSES),
        "required_bands": REQUIRED_S2_BANDS,
        "architecture": "BIFOLD-BigEarthNetv2-0/resnet18-s2-v0.2.0 (ConfigILM / ONNX)",
    }



def run_deterministic_fallback(
    image_array: Optional[np.ndarray],
    bands: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    bypass_reason: str = "Deterministic fallback requested",
) -> Dict[str, Any]:
    """
    Deterministic remote-sensing heuristic analysis across the 19 Corine Land Cover classes.
    Preserves existing deterministic baseline without fabricating deep learning predictions.
    Computes calibrated multi-label probabilities in [0.0, 1.0].
    Strictly distinguishes itself from neural network predictions.
    """
    # 1. Base prior probabilities derived from European Corine Land Cover distribution
    # Base frequencies ensure all 19 classes have valid probabilities in [0.0, 1.0]
    base_priors: Dict[str, float] = {
        "Continuous urban fabric": 0.05,
        "Discontinuous urban fabric": 0.12,
        "Industrial or commercial units": 0.06,
        "Arable land": 0.22,
        "Permanent crops": 0.08,
        "Pastures": 0.14,
        "Complex cultivation patterns": 0.10,
        "Land principally occupied by agriculture": 0.11,
        "Broad-leaved forest": 0.18,
        "Coniferous forest": 0.15,
        "Mixed forest": 0.16,
        "Natural grasslands and sclerophyllous vegetation": 0.07,
        "Transitional woodland-shrub": 0.09,
        "Beaches, dunes, sands": 0.03,
        "Inland wetlands": 0.04,
        "Coastal wetlands": 0.02,
        "Inland waters": 0.06,
        "Marine waters": 0.03,
        "Bare rock and sparsely vegetated areas": 0.04,
    }

    raw_scores = dict(base_priors)

    # 2. Extract empirical statistical features from raster data if available
    if image_array is not None and hasattr(image_array, "size") and image_array.size > 0:
        arr = np.asarray(image_array, dtype=np.float32)
        if arr.ndim >= 2:
            # Normalize to 0-1 range for statistical inspection
            arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
            if arr_max > arr_min:
                norm = (arr - arr_min) / (arr_max - arr_min)
            else:
                norm = np.zeros_like(arr)

            mean_val = float(np.nanmean(norm))
            std_val = float(np.nanstd(norm))

            # Channel-specific ratio heuristics
            num_channels = arr.shape[0] if arr.ndim == 3 and arr.shape[0] in (3, 4, 10, 12) else (
                arr.shape[-1] if arr.ndim == 3 else 1
            )

            if num_channels == 3:
                # 3-band RGB analysis (Channels: R=0, G=1, B=2 or last dim)
                r = norm[0] if arr.shape[0] == 3 else norm[..., 0]
                g = norm[1] if arr.shape[0] == 3 else norm[..., 1]
                b = norm[2] if arr.shape[0] == 3 else norm[..., 2]

                mean_r = float(np.nanmean(r))
                mean_g = float(np.nanmean(g))
                mean_b = float(np.nanmean(b))

                # Vegetation index approximation via visible greenness
                vis_veg = (mean_g - mean_r) / (mean_g + mean_r + 1e-6)
                # Water index approximation via blue prominence
                vis_water = (mean_b - mean_r) / (mean_b + mean_r + 1e-6)

                if vis_veg > 0.1:
                    raw_scores["Arable land"] += 0.25 * min(vis_veg * 2.0, 1.0)
                    raw_scores["Broad-leaved forest"] += 0.30 * min(vis_veg * 2.0, 1.0)
                    raw_scores["Pastures"] += 0.20 * min(vis_veg * 2.0, 1.0)
                    raw_scores["Continuous urban fabric"] *= 0.6
                if vis_water > 0.15:
                    raw_scores["Inland waters"] += 0.35 * min(vis_water * 2.5, 1.0)
                    raw_scores["Marine waters"] += 0.20 * min(vis_water * 2.5, 1.0)
                if std_val > 0.22 and mean_r > 0.35:
                    # High spatial variance & bright roofs -> Urban / Industrial
                    raw_scores["Discontinuous urban fabric"] += 0.25 * min(std_val * 2.0, 1.0)
                    raw_scores["Industrial or commercial units"] += 0.20 * min(std_val * 2.0, 1.0)

            elif num_channels >= 10:
                # Spectral band indices if multi-band
                b2 = norm[0] if arr.shape[0] >= 10 else norm[..., 0]  # Blue
                b4 = norm[2] if arr.shape[0] >= 10 else norm[..., 2]  # Red
                b8 = norm[6] if arr.shape[0] >= 10 else norm[..., 6]  # NIR

                ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
                ndwi = (b2 - b8) / (b2 + b8 + 1e-6)

                mean_ndvi = float(np.nanmean(ndvi))
                mean_ndwi = float(np.nanmean(ndwi))

                if mean_ndvi > 0.2:
                    raw_scores["Broad-leaved forest"] += 0.35 * min(mean_ndvi, 1.0)
                    raw_scores["Coniferous forest"] += 0.30 * min(mean_ndvi, 1.0)
                    raw_scores["Arable land"] += 0.25 * min(mean_ndvi, 1.0)
                if mean_ndwi > 0.1:
                    raw_scores["Inland waters"] += 0.40 * min(mean_ndwi * 2.0, 1.0)
                    raw_scores["Inland wetlands"] += 0.25 * min(mean_ndwi * 2.0, 1.0)

    # 3. Constrain each probability strictly to [0.0, 1.0]
    probabilities: Dict[str, float] = {}
    probability_vector: List[float] = []
    for cls_name in BIGEARTHNET_19_CLASSES:
        val = raw_scores.get(cls_name, 0.05)
        # Numerical clamp to valid probability interval [0.0, 1.0]
        clamped = float(np.clip(val, 0.0, 1.0))
        probabilities[cls_name] = round(clamped, 4)
        probability_vector.append(round(clamped, 4))

    # Build sorted ranking of active classes
    ranked = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top_predictions = [{"class": k, "probability": v} for k, v in ranked[:5]]

    return {
        "method": "Deterministic fallback analysis",
        "is_ai_prediction": False,
        "model_status": bigearthnet_loader.status.value,
        "bypass_reason": bypass_reason,
        "classes": BIGEARTHNET_19_CLASSES,
        "probabilities": probabilities,
        "probability_vector": probability_vector,
        "top_predictions": top_predictions,
        "band_count": len(bands) if bands else (image_array.shape[0] if hasattr(image_array, "shape") and len(image_array.shape) == 3 else None),
        "bands": bands,
    }


def predict_land_cover_probabilities(
    image_input: Any,
    bands: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    loader: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Public API interface for remote-sensing multi-label land cover predictions.
    
    Adheres strictly to the BigEarthNet ResNet-18 contract:
    - Target model: BIFOLD-BigEarthNetv2-0/resnet18-s2-v0.2.0
    - Model contract: (B, 10, H, W) -> 19 logits -> external Sigmoid -> 19 probabilities in [0.0, 1.0]
    - Strict input safety:
      Only runs BigEarthNet when verified as 10-band Sentinel-2 imagery.
      Bypasses 3-band RGB imagery to deterministic fallback.
      Bypasses unverified, misordered, or unknown band sequences to deterministic fallback.
      Never synthesizes, duplicates, or pads missing bands.
    - Preserves deterministic heuristic fallback analysis without crashing when model is unavailable.
    - Result clearly distinguishes "BigEarthNet ResNet-18 prediction" from "Deterministic fallback analysis".
    """
    active_loader = loader or bigearthnet_loader

    # Extract raster array and bands if wrapped in dictionary
    img_arr = image_input
    if isinstance(image_input, dict):
        img_arr = image_input.get("array") or image_input.get("data") or image_input.get("image")
        if bands is None:
            bands = image_input.get("bands") or image_input.get("band_names")

    # Step 1: Strict input safety verification
    is_valid_s2, safety_msg = validate_s2_10band_input(img_arr, bands)

    if not is_valid_s2:
        logger.info("BigEarthNet bypassed: %s", safety_msg)
        return run_deterministic_fallback(
            image_array=img_arr,
            bands=bands,
            metadata=metadata,
            bypass_reason=safety_msg,
        )

    # Step 2: Lazy model availability check
    if not active_loader.is_available:
        # Attempt lazy loading on-demand if checkpoint path configured
        success, m_status, detail = active_loader.load_model()
        if not success or not active_loader.is_available:
            logger.info("BigEarthNet model unavailable (%s: %s). Using fallback.", m_status, detail)
            return run_deterministic_fallback(
                image_array=img_arr,
                bands=bands,
                metadata=metadata,
                bypass_reason=f"BigEarthNet ResNet-18 model {m_status.value}: {detail}",
            )

    # Step 3: Neural Model Inference
    try:
        raw_probs = active_loader.run_inference(img_arr)
        # Flatten raw_probs to ensure 1D array of 19 floats in [0, 1]
        flat_probs = np.asarray(raw_probs).flatten()
        probs_list = [float(np.clip(p, 0.0, 1.0)) for p in flat_probs]

        if len(probs_list) != len(BIGEARTHNET_19_CLASSES):
            raise ValueError(
                f"Model output dimension mismatch: expected {len(BIGEARTHNET_19_CLASSES)} classes, got {len(probs_list)}"
            )

        probabilities: Dict[str, float] = {}
        for idx, cls_name in enumerate(BIGEARTHNET_19_CLASSES):
            probabilities[cls_name] = round(probs_list[idx], 4)

        ranked = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_predictions = [{"class": k, "probability": v} for k, v in ranked[:5]]

        return {
            "method": "BigEarthNet ResNet-18 prediction",
            "is_ai_prediction": True,
            "model_name": "BIFOLD-BigEarthNetv2-0/resnet18-s2-v0.2.0",
            "model_status": ModelStatus.AVAILABLE.value,
            "inference_engine": getattr(active_loader.engine, "value", str(active_loader.engine)),
            "classes": BIGEARTHNET_19_CLASSES,
            "probabilities": probabilities,
            "probability_vector": probs_list,
            "top_predictions": top_predictions,
            "band_count": 10,
            "bands": [b.upper() for b in bands] if bands else REQUIRED_S2_BANDS,
        }

    except Exception as err:
        logger.error("Error during BigEarthNet inference: %s. Reverting to fallback.", err)
        return run_deterministic_fallback(
            image_array=img_arr,
            bands=bands,
            metadata=metadata,
            bypass_reason=f"Neural inference exception: {err}",
        )
