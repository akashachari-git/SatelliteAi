"""
SatQuery AI - BigEarthNet ResNet-18 Dedicated Inference Service.
Connects verified BigEarthNet ResNet-18 10-band ONNX model to the production analysis path.

Key features:
- Cached ONNX Runtime session with CPUExecutionProvider (lazy loading).
- Strict 10-band Sentinel-2 input validation and RGB bypass.
- Spatial resizing to (120, 120) using bilinear interpolation per band if needed.
- Raw logits forward pass with external Sigmoid probability evaluation.
- 19-class Corine Land Cover taxonomy mapping.
- Clear distinction between real neural predictions and deterministic fallback.
"""
import os
import sys
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

# Ensure backend directory is in sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from backend.models.bigearthnet_loader import (
    BIGEARTHNET_19_CLASSES,
    REQUIRED_S2_BANDS,
    CANONICAL_MODEL_ID,
    ModelStatus,
    InferenceEngine,
    bigearthnet_loader,
    validate_s2_10band_input,
)
from backend.geospatial.raster_adapter import extract_raster_bands, CAPABILITY_MESSAGE
from backend.app.models.adaptation import run_deterministic_fallback
from backend.app.models.bigearthnet_tiler import bigearthnet_tiler, BigEarthNetTiler

logger = logging.getLogger("satquery.models.bigearthnet_inference")


def resize_raster_10band(array_10band: np.ndarray, target_size: Tuple[int, int] = (120, 120)) -> np.ndarray:
    """
    Resizes a (10, H, W) float32 raster to (10, target_H, target_W) using bilinear interpolation per band.
    Preserves exact band count, band order, and float32 values.
    """
    if array_10band.shape[1:] == target_size:
        return array_10band.astype(np.float32)

    from PIL import Image

    target_w, target_h = target_size[1], target_size[0]
    resized_channels = []
    for c in range(array_10band.shape[0]):
        band_slice = array_10band[c].astype(np.float32)
        img_band = Image.fromarray(band_slice)
        resized_band = img_band.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
        resized_channels.append(np.array(resized_band, dtype=np.float32))

    return np.stack(resized_channels, axis=0)


class BigEarthNetInferenceService:
    """
    Production-ready BigEarthNet ResNet-18 inference service.
    Caches ONNX Runtime session, validates input, performs genuine spatial
    tile/window inference via BigEarthNetTiler, and formats evidence-grounded predictions.
    """

    def __init__(self, loader=None, tiler=None):
        self._loader = loader or bigearthnet_loader
        self._tiler = tiler or bigearthnet_tiler

    @property
    def is_available(self) -> bool:
        return self._loader.is_available

    @property
    def model_status(self) -> ModelStatus:
        return self._loader.status

    @property
    def engine(self) -> InferenceEngine:
        return self._loader.engine

    def predict(
        self,
        image_source: Any,
        bands: Optional[List[str]] = None,
        query: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes BigEarthNet ResNet-18 inference on compatible Sentinel-2 10-band raster.
        Uses genuine spatial tile/window inference (120x120 px windows) across the raster.

        Returns structured result adhering to the scientific remote sensing contract:
        - Real AI prediction: "BigEarthNet ResNet-18" with tile-level and aggregate probabilities.
        - Fallback: "Deterministic fallback analysis" if input is incompatible or model unmounted.
        """
        # Step 1: Extract and validate 10-band Sentinel-2 raster
        is_compatible, aligned_10band, detected_bands, validation_msg = extract_raster_bands(
            image_source=image_source,
            bands=bands,
            metadata=metadata,
        )

        if not is_compatible or aligned_10band is None:
            logger.info("BigEarthNet bypassed: %s", validation_msg)
            fallback = run_deterministic_fallback(
                image_array=image_source if isinstance(image_source, np.ndarray) else None,
                bands=detected_bands,
                metadata=metadata,
                bypass_reason=validation_msg,
            )
            fallback["answer"] = f"BigEarthNet bypassed: {validation_msg}"
            return fallback

        # Step 2: Ensure model is loaded into memory (cached session)
        if not self._loader.is_available:
            success, status, load_msg = self._loader.load_model()
            if not success or not self._loader.is_available:
                logger.warning("BigEarthNet model unavailable: %s. Using fallback.", load_msg)
                fallback = run_deterministic_fallback(
                    image_array=aligned_10band,
                    bands=REQUIRED_S2_BANDS,
                    metadata=metadata,
                    bypass_reason=f"BigEarthNet ResNet-18 model {status.value}: {load_msg}",
                )
                fallback["answer"] = f"BigEarthNet ResNet-18 model unavailable ({status.value}): {load_msg}"
                return fallback

        # Step 3: Run spatial tile/window inference
        h, w = aligned_10band.shape[1], aligned_10band.shape[2]
        geotransform = metadata.get("geotransform") if metadata else None
        crs = metadata.get("crs") if metadata else None

        try:
            tile_res = self._tiler.run_tile_inference(
                raster_10band=aligned_10band,
                geotransform=geotransform,
                crs=crs,
            )

            tile_count = tile_res["tile_count"]
            tiles = tile_res["tiles"]
            class_aggregates = tile_res["class_aggregates"]
            image_probabilities = tile_res["image_probabilities"]
            dominant_classes = tile_res["dominant_classes"]

            if tile_count == 1 and not tiles[0]["is_padded"]:
                preprocessing_note = "Native 120×120 px single tile; 10-band structure preserved."
            else:
                preprocessing_note = (
                    f"Spatial windowing into {tile_count} tiles (120×120 px, stride {self._tiler.stride} px); "
                    "neutral zero-padding on boundaries; 10-band structure preserved."
                )

            # Generate natural language summary from genuine tile-aggregated model predictions
            top_class_info = dominant_classes[0]
            dominant_class = top_class_info["class"]
            dominant_prob = top_class_info["max_probability"] * 100.0
            dominant_cov = top_class_info["coverage_fraction"] * 100.0
            dominant_supp = top_class_info["supporting_tiles"]

            active_classes_str = ", ".join(
                [
                    f"{p['class']} ({p['max_probability']*100:.1f}%, {p['supporting_tiles']}/{tile_count} tiles)"
                    for p in dominant_classes[:3]
                ]
            )

            if tile_count == 1:
                answer = (
                    f"BigEarthNet ResNet-18 land-cover classification identified '{dominant_class}' "
                    f"as the primary surface class ({dominant_prob:.1f}% probability). "
                    f"Top identified Corine Land Cover classes: {active_classes_str}."
                )
            else:
                answer = (
                    f"BigEarthNet ResNet-18 tile-based land-cover analysis across {tile_count} spatial windows "
                    f"identified '{dominant_class}' as the dominant surface class "
                    f"(max tile probability: {dominant_prob:.1f}%, supporting {dominant_supp}/{tile_count} tiles, "
                    f"{dominant_cov:.1f}% spatial coverage). Top identified classes: {active_classes_str}."
                )

            # Construct structured evidence citations
            evidence = [
                f"Primary land-cover class: {dominant_class} (max probability: {dominant_prob:.1f}%, coverage: {dominant_cov:.1f}% across {dominant_supp}/{tile_count} tiles).",
                f"Top identified classes: {active_classes_str}.",
                f"Spatial windowing: {tile_count} tile(s) across {w}×{h} px raster (120×120 px window, stride {self._tiler.stride} px).",
                f"Aggregation method: Maximum tile probability with spatial coverage fraction across {tile_count} tiles.",
                f"Inference engine: ONNX Runtime CPUExecutionProvider (Model: {CANONICAL_MODEL_ID}).",
                f"Multispectral input: 10 authentic Sentinel-2 bands ({', '.join(REQUIRED_S2_BANDS)}).",
            ]

            probs_vector = [image_probabilities[cls_name] for cls_name in BIGEARTHNET_19_CLASSES]

            return {
                "model": "BigEarthNet ResNet-18",
                "model_source": CANONICAL_MODEL_ID,
                "input_type": "Sentinel-2 10-band",
                "is_ai_prediction": True,
                "model_status": self._loader.status.value,
                "inference_engine": self._loader.engine.value,
                "answer": answer,
                "confidence": float(round(dominant_prob, 1)),
                "predictions": dominant_classes,
                "classes": BIGEARTHNET_19_CLASSES,
                "probabilities": image_probabilities,
                "probability_vector": probs_vector,
                "top_predictions": dominant_classes,
                "tiles": tiles,
                "tile_count": tile_count,
                "tile_size": self._tiler.tile_size,
                "stride": self._tiler.stride,
                "class_aggregates": class_aggregates,
                "evidence": evidence,
                "evidence_metadata": {
                    "model_name": "BigEarthNet ResNet-18",
                    "model_source": CANONICAL_MODEL_ID,
                    "bands_used": REQUIRED_S2_BANDS,
                    "input_shape": [10, h, w],
                    "tile_count": tile_count,
                    "tile_size": self._tiler.tile_size,
                    "stride": self._tiler.stride,
                    "execution_engine": self._loader.engine.value,
                    "preprocessing": preprocessing_note,
                    "aggregation_method": "max_tile_probability",
                    "total_classes": len(BIGEARTHNET_19_CLASSES),
                },
                "band_count": 10,
                "bands": REQUIRED_S2_BANDS,
            }

        except Exception as err:
            logger.error("Error during BigEarthNet tile inference: %s. Reverting to fallback.", err)
            return run_deterministic_fallback(
                image_array=aligned_10band,
                bands=REQUIRED_S2_BANDS,
                metadata=metadata,
                bypass_reason=f"Neural inference exception: {err}",
            )


# Global singleton instance
bigearthnet_service = BigEarthNetInferenceService()
