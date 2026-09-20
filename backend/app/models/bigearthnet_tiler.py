"""
SatQuery AI - BigEarthNet Spatial Window & Tile Inference Engine.
Divides large 10-band Sentinel-2 rasters into spatial 120x120 windows,
evaluates each tile through the verified BigEarthNet ResNet-18 ONNX model,
and aggregates tile-level predictions into image-level land-cover evidence.

Scientific Guardrails:
- Does NOT fabricate detections or object bounding boxes (tiles represent
  areal land-cover parcels, not discrete objects).
- Does NOT invent geographic coordinates when CRS/transform is absent.
- Preserves genuine geotransform when present.
- Neutral constant zero-padding for boundary edge tiles (padded pixels are
  not counted as observed area).
- Does NOT alter the verified 10-band contract.
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
    bigearthnet_loader,
)

logger = logging.getLogger("satquery.models.bigearthnet_tiler")


class BigEarthNetTiler:
    """
    Spatial windowing and tile inference engine for BigEarthNet ResNet-18.
    """

    def __init__(
        self,
        tile_size: int = 120,
        stride: int = 120,
        loader=None,
        supporting_threshold: float = 0.15,
    ):
        self.tile_size = tile_size
        self.stride = stride
        self._loader = loader or bigearthnet_loader
        self.supporting_threshold = supporting_threshold

    def generate_tile_windows(
        self,
        height: int,
        width: int,
    ) -> List[Dict[str, Any]]:
        """
        Generates grid of spatial windows across (height, width).
        Handles edge tiles that extend beyond raster bounds.
        """
        windows = []
        row_idx = 0

        for y in range(0, height, self.stride):
            col_idx = 0
            y_min = y
            y_max = min(y + self.tile_size, height)
            tile_h = y_max - y_min

            for x in range(0, width, self.stride):
                x_min = x
                x_max = min(x + self.tile_size, width)
                tile_w = x_max - x_min

                is_padded = (tile_h < self.tile_size) or (tile_w < self.tile_size)
                pad_bottom = self.tile_size - tile_h
                pad_right = self.tile_size - tile_w

                windows.append({
                    "tile_id": f"tile_{row_idx}_{col_idx}",
                    "row": row_idx,
                    "column": col_idx,
                    "pixel_bounds": {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                    },
                    "original_height": tile_h,
                    "original_width": tile_w,
                    "padded_height": self.tile_size,
                    "padded_width": self.tile_size,
                    "is_padded": is_padded,
                    "pad_bottom": pad_bottom,
                    "pad_right": pad_right,
                    "valid_pixel_area": tile_h * tile_w,
                })
                col_idx += 1
            row_idx += 1

        return windows

    def extract_and_pad_tile(
        self,
        raster_10band: np.ndarray,
        window: Dict[str, Any],
    ) -> np.ndarray:
        """
        Extracts a (10, tile_h, tile_w) slice from raster and applies neutral
        zero-padding if tile_h < 120 or tile_w < 120.
        Returns a (10, 120, 120) float32 array.
        """
        pb = window["pixel_bounds"]
        tile_slice = raster_10band[:, pb["y_min"]:pb["y_max"], pb["x_min"]:pb["x_max"]].astype(np.float32)

        if not window["is_padded"]:
            return tile_slice

        # Pad spatial dimensions only: ((0, 0), (0, pad_bottom), (0, pad_right))
        padded = np.pad(
            tile_slice,
            ((0, 0), (0, window["pad_bottom"]), (0, window["pad_right"])),
            mode="constant",
            constant_values=0.0,
        )
        return padded.astype(np.float32)

    def compute_geo_bounds(
        self,
        pixel_bounds: Dict[str, int],
        geotransform: Optional[List[float]],
    ) -> Optional[Dict[str, float]]:
        """
        Converts pixel coordinates to geographic coordinates using affine geotransform:
        X_geo = GT[0] + X_pixel * GT[1] + Y_pixel * GT[2]
        Y_geo = GT[3] + X_pixel * GT[4] + Y_pixel * GT[5]
        If geotransform is absent, returns None. NEVER invents coordinates.
        """
        if not geotransform or len(geotransform) < 6:
            return None

        gt = geotransform
        x_min, y_min = pixel_bounds["x_min"], pixel_bounds["y_min"]
        x_max, y_max = pixel_bounds["x_max"], pixel_bounds["y_max"]

        # Corner coordinates
        geo_x_min = gt[0] + x_min * gt[1] + y_min * gt[2]
        geo_y_min = gt[3] + x_min * gt[4] + y_min * gt[5]
        geo_x_max = gt[0] + x_max * gt[1] + y_max * gt[2]
        geo_y_max = gt[3] + x_max * gt[4] + y_max * gt[5]

        return {
            "minX": round(min(geo_x_min, geo_x_max), 6),
            "maxX": round(max(geo_x_min, geo_x_max), 6),
            "minY": round(min(geo_y_min, geo_y_max), 6),
            "maxY": round(max(geo_y_min, geo_y_max), 6),
        }

    def run_tile_inference(
        self,
        raster_10band: np.ndarray,
        geotransform: Optional[List[float]] = None,
        crs: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Performs full tile-based inference across a (10, H, W) Sentinel-2 raster.

        Returns:
        {
            "tile_count": int,
            "tile_size": int,
            "stride": int,
            "raster_shape": [10, H, W],
            "tiles": [ ... list of tile-level predictions ... ],
            "class_aggregates": { class_name: { ... } },
            "image_probabilities": { class_name: prob },
            "dominant_classes": [ ... ],
            "aggregation_method": "max_tile_probability",
        }
        """
        if raster_10band.ndim != 3 or raster_10band.shape[0] != 10:
            raise ValueError(f"Expected 10-band raster of shape (10, H, W), got {raster_10band.shape}")

        height = raster_10band.shape[1]
        width = raster_10band.shape[2]

        # 1. Generate spatial windows
        windows = self.generate_tile_windows(height, width)
        tile_count = len(windows)
        logger.info("Generated %d spatial tile windows for %dx%d raster.", tile_count, width, height)

        # 2. Extract and pad tiles into batch
        tile_tensors = [self.extract_and_pad_tile(raster_10band, w) for w in windows]
        batch_array = np.stack(tile_tensors, axis=0)  # Shape: (N, 10, 120, 120)

        # 3. Execute ONNX inference on CPU (batch forward pass)
        if not self._loader.is_available:
            self._loader.load_model()

        # Loader run_inference handles ONNX/PyTorch and applies sigmoid -> (N, 19)
        raw_probs = self._loader.run_inference(batch_array)
        if raw_probs.ndim == 1:
            raw_probs = np.expand_dims(raw_probs, axis=0)

        # 4. Construct tile-level predictions
        tiles = []
        for i, win in enumerate(windows):
            probs_vector = [float(np.clip(raw_probs[i, c], 0.0, 1.0)) for c in range(19)]
            tile_probs = {
                cls_name: round(probs_vector[c], 4)
                for c, cls_name in enumerate(BIGEARTHNET_19_CLASSES)
            }
            ranked = sorted(tile_probs.items(), key=lambda x: x[1], reverse=True)
            top_preds = [{"class": k, "probability": v} for k, v in ranked[:5]]

            geo_bounds = self.compute_geo_bounds(win["pixel_bounds"], geotransform)
            geographic_bounds = None
            if geo_bounds and crs:
                try:
                    from backend.geospatial.georeference import GeoreferenceEngine
                    geographic_bounds = GeoreferenceEngine.projected_bounds_to_wgs84(geo_bounds, crs)
                except Exception:
                    pass

            tile_dict = {
                "tile_id": win["tile_id"],
                "row": win["row"],
                "column": win["column"],
                "pixel_bounds": win["pixel_bounds"],
                "original_size": [win["original_height"], win["original_width"]],
                "padded_size": [win["padded_height"], win["padded_width"]],
                "is_padded": win["is_padded"],
                "geo_bounds": geo_bounds,
                "geographic_bounds": geographic_bounds,
                "predictions": top_preds,
                "probabilities": tile_probs,
                "probability_vector": probs_vector,
            }
            tiles.append(tile_dict)

        # 5. Aggregate predictions across all tiles
        class_aggregates: Dict[str, Dict[str, Any]] = {}
        image_probabilities: Dict[str, float] = {}

        for cls_name in BIGEARTHNET_19_CLASSES:
            tile_probs_for_class = [t["probabilities"][cls_name] for t in tiles]
            max_prob = float(np.max(tile_probs_for_class))
            mean_prob = float(np.mean(tile_probs_for_class))
            supporting_tiles = sum(1 for p in tile_probs_for_class if p >= self.supporting_threshold)
            coverage_fraction = float(round(supporting_tiles / tile_count, 4))

            class_aggregates[cls_name] = {
                "class": cls_name,
                "model_probability": round(max_prob, 4),
                "max_probability": round(max_prob, 4),
                "mean_probability": round(mean_prob, 4),
                "supporting_tiles": supporting_tiles,
                "total_tiles": tile_count,
                "coverage_fraction": coverage_fraction,
            }
            # Image-level probability is the maximum probability observed across tiles
            image_probabilities[cls_name] = round(max_prob, 4)

        # Sort dominant classes by max_probability and coverage
        sorted_classes = sorted(
            class_aggregates.values(),
            key=lambda x: (x["max_probability"], x["coverage_fraction"]),
            reverse=True,
        )
        dominant_classes = sorted_classes[:5]

        return {
            "tile_count": tile_count,
            "tile_size": self.tile_size,
            "stride": self.stride,
            "raster_shape": [10, height, width],
            "tiles": tiles,
            "class_aggregates": class_aggregates,
            "image_probabilities": image_probabilities,
            "dominant_classes": dominant_classes,
            "aggregation_method": "max_tile_probability",
            "supporting_threshold": self.supporting_threshold,
            "geospatial_metadata": {
                "crs": crs,
                "has_geotransform": geotransform is not None,
            },
        }


# Global default tiler instance
bigearthnet_tiler = BigEarthNetTiler()
