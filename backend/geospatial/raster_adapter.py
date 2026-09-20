"""
SatQuery AI - Sentinel-2 & Multi-band Raster Adapter.
Maps uploaded raster files, arrays, and metadata to the official BigEarthNet
10-band Sentinel-2 spectral contract:
B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12.

Strict scientific guardrails:
- Never synthesizes missing Sentinel-2 bands from RGB.
- Never duplicates RGB channels to create 10 bands.
- Rejects 3-band RGB, single-band, or incompatible channel counts.
- Never assumes band ordering unless metadata explicitly establishes it.
"""
import os
import io
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from backend.models.bigearthnet_loader import (
    REQUIRED_S2_BANDS,
    normalize_band_name,
    validate_s2_10band_input,
)

logger = logging.getLogger("satquery.geospatial.raster_adapter")

CAPABILITY_MESSAGE = (
    "BigEarthNet land-cover analysis requires a compatible 10-band Sentinel-2 raster "
    "containing B02, B03, B04, B05, B06, B07, B08, B8A, B11 and B12."
)


def extract_raster_bands(
    image_source: Any,
    bands: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[np.ndarray], Optional[List[str]], str]:
    """
    Extracts, inspects, and aligns a raster input to the 10-band Sentinel-2 contract.

    Returns:
        (is_compatible, aligned_10band_array, normalized_bands, message)

    If compatible:
        aligned_10band_array has shape (10, H, W) with float32 dtype,
        matching REQUIRED_S2_BANDS in exact order.
    If incompatible:
        aligned_10band_array is None, and message describes the incompatibility.
    """
    detected_bands = list(bands) if bands else None
    raw_array: Optional[np.ndarray] = None

    # 1. Unpack dictionary container if provided
    if isinstance(image_source, dict):
        # Extract metadata and band names if embedded
        if not detected_bands:
            detected_bands = (
                image_source.get("bands")
                or image_source.get("band_names")
                or image_source.get("bandNames")
            )
        # Extract underlying raster data
        if "array" in image_source and image_source["array"] is not None:
            image_source = image_source["array"]
        elif "data" in image_source and image_source["data"] is not None:
            image_source = image_source["data"]
        elif "path" in image_source and image_source["path"] is not None:
            image_source = image_source["path"]
        elif "fileDataUri" in image_source and image_source["fileDataUri"] is not None:
            image_source = image_source["fileDataUri"]
        elif "filename" in image_source and os.path.exists(str(image_source["filename"])):
            image_source = image_source["filename"]

    # 2. Ingest raster array from source type
    if isinstance(image_source, np.ndarray):
        raw_array = image_source

    elif isinstance(image_source, str):
        # Case A: File path
        if os.path.isfile(image_source):
            ext = os.path.splitext(image_source.lower())[1]
            if ext in [".tif", ".tiff", ".geotiff"]:
                try:
                    import tifffile

                    with tifffile.TiffFile(image_source) as tif:
                        raw_array = tif.asarray()
                        # Inspect TIFF page tags for band descriptions if not already specified
                        if not detected_bands and len(tif.pages) >= 10:
                            page_descriptions = [
                                p.description.strip() for p in tif.pages if p.description
                            ]
                            if len(page_descriptions) >= 10:
                                detected_bands = page_descriptions
                except Exception as e:
                    logger.warning("tifffile failed to read '%s': %s", image_source, e)
            else:
                # RGB format (png, jpg, etc.)
                try:
                    from PIL import Image

                    with Image.open(image_source) as img:
                        raw_array = np.array(img)
                        if not detected_bands:
                            detected_bands = ["R", "G", "B"] if img.mode == "RGB" else list(img.mode)
                except Exception as e:
                    return False, None, None, f"Failed to load image from file '{image_source}': {e}"

        # Case B: Data URI or Base64 string
        elif image_source.startswith("data:") or len(image_source) > 100:
            try:
                data_str = image_source
                if "base64," in data_str:
                    data_str = data_str.split("base64,")[1]
                image_bytes = base64.b64decode(data_str)

                # Attempt reading as multi-band TIFF first
                try:
                    import tifffile

                    raw_array = tifffile.imread(io.BytesIO(image_bytes))
                except Exception:
                    from PIL import Image

                    with Image.open(io.BytesIO(image_bytes)) as img:
                        raw_array = np.array(img)
                        if not detected_bands:
                            detected_bands = ["R", "G", "B"] if img.mode == "RGB" else list(img.mode)
            except Exception as e:
                return False, None, None, f"Failed to decode base64 raster input: {e}"

    elif isinstance(image_source, bytes):
        try:
            import tifffile

            raw_array = tifffile.imread(io.BytesIO(image_source))
        except Exception:
            try:
                from PIL import Image

                with Image.open(io.BytesIO(image_source)) as img:
                    raw_array = np.array(img)
                    if not detected_bands:
                        detected_bands = ["R", "G", "B"] if img.mode == "RGB" else list(img.mode)
            except Exception as e:
                return False, None, None, f"Failed to decode bytes raster input: {e}"

    if raw_array is None:
        return False, None, None, f"Could not decode raster array from input. {CAPABILITY_MESSAGE}"

    # 3. Shape inspection & channel axis determination
    shape = raw_array.shape
    if raw_array.ndim == 2:
        return False, None, detected_bands, (
            f"Single-band 2D raster input cannot enter 10-band BigEarthNet model. {CAPABILITY_MESSAGE}"
        )

    # Detect channel position: (C, H, W) vs (H, W, C)
    if raw_array.ndim == 3:
        if shape[0] in [3, 4, 8, 10, 12, 13]:
            # Channels first: (C, H, W)
            channel_count = shape[0]
            channels_first = True
        elif shape[-1] in [3, 4, 8, 10, 12, 13]:
            # Channels last: (H, W, C)
            channel_count = shape[-1]
            channels_first = False
            raw_array = np.transpose(raw_array, (2, 0, 1))
        else:
            channel_count = shape[0]
            channels_first = True
    elif raw_array.ndim == 4:
        # Batch tensor: (B, C, H, W)
        if shape[1] in [3, 4, 8, 10, 12, 13]:
            channel_count = shape[1]
            raw_array = raw_array[0]  # Take first sample for single-raster pipeline
        elif shape[-1] in [3, 4, 8, 10, 12, 13]:
            channel_count = shape[-1]
            raw_array = np.transpose(raw_array[0], (2, 0, 1))
        else:
            channel_count = shape[1]
            raw_array = raw_array[0]
    else:
        return False, None, detected_bands, f"Unsupported array dimensionality: {raw_array.ndim}D"

    # 4. Strict scientific rejection rules
    # Rule 1: Reject 3-band RGB explicitly
    if channel_count == 3:
        return False, None, detected_bands or ["R", "G", "B"], (
            f"3-band RGB imagery is incompatible with BigEarthNet 10-band ResNet-18 model; "
            f"bypassing to deterministic fallback. {CAPABILITY_MESSAGE}"
        )

    # Rule 2: Reject channel counts other than 10 unless explicit Sentinel-2 10-band subset can be mapped
    if channel_count != 10:
        if not detected_bands:
            return False, None, None, (
                f"Incompatible band count: expected 10 Sentinel-2 bands, got {channel_count}; "
                f"bypassing model. {CAPABILITY_MESSAGE}"
            )

    # 5. Band identity verification and alignment
    if not detected_bands:
        # 10-band array with no band metadata cannot be assumed
        return False, None, None, (
            f"Band ordering is unknown or unverified for the {channel_count}-channel raster; "
            f"bypassing BigEarthNet model. {CAPABILITY_MESSAGE}"
        )

    normalized_bands = [normalize_band_name(b) for b in detected_bands]

    # Check if all 10 required bands are present in the detected bands
    missing_bands = [b for b in REQUIRED_S2_BANDS if b not in normalized_bands]
    if missing_bands:
        return False, None, normalized_bands, (
            f"Missing required Sentinel-2 bands: {missing_bands}. "
            f"Detected bands: {normalized_bands}. {CAPABILITY_MESSAGE}"
        )

    # If all 10 required bands are present, slice and reorder channels to exact REQUIRED_S2_BANDS order
    try:
        band_to_idx = {b: idx for idx, b in enumerate(normalized_bands)}
        aligned_slices = [raw_array[band_to_idx[req_b]] for req_b in REQUIRED_S2_BANDS]
        aligned_array = np.stack(aligned_slices, axis=0).astype(np.float32)
    except Exception as reorder_err:
        return False, None, normalized_bands, f"Failed to align raster bands: {reorder_err}"

    # Verify aligned array
    is_valid, msg = validate_s2_10band_input(aligned_array, REQUIRED_S2_BANDS)
    if not is_valid:
        return False, None, REQUIRED_S2_BANDS, msg

    return True, aligned_array, REQUIRED_S2_BANDS, "Verified compatible 10-band Sentinel-2 raster"
