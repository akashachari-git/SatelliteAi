from pathlib import Path
from typing import Dict, Any, List, Optional
from ..remote_sensing.raster_service import RasterMetadataService
from ..remote_sensing.satellite_validator import SatelliteImageValidator

class QueryValidator:
    """
    Validates pre-flight conditions before executing remote-sensing analysis:
    - Image presence and file existence
    - Raster validity and Earth observation authenticity
    - GeoTIFF and spectral band availability
    - Multi-image requirements for change detection
    - Operation support
    """

    @classmethod
    def validate_preconditions(
        cls,
        parsed_intent: Dict[str, Any],
        image_paths: List[str]
    ) -> Dict[str, Any]:
        # 1. Image presence check
        if not image_paths or len(image_paths) == 0:
            return {
                "valid": False,
                "error_type": "NO_IMAGE_LOADED",
                "message": "No satellite image is loaded. Please mount a GeoTIFF or satellite image before initiating analysis."
            }

        # 2. File existence check
        for idx, p in enumerate(image_paths):
            path_obj = Path(p)
            if not path_obj.exists():
                return {
                    "valid": False,
                    "error_type": "FILE_NOT_FOUND",
                    "message": f"Image file not found on server: {path_obj.name}. Please re-upload your satellite raster."
                }

        # 3. Image count check
        required_images = parsed_intent.get("required_image_count", 1)
        intent = parsed_intent.get("intent", "")

        if len(image_paths) < required_images:
            if intent == "change_detection":
                return {
                    "valid": False,
                    "error_type": "INSUFFICIENT_IMAGERY_FOR_CHANGE",
                    "message": (
                        "Change detection strictly requires two temporally co-registered images (T1 earlier and T2 later). "
                        f"Only {len(image_paths)} image was provided. Please mount an image pair to analyze changes."
                    )
                }
            return {
                "valid": False,
                "error_type": "INSUFFICIENT_IMAGES",
                "message": f"The requested operation '{intent}' requires at least {required_images} image(s), but only {len(image_paths)} was provided."
            }

        # 4. Raster metadata and satellite validity inspection
        raster_metas = []
        for idx, p in enumerate(image_paths):
            meta = RasterMetadataService.inspect_raster(p)
            raster_metas.append(meta)

            # Check if authentic satellite image
            is_sat, reason, _ = SatelliteImageValidator.validate_satellite_image(p, meta)
            if not is_sat:
                return {
                    "valid": False,
                    "error_type": "NON_SATELLITE_IMAGE",
                    "message": reason
                }

        # 5. Operation support check
        if intent == "unsupported":
            return {
                "valid": False,
                "error_type": "UNSUPPORTED_OPERATION",
                "message": (
                    "The query could not be routed to a supported remote sensing pipeline. "
                    "You can ask to detect water bodies, locate buildings, analyze vegetation vitality, "
                    "or compare two satellite images."
                )
            }

        return {
            "valid": True,
            "error_type": None,
            "message": "All remote sensing preconditions satisfied.",
            "raster_metadata": raster_metas
        }
