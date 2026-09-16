from typing import Dict, Any, Optional
from ..remote_sensing.image_processor import ImageProcessor
from ..models.adaptation import get_bigearthnet_adapter

class RemoteSensingCaptionerService:
    """
    Generates domain-aware captions and narrative descriptions of satellite scenes
    benchmarked against VRSBench / RSICD styles.
    """

    def __init__(self):
        self.adapter = get_bigearthnet_adapter()

    def generate_caption(self, image_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from ..remote_sensing.satellite_validator import SatelliteImageValidator
        is_sat, reason, _ = SatelliteImageValidator.validate_satellite_image(image_path, metadata)
        if not is_sat:
            return {
                "caption": reason,
                "short_summary": reason,
                "confidence": 0.0,
                "top_categories": [],
                "spectral_metrics": {},
                "model_used": "SatelliteImageValidator (Rejected)",
                "is_valid": False,
                "error": reason
            }

        arr = ImageProcessor.load_as_array(image_path)
        probs = self.adapter.predict_land_cover_probabilities(arr)
        features = self.adapter.extract_features(arr)

        top_1 = probs[0]
        top_2 = probs[1]
        top_3 = probs[2]

        caption_lines = [
            f"This satellite view mainly features **{top_1['label'].lower()}** alongside areas of **{top_2['label'].lower()}**."
        ]

        if features["mean_ndbi"] > 0.1:
            caption_lines.append("You can see dense clusters of buildings, roads, and urban neighborhoods across the scene.")
        elif features["mean_ndwi"] > 0.0:
            caption_lines.append("There are also visible water bodies or wet areas running through parts of the scene.")
        else:
            caption_lines.append(f"There are also minor patches of {top_3['label'].lower()} visible around the edges.")

        full_caption = " ".join(caption_lines)

        return {
            "caption": full_caption,
            "short_summary": f"A satellite view showing {top_1['label'].lower()} and {top_2['label'].lower()}.",
            "confidence": 0.90,
            "top_categories": probs[:5],
            "spectral_metrics": features,
            "model_used": "RemoteSensingCaptioner (BigEarthNet Aligned)"
        }
