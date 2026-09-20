"""
SatQuery AI - Optical + SAR Cross-Modal Fusion Specialist
Jointly analyzes Optical multispectral reflectance and SAR radar backscatter (VV/VH polarizations),
distinguishing which spectral/physical evidence was derived from Optical vs SAR sensors.
"""
from typing import Dict, Any, List, Optional
from .base import BaseRemoteSensingModel
from ..app.models.optical_sar_inference import optical_sar_service


class OpticalSARFusionModel(BaseRemoteSensingModel):
    """
    Specialist wrapper delegating to the evidence-grounded OpticalSARFusionService.
    """

    def __init__(self):
        super().__init__(
            "optical-sar-fusion-net",
            "Optical + SAR Multimodal Fusion Specialist (Evidence-Based Physical Fusion)"
        )
        self._is_loaded = True

    def load_weights(self, weights_path: Optional[str] = None) -> bool:
        self._weights_path = weights_path
        self._is_loaded = True
        return True

    def predict(
        self,
        query: str,
        inputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes genuine physical multimodal analysis between Optical and SAR inputs.
        """
        return optical_sar_service.predict(query=query, inputs=inputs, metadata=metadata)

    def extract_evidence(self, prediction: Dict[str, Any]) -> List[str]:
        return prediction.get("evidence", [])
