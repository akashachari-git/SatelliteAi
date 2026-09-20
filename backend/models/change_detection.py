"""
SatQuery AI - Bi-Temporal Change Detection & Understanding Specialist
Processes corresponding multi-temporal observation pairs (T1 Baseline and T2 Monitoring)
to detect candidate changes, quantify surface difference, and extract spatial change regions.
"""
from typing import Dict, Any, List, Optional
from .base import BaseRemoteSensingModel
from ..app.models.bitemporal_inference import bitemporal_service


class BiTemporalChangeModel(BaseRemoteSensingModel):
    """
    Specialist wrapper delegating to the evidence-grounded BiTemporalChangeService.
    """

    def __init__(self):
        super().__init__(
            "bitemporal-diff-net",
            "Bi-Temporal Difference Specialist (Classical Differencing + BigEarthNet Dual-State Prior)"
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
        Executes genuine bi-temporal difference analysis on uploaded image rasters.
        """
        return bitemporal_service.predict(query=query, inputs=inputs, metadata=metadata)

    def extract_evidence(self, prediction: Dict[str, Any]) -> List[str]:
        return prediction.get("evidence", [])
