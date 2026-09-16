from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

class BaseRemoteSensingModel(ABC):
    """
    Standard interface for all specialist remote sensing AI models
    in SatQuery AI.
    """

    def __init__(self, model_id: str, name: str, version: str = "1.0.0"):
        self.model_id = model_id
        self.name = name
        self.version = version
        self.is_loaded = False
        self.device = "CPU (Optimized)"

    @abstractmethod
    def load(self) -> bool:
        """Load model weights, config, and warm up runtime."""
        pass

    @abstractmethod
    def validate_input(self, inputs: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate whether the inputs meet model domain specifications."""
        pass

    @abstractmethod
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model inference and return structured prediction."""
        pass

    @abstractmethod
    def get_confidence(self, prediction: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> float:
        """Derive calibrated or estimated confidence score [0.0, 1.0]."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Return model provenance, training source, architecture, and capabilities."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "is_loaded": self.is_loaded,
            "device": self.device,
            "adaptation_source": "BigEarthNet-S2 (19-Class CORINE) / Remote Sensing Transfer",
            "domain": "Earth Observation & Geospatial VLM"
        }
