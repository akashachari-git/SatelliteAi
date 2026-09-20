"""
SatQuery AI - Remote Sensing Land Cover Adaptation Re-export.
Exposes predict_land_cover_probabilities from backend.app.models.adaptation.
"""
from backend.app.models.adaptation import (
    BIGEARTHNET_19_CLASSES,
    REQUIRED_S2_BANDS,
    ModelStatus,
    get_bigearthnet_status,
    run_deterministic_fallback,
    predict_land_cover_probabilities,
)

__all__ = [
    "BIGEARTHNET_19_CLASSES",
    "REQUIRED_S2_BANDS",
    "ModelStatus",
    "get_bigearthnet_status",
    "run_deterministic_fallback",
    "predict_land_cover_probabilities",
]
