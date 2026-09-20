"""
SatQuery AI - Geospatial Reference Module Re-export Shim.
Enables imports from `backend.app.geospatial.georeference` and `backend.geospatial.georeference`.
"""
from backend.geospatial.georeference import (
    GeoreferenceEngine,
    HAS_RASTERIO,
    HAS_PYPROJ,
)

__all__ = ["GeoreferenceEngine", "HAS_RASTERIO", "HAS_PYPROJ"]
