"""
SatQuery AI - Bounded Geospatial Raster & Metadata Cache
Provides bounded LRU caching for raster arrays and metadata.
Guarantees:
- Bounded memory footprint (max entries, eviction of oldest).
- Stable cache keys incorporating file path, modification time, and size.
- Thread-safe access.
- Avoids repeated disk I/O and TIFF header parsing across validation and inference.
"""
import os
import threading
from typing import Dict, Any, Optional, Tuple
import numpy as np


class BoundedRasterCache:
    """
    Thread-safe bounded in-memory cache for parsed rasters and metadata.
    """

    def __init__(self, max_raster_entries: int = 6, max_meta_entries: int = 32):
        self._max_raster_entries = max_raster_entries
        self._max_meta_entries = max_meta_entries
        self._raster_cache: Dict[Tuple[str, float, int], np.ndarray] = {}
        self._meta_cache: Dict[Tuple[str, float, int, str, str], Any] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _compute_key(filepath: str) -> Optional[Tuple[str, float, int]]:
        if not filepath or not isinstance(filepath, str) or not os.path.isfile(filepath):
            return None
        try:
            stat = os.stat(filepath)
            return (os.path.abspath(filepath), stat.st_mtime, stat.st_size)
        except Exception:
            return None

    def get_raster(self, filepath: str) -> Optional[np.ndarray]:
        key = self._compute_key(filepath)
        if not key:
            return None
        with self._lock:
            arr = self._raster_cache.get(key)
            if arr is not None:
                # Move to end (LRU touch)
                del self._raster_cache[key]
                self._raster_cache[key] = arr
                return arr
        return None

    def set_raster(self, filepath: str, arr: np.ndarray) -> None:
        key = self._compute_key(filepath)
        if not key or arr is None:
            return
        with self._lock:
            if key in self._raster_cache:
                del self._raster_cache[key]
            elif len(self._raster_cache) >= self._max_raster_entries:
                # Evict oldest
                oldest_key = next(iter(self._raster_cache))
                del self._raster_cache[oldest_key]
            self._raster_cache[key] = arr

    def get_metadata(self, filepath: str, role: str = "single", mode: str = "single") -> Optional[Any]:
        base_key = self._compute_key(filepath)
        if not base_key:
            return None
        full_key = (base_key[0], base_key[1], base_key[2], role, mode)
        with self._lock:
            meta = self._meta_cache.get(full_key)
            if meta is not None:
                del self._meta_cache[full_key]
                self._meta_cache[full_key] = meta
                return meta
        return None

    def set_metadata(self, filepath: str, meta: Any, role: str = "single", mode: str = "single") -> None:
        base_key = self._compute_key(filepath)
        if not base_key or meta is None:
            return
        full_key = (base_key[0], base_key[1], base_key[2], role, mode)
        with self._lock:
            if full_key in self._meta_cache:
                del self._meta_cache[full_key]
            elif len(self._meta_cache) >= self._max_meta_entries:
                oldest_key = next(iter(self._meta_cache))
                del self._meta_cache[oldest_key]
            self._meta_cache[full_key] = meta

    def clear(self) -> None:
        with self._lock:
            self._raster_cache.clear()
            self._meta_cache.clear()


# Global cache instance
raster_cache = BoundedRasterCache()
