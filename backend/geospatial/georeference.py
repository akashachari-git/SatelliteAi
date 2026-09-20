"""
SatQuery AI - Geospatial Reference & Coordinate Transformation Engine.
Provides rigorous georeferencing, affine transformations, projected-to-WGS84 conversions,
bounding-box reprojection, genuine area estimation, and cursor feature querying.

Strict scientific principles:
- NEVER substitutes a guessed CRS.
- NEVER fabricates coordinates, geographic bounds, or km² area.
- Explicitly marks geospatial status as 'available', 'partial', or 'unavailable'.
"""
import os
import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("satquery.geospatial.georeference")

# Try importing rasterio and pyproj; fall back gracefully if unavailable
try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    CRS = None
    Affine = None

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False


class GeoreferenceEngine:
    """
    Core engine for geospatial coordinate transformation and evidence grounding.
    """

    @classmethod
    def extract_geospatial_metadata(
        cls,
        raster_source: Any
    ) -> Dict[str, Any]:
        """
        Extracts genuine geospatial metadata from a raster file path, dataset, or metadata dictionary.
        Returns a dictionary with:
          - crs: Optional[str]
          - geotransform: Optional[List[float]] (GDAL 6-tuple: [c, a, b, f, d, e])
          - width: int
          - height: int
          - bounds: Optional[Dict[str, float]] (projected bounds)
          - geographic_bounds: Optional[Dict[str, float]] (WGS84 bounds)
          - resolution: Optional[str]
          - gsd_meters: Optional[float]
          - geospatial_status: 'available' | 'partial' | 'unavailable'
          - limitations: List[str]
        """
        crs_str: Optional[str] = None
        gt: Optional[List[float]] = None
        width: int = 0
        height: int = 0
        bounds: Optional[Dict[str, float]] = None
        geo_bounds: Optional[Dict[str, float]] = None
        resolution_str: Optional[str] = None
        gsd_m: Optional[float] = None
        limitations: List[str] = []

        # Case 1: raster_source is a file path
        if isinstance(raster_source, str) and os.path.isfile(raster_source):
            if HAS_RASTERIO:
                try:
                    with rasterio.open(raster_source) as src:
                        width = src.width
                        height = src.height
                        if src.crs:
                            crs_str = src.crs.to_string()
                        if src.transform:
                            # Affine to GDAL 6-tuple [c, a, b, f, d, e]
                            gt = [
                                float(src.transform.c),
                                float(src.transform.a),
                                float(src.transform.b),
                                float(src.transform.f),
                                float(src.transform.d),
                                float(src.transform.e),
                            ]
                            gsd_x = abs(src.transform.a)
                            gsd_y = abs(src.transform.e)
                            gsd_m = round(float((gsd_x + gsd_y) / 2.0), 3)
                            resolution_str = f"{gsd_m}m GSD"
                        b = src.bounds
                        bounds = {
                            "minX": float(b.left),
                            "minY": float(b.bottom),
                            "maxX": float(b.right),
                            "maxY": float(b.top),
                        }
                except Exception as e:
                    logger.debug("rasterio could not read '%s': %s", raster_source, e)

            # Fallback to tifffile if rasterio didn't extract transform
            if gt is None:
                try:
                    import tifffile
                    with tifffile.TiffFile(raster_source) as tif:
                        page = tif.pages[0]
                        width = page.shape[1] if len(page.shape) >= 2 else 0
                        height = page.shape[0] if len(page.shape) >= 2 else 0
                        # Check ModelTiepoint and PixelScale
                        if hasattr(page, "geotiff_tags") and page.geotiff_tags:
                            geo_tags = page.geotiff_tags
                            # Check GeoKeyDirectory or tags
                            pass
                except Exception:
                    pass

        # Case 2: raster_source is a dictionary of metadata
        elif isinstance(raster_source, dict):
            crs_str = raster_source.get("crs")
            gt = raster_source.get("geotransform")
            width = int(raster_source.get("width") or 0)
            height = int(raster_source.get("height") or 0)
            bounds = raster_source.get("bounds")
            resolution_str = raster_source.get("resolution")

            if gt and len(gt) >= 6:
                gsd_x = abs(gt[1])
                gsd_y = abs(gt[5])
                gsd_m = round(float((gsd_x + gsd_y) / 2.0), 3)
                if not resolution_str:
                    resolution_str = f"{gsd_m}m GSD"

        # Determine geospatial status
        has_crs = bool(crs_str and not crs_str.startswith("Local") and crs_str.lower() != "none")
        has_gt = bool(gt and len(gt) >= 6 and (gt[1] != 0 or gt[5] != 0))

        if has_crs and has_gt:
            # Compute geographic bounds in WGS84
            if bounds:
                geo_bounds = cls.projected_bounds_to_wgs84(bounds, crs_str)
            elif width > 0 and height > 0:
                # Compute projected bounds from transform
                c, a, b, f, d, e = gt[:6]
                corners_x = [c, c + width * a, c + width * a + height * b, c + height * b]
                corners_y = [f, f + width * d, f + width * d + height * e, f + height * e]
                bounds = {
                    "minX": float(min(corners_x)),
                    "maxX": float(max(corners_x)),
                    "minY": float(min(corners_y)),
                    "maxY": float(max(corners_y)),
                }
                geo_bounds = cls.projected_bounds_to_wgs84(bounds, crs_str)

            status = "available"
        elif has_gt or has_crs:
            status = "partial"
            if not has_crs:
                limitations.append("Missing CRS: Spatial transform exists but projection is undefined.")
            if not has_gt:
                limitations.append("Missing Affine Transform: CRS is known but pixel-to-world mapping is absent.")
        else:
            status = "unavailable"
            limitations.append("Source raster does not contain valid georeferencing metadata (CRS/Affine).")

        return {
            "crs": crs_str if has_crs else None,
            "geotransform": gt if has_gt else None,
            "width": width,
            "height": height,
            "bounds": bounds,
            "geographic_bounds": geo_bounds,
            "resolution": resolution_str,
            "gsd_meters": gsd_m,
            "geospatial_status": status,
            "limitations": limitations,
        }

    @classmethod
    def pixel_to_projected(
        cls,
        col: float,
        row: float,
        geotransform: Optional[List[float]]
    ) -> Optional[Tuple[float, float]]:
        """
        Converts pixel coordinates (col=X, row=Y) to native projected coordinates (X_proj, Y_proj).
        X_proj = c + col * a + row * b
        Y_proj = f + col * d + row * e
        where geotransform = [c, a, b, f, d, e].
        """
        if not geotransform or len(geotransform) < 6:
            return None
        c, a, b, f, d, e = geotransform[:6]
        x_proj = c + col * a + row * b
        y_proj = f + col * d + row * e
        return (float(x_proj), float(y_proj))

    @classmethod
    def projected_to_wgs84(
        cls,
        x: float,
        y: float,
        crs: Optional[str]
    ) -> Optional[Tuple[float, float]]:
        """
        Converts native projected coordinates (x, y) to WGS84 (latitude, longitude).
        Returns: (latitude, longitude) or None if transformation is not possible.
        """
        if not crs or not HAS_PYPROJ:
            return None

        # Clean CRS string (e.g. 'EPSG:32643 (WGS 84 / UTM zone 43N)' -> 'EPSG:32643')
        crs_clean = crs.split()[0].strip()
        if crs_clean.startswith("Local") or crs_clean.lower() == "none":
            return None

        try:
            # If already WGS84 geographic
            if crs_clean.upper() in ["EPSG:4326", "WGS84", "OGC:CRS84"]:
                # In EPSG:4326, x is longitude, y is latitude
                return (float(y), float(x))

            transformer = pyproj.Transformer.from_crs(crs_clean, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(x, y)
            if math.isnan(lat) or math.isnan(lon) or math.isinf(lat) or math.isinf(lon):
                return None
            return (float(lat), float(lon))
        except Exception as e:
            logger.debug("Coordinate reprojection failed for CRS '%s': %s", crs, e)
            return None

    @classmethod
    def pixel_to_wgs84(
        cls,
        col: float,
        row: float,
        geotransform: Optional[List[float]],
        crs: Optional[str]
    ) -> Optional[Tuple[float, float]]:
        """
        Directly converts pixel coordinates (col, row) to WGS84 (latitude, longitude).
        """
        proj_pt = cls.pixel_to_projected(col, row, geotransform)
        if not proj_pt or not crs:
            return None
        return cls.projected_to_wgs84(proj_pt[0], proj_pt[1], crs)

    @classmethod
    def projected_bounds_to_wgs84(
        cls,
        bounds: Dict[str, float],
        crs: Optional[str]
    ) -> Optional[Dict[str, float]]:
        """
        Converts projected bounding box {'minX', 'minY', 'maxX', 'maxY'} to WGS84
        {'west': min_lon, 'south': min_lat, 'east': max_lon, 'north': max_lat}.
        """
        if not bounds or not crs:
            return None

        min_x, min_y = bounds["minX"], bounds["minY"]
        max_x, max_y = bounds["maxX"], bounds["maxY"]

        # Sample all 4 corners to handle non-axis-aligned or inverted transforms
        corners = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
        ]
        lats: List[float] = []
        lons: List[float] = []
        for x, y in corners:
            pt = cls.projected_to_wgs84(x, y, crs)
            if pt is None:
                return None
            lats.append(pt[0])
            lons.append(pt[1])

        return {
            "west": round(float(min(lons)), 6),
            "south": round(float(min(lats)), 6),
            "east": round(float(max(lons)), 6),
            "north": round(float(max(lats)), 6),
        }

    @classmethod
    def pixel_bbox_to_geographic(
        cls,
        pixel_bbox: Dict[str, float],
        geotransform: Optional[List[float]],
        crs: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Converts a pixel bounding box {'x': min_x, 'y': min_y, 'width': w, 'height': h}
        into both projected bounds and WGS84 geographic bounds.
        Returns:
        {
            "projected_bbox": {"minX": ..., "minY": ..., "maxX": ..., "maxY": ...},
            "geographic_bbox": {"west": ..., "south": ..., "east": ..., "north": ...},
            "center_lat_lon": {"latitude": ..., "longitude": ...}
        }
        or None if georeferencing is unavailable.
        """
        if not geotransform or not crs:
            return None

        x = float(pixel_bbox.get("x", 0))
        y = float(pixel_bbox.get("y", 0))
        w = float(pixel_bbox.get("width", 0))
        h = float(pixel_bbox.get("height", 0))

        # Four corners in pixel space
        corners_px = [
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h),
        ]

        proj_corners = [cls.pixel_to_projected(cx, cy, geotransform) for cx, cy in corners_px]
        if any(pt is None for pt in proj_corners):
            return None

        xs = [pt[0] for pt in proj_corners]  # type: ignore
        ys = [pt[1] for pt in proj_corners]  # type: ignore

        projected_bounds = {
            "minX": round(float(min(xs)), 3),
            "minY": round(float(min(ys)), 3),
            "maxX": round(float(max(xs)), 3),
            "maxY": round(float(max(ys)), 3),
        }

        geo_bounds = cls.projected_bounds_to_wgs84(projected_bounds, crs)
        if not geo_bounds:
            return None

        # Center point
        center_col = x + w / 2.0
        center_row = y + h / 2.0
        center_pt = cls.pixel_to_wgs84(center_col, center_row, geotransform, crs)

        return {
            "projected_bbox": projected_bounds,
            "geographic_bbox": geo_bounds,
            "center_lat_lon": (
                {"latitude": round(center_pt[0], 6), "longitude": round(center_pt[1], 6)}
                if center_pt else None
            ),
        }

    @classmethod
    def calculate_pixel_area(
        cls,
        pixel_count: int,
        geotransform: Optional[List[float]],
        crs: Optional[str],
        center_latitude: Optional[float] = None
    ) -> Optional[Dict[str, float]]:
        """
        Calculates genuine ground area for a given count of pixels.
        Only calculates area when valid geotransform and CRS exist.
        - For projected CRS: calculates using ground sampling dimensions:
            pixel_area_m2 = |a * e - b * d|
        - For geographic CRS (e.g. EPSG:4326): uses cosine-adjusted spherical projection:
            pixel_area_m2 = (|a| * 111320 * cos(lat)) * (|e| * 111320)
        Returns:
            {"area_m2": ..., "area_km2": ..., "area_hectares": ...}
        or None if georeferencing is missing. NEVER fabricates area!
        """
        if pixel_count <= 0 or not geotransform or len(geotransform) < 6 or not crs:
            return None

        crs_clean = crs.split()[0].strip()
        if crs_clean.startswith("Local") or crs_clean.lower() == "none":
            return None

        c, a, b, f, d, e = geotransform[:6]

        # Check if CRS is geographic (degrees)
        is_geographic = False
        if crs_clean.upper() in ["EPSG:4326", "WGS84", "OGC:CRS84"] or "degree" in crs.lower():
            is_geographic = True

        if is_geographic:
            # Lat/lon degrees
            ref_lat = center_latitude if center_latitude is not None else f
            lat_rad = math.radians(ref_lat)
            m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
            m_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)

            pixel_w_m = abs(a) * m_per_deg_lon
            pixel_h_m = abs(e) * m_per_deg_lat
            single_pixel_area_m2 = pixel_w_m * pixel_h_m
        else:
            # Projected CRS: units are meters (e.g. UTM)
            # Area of parallelogram formed by transform basis vectors
            single_pixel_area_m2 = abs(a * e - b * d)

        if single_pixel_area_m2 <= 0:
            return None

        total_area_m2 = float(pixel_count) * single_pixel_area_m2
        total_area_km2 = total_area_m2 / 1_000_000.0
        total_area_ha = total_area_m2 / 10_000.0

        return {
            "area_m2": round(total_area_m2, 2),
            "area_km2": round(total_area_km2, 6),
            "area_hectares": round(total_area_ha, 4),
        }

    @classmethod
    def query_cursor_feature(
        cls,
        col: float,
        row: float,
        metadata: Optional[Dict[str, Any]] = None,
        features: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Exposes backend utility for frontend cursor hover / inspection:
        pixel -> latitude/longitude -> relevant evidence region.
        If no georeference exists, returns explicit unavailable message rather than invented coordinates.
        """
        meta = metadata or {}
        gt = meta.get("geotransform")
        crs = meta.get("crs")

        pt = cls.pixel_to_wgs84(col, row, gt, crs)
        if pt is not None:
            coords = {
                "latitude": round(pt[0], 6),
                "longitude": round(pt[1], 6),
            }
            geo_status = "available"
        else:
            coords = None
            geo_status = "unavailable"

        # Find any intersecting feature or bounding box
        matched_feature = None
        evidence_source = "Pixel Grid"
        if features:
            for feat in features:
                fx = feat.get("x", 0)
                fy = feat.get("y", 0)
                fw = feat.get("width", 0)
                fh = feat.get("height", 0)
                if fx <= col <= (fx + fw) and fy <= row <= (fy + fh):
                    matched_feature = feat.get("label") or feat.get("id")
                    evidence_source = feat.get("description") or "Detected Feature Region"
                    break

        result: Dict[str, Any] = {
            "pixel": {"x": float(col), "y": float(row)},
            "coordinates": coords,
            "feature": matched_feature,
            "evidence_source": evidence_source,
            "geospatial_status": geo_status,
        }

        if geo_status == "unavailable":
            result["message"] = "Geolocation unavailable for this raster"

        return result
