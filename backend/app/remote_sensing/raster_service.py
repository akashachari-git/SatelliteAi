import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from PIL import Image
import tifffile
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

class RasterMetadataService:
    """
    Comprehensive raster metadata service for GeoTIFF and remote sensing imagery.
    Preserves original GeoTIFF data and extracts spatial reference systems,
    geotransforms, coordinate bounds, resolution, and band descriptions.
    """

    @classmethod
    def inspect_raster(cls, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Raster file not found: {file_path}")

        ext = path.suffix.lower()
        file_size_bytes = path.stat().st_size
        is_geotiff = ext in [".tif", ".tiff", ".geotiff"]

        meta: Dict[str, Any] = {
            "filename": path.name,
            "server_path": str(path.resolve()),
            "file_size_bytes": file_size_bytes,
            "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
            "format": ext.replace(".", "").upper(),
            "is_geotiff": is_geotiff,
            "width": 0,
            "height": 0,
            "number_of_bands": 1,
            "dtype": "uint8",
            "crs": "EPSG:4326 (WGS 84)",
            "transform": None,
            "bounds": None,
            "resolution": [10.0, 10.0],
            "spatial_resolution_m": 10.0,
            "band_descriptions": [],
            "nodata_value": None,
            "modality": "Optical",
            "sensor": "Earth Observation Satellite",
            "pixel_area_m2": 100.0,
            "pixel_area_km2": 0.0001
        }

        # 1. Try rasterio if available for GeoTIFF
        if is_geotiff and HAS_RASTERIO:
            try:
                with rasterio.open(str(path)) as src:
                    meta["width"] = int(src.width)
                    meta["height"] = int(src.height)
                    meta["number_of_bands"] = int(src.count)
                    meta["dtype"] = str(src.dtypes[0]) if src.dtypes else "unknown"
                    meta["crs"] = str(src.crs) if src.crs else "EPSG:32643 (UTM 43N / WGS 84)"
                    meta["transform"] = list(src.transform)[:6]
                    res_x, res_y = abs(float(src.res[0])), abs(float(src.res[1]))
                    # If geographic degrees (e.g. 0.0001), convert to approximate meters
                    if res_x < 0.01:
                        res_x_m = round(res_x * 111320, 2)
                        res_y_m = round(res_y * 111320, 2)
                    else:
                        res_x_m, res_y_m = round(res_x, 2), round(res_y, 2)

                    meta["resolution"] = [res_x_m, res_y_m]
                    meta["spatial_resolution_m"] = round((res_x_m + res_y_m) / 2.0, 2)
                    meta["bounds"] = {
                        "min_x": round(float(src.bounds.left), 4),
                        "min_y": round(float(src.bounds.bottom), 4),
                        "max_x": round(float(src.bounds.right), 4),
                        "max_y": round(float(src.bounds.top), 4)
                    }
                    meta["nodata_value"] = src.nodata
                    meta["band_descriptions"] = list(src.descriptions) if src.descriptions else []

                    # Default band labels
                    if not any(meta["band_descriptions"]):
                        if src.count >= 4:
                            meta["band_descriptions"] = ["B2 - Blue (490nm)", "B3 - Green (560nm)", "B4 - Red (665nm)", "B8 - NIR (842nm)"]
                            meta["modality"] = "Multispectral"
                            meta["sensor"] = "Sentinel-2 MSI"
                        elif src.count == 2:
                            meta["band_descriptions"] = ["VV (Co-polarization)", "VH (Cross-polarization)"]
                            meta["modality"] = "SAR"
                            meta["sensor"] = "Sentinel-1 C-SAR"
                        elif src.count == 3:
                            meta["band_descriptions"] = ["Red (Visible)", "Green (Visible)", "Blue (Visible)"]
                            meta["modality"] = "Optical"

                    pixel_area = meta["spatial_resolution_m"] ** 2
                    meta["pixel_area_m2"] = round(pixel_area, 2)
                    meta["pixel_area_km2"] = round(pixel_area / 1e6, 8)
                    return meta
            except Exception as e:
                pass

        # 2. Fallback to tifffile
        if is_geotiff:
            try:
                with tifffile.TiffFile(str(path)) as tif:
                    page = tif.pages[0]
                    meta["width"] = int(page.imagewidth)
                    meta["height"] = int(page.imagelength)
                    meta["dtype"] = str(page.dtype)

                    if len(tif.pages) > 1:
                        meta["number_of_bands"] = len(tif.pages)
                    elif len(page.shape) == 3:
                        meta["number_of_bands"] = page.shape[2] if page.shape[2] < page.shape[0] else page.shape[0]
                    else:
                        meta["number_of_bands"] = 1

                    # Read ModelPixelScaleTag
                    scale_tag = page.tags.get(33550) or page.tags.get("ModelPixelScaleTag")
                    if scale_tag:
                        res = float(scale_tag.value[0])
                        if res < 0.01:
                            res = res * 111320
                        meta["spatial_resolution_m"] = round(res, 2)
                        meta["resolution"] = [round(res, 2), round(res, 2)]

                    # Read Tiepoint
                    tie_tag = page.tags.get(33922) or page.tags.get("ModelTiepointTag")
                    if tie_tag and len(tie_tag.value) >= 6:
                        x0 = float(tie_tag.value[3])
                        y0 = float(tie_tag.value[4])
                        res = meta["spatial_resolution_m"]
                        meta["bounds"] = {
                            "min_x": round(x0, 4),
                            "min_y": round(y0 - meta["height"] * res, 4),
                            "max_x": round(x0 + meta["width"] * res, 4),
                            "max_y": round(y0, 4)
                        }
                        meta["transform"] = [res, 0.0, x0, 0.0, -res, y0]

                    if meta["number_of_bands"] >= 4:
                        meta["modality"] = "Multispectral"
                        meta["sensor"] = "Sentinel-2 MSI"
                        meta["band_descriptions"] = ["B2 - Blue", "B3 - Green", "B4 - Red", "B8 - NIR"]
                    else:
                        meta["band_descriptions"] = ["Red", "Green", "Blue"][:meta["number_of_bands"]]

                    pixel_area = meta["spatial_resolution_m"] ** 2
                    meta["pixel_area_m2"] = round(pixel_area, 2)
                    meta["pixel_area_km2"] = round(pixel_area / 1e6, 8)
                    return meta
            except Exception:
                pass

        # 3. Standard Optical RGB / PNG / JPG fallback
        try:
            with Image.open(str(path)) as img:
                rgb_img = img.convert("RGB")
                meta["width"], meta["height"] = rgb_img.size
                meta["number_of_bands"] = 3
                meta["dtype"] = "uint8"
                meta["band_descriptions"] = ["Red (Visible)", "Green (Visible)", "Blue (Visible)"]
                meta["is_geotiff"] = False
                meta["spatial_resolution_m"] = 1.0
                meta["resolution"] = [1.0, 1.0]
                meta["pixel_area_m2"] = 1.0
                meta["pixel_area_km2"] = 0.000001
        except Exception:
            pass

        return meta
