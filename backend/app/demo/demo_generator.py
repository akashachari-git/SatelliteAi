import os
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import tifffile
from PIL import Image, ImageDraw
from ..config import DEMO_DATA_DIR

class DemoDataGenerator:
    """
    Generates genuine GeoTIFF remote-sensing demonstration rasters
    with realistic spatial features, radiometric properties, and geospatial metadata tags.
    """

    @classmethod
    def ensure_demo_files(cls) -> Dict[str, Dict[str, Any]]:
        """
        Creates sample GeoTIFF files if not already present.
        Returns the curated scenario catalog.
        """
        scenarios = {
            "scenario_1_urban": {
                "id": "scenario_1_urban",
                "title": "Scenario 1: Urban Land-Cover VQA",
                "description": "High-resolution optical satellite capture over a metropolitan development zone.",
                "default_query": "Describe the major land-cover types visible in this image.",
                "suggested_queries": [
                    "Describe the major land-cover types visible in this image.",
                    "Are there buildings and impervious surfaces visible?",
                    "What type of land cover dominates this scene?"
                ],
                "image_count": 1,
                "task": "SINGLE_VQA",
                "images": [
                    {
                        "filename": "urban_satellite_eo.tif",
                        "modality": "Optical",
                        "description": "Multispectral RGB optical Earth observation raster (10m GSD)"
                    }
                ]
            },
            "scenario_2_water": {
                "id": "scenario_2_water",
                "title": "Scenario 2: Text-Guided Water Grounding",
                "description": "Satellite imagery featuring a reservoir and river system for spatial localization.",
                "default_query": "Highlight the water body referred to in the query.",
                "suggested_queries": [
                    "Highlight the water body referred to in the query.",
                    "Where is the lake located?",
                    "Is there a river in this image?"
                ],
                "image_count": 1,
                "task": "GROUNDING",
                "images": [
                    {
                        "filename": "water_reservoir_eo.tif",
                        "modality": "Optical",
                        "description": "Optical Earth observation with high NDWI absorption contrast"
                    }
                ]
            },
            "scenario_3_change": {
                "id": "scenario_3_change",
                "title": "Scenario 3: Bi-Temporal Urban Expansion",
                "description": "Temporal pair (2021 Reference vs 2023 Monitoring) capturing rapid suburban infrastructure expansion.",
                "default_query": "What changed between these two dates, and where did the change occur?",
                "suggested_queries": [
                    "What changed between these two dates, and where did the change occur?",
                    "Has the built-up area increased, decreased, or remained unchanged?",
                    "Where did the most significant development take place?"
                ],
                "image_count": 2,
                "task": "CHANGE_ANALYSIS",
                "images": [
                    {
                        "filename": "temporal_t1_2021.tif",
                        "modality": "Optical",
                        "description": "T1 Capture (2021-02-10): Sparse vegetation & undeveloped eastern sector"
                    },
                    {
                        "filename": "temporal_t2_2023.tif",
                        "modality": "Optical",
                        "description": "T2 Capture (2023-11-18): New road networks, commercial fabric in east"
                    }
                ]
            },
            "scenario_4_optical_sar": {
                "id": "scenario_4_optical_sar",
                "title": "Scenario 4: Cross-Modal Optical + SAR Fusion",
                "description": "Co-registered Sentinel-2 Optical and Sentinel-1 C-SAR backscatter for structural & hydrological discernment.",
                "default_query": "Identify built-up and water-covered regions using both images.",
                "suggested_queries": [
                    "Identify built-up and water-covered regions using both images.",
                    "Which regions are more clearly identified using SAR?",
                    "Use the optical and SAR images together to identify urban structures."
                ],
                "image_count": 2,
                "task": "OPTICAL_SAR_ANALYSIS",
                "images": [
                    {
                        "filename": "crossmodal_optical.tif",
                        "modality": "Optical",
                        "description": "Sentinel-2 MSI Optical surface reflectance"
                    },
                    {
                        "filename": "crossmodal_sar_s1.tif",
                        "modality": "SAR",
                        "description": "Sentinel-1 C-SAR microwave backscatter (VV polarization)"
                    }
                ]
            }
        }

        # Generate each raster file with authentic geospatial tags
        cls._create_urban_raster(DEMO_DATA_DIR / "urban_satellite_eo.tif")
        cls._create_water_raster(DEMO_DATA_DIR / "water_reservoir_eo.tif")
        cls._create_temporal_t1(DEMO_DATA_DIR / "temporal_t1_2021.tif")
        cls._create_temporal_t2(DEMO_DATA_DIR / "temporal_t2_2023.tif")
        cls._create_crossmodal_optical(DEMO_DATA_DIR / "crossmodal_optical.tif")
        cls._create_crossmodal_sar(DEMO_DATA_DIR / "crossmodal_sar_s1.tif")

        return scenarios

    @classmethod
    def _save_geotiff(cls, file_path: Path, array: np.ndarray, res_m: float = 10.0, origin_x: float = 432000.0, origin_y: float = 1425000.0):
        """Saves a numpy array as a valid GeoTIFF with ModelPixelScale and ModelTiepoint tags."""
        if file_path.exists():
            return

        h, w = array.shape[:2]
        # Standard GeoTIFF tags
        # Tag 33550: ModelPixelScaleTag (scale_x, scale_y, scale_z)
        # Tag 33922: ModelTiepointTag (i, j, k, x, y, z)
        # Tag 34735: GeoKeyDirectoryTag (UTM Zone 43N / WGS84)
        extratags = [
            (33550, 'd', 3, (res_m, res_m, 0.0), False),
            (33922, 'd', 6, (0.0, 0.0, 0.0, origin_x, origin_y, 0.0), False),
            (34735, 'H', 24, (
                1, 1, 0, 5,
                1024, 0, 1, 1,      # GTModelTypeGeoKey: ModelTypeProjected
                1025, 0, 1, 1,      # GTRasterTypeGeoKey: RasterPixelIsArea
                2048, 0, 1, 4326,   # GeographicTypeGeoKey: WGS 84
                3072, 0, 1, 32643,  # ProjectedCSTypeGeoKey: WGS 84 / UTM zone 43N
                3076, 0, 1, 9001    # ProjLinearUnitsGeoKey: Linear_Meter
            ), False)
        ]

        tifffile.imwrite(
            str(file_path),
            array.astype(np.uint8),
            extratags=extratags,
            photometric='rgb' if array.ndim == 3 else 'minisblack'
        )

    @classmethod
    def _create_urban_raster(cls, path: Path):
        w, h = 512, 512
        arr = np.ones((h, w, 3), dtype=np.uint8) * 120 # Slate gray base
        # Road grid
        for x in range(0, w, 64):
            arr[:, max(0, x-4):min(w, x+4), :] = 45 # Dark asphalt
        for y in range(0, h, 64):
            arr[max(0, y-4):min(h, y+4), :, :] = 45
        # Buildings blocks (bright roofs, albedo)
        for i in range(16, w-32, 64):
            for j in range(16, h-32, 64):
                arr[j:j+36, i:i+36, 0] = np.random.randint(180, 230)
                arr[j:j+36, i:i+36, 1] = np.random.randint(170, 210)
                arr[j:j+36, i:i+36, 2] = np.random.randint(160, 200)
        # Small park patch in corner
        arr[380:480, 380:480, 0] = 35
        arr[380:480, 380:480, 1] = 145
        arr[380:480, 380:480, 2] = 45
        cls._save_geotiff(path, arr)

    @classmethod
    def _create_water_raster(cls, path: Path):
        w, h = 512, 512
        arr = np.ones((h, w, 3), dtype=np.uint8)
        # Surrounding green/brown terrain
        arr[:, :, 0] = 130 # Red
        arr[:, :, 1] = 175 # Green
        arr[:, :, 2] = 95  # Blue
        
        # River / reservoir body curving across center
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        # Meandering curve: |y - (256 + 80*sin(x/60))| < 45
        river_mask = np.abs(y_coords - (256 + 90 * np.sin(x_coords / 65.0))) < 55
        # Lake expansion on the right
        lake_mask = ((x_coords - 380)**2 + (y_coords - 240)**2) < 80**2
        water = river_mask | lake_mask
        
        arr[water, 0] = 15
        arr[water, 1] = 75
        arr[water, 2] = 160
        cls._save_geotiff(path, arr)

    @classmethod
    def _create_temporal_t1(cls, path: Path):
        # 2021 reference: largely green rural landscape, only a few small structures in west
        w, h = 512, 512
        arr = np.ones((h, w, 3), dtype=np.uint8)
        arr[:, :, 0] = 90
        arr[:, :, 1] = 165
        arr[:, :, 2] = 70
        # West small settlement
        arr[100:220, 40:160, 0] = 170
        arr[100:220, 40:160, 1] = 160
        arr[100:220, 40:160, 2] = 150
        # Light dirt path
        arr[150:160, :, :] = 140
        cls._save_geotiff(path, arr)

    @classmethod
    def _create_temporal_t2(cls, path: Path):
        # 2023 monitoring: large new urban construction in EASTERN sector
        w, h = 512, 512
        arr = np.ones((h, w, 3), dtype=np.uint8)
        arr[:, :, 0] = 90
        arr[:, :, 1] = 165
        arr[:, :, 2] = 70
        # Old West settlement
        arr[100:220, 40:160, 0] = 170
        arr[100:220, 40:160, 1] = 160
        arr[100:220, 40:160, 2] = 150
        arr[150:160, :, :] = 140
        # NEW EASTERN URBAN SECTOR (major change)
        arr[80:420, 280:480, 0] = 210 # High albedo concrete/roofing
        arr[80:420, 280:480, 1] = 195
        arr[80:420, 280:480, 2] = 190
        # Highway
        arr[:, 340:355, :] = 40
        cls._save_geotiff(path, arr)

    @classmethod
    def _create_crossmodal_optical(cls, path: Path):
        # Optical Sentinel-2 style: Urban in center, river in south, forest in north
        w, h = 512, 512
        arr = np.ones((h, w, 3), dtype=np.uint8)
        # Forest north
        arr[:180, :, 0] = 30
        arr[:180, :, 1] = 135
        arr[:180, :, 2] = 40
        # Built-up center
        arr[180:360, :, 0] = 175
        arr[180:360, :, 1] = 165
        arr[180:360, :, 2] = 155
        # Water south
        arr[360:, :, 0] = 20
        arr[360:, :, 1] = 70
        arr[360:, :, 2] = 150
        cls._save_geotiff(path, arr)

    @classmethod
    def _create_crossmodal_sar(cls, path: Path):
        # Sentinel-1 C-SAR Grayscale backscatter:
        # Water south: specular reflection -> very dark (backscatter ~ 10-30)
        # Built-up center: dihedral double bounce -> very bright (backscatter ~ 200-255)
        # Forest north: volume scattering -> medium gray with speckle noise (~ 80-130)
        w, h = 512, 512
        arr = np.zeros((h, w), dtype=np.uint8)
        # North forest volume scatter + speckle
        speckle = np.random.rayleigh(scale=25, size=(180, w)).clip(0, 80).astype(np.uint8)
        arr[:180, :] = 85 + speckle
        # Center built-up: very bright double-bounce
        arr[180:360, :] = np.random.randint(190, 255, size=(180, w), dtype=np.uint8)
        # South water: calm dark surface
        arr[360:, :] = np.random.randint(5, 28, size=(h - 360, w), dtype=np.uint8)
        cls._save_geotiff(path, arr)
