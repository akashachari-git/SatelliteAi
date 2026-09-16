import numpy as np
from typing import Dict, Any, Tuple, Optional

class SpectralIndicesCalculator:
    """
    Computes standard earth observation indices for optical and SAR imagery.
    """

    @staticmethod
    def has_multispectral_bands(image_arr: np.ndarray, min_bands: int = 4) -> bool:
        """Checks if the array contains authentic multispectral bands (at least 4 bands)."""
        return image_arr is not None and image_arr.ndim == 3 and image_arr.shape[2] >= min_bands

    @staticmethod
    def calculate_ndvi(image_arr: np.ndarray, red_band: int = 2, nir_band: int = 3) -> Optional[np.ndarray]:
        """
        Normalized Difference Vegetation Index = (NIR - Red) / (NIR + Red)
        STRICT: Only calculated when true multispectral bands (specifically NIR) are available.
        Returns None for standard 3-band RGB or single-band imagery.
        """
        if not SpectralIndicesCalculator.has_multispectral_bands(image_arr, min_bands=4):
            return None

        # Verify bands within range
        max_band = max(red_band, nir_band)
        if image_arr.shape[2] <= max_band:
            red_band, nir_band = 0, 1

        red = image_arr[:, :, red_band].astype(np.float32)
        nir = image_arr[:, :, nir_band].astype(np.float32)

        denom = nir + red
        denom[denom == 0] = 1e-6
        ndvi = (nir - red) / denom
        return np.clip(ndvi, -1.0, 1.0)

    @staticmethod
    def calculate_ndwi(image_arr: np.ndarray, green_band: int = 1, nir_band: int = 3) -> Optional[np.ndarray]:
        """
        Normalized Difference Water Index = (Green - NIR) / (Green + NIR)
        STRICT: Only calculated when true multispectral bands (Green and NIR) are available.
        Returns None for standard 3-band RGB imagery.
        """
        if not SpectralIndicesCalculator.has_multispectral_bands(image_arr, min_bands=4):
            return None

        green = image_arr[:, :, green_band].astype(np.float32)
        nir = image_arr[:, :, nir_band].astype(np.float32)

        denom = green + nir
        denom[denom == 0] = 1e-6
        ndwi = (green - nir) / denom
        return np.clip(ndwi, -1.0, 1.0)

    @staticmethod
    def calculate_ndbi(image_arr: np.ndarray, swir_band: int = 4, nir_band: int = 3) -> Optional[np.ndarray]:
        """
        Normalized Difference Built-up Index (NDBI) = (SWIR - NIR) / (SWIR + NIR)
        STRICT: Only calculated when SWIR and NIR multispectral bands are available.
        Returns None for standard 3-band RGB imagery.
        """
        if not SpectralIndicesCalculator.has_multispectral_bands(image_arr, min_bands=5):
            return None

        swir = image_arr[:, :, swir_band].astype(np.float32)
        nir = image_arr[:, :, nir_band].astype(np.float32)

        denom = swir + nir
        denom[denom == 0] = 1e-6
        ndbi = (swir - nir) / denom
        return np.clip(ndbi, -1.0, 1.0)

    @staticmethod
    def calculate_sar_dielectric_profile(sar_arr: np.ndarray) -> Dict[str, Any]:
        """
        Extracts backscatter statistics from SAR imagery:
        - Water/smooth surface: specular reflection -> very low backscatter (dark)
        - Urban/structures: double-bounce -> very high backscatter (bright)
        - Vegetation: volume scattering -> medium backscatter
        """
        if sar_arr.ndim == 3:
            gray = sar_arr[:, :, 0].astype(np.float32)
        else:
            gray = sar_arr.astype(np.float32)

        mean_db = float(np.mean(gray))
        std_db = float(np.std(gray))
        p10 = float(np.percentile(gray, 10))
        p90 = float(np.percentile(gray, 90))

        # Classify backscatter regimes
        water_mask = gray < (mean_db - 0.6 * std_db)
        urban_mask = gray > (mean_db + 0.8 * std_db)
        veg_mask = ~(water_mask | urban_mask)

        return {
            "mean_backscatter": round(mean_db, 2),
            "std_backscatter": round(std_db, 2),
            "water_surface_pct": round(float(np.mean(water_mask) * 100), 2),
            "urban_double_bounce_pct": round(float(np.mean(urban_mask) * 100), 2),
            "volume_scattering_pct": round(float(np.mean(veg_mask) * 100), 2),
            "water_mask": water_mask,
            "urban_mask": urban_mask
        }
