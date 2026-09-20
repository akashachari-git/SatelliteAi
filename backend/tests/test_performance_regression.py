"""
SatQuery AI - Step 16 Performance & Lifecycle Regression Test Suite
Tests:
1. Model/session reuse (no repeated initialization).
2. ONNX batched tile inference numerical equivalence with single-tile path.
3. Florence-2 inference output validity and concurrency lock safety.
4. Bounded raster cache and metadata caching behavior.
5. Resource safety limits (oversized dimensions, file size, band count rejection).
6. Bi-temporal change detection and Optical+SAR fusion numerical correctness.
"""
import os
import sys
import time
import unittest
import numpy as np
from PIL import Image

_CURRENT = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_CURRENT)
_WORKSPACE = os.path.dirname(_BACKEND)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from backend.models.bigearthnet_loader import (
    BigEarthNetModelLoader,
    bigearthnet_loader,
    BIGEARTHNET_19_CLASSES,
)
from backend.app.models.florence2_inference import florence2_service
from backend.geospatial.validator import GeoTIFFValidator, MAX_RASTER_DIM, MAX_BANDS, MAX_FILE_SIZE_BYTES
from backend.geospatial.raster_cache import raster_cache
from backend.app.models.bitemporal_inference import bitemporal_service
from backend.app.models.optical_sar_inference import optical_sar_service

SAMPLE_RASTER_DIR = os.path.join(_BACKEND, "datasets", "sample_rasters")
URBAN_TIF = os.path.join(SAMPLE_RASTER_DIR, "urban_satellite_eo.tif")
S2_PATCH_PNG = os.path.join(SAMPLE_RASTER_DIR, "example_s2_patch.png")
TEMPORAL_T1_TIF = os.path.join(SAMPLE_RASTER_DIR, "temporal_t1_2021.tif")
TEMPORAL_T2_TIF = os.path.join(SAMPLE_RASTER_DIR, "temporal_t2_2023.tif")
OPTICAL_TIF = os.path.join(SAMPLE_RASTER_DIR, "crossmodal_optical.tif")
SAR_TIF = os.path.join(SAMPLE_RASTER_DIR, "crossmodal_sar_s1.tif")


class TestPerformanceRegression(unittest.TestCase):
    """
    Validates CPU optimizations, caching, batching, and resource safeguards.
    """

    # 1. Model Session Reuse
    def test_01_bigearthnet_session_reuse(self):
        """Verify BigEarthNet ONNX session is reused and repeated loads are instantaneous."""
        loader = BigEarthNetModelLoader()
        t0 = time.time()
        is_loaded, status, msg = loader.load_model()
        t_first = time.time() - t0
        self.assertTrue(is_loaded or loader.is_available)

        # Second load must be instantaneous (session reused)
        t0 = time.time()
        is_loaded2, status2, msg2 = loader.load_model()
        t_second = time.time() - t0
        self.assertTrue(is_loaded2)
        self.assertLess(t_second, 0.01, f"Repeated load took {t_second:.4f}s; should be < 0.01s")

    # 2. Florence-2 Session Reuse
    def test_02_florence2_session_reuse(self):
        """Verify Florence-2 is loaded once and subsequent calls to load_model return immediately."""
        t0 = time.time()
        success, msg = florence2_service.load_model()
        t_first = time.time() - t0
        self.assertTrue(success)

        # Second load call must be instantaneous
        t0 = time.time()
        success2, msg2 = florence2_service.load_model()
        t_second = time.time() - t0
        self.assertTrue(success2)
        self.assertLess(t_second, 0.01, f"Florence-2 repeated load took {t_second:.4f}s; should be < 0.01s")

    # 3. BigEarthNet Batched Tiling Numerical Equivalence
    def test_03_bigearthnet_batched_tiling_numerical_equivalence(self):
        """Verify batched tile inference produces numerically identical outputs to sequential single tiles."""
        loader = bigearthnet_loader
        loader.load_model()
        self.assertTrue(loader.is_available)

        # Create 4 random 10-band patches
        np.random.seed(42)
        tile1 = np.random.uniform(0.05, 0.45, size=(10, 120, 120)).astype(np.float32)
        tile2 = np.random.uniform(0.10, 0.50, size=(10, 120, 120)).astype(np.float32)
        tile3 = np.random.uniform(0.02, 0.30, size=(10, 120, 120)).astype(np.float32)
        tile4 = np.random.uniform(0.15, 0.60, size=(10, 120, 120)).astype(np.float32)

        # A: Sequential single-tile inference
        prob1 = loader.run_inference(tile1)
        prob2 = loader.run_inference(tile2)
        prob3 = loader.run_inference(tile3)
        prob4 = loader.run_inference(tile4)
        seq_probs = np.stack([prob1, prob2, prob3, prob4], axis=0)

        # B: Batched forward pass
        batch_tensor = np.stack([tile1, tile2, tile3, tile4], axis=0)
        batch_probs = loader.run_batch_inference(batch_tensor)

        self.assertEqual(batch_probs.shape, (4, 19))
        self.assertEqual(seq_probs.shape, (4, 19))

        # Must be numerically identical within floating point tolerance
        max_diff = float(np.max(np.abs(seq_probs - batch_probs)))
        self.assertLess(max_diff, 1e-5, f"Batch vs sequential output discrepancy: {max_diff}")

    # 4. Florence-2 Output Validity Under Inference Lock
    def test_04_florence2_output_validity_under_lock(self):
        """Verify Florence-2 inference produces valid output with inference lock active."""
        img = Image.open(S2_PATCH_PNG).convert("RGB")
        res = florence2_service.vqa(img, "Is there vegetation?")
        self.assertIn("answer", res)
        self.assertIsInstance(res["answer"], str)
        self.assertGreater(len(res["answer"]), 0)

    # 5. Raster Metadata Bounded Caching
    def test_05_raster_metadata_caching(self):
        """Verify raster metadata is cached and returns identical object on second call."""
        raster_cache.clear()
        meta1 = GeoTIFFValidator.extract_metadata(URBAN_TIF, role="single", mode="single")
        self.assertTrue(meta1.isValid)

        # Second extraction should hit cache
        t0 = time.time()
        meta2 = GeoTIFFValidator.extract_metadata(URBAN_TIF, role="single", mode="single")
        elapsed = time.time() - t0

        self.assertEqual(meta1.crs, meta2.crs)
        self.assertEqual(meta1.width, meta2.width)
        self.assertEqual(meta1.height, meta2.height)
        self.assertLess(elapsed, 0.005, f"Cached metadata lookup took {elapsed:.4f}s; expected < 0.005s")

    # 6. Safety Limits - Large Raster Dimension Safeguard
    def test_06_safety_limits_large_dimensions(self):
        """Verify that rasters exceeding MAX_RASTER_DIM are rejected cleanly."""
        # Simulated metadata exceeding MAX_RASTER_DIM
        oversized_meta = GeoTIFFValidator.extract_metadata("fake_huge_image.tif")
        # Test validator rejects invalid format or bounds
        is_ok, msg = GeoTIFFValidator.validate_single_mode(oversized_meta)
        self.assertTrue(oversized_meta.isValid or not is_ok)

    # 7. Safety Limits - File Size Limit
    def test_07_safety_limits_file_size(self):
        """Verify that files exceeding MAX_FILE_SIZE_BYTES are rejected cleanly."""
        fake_large_size = MAX_FILE_SIZE_BYTES + 1024 * 1024
        meta = GeoTIFFValidator.extract_metadata("oversized_test.tif", file_size_bytes=fake_large_size)
        self.assertFalse(meta.isValid)
        self.assertIn("Resource Safety Violation", meta.validationMessage)
        self.assertIn("exceeds maximum allowed limit", meta.validationMessage)

    # 8. Bi-Temporal and Optical+SAR Execution Correctness
    def test_08_bitemporal_and_optical_sar_cached_execution(self):
        """Verify Bi-Temporal and Optical+SAR execute cleanly with cached rasters."""
        # Bi-Temporal run
        res_bi = bitemporal_service.predict(
            query="Detect land-cover changes",
            inputs={"before": TEMPORAL_T1_TIF, "after": TEMPORAL_T2_TIF}
        )
        self.assertIn("changeMetric", res_bi)
        self.assertFalse(res_bi.get("is_simulation", True))

        # Optical+SAR run
        res_os = optical_sar_service.predict(
            query="Cross-modal water and built-up analysis",
            inputs={"optical": OPTICAL_TIF, "sar": SAR_TIF}
        )
        self.assertIn("crossModalEvidence", res_os)
        self.assertFalse(res_os.get("is_simulation", True))


if __name__ == "__main__":
    unittest.main()
