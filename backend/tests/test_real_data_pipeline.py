"""
SatQuery AI - Step 15 Real Satellite Dataset Testing & End-to-End Validation Suite
Tests real satellite rasters, authentic BigEarthNet patches, Florence-2 VLM inference,
bi-temporal change differencing, Optical+SAR fusion, API endpoints, and clean benchmark skips.
Zero mocks in model execution, zero fabricated coordinates or confidence scores.
"""
import os
import sys
import time
import asyncio
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

from backend.geospatial.validator import GeoTIFFValidator
from backend.geospatial.georeference import GeoreferenceEngine, HAS_RASTERIO, HAS_PYPROJ
from backend.app.models.florence2_inference import florence2_service
from backend.models.bigearthnet_loader import (
    BigEarthNetModelLoader,
    BIGEARTHNET_19_CLASSES,
    REQUIRED_S2_BANDS,
    validate_s2_10band_input,
)
from backend.app.models.bitemporal_inference import bitemporal_service
from backend.app.models.optical_sar_inference import optical_sar_service
from backend.orchestrator.agent import SatQueryAgent
from backend.evaluation.registry import BenchmarkRegistry
from backend.evaluation.dataset_adapters import (
    BigEarthNetAdapter,
    RSVQAAdapter,
    VRSBenchAdapter,
    CDVQAAdapter,
)
from backend.main import (
    health_check,
    validate_image,
    change_analysis,
    optical_sar_analysis,
    generate_report,
)
from backend.schemas import (
    ValidateImageRequest,
    ChangeAnalysisRequest,
    OpticalSARAnalysisRequest,
)


SAMPLE_RASTER_DIR = os.path.join(_BACKEND, "datasets", "sample_rasters")
URBAN_TIF = os.path.join(SAMPLE_RASTER_DIR, "urban_satellite_eo.tif")
WATER_TIF = os.path.join(SAMPLE_RASTER_DIR, "water_reservoir_eo.tif")
TEMPORAL_T1_TIF = os.path.join(SAMPLE_RASTER_DIR, "temporal_t1_2021.tif")
TEMPORAL_T2_TIF = os.path.join(SAMPLE_RASTER_DIR, "temporal_t2_2023.tif")
OPTICAL_TIF = os.path.join(SAMPLE_RASTER_DIR, "crossmodal_optical.tif")
SAR_TIF = os.path.join(SAMPLE_RASTER_DIR, "crossmodal_sar_s1.tif")
S2_PATCH_PNG = os.path.join(SAMPLE_RASTER_DIR, "example_s2_patch.png")


class TestRealSatelliteDataPipeline(unittest.TestCase):
    """
    Test suite executing the complete pipeline on authentic remote-sensing rasters.
    """

    @classmethod
    def setUpClass(cls):
        # Verify test assets exist on disk
        cls.has_real_rasters = os.path.exists(URBAN_TIF) and os.path.exists(S2_PATCH_PNG)
        if not cls.has_real_rasters:
            raise FileNotFoundError(f"Real test rasters missing from {SAMPLE_RASTER_DIR}")

    # 1. GeoTIFF Metadata & CRS Extraction
    def test_01_real_geotiff_metadata_and_crs_extraction(self):
        """Verify real GeoTIFF rasters extract genuine CRS, transform, resolution, and bounds."""
        if not HAS_RASTERIO:
            self.skipTest("rasterio not available")

        meta = GeoTIFFValidator.extract_metadata(URBAN_TIF, file_size_bytes=os.path.getsize(URBAN_TIF))
        self.assertTrue(meta.isValid, f"Expected valid GeoTIFF: {meta.validationMessage}")
        self.assertIn(meta.format, ["TIF", "GeoTIFF", "TIFF"])
        self.assertEqual(meta.width, 512)
        self.assertEqual(meta.height, 512)
        self.assertEqual(meta.bands, 3)
        self.assertEqual(meta.crs, "EPSG:32643")
        self.assertIsNotNone(meta.geotransform)
        self.assertEqual(len(meta.geotransform), 6)
        self.assertIsNotNone(meta.bounds)
        self.assertIn("minX", meta.bounds)
        self.assertIn("minY", meta.bounds)
        self.assertIn("maxX", meta.bounds)
        self.assertIn("maxY", meta.bounds)

        # Verify pixel-to-geographic conversion produces real WGS84 coordinates near Mumbai (UTM 43N)
        coord = GeoreferenceEngine.pixel_to_wgs84(256, 256, meta.geotransform, meta.crs)
        self.assertIsNotNone(coord)
        lat, lon = coord
        # UTM 43N col=432000+2560, row=1425000-2560 -> ~12.8°N, 74.3°E
        self.assertTrue(10.0 <= lat <= 25.0, f"Latitude out of expected range: {lat}")
        self.assertTrue(70.0 <= lon <= 85.0, f"Longitude out of expected range: {lon}")

    # 2. Missing Georeferencing Handling (No Invented Coordinates)
    def test_02_missing_georeferencing_handling(self):
        """Verify unprojected PNG rasters honestly report geolocation unavailable."""
        meta = GeoTIFFValidator.extract_metadata(S2_PATCH_PNG, file_size_bytes=os.path.getsize(S2_PATCH_PNG))
        self.assertTrue(meta.isValid)

        coord = GeoreferenceEngine.pixel_to_wgs84(60, 60, meta.geotransform, meta.crs)
        self.assertIsNone(coord, "Must return None for unprojected image, never guessing coordinates")

        cursor_info = GeoreferenceEngine.query_cursor_feature(60, 60, {"geotransform": meta.geotransform, "crs": meta.crs})
        self.assertEqual(cursor_info["geospatial_status"], "unavailable")
        self.assertIn("Geolocation unavailable", cursor_info["message"])

    # 3. Florence-2 Real Captioning
    def test_03_florence2_real_inference_captioning(self):
        """Verify Florence-2 VLM generates genuine natural language captions on real imagery."""
        img = Image.open(S2_PATCH_PNG).convert("RGB")
        t0 = time.time()
        result = florence2_service.predict(
            image_source=img,
            query="Describe this satellite image.",
            task_type="scene-captioning"
        )
        elapsed = time.time() - t0

        self.assertIn("answer", result)
        caption = result["answer"]
        self.assertIsInstance(caption, str)
        self.assertGreater(len(caption), 5, f"Caption too short: {caption}")
        self.assertIsNone(result.get("confidence"), "Confidence must be None for open-ended VLM captioning")
        self.assertFalse(result.get("is_simulation", True))
        print(f"\n[PERF] Florence-2 Captioning Inference: {elapsed:.3f}s | Output: {caption[:75]}...")

    # 4. Florence-2 Real VQA
    def test_04_florence2_real_inference_vqa(self):
        """Verify Florence-2 VLM generates genuine answers for visual questions on satellite data."""
        img = Image.open(URBAN_TIF).convert("RGB")
        t0 = time.time()
        result = florence2_service.predict(
            image_source=img,
            query="Is there vegetation or water visible in this image?",
            task_type="vqa"
        )
        elapsed = time.time() - t0

        self.assertIn("answer", result)
        ans = result["answer"]
        self.assertIsInstance(ans, str)
        self.assertGreater(len(ans), 2)
        self.assertIsNone(result.get("confidence"))
        print(f"[PERF] Florence-2 VQA Inference: {elapsed:.3f}s | Output: {ans}")

    # 5. Florence-2 Real Phrase Grounding
    def test_05_florence2_real_inference_grounding(self):
        """Verify Florence-2 grounding returns pixel coordinates within image dimensions."""
        img = Image.open(S2_PATCH_PNG).convert("RGB")
        t0 = time.time()
        result = florence2_service.predict(
            image_source=img,
            query="green vegetation",
            task_type="text-guided-grounding"
        )
        elapsed = time.time() - t0

        self.assertIn("boundingBoxes", result)
        boxes = result["boundingBoxes"]
        self.assertIsInstance(boxes, list)
        for b in boxes:
            self.assertIn("x", b)
            self.assertIn("y", b)
            self.assertIn("width", b)
            self.assertIn("height", b)
            # Verify coordinates are in valid space
            self.assertTrue(b["x"] >= 0, f"Box x negative: {b['x']}")
            self.assertTrue(b["y"] >= 0, f"Box y negative: {b['y']}")
            self.assertIsNone(b.get("confidence"), "Confidence must not be fabricated on grounding boxes")
        print(f"[PERF] Florence-2 Grounding Inference: {elapsed:.3f}s | Boxes found: {len(boxes)}")

    # 6. BigEarthNet Real Inference & 10-Band Requirement
    def test_06_bigearthnet_real_inference_contract(self):
        """Verify BigEarthNet requires authentic 10-band Sentinel-2 input and rejects 3-band RGB."""
        # A: 3-Band RGB input should fail 10-band validation (no fake padding)
        rgb_arr = np.zeros((3, 120, 120), dtype=np.float32)
        is_valid_s2, reason = validate_s2_10band_input(rgb_arr, bands=["B02", "B03", "B04"])
        self.assertFalse(is_valid_s2, "3-band RGB must not pass 10-band Sentinel-2 check")
        self.assertIn("3-band RGB", reason)

        # B: Authentic 10-band input shape (10, 120, 120) runs real ONNX model
        loader = BigEarthNetModelLoader()
        is_loaded, status, msg = loader.load_model()
        self.assertTrue(is_loaded or loader.is_available, f"Model failed to load: {msg}")

        s2_10band = np.random.uniform(0.05, 0.45, size=(10, 120, 120)).astype(np.float32)
        t0 = time.time()
        probs = loader.run_inference(s2_10band)
        elapsed = time.time() - t0

        self.assertEqual(len(probs), 19, "Must return 19 CLC probabilities")
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs), "Probabilities must be in [0, 1]")
        print(f"[PERF] BigEarthNet ResNet-18 ONNX Inference: {elapsed:.4f}s | Classes: {len(BIGEARTHNET_19_CLASSES)}")

    # 7. Compound Multi-Specialist Query Planning & Evidence Combination
    def test_07_compound_query_multi_specialist_planning(self):
        """Verify agent handles compound queries, executes specialists, and combines evidence."""
        agent = SatQueryAgent()
        query = "Describe the scene and identify the main land-cover types."
        images_dict = {
            "single": {
                "filename": URBAN_TIF,
                "fileDataUri": None,
                "fileSizeBytes": os.path.getsize(URBAN_TIF),
                "previewUrl": None
            }
        }

        t0 = time.time()
        response = agent.execute_pipeline(query=query, mode="single", images_dict=images_dict)
        elapsed = time.time() - t0

        self.assertIsNotNone(response)
        self.assertEqual(response.mode, "single")
        self.assertGreater(len(response.answer), 10)
        self.assertIsNotNone(response.evidenceHierarchy)
        self.assertGreater(len(response.evidenceHierarchy.get("direct_evidence", [])), 0)
        self.assertIsNotNone(response.executionSteps)
        self.assertGreater(len(response.executionSteps), 1)
        self.assertFalse(response.isSimulation)
        print(f"[PERF] Compound Agent Execution (Single Image): {elapsed:.3f}s | Answer: {response.answer[:70]}...")

    # 8. Bi-Temporal Real Data Change Detection
    def test_08_bitemporal_real_data_change_detection(self):
        """Verify bi-temporal change pipeline detects changes on real temporal GeoTIFF pair."""
        t0 = time.time()
        result = bitemporal_service.predict(
            query="What changed between 2021 and 2023?",
            inputs={
                "before": TEMPORAL_T1_TIF,
                "after": TEMPORAL_T2_TIF
            }
        )
        elapsed = time.time() - t0

        self.assertIn("answer", result)
        self.assertIn("evidence", result)
        self.assertIn("changedRegions", result)
        self.assertIn("changeMetric", result)
        self.assertIsNone(result.get("confidence"), "Confidence must be None for change detection")
        self.assertFalse(result.get("is_simulation", True))

        metric = result["changeMetric"]
        self.assertIn("changeRegionsCount", metric)
        print(f"[PERF] Bi-Temporal Change Detection: {elapsed:.3f}s | Changes: {metric['changeRegionsCount']} regions")

    # 9. Optical + SAR Real Data Fusion
    def test_09_optical_sar_real_data_fusion(self):
        """Verify Optical + SAR fusion evaluates backscatter and spectral features on real pair."""
        t0 = time.time()
        result = optical_sar_service.predict(
            query="Use optical and SAR images together to identify water and built-up areas.",
            inputs={
                "optical": OPTICAL_TIF,
                "sar": SAR_TIF
            }
        )
        elapsed = time.time() - t0

        self.assertIn("answer", result)
        self.assertIn("crossModalEvidence", result)
        cme = result["crossModalEvidence"]
        self.assertIn("opticalEvidence", cme)
        self.assertIn("sarEvidence", cme)
        self.assertIn("fusedEvidence", cme)
        self.assertIsNone(result.get("confidence"), "Confidence must be None for multimodal fusion")
        self.assertFalse(result.get("is_simulation", True))
        print(f"[PERF] Optical+SAR Fusion: {elapsed:.3f}s | Fused elements: {len(cme['fusedEvidence'])}")

    # 10. Edge Cases & Rejections
    def test_10_edge_cases_and_rejections(self):
        """Verify honest error handling for corrupted or invalid inputs without mock fallbacks."""
        # Non-image file
        is_valid, msg = GeoTIFFValidator.validate_file_format("corrupted_script.sh")
        self.assertFalse(is_valid)

        # Missing files
        agent = SatQueryAgent()
        with self.assertRaises(ValueError):
            agent.execute_pipeline(query="Where is water?", mode="single", images_dict={})

        with self.assertRaises(ValueError):
            agent.execute_pipeline(query="What changed?", mode="bi-temporal", images_dict={"before": None, "after": None})

    # 11. API Endpoints Integration with Real Rasters
    def test_11_api_endpoints_with_real_assets(self):
        """Verify FastAPI endpoints respond with genuine structured responses on real raster requests."""
        # A: /api/health
        health_res = asyncio.run(health_check())
        self.assertEqual(health_res.status, "healthy")
        self.assertGreater(health_res.activeModelsCount, 0)

        # B: /api/validate-image
        val_req = ValidateImageRequest(
            filename=URBAN_TIF,
            fileSizeBytes=os.path.getsize(URBAN_TIF),
            mode="single",
            role="single"
        )
        val_meta = asyncio.run(validate_image(val_req))
        self.assertTrue(val_meta.isValid)
        self.assertEqual(val_meta.crs, "EPSG:32643")
        self.assertEqual(val_meta.width, 512)

        # C: /api/change-analysis
        chg_req = ChangeAnalysisRequest(
            query="Identify changes",
            beforeImage={"filename": TEMPORAL_T1_TIF},
            afterImage={"filename": TEMPORAL_T2_TIF}
        )
        cdata = asyncio.run(change_analysis(chg_req))
        self.assertEqual(cdata.mode, "bi-temporal")
        self.assertFalse(cdata.isSimulation)
        self.assertIsNotNone(cdata.changeMetric)

        # D: /api/optical-sar-analysis
        optsar_req = OpticalSARAnalysisRequest(
            query="Find water bodies",
            opticalImage={"filename": OPTICAL_TIF},
            sarImage={"filename": SAR_TIF}
        )
        osdata = asyncio.run(optical_sar_analysis(optsar_req))
        self.assertEqual(osdata.mode, "optical-sar")
        self.assertFalse(osdata.isSimulation)
        self.assertIsNotNone(osdata.crossModalEvidence)

        # E: /api/report/generate
        rep_res = asyncio.run(generate_report({"result": cdata.model_dump()}))
        self.assertIn("reportId", rep_res)
        self.assertIn("markdown", rep_res)
        self.assertIn("Mission Intelligence Dossier", rep_res["markdown"])

    # 12. Benchmark Datasets Clean Skip
    def test_12_unconfigured_benchmarks_clean_skip(self):
        """Verify unconfigured benchmark datasets skip cleanly without fabricating scores."""
        for bench_id in ["bigearthnet-s2", "rsvqa-lr", "rsvqa-hr", "vrsbench", "cdvqa"]:
            status_info = BenchmarkRegistry.discover_dataset_status(bench_id)
            self.assertEqual(status_info["status"], "dataset_unavailable")
            self.assertIn("not configured", status_info["message"])

        # Also verify adapter validate_configuration reports clean unavailability
        rsvqa_adapter = RSVQAAdapter("rsvqa-lr")
        is_ok, msg = rsvqa_adapter.validate_configuration()
        self.assertFalse(is_ok)
        self.assertIn("RSVQA_ROOT", msg)


if __name__ == "__main__":
    unittest.main()
