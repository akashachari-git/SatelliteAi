"""
Step 5 Test Suite: Integration of Verified Real BigEarthNet Model into SatQuery Pipeline.
Validates:
A. Valid 10-band Sentinel-2 input -> BigEarthNet runs successfully.
B. RGB 3-band image -> BigEarthNet is rejected.
C. 8-band image -> BigEarthNet is rejected.
D. 10-band image with incorrect/missing band identity -> BigEarthNet is rejected.
E. Model output shape -> exactly 19 outputs.
F. Probabilities -> every probability is between 0 and 1.
G. Model session reuse -> model session is cached, not reloaded per inference.
H. Band reordering -> adapter correctly aligns scrambled band sequences.
I. Spatial resizing -> non-120x120 rasters are resized to 120x120.
"""
import unittest
import numpy as np

from backend.models.bigearthnet_loader import (
    BIGEARTHNET_19_CLASSES,
    REQUIRED_S2_BANDS,
    bigearthnet_loader,
)
from backend.geospatial.raster_adapter import (
    extract_raster_bands,
    CAPABILITY_MESSAGE,
)
from backend.app.models.bigearthnet_inference import (
    bigearthnet_service,
    resize_raster_10band,
)
from backend.orchestrator.agent import SatQueryAgent


def create_synthetic_sentinel2_test_data(height: int = 120, width: int = 120) -> np.ndarray:
    """
    Creates a realistic synthetic 10-band Sentinel-2 raster for pipeline testing.
    LABELED CLEARLY AS TEST DATA: NOT A REAL-WORLD SATELLITE OBSERVATION.

    Simulates characteristic multispectral reflectance:
    - Top half: Vegetated canopy (chlorophyll absorption in B04, red-edge surge in B05-B07, NIR peak in B08/B8A).
    - Bottom half: Inland water (low reflectance in visible, strong absorption in NIR/SWIR).
    """
    raster = np.zeros((10, height, width), dtype=np.float32)
    half_h = height // 2

    # Band index mapping:
    # 0: B02 (Blue), 1: B03 (Green), 2: B04 (Red)
    # 3: B05 (RE1),  4: B06 (RE2),   5: B07 (RE3)
    # 6: B08 (NIR),  7: B8A (Narrow NIR)
    # 8: B11 (SWIR1), 9: B12 (SWIR2)

    # Top half: Vegetation signature (reflectance in [0, 1])
    raster[0, :half_h, :] = 0.04  # B02 Blue
    raster[1, :half_h, :] = 0.08  # B03 Green
    raster[2, :half_h, :] = 0.04  # B04 Red
    raster[3, :half_h, :] = 0.22  # B05 RedEdge 1
    raster[4, :half_h, :] = 0.42  # B06 RedEdge 2
    raster[5, :half_h, :] = 0.48  # B07 RedEdge 3
    raster[6, :half_h, :] = 0.52  # B08 NIR
    raster[7, :half_h, :] = 0.53  # B8A Narrow NIR
    raster[8, :half_h, :] = 0.20  # B11 SWIR 1
    raster[9, :half_h, :] = 0.10  # B12 SWIR 2

    # Bottom half: Water signature (strong NIR/SWIR absorption)
    raster[0, half_h:, :] = 0.06  # B02 Blue
    raster[1, half_h:, :] = 0.05  # B03 Green
    raster[2, half_h:, :] = 0.02  # B04 Red
    raster[3, half_h:, :] = 0.01  # B05 RedEdge 1
    raster[4, half_h:, :] = 0.005 # B06 RedEdge 2
    raster[5, half_h:, :] = 0.005 # B07 RedEdge 3
    raster[6, half_h:, :] = 0.004 # B08 NIR
    raster[7, half_h:, :] = 0.003 # B8A Narrow NIR
    raster[8, half_h:, :] = 0.002 # B11 SWIR 1
    raster[9, half_h:, :] = 0.001 # B12 SWIR 2

    return raster


class TestBigEarthNetPipeline(unittest.TestCase):
    """Integration test suite for BigEarthNet ResNet-18 pipeline execution."""

    @classmethod
    def setUpClass(cls):
        # Create realistic synthetic 10-band Sentinel-2 test data
        cls.test_s2_data = create_synthetic_sentinel2_test_data(120, 120)
        cls.agent = SatQueryAgent()

    def test_A_valid_10band_sentinel2_runs_successfully(self):
        """
        A. Valid 10-band Sentinel-2 input -> BigEarthNet runs real inference
        via both the inference service and the agent orchestrator.
        """
        # 1. Direct Service Call
        res = bigearthnet_service.predict(
            image_source=self.test_s2_data,
            bands=REQUIRED_S2_BANDS,
            query="Analyze land cover and vegetation classes",
        )

        self.assertTrue(res["is_ai_prediction"])
        self.assertEqual(res["model"], "BigEarthNet ResNet-18")
        self.assertEqual(res["input_type"], "Sentinel-2 10-band")
        self.assertEqual(len(res["predictions"]), 5)
        self.assertEqual(len(res["probabilities"]), 19)
        self.assertIn("BigEarthNet", res["answer"])
        self.assertGreater(len(res["evidence"]), 0)

        # 2. Agent Orchestrator Pipeline Call
        pipeline_res = self.agent.execute_pipeline(
            query="What is the dominant land cover class in this Sentinel-2 scene?",
            mode="single",
            images_dict={
                "single": {
                    "name": "sentinel2_test_data.tif",
                    "array": self.test_s2_data,
                    "bands": REQUIRED_S2_BANDS,
                }
            },
        )

        self.assertEqual(pipeline_res.taskType, "land-cover-classification")
        self.assertEqual(pipeline_res.selectedModel, "BigEarthNet ResNet-18 Land-Cover Specialist")
        self.assertFalse(pipeline_res.isSimulation)
        self.assertIn("BigEarthNet", pipeline_res.answer)

        # Verify execution trace contains all 9 required fields
        step6 = next((s for s in pipeline_res.executionSteps if s.id == "step-6"), None)
        self.assertIsNotNone(step6)
        labels = [d["label"] for d in step6.details]
        required_labels = [
            "Input Validation Result",
            "Detected Sensor / Type",
            "Detected Bands",
            "Selected Model",
            "Model Version / Source",
            "Preprocessing",
            "Inference Status",
            "Prediction Count",
            "Fallback Status",
        ]
        for req_lbl in required_labels:
            self.assertIn(req_lbl, labels, f"Missing execution trace label: {req_lbl}")

    def test_B_rgb_3band_image_is_rejected(self):
        """
        B. RGB 3-band image -> BigEarthNet is rejected with capability message.
        """
        rgb_data = np.random.uniform(0, 255, size=(3, 120, 120)).astype(np.float32)

        # Check adapter rejection
        is_compat, arr, bands, msg = extract_raster_bands(rgb_data, bands=["R", "G", "B"])
        self.assertFalse(is_compat)
        self.assertIn("RGB", msg)
        self.assertIn("BigEarthNet land-cover analysis requires", msg)

        # Check agent pipeline rejection
        pipeline_res = self.agent.execute_pipeline(
            query="Classify land cover",
            mode="single",
            images_dict={
                "single": {
                    "name": "photo_rgb.jpg",
                    "array": rgb_data,
                    "bands": ["R", "G", "B"],
                }
            },
        )

        self.assertEqual(pipeline_res.taskType, "land-cover-classification")
        self.assertEqual(
            pipeline_res.selectedModel,
            "Deterministic Fallback Analysis (Classical Heuristic)",
        )
        self.assertTrue(pipeline_res.isSimulation)
        self.assertIn("BigEarthNet land-cover analysis requires", pipeline_res.answer)

    def test_C_8band_image_is_rejected(self):
        """
        C. 8-band image -> BigEarthNet is rejected.
        """
        eight_band_data = np.random.uniform(0, 1, size=(8, 120, 120)).astype(np.float32)
        eight_bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08"]

        is_compat, arr, bands, msg = extract_raster_bands(eight_band_data, bands=eight_bands)
        self.assertFalse(is_compat)
        self.assertIn("BigEarthNet land-cover analysis requires", msg)

        res = bigearthnet_service.predict(image_source=eight_band_data, bands=eight_bands)
        self.assertFalse(res["is_ai_prediction"])
        self.assertEqual(res["method"], "Deterministic fallback analysis")

    def test_D_10band_image_with_incorrect_or_missing_band_identity_is_rejected(self):
        """
        D. 10-band image with incorrect or missing band identity -> BigEarthNet is rejected.
        """
        ten_band_data = np.random.uniform(0, 1, size=(10, 120, 120)).astype(np.float32)

        # Case 1: No band identity provided (cannot guess/assume)
        is_compat_none, _, _, msg_none = extract_raster_bands(ten_band_data, bands=None)
        self.assertFalse(is_compat_none)
        self.assertIn("unknown or unverified", msg_none.lower())

        # Case 2: 10 bands but wrong bands (e.g. B01 through B10, missing B11 and B12)
        wrong_bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10"]
        is_compat_wrong, _, _, msg_wrong = extract_raster_bands(ten_band_data, bands=wrong_bands)
        self.assertFalse(is_compat_wrong)
        self.assertIn("Missing required Sentinel-2 bands", msg_wrong)

    def test_E_model_output_shape_exactly_19(self):
        """
        E. Model output shape has exactly 19 outputs.
        """
        res = bigearthnet_service.predict(
            image_source=self.test_s2_data,
            bands=REQUIRED_S2_BANDS,
        )
        self.assertEqual(len(res["probabilities"]), 19)
        self.assertEqual(len(res["probability_vector"]), 19)
        self.assertEqual(len(res["classes"]), 19)
        self.assertEqual(res["classes"], BIGEARTHNET_19_CLASSES)

    def test_F_probabilities_between_0_and_1(self):
        """
        F. Every probability is strictly between 0 and 1.
        """
        res = bigearthnet_service.predict(
            image_source=self.test_s2_data,
            bands=REQUIRED_S2_BANDS,
        )
        for cls_name, prob in res["probabilities"].items():
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

        for prob in res["probability_vector"]:
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_G_model_session_reuse(self):
        """
        G. Model session is reused across inferences and not reloaded every request.
        """
        # Ensure model is loaded
        bigearthnet_service.predict(image_source=self.test_s2_data, bands=REQUIRED_S2_BANDS)
        session_id_1 = id(bigearthnet_service._loader._onnx_session)

        # Second inference
        bigearthnet_service.predict(image_source=self.test_s2_data, bands=REQUIRED_S2_BANDS)
        session_id_2 = id(bigearthnet_service._loader._onnx_session)

        self.assertEqual(
            session_id_1,
            session_id_2,
            "ONNX Runtime session was recreated; session caching failed.",
        )

    def test_H_band_reordering_adapter(self):
        """
        H. Scrambled but complete 10-band Sentinel-2 input is correctly aligned by the adapter.
        """
        # Reorder bands in reverse
        scrambled_bands = list(reversed(REQUIRED_S2_BANDS))
        scrambled_slices = [self.test_s2_data[REQUIRED_S2_BANDS.index(b)] for b in scrambled_bands]
        scrambled_data = np.stack(scrambled_slices, axis=0)

        is_compat, aligned, normalized, msg = extract_raster_bands(scrambled_data, bands=scrambled_bands)
        self.assertTrue(is_compat)
        self.assertEqual(normalized, REQUIRED_S2_BANDS)
        self.assertEqual(aligned.shape, (10, 120, 120))

        # Check values match original
        np.testing.assert_array_equal(aligned, self.test_s2_data)

    def test_I_spatial_resizing_adapter(self):
        """
        I. Non-120x120 10-band raster (e.g. 64x64) is automatically resized to 120x120.
        """
        small_data = create_synthetic_sentinel2_test_data(64, 64)
        self.assertEqual(small_data.shape, (10, 64, 64))

        res = bigearthnet_service.predict(
            image_source=small_data,
            bands=REQUIRED_S2_BANDS,
        )
        self.assertTrue(res["is_ai_prediction"])
        self.assertEqual(len(res["probabilities"]), 19)
        self.assertIn("120×120", res["evidence_metadata"]["preprocessing"])


if __name__ == "__main__":
    unittest.main()
