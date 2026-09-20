"""
SatQuery AI - Bi-Temporal Change Detection Pipeline Tests
Validates:
1. Identical images -> zero significant change detected.
2. Known synthetic numerical difference -> detected candidate change with accurate pixel counts and bounding boxes.
3. Dimension mismatch handling with honest resampling warning.
4. Gross aspect ratio mismatch rejection.
5. Incompatible CRS rejection.
6. Missing georeferencing handling (pixel coordinates preserved; no guessed geocoordinates).
7. BigEarthNet before/after dual-state evaluation when 10-band Sentinel-2 inputs exist.
8. RGB-only inputs avoid fake BigEarthNet classification.
9. No fabricated confidence scores.
10. Temporal query routing in SatQueryAgent.
11. Invalid input rejection (no silent fake fallbacks).
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

from backend.app.models.bitemporal_inference import BiTemporalChangeService, bitemporal_service
from backend.models.change_detection import BiTemporalChangeModel
from backend.orchestrator.agent import SatQueryAgent
from backend.registry.tools import ToolRegistry
from backend.models.bigearthnet_loader import REQUIRED_S2_BANDS


class TestBiTemporalPipeline(unittest.TestCase):
    def setUp(self):
        self.service = BiTemporalChangeService()
        self.agent = SatQueryAgent()

        # Synthetic 120x120 RGB images
        self.img_t1_rgb = np.ones((120, 120, 3), dtype=np.uint8) * 120
        self.img_t2_rgb_identical = np.ones((120, 120, 3), dtype=np.uint8) * 120

        # T2 with a distinct 30x30 change patch (900 px) at y=20:50, x=40:70
        self.img_t2_rgb_changed = np.ones((120, 120, 3), dtype=np.uint8) * 120
        self.img_t2_rgb_changed[20:50, 40:70, :] = 240

        # Synthetic 10-band Sentinel-2 rasters
        self.s2_t1 = np.ones((10, 120, 120), dtype=np.float32) * 500.0
        self.s2_t2 = np.ones((10, 120, 120), dtype=np.float32) * 500.0
        self.s2_t2[2, 20:50, 40:70] = 2500.0  # B04 Red shift
        self.s2_t2[1, 20:50, 40:70] = 1800.0  # B03 Green shift

    # ------------------------------------------------------------------
    # 1. Identical Images
    # ------------------------------------------------------------------
    def test_01_identical_images_yield_zero_change(self):
        """Verify identical temporal observations detect zero significant change."""
        res = self.service.predict(
            query="What changed between these images?",
            inputs={"before": self.img_t1_rgb, "after": self.img_t2_rgb_identical}
        )
        self.assertEqual(res["changed_pixel_count"], 0)
        self.assertEqual(res["change_coverage_percentage"], 0.0)
        self.assertEqual(len(res["boundingBoxes"]), 0)
        self.assertEqual(res["changeDirection"], "No Significant Change")
        self.assertIsNone(res["confidence"], "Confidence must be None (no fabricated score).")

    # ------------------------------------------------------------------
    # 2. Known Synthetic Difference
    # ------------------------------------------------------------------
    def test_02_known_synthetic_difference_detected(self):
        """Verify a known 30x30 change patch is correctly localized and quantified."""
        res = self.service.predict(
            query="What changed?",
            inputs={"before": self.img_t1_rgb, "after": self.img_t2_rgb_changed}
        )
        self.assertGreater(res["changed_pixel_count"], 800)
        self.assertLessEqual(res["changed_pixel_count"], 900)
        self.assertGreater(res["change_coverage_percentage"], 5.0)
        self.assertEqual(len(res["boundingBoxes"]), 1)

        box = res["boundingBoxes"][0]
        # Verify box coordinates match the inserted patch: y in [20, 50], x in [40, 70]
        self.assertAlmostEqual(box["x"], 40.0, delta=2.0)
        self.assertAlmostEqual(box["y"], 20.0, delta=2.0)
        self.assertAlmostEqual(box["width"], 30.0, delta=2.0)
        self.assertAlmostEqual(box["height"], 30.0, delta=2.0)
        self.assertIsNone(box["confidence"], "No fabricated box confidence score.")

    # ------------------------------------------------------------------
    # 3. Dimension Mismatch & Approximate Resampling
    # ------------------------------------------------------------------
    def test_03_dimension_mismatch_resampled_with_warning(self):
        """Verify slightly mismatched image dimensions are resampled with an explicit note."""
        t2_smaller = np.ones((100, 100, 3), dtype=np.uint8) * 120
        res = self.service.predict(
            query="Compare these images",
            inputs={"before": self.img_t1_rgb, "after": t2_smaller}
        )
        self.assertIn("Spatial Alignment", res["alignment_note"])
        self.assertIn("Resampled T2", res["alignment_note"])

    def test_04_gross_aspect_ratio_mismatch_rejected(self):
        """Verify severely mismatched aspect ratios are rejected with clear error."""
        t2_skewed = np.ones((60, 180, 3), dtype=np.uint8) * 120  # aspect ratio 3.0 vs 1.0
        with self.assertRaises(ValueError) as ctx:
            self.service.predict(
                query="What changed?",
                inputs={"before": self.img_t1_rgb, "after": t2_skewed}
            )
        self.assertIn("Spatial Incompatibility", str(ctx.exception))

    # ------------------------------------------------------------------
    # 4. Incompatible CRS Rejection
    # ------------------------------------------------------------------
    def test_05_incompatible_crs_rejected(self):
        """Verify conflicting CRS projections between observations are rejected."""
        with self.assertRaises(ValueError) as ctx:
            self.service.predict(
                query="What changed?",
                inputs={"before": self.img_t1_rgb, "after": self.img_t2_rgb_changed},
                metadata={"crs_before": "EPSG:32643 (UTM Zone 43N)", "crs_after": "EPSG:32644 (UTM Zone 44N)"}
            )
        self.assertIn("CRS Incompatibility", str(ctx.exception))

    # ------------------------------------------------------------------
    # 5. Missing Georeferencing Handling
    # ------------------------------------------------------------------
    def test_06_missing_georeferencing_uses_pixel_space(self):
        """Verify images without georeferencing output bounding boxes in pixel space without guessed lat/lon."""
        res = self.service.predict(
            query="What changed?",
            inputs={"before": self.img_t1_rgb, "after": self.img_t2_rgb_changed}
        )
        for box in res["boundingBoxes"]:
            self.assertIsInstance(box["x"], float)
            self.assertIsInstance(box["y"], float)
            self.assertNotIn("° N", box["description"])
            self.assertNotIn("° E", box["description"])

    # ------------------------------------------------------------------
    # 6. BigEarthNet Dual-State Prior on 10-Band Sentinel-2
    # ------------------------------------------------------------------
    def test_07_bigearthnet_dual_state_prior_when_10band_present(self):
        """Verify 10-band Sentinel-2 inputs trigger independent BigEarthNet before/after comparison."""
        mock_pred_b = {
            "probabilities": {
                "Continuous urban fabric": 10.0,
                "Discontinuous urban fabric": 15.0,
                "Coniferous forest": 55.0,
            }
        }
        mock_pred_a = {
            "probabilities": {
                "Continuous urban fabric": 35.0,  # +25% shift
                "Discontinuous urban fabric": 15.0,
                "Coniferous forest": 20.0,        # -35% shift
            }
        }

        with patch("backend.app.models.bitemporal_inference.bigearthnet_service.predict") as mock_predict:
            mock_predict.side_effect = [mock_pred_b, mock_pred_a]
            res = self.service.predict(
                query="Has urban area changed?",
                inputs={
                    "before": {"array": self.s2_t1, "bands": REQUIRED_S2_BANDS},
                    "after": {"array": self.s2_t2, "bands": REQUIRED_S2_BANDS}
                }
            )
            self.assertTrue(res["bigearthnet_used"])
            self.assertGreater(len(res["bigearthnet_shifts"]), 0)
            shifts_classes = [s["class"] for s in res["bigearthnet_shifts"]]
            self.assertIn("Coniferous forest", shifts_classes)
            self.assertIn("Continuous urban fabric", shifts_classes)

    def test_08_rgb_inputs_avoid_fake_bigearthnet_classification(self):
        """Verify RGB-only inputs explicitly state BigEarthNet is unavailable."""
        res = self.service.predict(
            query="What changed?",
            inputs={"before": self.img_t1_rgb, "after": self.img_t2_rgb_changed}
        )
        self.assertFalse(res["bigearthnet_used"])
        evidence_text = " ".join(res["evidence"])
        self.assertIn("BigEarthNet 10-band classification unavailable", evidence_text)

    # ------------------------------------------------------------------
    # 7. No Fabricated Confidence Scores
    # ------------------------------------------------------------------
    def test_09_no_fabricated_confidence_scores(self):
        """Verify results omit fabricated confidence numbers."""
        res = self.service.predict(
            query="What changed?",
            inputs={"before": self.img_t1_rgb, "after": self.img_t2_rgb_changed}
        )
        self.assertIsNone(res["confidence"])
        for box in res["boundingBoxes"]:
            self.assertIsNone(box["confidence"])

    # ------------------------------------------------------------------
    # 8. Temporal Query Routing
    # ------------------------------------------------------------------
    def test_10_temporal_query_routing(self):
        """Verify temporal questions route to change-analysis and change-based-vqa."""
        task1 = self.agent.classify_task("What changed?", "bi-temporal", 2, ["Optical RGB", "Optical RGB"])
        self.assertEqual(task1, "change-analysis")

        task2 = self.agent.classify_task("Has vegetation changed?", "bi-temporal", 2, ["Optical RGB", "Optical RGB"])
        self.assertEqual(task2, "change-based-vqa")

        task3 = self.agent.classify_task("Compare these images", "bi-temporal", 2, ["Optical RGB", "Optical RGB"])
        self.assertEqual(task3, "change-analysis")

        task4 = self.agent.classify_task("Has water increased or decreased?", "bi-temporal", 2, ["Optical RGB", "Optical RGB"])
        self.assertEqual(task4, "change-based-vqa")

    # ------------------------------------------------------------------
    # 9. Invalid Input Rejection
    # ------------------------------------------------------------------
    def test_11_invalid_input_rejection_no_silent_fallback(self):
        """Verify missing or corrupted temporal inputs raise clear ValueError."""
        with self.assertRaises(ValueError):
            self.service.predict(
                query="What changed?",
                inputs={"before": None, "after": self.img_t2_rgb_changed}
            )

        with self.assertRaises(ValueError):
            self.service.predict(
                query="What changed?",
                inputs={"before": self.img_t1_rgb, "after": None}
            )

    # ------------------------------------------------------------------
    # 10. Agent End-to-End Execution
    # ------------------------------------------------------------------
    def test_12_agent_end_to_end_bitemporal_execution(self):
        """Verify agent.execute_pipeline executes real bi-temporal change analysis."""
        resp = self.agent.execute_pipeline(
            query="What changed?",
            mode="bi-temporal",
            images_dict={
                "before": {"array": self.img_t1_rgb, "name": "t1.png"},
                "after": {"array": self.img_t2_rgb_changed, "name": "t2.png"}
            }
        )
        self.assertEqual(resp.taskType, "change-analysis")
        self.assertEqual(resp.selectedModel, "Bi-Temporal Difference Specialist")
        self.assertIsNone(resp.confidence, "Confidence must be None.")
        self.assertFalse(resp.isSimulation)
        self.assertIsNotNone(resp.changeMetric)
        self.assertGreater(resp.changeMetric.netChangePercentage, 0.0)
        self.assertGreaterEqual(len(resp.boundingBoxes or []), 1)

        # Verify trace steps
        step_titles = [s.title for s in resp.executionSteps]
        self.assertIn("Change specialist selected", step_titles)
        self.assertIn("Temporal analysis", step_titles)
        self.assertIn("Evidence extraction", step_titles)
        self.assertIn("Response generation", step_titles)


if __name__ == "__main__":
    unittest.main()
