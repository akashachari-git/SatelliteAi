"""
SatQuery AI - Optical + SAR Multimodal Fusion Pipeline Tests
Validates:
1. Valid optical + SAR pair processing and feature corroboration.
2. Optical-only input rejection for multimodal route.
3. SAR-only input rejection for multimodal route.
4. Incompatible CRS rejection.
5. Incompatible dimensions/aspect ratio rejection.
6. Missing georeferencing handling (pixel space preserved; no guessed lat/lon).
7. Genuine VV/VH polarimetric and dB handling.
8. Rejection of invalid SAR / RGB masquerading as SAR.
9. Optical spectral indices (NDVI, NDWI, NDBI) when 10-band Sentinel-2 exists.
10. Cross-modal fused evidence generation.
11. Query routing for optical-SAR queries in SatQueryAgent.
12. No fabricated confidence scores.
13. No fake model/checkpoint claims.
"""
import unittest
import numpy as np
from PIL import Image

from backend.app.models.optical_sar_inference import OpticalSARFusionService, optical_sar_service
from backend.models.optical_sar_fusion import OpticalSARFusionModel
from backend.orchestrator.agent import SatQueryAgent
from backend.registry.tools import ToolRegistry
from backend.models.bigearthnet_loader import REQUIRED_S2_BANDS


class TestOpticalSARPipeline(unittest.TestCase):
    def setUp(self):
        self.service = OpticalSARFusionService()
        self.agent = SatQueryAgent()

        # Synthetic 120x120 Optical RGB image
        self.opt_rgb = np.ones((120, 120, 3), dtype=np.uint8) * 150
        # Add a dark water patch (low albedo) at [20:50, 20:50]
        self.opt_rgb[20:50, 20:50, :] = 20
        # Add a bright built-up patch (high albedo) at [70:100, 70:100]
        self.opt_rgb[70:100, 70:100, :] = 240

        # Synthetic 120x120 SAR dual-pol (VV, VH) array in dB
        # Background: intermediate backscatter (-12 dB)
        self.sar_dual = np.ones((2, 120, 120), dtype=np.float32) * -12.0
        # Specular extinction (water candidate: -25 dB) matching optical water
        self.sar_dual[0, 20:50, 20:50] = -25.0
        self.sar_dual[1, 20:50, 20:50] = -30.0
        # Double-bounce corner reflection (built-up candidate: -3 dB) matching optical built-up
        self.sar_dual[0, 70:100, 70:100] = -3.0
        self.sar_dual[1, 70:100, 70:100] = -8.0

        # Synthetic 10-band Sentinel-2 array
        self.s2_10band = np.ones((10, 120, 120), dtype=np.float32) * 1000.0
        # Water patch: high NDWI (Green B03=1500, NIR B08=200)
        self.s2_10band[1, 20:50, 20:50] = 1500.0  # B03 Green
        self.s2_10band[6, 20:50, 20:50] = 200.0   # B08 NIR
        # Built-up patch: high NDBI (SWIR B11=2500, NIR B08=1000)
        self.s2_10band[8, 70:100, 70:100] = 2500.0 # B11 SWIR
        self.s2_10band[6, 70:100, 70:100] = 1000.0 # B08 NIR

    # ------------------------------------------------------------------
    # 1. Valid Optical + SAR Pair
    # ------------------------------------------------------------------
    def test_01_valid_optical_sar_pair_processing(self):
        """Verify valid optical and SAR pair generates genuine corroborated cross-modal evidence."""
        res = self.service.predict(
            query="Analyze optical and SAR together",
            inputs={
                "optical": self.opt_rgb,
                "sar": {"array": self.sar_dual, "sensor": "Sentinel-1", "modality": "SAR"}
            }
        )
        self.assertIn("crossModalEvidence", res)
        cme = res["crossModalEvidence"]
        self.assertIn("opticalEvidence", cme)
        self.assertIn("sarEvidence", cme)
        self.assertIn("fusedEvidence", cme)
        self.assertIn("corroboratingFeatures", cme)

        # Check detected bounding boxes
        self.assertGreaterEqual(len(res["boundingBoxes"]), 1)
        self.assertIsNone(res["confidence"], "No fabricated confidence score.")

    # ------------------------------------------------------------------
    # 2. Input Validation Rejections
    # ------------------------------------------------------------------
    def test_02_optical_only_input_rejected(self):
        """Verify optical-only input is rejected for multimodal route."""
        with self.assertRaises(ValueError) as ctx:
            self.service.predict(
                query="Analyze optical and SAR",
                inputs={"optical": self.opt_rgb, "sar": None}
            )
        self.assertIn("Missing SAR observation", str(ctx.exception))

    def test_03_sar_only_input_rejected(self):
        """Verify SAR-only input is rejected for multimodal route."""
        with self.assertRaises(ValueError) as ctx:
            self.service.predict(
                query="Analyze optical and SAR",
                inputs={"optical": None, "sar": self.sar_dual}
            )
        self.assertIn("Missing Optical observation", str(ctx.exception))

    def test_04_incompatible_crs_rejected(self):
        """Verify conflicting CRS projections between Optical and SAR are rejected."""
        with self.assertRaises(ValueError) as ctx:
            self.service.predict(
                query="Analyze optical and SAR",
                inputs={"optical": self.opt_rgb, "sar": {"array": self.sar_dual, "sensor": "Sentinel-1"}},
                metadata={"crs_optical": "EPSG:32643", "crs_sar": "EPSG:32644"}
            )
        self.assertIn("CRS Incompatibility", str(ctx.exception))

    def test_05_incompatible_aspect_ratio_rejected(self):
        """Verify severely mismatched aspect ratios are rejected with clear error."""
        sar_skewed = np.ones((60, 180), dtype=np.float32) * -12.0  # aspect ratio 3.0 vs 1.0
        with self.assertRaises(ValueError) as ctx:
            self.service.predict(
                query="Analyze optical and SAR",
                inputs={"optical": self.opt_rgb, "sar": {"array": sar_skewed, "sensor": "Sentinel-1"}}
            )
        self.assertIn("Spatial Incompatibility", str(ctx.exception))

    def test_06_invalid_sar_rgb_masquerading_rejected(self):
        """Verify optical photo submitted as SAR without radar metadata is rejected."""
        fake_sar_rgb = np.zeros((120, 120, 3), dtype=np.uint8)
        fake_sar_rgb[:, :, 0] = 200  # Red
        fake_sar_rgb[:, :, 1] = 50   # Green
        with self.assertRaises(ValueError) as ctx:
            self.service.predict(
                query="Analyze optical and SAR",
                inputs={"optical": self.opt_rgb, "sar": fake_sar_rgb}
            )
        self.assertIn("Invalid SAR Input", str(ctx.exception))

    # ------------------------------------------------------------------
    # 3. Missing Georeferencing Preserves Pixel Space
    # ------------------------------------------------------------------
    def test_07_missing_georeferencing_uses_pixel_space(self):
        """Verify bounding boxes remain strictly in image pixel space without guessed coordinates."""
        res = self.service.predict(
            query="Analyze optical and SAR",
            inputs={"optical": self.opt_rgb, "sar": {"array": self.sar_dual, "sensor": "Sentinel-1"}}
        )
        for box in res["boundingBoxes"]:
            self.assertIsInstance(box["x"], float)
            self.assertIsInstance(box["y"], float)
            self.assertNotIn("° N", box["description"])
            self.assertNotIn("° E", box["description"])

    # ------------------------------------------------------------------
    # 4. Genuine VV/VH and dB Handling
    # ------------------------------------------------------------------
    def test_08_genuine_vv_vh_and_db_handling(self):
        """Verify dual-polarization VV/VH backscatter and cross-ratio are evaluated."""
        res = self.service.predict(
            query="Analyze radar backscatter",
            inputs={"optical": self.opt_rgb, "sar": {"array": self.sar_dual, "sensor": "Sentinel-1"}}
        )
        sar_ev = " ".join(res["crossModalEvidence"]["sarEvidence"])
        self.assertIn("Dual-Pol VV/VH", sar_ev)
        self.assertIn("Mean VV backscatter", sar_ev)
        self.assertIn("VV/VH cross-polarization ratio", sar_ev)

    # ------------------------------------------------------------------
    # 5. Optical Spectral Evidence (Sentinel-2 10-Band)
    # ------------------------------------------------------------------
    def test_09_multispectral_indices_computed_when_available(self):
        """Verify 10-band Sentinel-2 inputs trigger genuine NDVI, NDWI, and NDBI extraction."""
        res = self.service.predict(
            query="Analyze optical and SAR",
            inputs={
                "optical": {"array": self.s2_10band, "bands": REQUIRED_S2_BANDS},
                "sar": {"array": self.sar_dual, "sensor": "Sentinel-1"}
            }
        )
        opt_ev = " ".join(res["crossModalEvidence"]["opticalEvidence"])
        self.assertIn("NDVI", opt_ev)
        self.assertIn("NDWI", opt_ev)
        self.assertIn("NDBI", opt_ev)

    # ------------------------------------------------------------------
    # 6. Fused Cross-Modal Evidence Generation
    # ------------------------------------------------------------------
    def test_10_cross_modal_fused_evidence_generation(self):
        """Verify cross-sensor corroboration isolates water and built-up structures."""
        res = self.service.predict(
            query="Analyze optical and SAR",
            inputs={"optical": self.opt_rgb, "sar": {"array": self.sar_dual, "sensor": "Sentinel-1"}}
        )
        fused_ev = " ".join(res["crossModalEvidence"]["fusedEvidence"])
        self.assertIn("Evidence-based optical + SAR fusion", fused_ev)
        self.assertIn("CORROBORATED WATER", fused_ev)
        self.assertIn("CORROBORATED BUILT-UP", fused_ev)

    # ------------------------------------------------------------------
    # 7. Query Routing in SatQueryAgent
    # ------------------------------------------------------------------
    def test_11_optical_sar_query_routing(self):
        """Verify natural language optical+SAR queries route to optical-sar-analysis."""
        queries = [
            "Analyze optical and SAR together",
            "Compare optical and radar imagery",
            "What does SAR reveal that optical does not?",
            "Analyze this optical SAR pair",
            "Combine optical and SAR evidence",
        ]
        for q in queries:
            task = self.agent.classify_task(q, "optical-sar", 2, ["Optical RGB", "SAR"])
            self.assertEqual(task, "optical-sar-analysis", f"Query '{q}' must route to optical-sar-analysis.")

    # ------------------------------------------------------------------
    # 8. No Fabricated Confidence or Neural Claims
    # ------------------------------------------------------------------
    def test_12_no_fabricated_confidence_or_neural_claims(self):
        """Verify responses omit fake confidence and do not claim fictitious 620M models."""
        res = self.service.predict(
            query="Analyze optical and SAR",
            inputs={"optical": self.opt_rgb, "sar": {"array": self.sar_dual, "sensor": "Sentinel-1"}}
        )
        self.assertIsNone(res["confidence"])
        self.assertNotIn("620M", str(res))
        self.assertNotIn("CrossSens-Fusion", str(res["answer"]))

    # ------------------------------------------------------------------
    # 9. Agent End-to-End Execution
    # ------------------------------------------------------------------
    def test_13_agent_end_to_end_optical_sar_execution(self):
        """Verify agent.execute_pipeline executes real Optical + SAR multimodal analysis."""
        resp = self.agent.execute_pipeline(
            query="Analyze optical and SAR together",
            mode="optical-sar",
            images_dict={
                "optical": {"array": self.opt_rgb, "name": "opt.png"},
                "sar": {"array": self.sar_dual, "name": "sar.tif", "sensor": "Sentinel-1", "modality": "SAR"}
            }
        )
        self.assertEqual(resp.taskType, "optical-sar-analysis")
        self.assertEqual(resp.selectedModel, "Optical + SAR Multimodal Fusion Specialist")
        self.assertIsNone(resp.confidence, "Confidence must be None (no fabricated score).")
        self.assertFalse(resp.isSimulation)
        self.assertIsNotNone(resp.crossModalEvidence)
        self.assertGreaterEqual(len(resp.boundingBoxes or []), 1)

        # Check trace steps
        step_titles = [s.title for s in resp.executionSteps]
        self.assertIn("Optical + SAR pair detected", step_titles)
        self.assertIn("Optical-SAR specialist selected", step_titles)
        self.assertIn("Multimodal processing", step_titles)
        self.assertIn("Response generated", step_titles)


if __name__ == "__main__":
    unittest.main()
