"""
Unit tests for STEP 11: Agentic Orchestration & Evidence-Aware Multi-Specialist Controller.
Tests:
1. Single specialist routing
2. Multi-specialist planning
3. Input validation before execution
4. Caption + Land-Cover combination execution & evidence synthesis
5. Temporal + BigEarthNet combination
6. Optical + SAR routing (ensures no routing to Florence-2 alone)
7. Missing image validation (rejects 1-image comparison or 1-image fusion)
8. Insufficient band validation (RGB image rejects BigEarthNet with clear limitation)
9. Disagreement detection and reporting between specialists
10. Structured execution trace with real actions and statuses
11. No fabricated confidence scores (confidence is None)
12. Unsupported / out-of-domain query handling
"""
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from backend.orchestrator.agent import SatQueryAgent
from backend.orchestrator.planner import AgentPlanner, AgentPlan
from backend.orchestrator.evidence_combiner import EvidenceCombiner, EvidenceHierarchy
from backend.models.bigearthnet_loader import REQUIRED_S2_BANDS


class TestOrchestratorPlanning(unittest.TestCase):
    def setUp(self):
        self.agent = SatQueryAgent()
        self.planner = AgentPlanner()
        self.combiner = EvidenceCombiner()

        # Synthetic test data for unit tests
        self.test_rgb = np.zeros((120, 120, 3), dtype=np.uint8)
        self.test_rgb[:, :, 1] = 180  # Greenish

        self.test_10band = np.zeros((10, 120, 120), dtype=np.float32)
        self.test_10band[0] = 500.0   # B02 (Blue)
        self.test_10band[1] = 700.0   # B03 (Green)
        self.test_10band[2] = 400.0   # B04 (Red)
        self.test_10band[6] = 2800.0  # B08 (NIR)
        self.test_10band[8] = 900.0   # B11 (SWIR)

        self.test_sar = np.zeros((2, 120, 120), dtype=np.float32)
        self.test_sar[0] = -12.0  # VV dB
        self.test_sar[1] = -18.0  # VH dB

    # ------------------------------------------------------------------
    # 1. Single Specialist Routing
    # ------------------------------------------------------------------
    def test_01_single_specialist_routing(self):
        """Verify individual intents correctly route to their designated single specialist."""
        # Caption -> Florence-2
        plan_cap = self.planner.create_plan("Describe this satellite image", "single", 1, {})
        self.assertEqual(plan_cap.primary_intent, "scene-captioning")
        self.assertEqual(plan_cap.selected_tools, ["florence2-vlm"])
        self.assertFalse(plan_cap.is_multi_specialist)

        # Grounding -> Florence-2
        plan_grd = self.planner.create_plan("Locate the buildings", "single", 1, {})
        self.assertEqual(plan_grd.primary_intent, "text-guided-grounding")
        self.assertIn("florence2-vlm", plan_grd.selected_tools)

        # VQA -> Florence-2
        plan_vqa = self.planner.create_plan("How many aircraft are on the apron?", "single", 1, {})
        self.assertEqual(plan_vqa.primary_intent, "vqa")
        self.assertEqual(plan_vqa.selected_tools, ["florence2-vlm"])

        # Land Cover -> BigEarthNet
        plan_lc = self.planner.create_plan("Identify the Corine land cover classes", "single", 1, {}, is_s2_10band=True)
        self.assertEqual(plan_lc.primary_intent, "land-cover-classification")
        self.assertEqual(plan_lc.selected_tools, ["bigearthnet-classifier"])

    # ------------------------------------------------------------------
    # 2. Multi-Specialist Planning
    # ------------------------------------------------------------------
    def test_02_multi_specialist_planning(self):
        """Verify compound queries produce multi-specialist plans with dependencies."""
        query = "What land cover is present and what does the image show?"
        plan = self.planner.create_plan(query, "single", 1, {}, is_s2_10band=True)

        self.assertTrue(plan.is_multi_specialist)
        self.assertIn("florence2-vlm", plan.selected_tools)
        self.assertIn("bigearthnet-classifier", plan.selected_tools)
        self.assertIn("caption", plan.intents)
        self.assertIn("land-cover", plan.intents)
        self.assertEqual(plan.execution_order, ["florence2-vlm", "bigearthnet-classifier"])

    # ------------------------------------------------------------------
    # 3. Input Validation Before Execution
    # ------------------------------------------------------------------
    def test_03_input_validation_before_execution(self):
        """Verify planner validates input feasibility before model execution."""
        # Query requires 2 images, but only 1 provided
        with self.assertRaises(ValueError) as ctx:
            self.planner.create_plan("Compare these two images and tell me what changed", "single", 1, {})
        self.assertIn("requires 2 observations", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            self.planner.create_plan("Analyze the optical and SAR images together", "single", 1, {})
        self.assertIn("requires 2 observations", str(ctx.exception))

    # ------------------------------------------------------------------
    # 4. Caption + Land-Cover Combination Execution & Synthesis
    # ------------------------------------------------------------------
    def test_04_caption_and_land_cover_combination(self):
        """Verify multi-specialist execution combines caption and land-cover evidence."""
        mock_florence_res = {
            "model": "Florence-2",
            "model_name": "Florence-2 Vision-Language Specialist",
            "task": "scene-captioning",
            "task_label": "Image Captioning",
            "answer": "An aerial view of green agricultural fields and surrounding roads.",
            "confidence": None,
            "evidence": ["VLM Task: Image Captioning", "Detected agricultural fields"],
            "boundingBoxes": None,
            "execution_duration_ms": 350,
            "is_simulation": False
        }
        with patch("backend.orchestrator.agent.florence2_service.predict", return_value=mock_florence_res):
            resp = self.agent.execute_pipeline(
                query="Describe this image and identify the land-cover classes.",
                mode="single",
                images_dict={
                    "single": {
                        "array": self.test_10band,
                        "bands": REQUIRED_S2_BANDS,
                        "name": "sentinel2_scene.tif"
                    }
                }
            )
            self.assertIsNotNone(resp.agentPlan)
            self.assertTrue(resp.agentPlan["is_multi_specialist"])
            self.assertIn("florence2-vlm", resp.agentPlan["selected_tools"])
            self.assertIn("bigearthnet-classifier", resp.agentPlan["selected_tools"])

            # Check evidence hierarchy
            self.assertIsNotNone(resp.evidenceHierarchy)
            self.assertGreater(len(resp.evidenceHierarchy["direct_evidence"]), 1)
            self.assertIsNone(resp.confidence)

    # ------------------------------------------------------------------
    # 5. Temporal + BigEarthNet Combination
    # ------------------------------------------------------------------
    def test_05_temporal_and_bigearthnet_combination(self):
        """Verify temporal change queries on 10-band pairs incorporate BigEarthNet dual-state prior."""
        resp = self.agent.execute_pipeline(
            query="Compare these two images and tell me what changed.",
            mode="bi-temporal",
            images_dict={
                "before": {"array": self.test_10band, "bands": REQUIRED_S2_BANDS, "name": "t1.tif"},
                "after": {"array": self.test_10band, "bands": REQUIRED_S2_BANDS, "name": "t2.tif"}
            }
        )
        self.assertEqual(resp.taskType, "change-analysis")
        self.assertIsNotNone(resp.agentPlan)
        self.assertEqual(resp.agentPlan["primary_intent"], "bi-temporal")
        self.assertIsNotNone(resp.evidenceHierarchy)

    # ------------------------------------------------------------------
    # 6. Optical + SAR Routing
    # ------------------------------------------------------------------
    def test_06_optical_sar_routing(self):
        """Verify optical+SAR queries route exclusively to optical-sar specialist, never Florence-2 alone."""
        queries = [
            "Analyze the optical and SAR images together.",
            "Compare optical and radar imagery",
            "What does SAR reveal that optical does not?",
            "Combine optical and SAR evidence"
        ]
        for q in queries:
            task = self.agent.classify_task(q, "optical-sar", 2, ["Optical RGB", "SAR"])
            self.assertEqual(task, "optical-sar-analysis", f"Query '{q}' must route to optical-sar-analysis.")

    # ------------------------------------------------------------------
    # 7. Missing Image Validation
    # ------------------------------------------------------------------
    def test_07_missing_image_validation(self):
        """Verify missing temporal or multimodal inputs are rejected with clear errors."""
        with self.assertRaises(ValueError) as ctx:
            self.agent.execute_pipeline(
                query="Compare these two images",
                mode="single",
                images_dict={"single": {"array": self.test_rgb}}
            )
        self.assertIn("requires 2 observations", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            self.agent.execute_pipeline(
                query="Analyze optical and SAR together",
                mode="single",
                images_dict={"single": {"array": self.test_rgb}}
            )
        self.assertIn("requires 2 observations", str(ctx.exception))

    # ------------------------------------------------------------------
    # 8. Insufficient Band Validation
    # ------------------------------------------------------------------
    def test_08_insufficient_band_validation(self):
        """Verify RGB-only input rejects BigEarthNet deep learning and records clear limitation."""
        resp = self.agent.execute_pipeline(
            query="Classify this Sentinel-2 image",
            mode="single",
            images_dict={"single": {"array": self.test_rgb, "name": "rgb_photo.jpg"}}
        )
        self.assertEqual(resp.selectedModel, "Deterministic Fallback Analysis (Classical Heuristic)")
        self.assertTrue(resp.isSimulation)
        self.assertIn("Incompatible band count", str(resp.executionSteps))
        self.assertIsNotNone(resp.agentPlan)
        self.assertTrue(any("Incompatible raster" in lim for lim in resp.agentPlan["limitations"]))

    # ------------------------------------------------------------------
    # 9. Disagreement Detection and Reporting
    # ------------------------------------------------------------------
    def test_09_disagreement_detection_and_reporting(self):
        """Verify conflicts between Florence-2 text and BigEarthNet classes are detected and reported."""
        # Case 1: Florence says water, BigEarthNet says Urban with high score
        florence_caption = "A wide lake surrounded by calm water."
        ben_predictions = [
            {"label": "Continuous urban fabric", "probability": 0.88},
            {"label": "Industrial or commercial units", "probability": 0.10}
        ]
        disagreements = self.combiner.detect_disagreements(florence_caption, ben_predictions)
        self.assertGreaterEqual(len(disagreements), 1)
        self.assertIn("Disagreement between specialists", disagreements[0])
        self.assertIn("water", disagreements[0])
        self.assertIn("Continuous urban fabric", disagreements[0])

        # Verify combined hierarchy reflects disagreement without picking a side
        hierarchy = self.combiner.combine(
            query="Describe this image and land cover",
            results_by_tool={
                "florence2-vlm": {"answer": florence_caption, "evidence": [florence_caption]},
                "bigearthnet-classifier": {"answer": "Continuous urban fabric", "predictions": ben_predictions}
            }
        )
        self.assertGreater(len(hierarchy.disagreements), 0)
        self.assertIn("Disagreement observed between specialist models", hierarchy.synthesized_answer)
        self.assertIn("Both specialist perspectives are retained", hierarchy.synthesized_answer)

    # ------------------------------------------------------------------
    # 10. Structured Execution Trace
    # ------------------------------------------------------------------
    def test_10_structured_execution_trace(self):
        """Verify execution trace contains genuine step metrics and auditable provenance."""
        mock_florence_res = {
            "model": "Florence-2",
            "model_name": "Florence-2 Vision-Language Specialist",
            "task": "scene-captioning",
            "task_label": "Image Captioning",
            "answer": "An aerial view of green agricultural fields.",
            "confidence": None,
            "evidence": ["VLM Task: Image Captioning"],
            "boundingBoxes": None,
            "execution_duration_ms": 300,
            "is_simulation": False
        }
        with patch("backend.orchestrator.agent.florence2_service.predict", return_value=mock_florence_res):
            resp = self.agent.execute_pipeline(
                query="Describe this image",
                mode="single",
                images_dict={"single": {"array": self.test_rgb}}
            )
            steps = resp.executionSteps
            self.assertGreaterEqual(len(steps), 5)
            for s in steps:
                self.assertIsInstance(s.stepNumber, int)
                self.assertIn(s.status, ["completed", "failed"])
                self.assertIsInstance(s.durationMs, int)
                self.assertGreater(len(s.summary), 0)

    # ------------------------------------------------------------------
    # 11. No Fabricated Confidence
    # ------------------------------------------------------------------
    def test_11_no_fabricated_confidence(self):
        """Verify confidence is strictly None for open-ended VLM and multi-specialist synthesis."""
        mock_florence_res = {
            "model": "Florence-2",
            "model_name": "Florence-2 Vision-Language Specialist",
            "task": "scene-captioning",
            "task_label": "Image Captioning",
            "answer": "An aerial view of green agricultural fields.",
            "confidence": None,
            "evidence": ["VLM Task: Image Captioning"],
            "boundingBoxes": None,
            "execution_duration_ms": 300,
            "is_simulation": False
        }
        with patch("backend.orchestrator.agent.florence2_service.predict", return_value=mock_florence_res):
            resp = self.agent.execute_pipeline(
                query="What land cover is present and what does the image show?",
                mode="single",
                images_dict={"single": {"array": self.test_rgb}}
            )
            self.assertIsNone(resp.confidence, "Confidence must be None (no fabricated score).")

    # ------------------------------------------------------------------
    # 12. Unsupported Query Handling
    # ------------------------------------------------------------------
    def test_12_unsupported_query_handling(self):
        """Verify out-of-domain queries record clear limitations rather than hallucinating."""
        plan = self.planner.create_plan(
            query="Predict tomorrow's stock price for this port",
            mode="single",
            image_count=1,
            images_dict={}
        )
        self.assertTrue(any("out-of-domain" in lim for lim in plan.limitations))


if __name__ == "__main__":
    unittest.main()
