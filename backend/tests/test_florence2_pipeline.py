"""
SatQuery AI - Florence-2 VLM Backend Integration Tests
Validates:
1. Lazy loading of Florence-2 service
2. Tool registry registration and capability selection
3. Query-intent classification (Captioning vs VQA vs Grounding vs BigEarthNet)
4. Image preparation (Native RGB vs Sentinel-2 10-band True-Color RGB composite)
5. Structured task execution (Captioning, VQA, Phrase Grounding)
6. Agent pipeline end-to-end integration and execution traces
7. BigEarthNet isolation and preservation
8. Input validation and error handling (no fake fallbacks)
"""
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

from backend.app.models.florence2_inference import Florence2InferenceService, florence2_service
from backend.registry.tools import ToolRegistry
from backend.orchestrator.agent import SatQueryAgent
from backend.models.bigearthnet_loader import REQUIRED_S2_BANDS


class TestFlorence2Pipeline(unittest.TestCase):
    def setUp(self):
        self.agent = SatQueryAgent()
        self.test_rgb_image = Image.new("RGB", (120, 120), color=(100, 150, 200))
        self.test_10band_array = np.ones((10, 120, 120), dtype=np.float32) * 500.0
        # Make Red, Green, Blue distinct
        self.test_10band_array[2] = 1200.0  # B04 Red
        self.test_10band_array[1] = 800.0   # B03 Green
        self.test_10band_array[0] = 400.0   # B02 Blue

    # ------------------------------------------------------------------
    # 1. Lazy Loading & Availability
    # ------------------------------------------------------------------
    def test_01_lazy_loading_and_availability(self):
        """Verify service does not load weights until requested and reports availability."""
        fresh_service = Florence2InferenceService()
        self.assertFalse(fresh_service._is_loaded, "Model must not be loaded at instantiation time.")
        self.assertTrue(fresh_service.is_available, "Local Florence-2 checkpoint must be recognized as available.")

    # ------------------------------------------------------------------
    # 2. Tool Registry Selection
    # ------------------------------------------------------------------
    def test_02_tool_registry_selection(self):
        """Verify tool registry routes VLM tasks to florence2-vlm and preserves others."""
        # Florence-2 tasks
        vlm_tasks = [
            "scene-captioning", "image_captioning",
            "vqa", "visual_question_answering",
            "text-guided-grounding", "phrase_grounding"
        ]
        for task in vlm_tasks:
            tool = ToolRegistry.select_tool_for_task(task, "single")
            self.assertEqual(
                tool["id"],
                "florence2-vlm",
                f"Task '{task}' must route to florence2-vlm."
            )
            self.assertEqual(tool["name"], "Florence-2 Vision-Language Specialist")

        # BigEarthNet preservation
        clc_tool = ToolRegistry.select_tool_for_task("land-cover-classification", "single")
        self.assertEqual(clc_tool["id"], "bigearthnet-classifier")

        # Temporal & Multimodal preservation
        change_tool = ToolRegistry.select_tool_for_task("change-analysis", "bi-temporal")
        self.assertEqual(change_tool["id"], "bitemporal-diff-net")

        fusion_tool = ToolRegistry.select_tool_for_task("optical-sar-analysis", "optical-sar")
        self.assertEqual(fusion_tool["id"], "optical-sar-fusion-net")

    # ------------------------------------------------------------------
    # 3. Query-Intent Classification
    # ------------------------------------------------------------------
    def test_03_query_intent_classification(self):
        """Verify natural language query intents route to Florence-2 vs BigEarthNet."""
        # Captioning queries
        caption_queries = [
            "Describe this image",
            "What is shown in this image?",
            "Give me a scene description",
            "What does this satellite image contain?",
            "Provide a caption for this scene",
        ]
        for q in caption_queries:
            task = self.agent.classify_task(q, "single", 1, ["Optical RGB"])
            self.assertEqual(task, "scene-captioning", f"Query '{q}' must be classified as scene-captioning.")

        # VQA queries
        vqa_queries = [
            "Is there water in this image?",
            "Are there buildings?",
            "What is visible near the center?",
            "Does this image contain vegetation?",
        ]
        for q in vqa_queries:
            task = self.agent.classify_task(q, "single", 1, ["Optical RGB"])
            self.assertEqual(task, "vqa", f"Query '{q}' must be classified as vqa.")

        # Grounding queries
        grounding_queries = [
            "Where are the buildings?",
            "Locate the roads",
            "Show me the water",
            "Find the vegetation",
            "Highlight the airport runway",
        ]
        for q in grounding_queries:
            task = self.agent.classify_task(q, "single", 1, ["Optical RGB"])
            self.assertEqual(task, "text-guided-grounding", f"Query '{q}' must be classified as text-guided-grounding.")

        # Land-cover / BigEarthNet queries (must NOT route to Florence-2)
        clc_queries = [
            "What land-cover classes are present?",
            "Classify this Sentinel-2 image",
            "What type of land cover is this?",
            "Run Corine CLC classification",
        ]
        for q in clc_queries:
            task = self.agent.classify_task(q, "single", 1, ["Sentinel-2 Multispectral (10-Band)"])
            self.assertEqual(task, "land-cover-classification", f"Query '{q}' must be classified as land-cover-classification.")

    # ------------------------------------------------------------------
    # 4. Image Preparation: Native RGB vs Multispectral Composite
    # ------------------------------------------------------------------
    def test_04_image_preparation_native_rgb(self):
        """Verify native RGB image input is ingested directly with honest provenance."""
        pil_img, modality, meta = Florence2InferenceService.prepare_image(self.test_rgb_image)
        self.assertIsInstance(pil_img, Image.Image)
        self.assertEqual(pil_img.mode, "RGB")
        self.assertEqual(pil_img.size, (120, 120))
        self.assertEqual(modality, "Optical RGB")
        self.assertFalse(meta.get("is_multispectral_derived"))

    def test_05_image_preparation_10band_sentinel2(self):
        """Verify 10-band Sentinel-2 raster is converted to authentic B04-B03-B02 RGB composite."""
        pil_img, modality, meta = Florence2InferenceService.prepare_image(self.test_10band_array)
        self.assertIsInstance(pil_img, Image.Image)
        self.assertEqual(pil_img.mode, "RGB")
        self.assertEqual(pil_img.size, (120, 120))
        self.assertIn("Sentinel-2 Derived True-Color RGB Composite", modality)
        self.assertTrue(meta.get("is_multispectral_derived"))

    # ------------------------------------------------------------------
    # 5. Florence-2 Task Outputs (Caption, VQA, Grounding)
    # ------------------------------------------------------------------
    def test_06_caption_output_schema(self):
        """Verify caption output contains required fields and no fake confidence."""
        mock_output = {
            "<MORE_DETAILED_CAPTION>": "An aerial view of rural agricultural land with green fields."
        }
        with patch.object(florence2_service, "_generate", return_value=(mock_output, 0.45)):
            res = florence2_service.caption(self.test_rgb_image, detailed=True)
            self.assertEqual(res["task"], "image_captioning")
            self.assertEqual(res["caption"], "An aerial view of rural agricultural land with green fields.")
            self.assertIn("latency_ms", res)
            self.assertNotIn("confidence", res)

    def test_07_vqa_output_schema_and_sanitization(self):
        """Verify VQA output cleans leading vocabulary artifacts like 'QA>'."""
        mock_output = {
            "<VQA>": "QA>Dense green vegetation and small farm structures."
        }
        with patch.object(florence2_service, "_generate", return_value=(mock_output, 0.32)):
            res = florence2_service.vqa(self.test_rgb_image, "What is visible?")
            self.assertEqual(res["task"], "visual_question_answering")
            self.assertEqual(res["answer"], "Dense green vegetation and small farm structures.")
            self.assertFalse(res["answer"].startswith("QA>"))
            self.assertNotIn("confidence", res)

    def test_08_grounding_output_pixel_coordinates(self):
        """Verify grounding produces bounding boxes in image pixel space without fabricated geocoordinates."""
        mock_output = {
            "<CAPTION_TO_PHRASE_GROUNDING>": {
                "bboxes": [[10.5, 20.0, 50.5, 80.0]],
                "labels": ["roads"]
            }
        }
        with patch.object(florence2_service, "_generate", return_value=(mock_output, 0.38)):
            res = florence2_service.ground(self.test_rgb_image, "roads")
            self.assertEqual(res["task"], "phrase_grounding")
            boxes = res["boundingBoxes"]
            self.assertEqual(len(boxes), 1)
            box = boxes[0]
            self.assertEqual(box["label"], "roads")
            self.assertEqual(box["x"], 10.5)
            self.assertEqual(box["y"], 20.0)
            self.assertEqual(box["width"], 40.0)
            self.assertEqual(box["height"], 60.0)
            self.assertIsNone(box["confidence"], "No fabricated confidence score.")
            self.assertEqual(res["image_dimensions"], [120, 120])

    # ------------------------------------------------------------------
    # 6. Agent Pipeline End-to-End Integration
    # ------------------------------------------------------------------
    def test_09_agent_pipeline_routes_to_florence2_with_trace(self):
        """Verify agent executes Florence-2 with full trace for captioning, VQA, and grounding."""
        mock_cap_res = {
            "model": "Florence-2",
            "model_name": "Florence-2 Vision-Language Specialist",
            "model_source": "microsoft/Florence-2-base",
            "task": "scene-captioning",
            "task_label": "Image Captioning",
            "answer": "An aerial view of green agricultural fields.",
            "confidence": None,
            "evidence": ["VLM Task: Image Captioning", "Input Representation: Optical RGB"],
            "boundingBoxes": None,
            "imageOverlayType": "none",
            "execution_duration_ms": 420,
            "input_modality": "Optical RGB",
            "image_dimensions": [120, 120],
            "is_multispectral_derived": False,
            "is_simulation": False,
        }

        with patch.object(florence2_service, "predict", return_value=mock_cap_res):
            resp = self.agent.execute_pipeline(
                query="Describe this image",
                mode="single",
                images_dict={"single": {"array": np.array(self.test_rgb_image)}}
            )
            self.assertEqual(resp.selectedModel, "Florence-2 Vision-Language Specialist")
            self.assertEqual(resp.taskType, "scene-captioning")
            self.assertEqual(resp.answer, "An aerial view of green agricultural fields.")
            self.assertIsNone(resp.confidence, "Confidence must be None (no fabricated score).")
            self.assertFalse(resp.isSimulation)

            # Check trace steps
            step_titles = [s.title for s in resp.executionSteps]
            self.assertIn("Specialist model selected", step_titles)
            self.assertIn("Model execution", step_titles)
            self.assertIn("Response generation", step_titles)

    def test_10_agent_pipeline_preserves_bigearthnet_for_landcover(self):
        """Verify land-cover queries on 10-band Sentinel-2 input still route to BigEarthNet ResNet-18."""
        resp = self.agent.execute_pipeline(
            query="Classify this Sentinel-2 image",
            mode="single",
            images_dict={
                "single": {
                    "array": self.test_10band_array,
                    "bands": REQUIRED_S2_BANDS,
                    "name": "s2_patch.tif"
                }
            }
        )
        self.assertEqual(resp.selectedModel, "BigEarthNet ResNet-18 Land-Cover Specialist")
        self.assertEqual(resp.taskType, "land-cover-classification")
        self.assertIsNotNone(resp.confidence, "BigEarthNet provides genuine probabilistic confidence.")
        self.assertFalse(resp.isSimulation)

    # ------------------------------------------------------------------
    # 7. Error Handling: Incompatible Input & Model Unavailable
    # ------------------------------------------------------------------
    def test_11_invalid_input_raises_error_no_fake_fallback(self):
        """Verify invalid input raises clear validation error rather than returning fake output."""
        with self.assertRaises(ValueError):
            florence2_service.predict(
                image_source=None,
                query="Describe this image",
                task_type="scene-captioning"
            )

    def test_12_model_unavailable_raises_error_no_fake_fallback(self):
        """Verify missing model raises RuntimeError rather than falling back to static template."""
        with patch.object(florence2_service, "load_model", return_value=(False, "Weights missing")):
            with patch.object(florence2_service, "_is_loaded", False):
                with self.assertRaises(RuntimeError):
                    florence2_service.predict(
                        image_source=self.test_rgb_image,
                        query="Describe this image",
                        task_type="scene-captioning"
                    )


if __name__ == "__main__":
    unittest.main()
