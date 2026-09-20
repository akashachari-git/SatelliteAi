"""
Unit tests for BigEarthNet ResNet-18 Integration (Step 1).
Validates:
1. 3-band RGB cannot enter BigEarthNet (strictly bypassed)
2. Incompatible band count is rejected
3. Unknown or unverified band ordering is rejected
4. Model unavailable does not crash backend
5. Adapter fallback still works and outputs 19 probabilities in [0, 1]
6. Successful model output contains 19 probabilities in [0, 1] via sigmoid
7. Backend health endpoint functions properly
"""
import unittest
import asyncio
import numpy as np

from backend.models.bigearthnet_loader import (
    BIGEARTHNET_19_CLASSES,
    REQUIRED_S2_BANDS,
    ModelStatus,
    InferenceEngine,
    BigEarthNetModelLoader,
    DEFAULT_ONNX_PATH,
    DEFAULT_CHECKPOINT_DIR,
    validate_s2_10band_input,
    normalize_band_name,
)
from backend.app.models.adaptation import (
    get_bigearthnet_status,
    run_deterministic_fallback,
    predict_land_cover_probabilities,
)
from backend.main import health_check


class TestBigEarthNetIntegration(unittest.TestCase):
    """Test suite for BigEarthNet ResNet-18 loading and land cover adaptation."""

    def test_rgb_3band_cannot_enter_bigearthnet(self):
        """3-band RGB images must be rejected from BigEarthNet and redirected to fallback."""
        rgb_image = np.zeros((3, 120, 120), dtype=np.float32)
        rgb_bands = ["red", "green", "blue"]

        # Check validation function directly
        is_valid, msg = validate_s2_10band_input(rgb_image, rgb_bands)
        self.assertFalse(is_valid)
        self.assertIn("RGB", msg)

        # Check public predict interface
        result = predict_land_cover_probabilities(rgb_image, bands=rgb_bands)
        self.assertEqual(result["method"], "Deterministic fallback analysis")
        self.assertFalse(result["is_ai_prediction"])
        self.assertIn("RGB", result["bypass_reason"])
        self.assertEqual(len(result["probabilities"]), 19)

    def test_incompatible_band_count_is_rejected(self):
        """Inputs with band counts other than exactly 10 must be rejected."""
        # Case A: 4-band input (e.g. RGB + NIR)
        four_bands = ["B02", "B03", "B04", "B08"]
        img_4 = np.zeros((4, 120, 120), dtype=np.float32)
        is_valid, msg = validate_s2_10band_input(img_4, four_bands)
        self.assertFalse(is_valid)
        self.assertIn("10", msg)

        # Case B: 12-band full Sentinel-2 input without 10-band selection
        twelve_bands = [
            "B01", "B02", "B03", "B04", "B05", "B06",
            "B07", "B08", "B8A", "B09", "B11", "B12"
        ]
        img_12 = np.zeros((12, 120, 120), dtype=np.float32)
        is_valid_12, msg_12 = validate_s2_10band_input(img_12, twelve_bands)
        self.assertFalse(is_valid_12)

        # Verify through public adapter
        res_4 = predict_land_cover_probabilities(img_4, bands=four_bands)
        self.assertEqual(res_4["method"], "Deterministic fallback analysis")
        self.assertFalse(res_4["is_ai_prediction"])

    def test_unknown_band_ordering_is_rejected(self):
        """Unknown, missing, or misordered band lists must be rejected."""
        img_10 = np.zeros((10, 120, 120), dtype=np.float32)

        # Case A: None / missing band names
        is_valid_none, msg_none = validate_s2_10band_input(img_10, bands=None)
        self.assertFalse(is_valid_none)
        self.assertIn("unverified", msg_none.lower())

        # Case B: Reversed / arbitrary order
        reversed_bands = list(reversed(REQUIRED_S2_BANDS))
        is_valid_rev, msg_rev = validate_s2_10band_input(img_10, bands=reversed_bands)
        self.assertFalse(is_valid_rev)
        self.assertIn("misordered", msg_rev.lower())

        # Verify via predict_land_cover_probabilities
        res_unknown = predict_land_cover_probabilities(img_10, bands=None)
        self.assertEqual(res_unknown["method"], "Deterministic fallback analysis")
        self.assertFalse(res_unknown["is_ai_prediction"])

    def test_model_unavailable_does_not_crash_backend(self):
        """When checkpoint is unmounted, backend must report UNAVAILABLE and not crash."""
        unmounted_loader = BigEarthNetModelLoader(checkpoint_path="nonexistent/path/to/checkpoint")
        self.assertEqual(unmounted_loader.status, ModelStatus.UNAVAILABLE)
        self.assertFalse(unmounted_loader.is_available)

        # Even with valid 10-band input, unmounted model gracefully falls back to deterministic analysis
        valid_10_img = np.zeros((10, 120, 120), dtype=np.float32)
        result = predict_land_cover_probabilities(valid_10_img, bands=REQUIRED_S2_BANDS, loader=unmounted_loader)

        self.assertIsNotNone(result)
        self.assertEqual(result["method"], "Deterministic fallback analysis")
        self.assertFalse(result["is_ai_prediction"])
        self.assertEqual(len(result["probabilities"]), 19)

    def test_adapter_fallback_still_works(self):
        """Deterministic fallback analysis must compute 19 valid probabilities in [0.0, 1.0]."""
        test_img = np.random.uniform(0.0, 1.0, size=(3, 64, 64)).astype(np.float32)
        fallback = run_deterministic_fallback(
            image_array=test_img,
            bands=["B04", "B03", "B02"],
            bypass_reason="Test heuristic verification",
        )

        self.assertEqual(fallback["method"], "Deterministic fallback analysis")
        self.assertFalse(fallback["is_ai_prediction"])
        self.assertEqual(len(fallback["classes"]), 19)
        self.assertEqual(len(fallback["probabilities"]), 19)
        self.assertEqual(len(fallback["probability_vector"]), 19)

        # Ensure all 19 probabilities are within [0.0, 1.0]
        for cls_name, prob in fallback["probabilities"].items():
            self.assertIn(cls_name, BIGEARTHNET_19_CLASSES)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

        for prob in fallback["probability_vector"]:
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

        self.assertGreater(len(fallback["top_predictions"]), 0)

    def test_successful_model_output_contract(self):
        """
        Validates the model contract using a mock classifier:
        Input: (B, 10, H, W) -> 19 logits -> Sigmoid -> 19 probabilities in [0, 1].
        """
        # Create a mock loader that returns simulated 19 raw logits
        class MockClassifier:
            def __call__(self, x):
                # Returns raw logits (some negative, some positive)
                # Raw logits: [-2.5, 1.2, -0.8, 3.1, ...]
                batch_size = x.shape[0] if (hasattr(x, "ndim") and x.ndim == 4) else 1
                logits = np.array([
                    -2.5, 1.2, -0.8, 3.1, -1.5, 0.5, -3.0, 2.0, -0.2, 0.9,
                    -1.1, 0.3, -2.0, 1.8, -0.5, 0.1, -1.9, 2.7, -0.4
                ], dtype=np.float32)
                if batch_size > 1:
                    logits = np.tile(logits, (batch_size, 1))
                return logits

        class MockLoader(BigEarthNetModelLoader):
            def __init__(self):
                super().__init__()
                self._status = ModelStatus.AVAILABLE
                self._model = MockClassifier()

            @property
            def is_available(self):
                return True

            def run_inference(self, input_tensor):
                # Apply model and sigmoid externally
                logits = self._model(input_tensor)
                # Sigmoid: 1 / (1 + exp(-logits))
                probs = 1.0 / (1.0 + np.exp(-logits))
                return probs

        mock_loader = MockLoader()
        s2_input = np.random.uniform(0.0, 1.0, size=(10, 120, 120)).astype(np.float32)

        result = predict_land_cover_probabilities(
            image_input=s2_input,
            bands=REQUIRED_S2_BANDS,
            loader=mock_loader,
        )

        # Verify contract compliance
        self.assertEqual(result["method"], "BigEarthNet ResNet-18 prediction")
        self.assertTrue(result["is_ai_prediction"])
        self.assertEqual(result["model_status"], "AVAILABLE")
        self.assertEqual(len(result["probabilities"]), 19)
        self.assertEqual(len(result["probability_vector"]), 19)

        # Verify probabilities in [0, 1]
        for cls_name, prob in result["probabilities"].items():
            self.assertIn(cls_name, BIGEARTHNET_19_CLASSES)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

        for prob in result["probability_vector"]:
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

        # Verify classes match nomenclature
        self.assertEqual(result["classes"], BIGEARTHNET_19_CLASSES)

    def test_backend_health_check(self):
        """Backend health endpoint must report healthy status without exceptions."""
        res = asyncio.run(health_check())
        self.assertEqual(res.status, "healthy")
        self.assertEqual(res.version, "2.0.0")
        self.assertGreaterEqual(res.activeModelsCount, 0)


class TestBigEarthNetCheckpointLoading(unittest.TestCase):
    """
    Step 3 Verification: Genuinely tests the official BigEarthNet ResNet-18 checkpoint.
    Verifies:
    1. Checkpoint loads successfully.
    2. strict=True weight compatibility: 0 missing keys, 0 unexpected keys.
    3. Parameter count is exactly 11,208,211.
    4. Model accepts tensor shape (1, 10, 120, 120).
    5. Forward pass completes on CPU.
    6. Output has exactly 19 logits.
    7. Sigmoid probabilities are in [0, 1].
    8. 3-band RGB is rejected/bypassed rather than padded.
    """

    @classmethod
    def setUpClass(cls):
        import os
        from backend.models.bigearthnet_loader import DEFAULT_CHECKPOINT_DIR

        cls.checkpoint_path = DEFAULT_CHECKPOINT_DIR
        cls.checkpoint_exists = os.path.isdir(cls.checkpoint_path) and os.path.exists(
            os.path.join(cls.checkpoint_path, "model.safetensors")
        )

    def test_01_checkpoint_loads_successfully(self):
        """Verify checkpoint loads via BigEarthNetv2_0_ImageClassifier.from_pretrained."""
        if not self.checkpoint_exists:
            self.skipTest(f"Checkpoint not found at {self.checkpoint_path}")
        from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
        model = BigEarthNetv2_0_ImageClassifier.from_pretrained(self.checkpoint_path)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, BigEarthNetv2_0_ImageClassifier)

    def test_02_strict_weight_compatibility(self):
        """Verify strict=True loading yields 0 missing keys and 0 unexpected keys."""
        if not self.checkpoint_exists:
            self.skipTest(f"Checkpoint not found at {self.checkpoint_path}")
        import os
        import safetensors.torch
        from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
        model = BigEarthNetv2_0_ImageClassifier.from_pretrained(self.checkpoint_path)
        weights = safetensors.torch.load_file(os.path.join(self.checkpoint_path, "model.safetensors"))
        load_result = model.load_state_dict(weights, strict=True)
        self.assertEqual(len(load_result.missing_keys), 0, f"Missing keys: {load_result.missing_keys}")
        self.assertEqual(len(load_result.unexpected_keys), 0, f"Unexpected keys: {load_result.unexpected_keys}")

    def test_03_parameter_count(self):
        """Verify parameter count remains exactly 11,208,211."""
        if not self.checkpoint_exists:
            self.skipTest(f"Checkpoint not found at {self.checkpoint_path}")
        from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
        model = BigEarthNetv2_0_ImageClassifier.from_pretrained(self.checkpoint_path)
        param_count = sum(p.numel() for p in model.parameters())
        self.assertEqual(param_count, 11208211)

    def test_04_forward_pass_and_tensor_shape(self):
        """Verify model accepts (1, 10, 120, 120), runs on CPU, outputs 19 logits with sigmoid in [0, 1]."""
        if not self.checkpoint_exists:
            self.skipTest(f"Checkpoint not found at {self.checkpoint_path}")
        import torch
        from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
        model = BigEarthNetv2_0_ImageClassifier.from_pretrained(self.checkpoint_path)
        model.eval()
        model.to("cpu")

        # Deterministic synthetic 10-band tensor
        x = torch.ones((1, 10, 120, 120), dtype=torch.float32) * 0.5
        with torch.no_grad():
            logits = model(x)

        # Output shape must be (1, 19)
        self.assertEqual(logits.shape, (1, 19))
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

        # Sigmoid probabilities must be in [0, 1]
        probs = torch.sigmoid(logits)
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())

    def test_05_rgb_rejection_no_padding(self):
        """Verify 3-band RGB is rejected rather than padded."""
        rgb_tensor = np.zeros((3, 120, 120), dtype=np.float32)
        is_valid, msg = validate_s2_10band_input(rgb_tensor, ["B04", "B03", "B02"])
        self.assertFalse(is_valid)
        self.assertIn("RGB", msg)



class TestBigEarthNetONNX(unittest.TestCase):
    """
    Step 4 Verification: ONNX Conversion and Numerical Equivalence.
    1. ONNX file exists on disk.
    2. ONNX model loads successfully with onnxruntime.
    3. ONNX Runtime uses CPUExecutionProvider.
    4. Input tensor has 10 channels.
    5. Output tensor has 19 logits.
    6. PyTorch vs ONNX numerical equivalence (rtol=1e-4, atol=1e-5).
    7. Same deterministic input produces equivalent outputs.
    8. Sigmoid probabilities remain in [0, 1].
    9. RGB input is rejected/bypassed by safety validation.
    10. Loader uses ONNX as primary and falls back cleanly to PyTorch.
    """

    @classmethod
    def setUpClass(cls):
        import os
        import onnxruntime as ort
        from backend.models.bigearthnet_loader import DEFAULT_ONNX_PATH, DEFAULT_CHECKPOINT_DIR

        cls.onnx_path = DEFAULT_ONNX_PATH
        cls.onnx_exists = os.path.isfile(cls.onnx_path)
        cls.checkpoint_path = DEFAULT_CHECKPOINT_DIR
        cls.checkpoint_exists = os.path.isdir(cls.checkpoint_path) and os.path.isfile(
            os.path.join(cls.checkpoint_path, "model.safetensors")
        )
        if cls.onnx_exists:
            cls.session = ort.InferenceSession(cls.onnx_path, providers=["CPUExecutionProvider"])
        else:
            cls.session = None

    def test_01_onnx_file_exists(self):
        """Verify the ONNX model file exists on disk with expected size."""
        import os

        self.assertTrue(self.onnx_exists, f"ONNX file not found at {self.onnx_path}")
        file_size = os.path.getsize(self.onnx_path)
        self.assertGreater(file_size, 10 * 1024 * 1024, f"ONNX file size too small: {file_size} bytes")

    def test_02_onnx_model_loads_successfully(self):
        """Verify ONNX model loads successfully via onnxruntime."""
        if not self.onnx_exists:
            self.skipTest(f"ONNX model not found at {self.onnx_path}")
        self.assertIsNotNone(self.session)

    def test_03_onnx_runtime_cpu_provider(self):
        """Verify ONNX Runtime uses CPUExecutionProvider."""
        if not self.onnx_exists or self.session is None:
            self.skipTest("ONNX session not initialized")
        providers = self.session.get_providers()
        self.assertIn("CPUExecutionProvider", providers)

    def test_04_input_10_channels_output_19_logits(self):
        """Verify ONNX input is 10 channels float32 and output is 19 logits float32."""
        if not self.onnx_exists or self.session is None:
            self.skipTest("ONNX session not initialized")
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]

        self.assertEqual(input_info.shape[1], 10, f"Expected 10 input channels, got {input_info.shape}")
        self.assertEqual(input_info.shape[2], 120)
        self.assertEqual(input_info.shape[3], 120)
        self.assertEqual(input_info.type, "tensor(float)")

        self.assertEqual(output_info.shape[1], 19, f"Expected 19 logits output, got {output_info.shape}")
        self.assertEqual(output_info.type, "tensor(float)")

    def test_05_pytorch_vs_onnx_numerical_equivalence(self):
        """
        MANDATORY: Verify PyTorch vs ONNX numerical equivalence on identical deterministic input.
        Target: np.testing.assert_allclose(torch_logits, onnx_logits, rtol=1e-4, atol=1e-5).
        """
        if not self.onnx_exists or not self.checkpoint_exists:
            self.skipTest("Both ONNX model and PyTorch checkpoint must exist for numerical comparison")

        import torch
        from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

        # 1. Load PyTorch model
        pt_model = BigEarthNetv2_0_ImageClassifier.from_pretrained(self.checkpoint_path)
        pt_model.eval()
        pt_model.to("cpu")

        # 2. Generate deterministic input tensor
        torch.manual_seed(42)
        dummy_input = torch.ones((1, 10, 120, 120), dtype=torch.float32) * 0.5

        # 3. PyTorch forward pass
        with torch.no_grad():
            torch_logits = pt_model(dummy_input).cpu().numpy()

        # 4. ONNX forward pass
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        ort_logits = self.session.run([output_name], {input_name: dummy_input.numpy()})[0]

        # 5. Numerical error analysis
        abs_error = np.abs(torch_logits - ort_logits)
        max_abs = float(np.max(abs_error))
        mean_abs = float(np.mean(abs_error))

        # 6. Strict assertion
        np.testing.assert_allclose(
            torch_logits,
            ort_logits,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"Numerical mismatch: max_abs={max_abs:.8e}, mean_abs={mean_abs:.8e}",
        )
        self.assertLess(max_abs, 1e-4)

    def test_06_same_deterministic_input_equivalent_probabilities(self):
        """Verify sigmoid probabilities from ONNX and PyTorch are numerically equivalent."""
        if not self.onnx_exists or not self.checkpoint_exists:
            self.skipTest("Both models required")

        import torch
        from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

        pt_model = BigEarthNetv2_0_ImageClassifier.from_pretrained(self.checkpoint_path)
        pt_model.eval()
        pt_model.to("cpu")

        dummy_input = torch.ones((1, 10, 120, 120), dtype=torch.float32) * 0.5
        with torch.no_grad():
            pt_logits = pt_model(dummy_input).cpu().numpy()
        pt_probs = 1.0 / (1.0 + np.exp(-pt_logits))

        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        ort_logits = self.session.run([output_name], {input_name: dummy_input.numpy()})[0]
        ort_probs = 1.0 / (1.0 + np.exp(-ort_logits))

        np.testing.assert_allclose(pt_probs, ort_probs, rtol=1e-4, atol=1e-5)

    def test_07_sigmoid_probabilities_in_range(self):
        """Verify all ONNX output probabilities after sigmoid are strictly in [0.0, 1.0]."""
        if not self.onnx_exists or self.session is None:
            self.skipTest("ONNX session not initialized")

        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        dummy = np.random.uniform(0.0, 1.0, size=(1, 10, 120, 120)).astype(np.float32)
        ort_logits = self.session.run([output_name], {input_name: dummy})[0]
        ort_probs = 1.0 / (1.0 + np.exp(-ort_logits))

        self.assertEqual(ort_probs.shape, (1, 19))
        self.assertTrue((ort_probs >= 0.0).all())
        self.assertTrue((ort_probs <= 1.0).all())
        self.assertFalse(np.isnan(ort_probs).any())

    def test_08_rgb_input_rejected_bypassed(self):
        """Verify RGB input is rejected by safety validation and routed to fallback."""
        rgb_tensor = np.zeros((3, 120, 120), dtype=np.float32)
        is_valid, msg = validate_s2_10band_input(rgb_tensor, ["B04", "B03", "B02"])
        self.assertFalse(is_valid)
        self.assertIn("RGB", msg)

        result = predict_land_cover_probabilities(rgb_tensor, bands=["B04", "B03", "B02"])
        self.assertEqual(result["method"], "Deterministic fallback analysis")
        self.assertFalse(result["is_ai_prediction"])

    def test_09_loader_primary_onnx_and_pytorch_fallback(self):
        """Verify BigEarthNetModelLoader uses ONNX primarily and falls back to PyTorch."""
        # 1. Primary ONNX test
        loader = BigEarthNetModelLoader(onnx_path=self.onnx_path, checkpoint_path=self.checkpoint_path)
        success, status, msg = loader.load_model()
        self.assertTrue(success)
        self.assertEqual(loader.status, ModelStatus.AVAILABLE)
        self.assertEqual(loader.engine, InferenceEngine.ONNX)

        # Run inference via ONNX
        sample_input = np.ones((10, 120, 120), dtype=np.float32) * 0.5
        probs = loader.run_inference(sample_input)
        self.assertEqual(len(probs), 19)
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

        # 2. PyTorch Fallback test
        fallback_loader = BigEarthNetModelLoader(
            onnx_path="nonexistent/model.onnx",
            checkpoint_path=self.checkpoint_path,
        )
        fb_success, fb_status, fb_msg = fallback_loader.load_model()
        self.assertTrue(fb_success)
        self.assertEqual(fallback_loader.status, ModelStatus.AVAILABLE)
        self.assertEqual(fallback_loader.engine, InferenceEngine.PYTORCH)

        fb_probs = fallback_loader.run_inference(sample_input)
        self.assertEqual(len(fb_probs), 19)
        self.assertTrue((fb_probs >= 0.0).all() and (fb_probs <= 1.0).all())

        # Check numerical agreement between loader ONNX and loader PyTorch
        np.testing.assert_allclose(probs, fb_probs, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

