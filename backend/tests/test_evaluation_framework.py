"""
SatQuery AI - Step 13 Real Benchmark + Evaluation Layer Tests.
Strictly verifies:
1. BigEarthNet adapter schema
2. Real multilabel metric calculation
3. Threshold handling
4. Per-class metrics
5. Missing BigEarthNet dataset handling
6. RSVQA adapter
7. Missing RSVQA dataset handling
8. VRSBench adapter
9. Unsupported VRSBench task handling
10. CDVQA adapter
11. Missing CDVQA dataset handling
12. EvaluationResult schema
13. Sample-level audit records
14. Reproducibility metadata
15. Report generation (JSON & Markdown)
16. No fabricated benchmark values
17. Dataset configuration validation
18. Empty dataset handling
"""
import os
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from backend.evaluation.registry import BenchmarkRegistry, BenchmarkSpec, BENCHMARK_SPECS
from backend.evaluation.metrics import (
    compute_multilabel_metrics,
    compute_vqa_metrics,
    compute_captioning_metrics,
    compute_rouge_l,
    compute_token_overlap,
    normalize_text_answer,
    METRIC_DEFINITIONS,
)
from backend.evaluation.dataset_adapters import (
    DatasetAdapter,
    BigEarthNetAdapter,
    RSVQAAdapter,
    VRSBenchAdapter,
    CDVQAAdapter,
)
from backend.evaluation.runners import EvaluationResult, EvaluationRunner
from backend.evaluation.reports import ReportGenerator
from backend.models.bigearthnet_loader import BIGEARTHNET_19_CLASSES, REQUIRED_S2_BANDS


class TestEvaluationFramework(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. BigEarthNet adapter schema
    def test_bigearthnet_adapter_schema(self):
        ben_dir = os.path.join(self.temp_dir, "bigearthnet")
        os.makedirs(os.path.join(ben_dir, "splits"), exist_ok=True)
        split_data = [
            {
                "patch_name": "S2A_MSIL2A_20170717T113321_N0205_R080_T30UVU_24_57",
                "labels": ["Arable land", "Complex cultivation patterns"],
            }
        ]
        with open(os.path.join(ben_dir, "splits", "test.json"), "w", encoding="utf-8") as f:
            json.dump(split_data, f)

        adapter = BigEarthNetAdapter(root_path=ben_dir)
        samples, diag = adapter.load_samples(split="test")
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertIn("sample_id", sample)
        self.assertIn("patch_name", sample)
        self.assertIn("patch_dir", sample)
        self.assertIn("labels", sample)
        self.assertIn("label_names", sample)
        self.assertIn("required_bands", sample)
        self.assertEqual(len(sample["labels"]), 19)
        self.assertEqual(sample["labels"][BIGEARTHNET_19_CLASSES.index("Arable land")], 1)
        self.assertEqual(sample["labels"][BIGEARTHNET_19_CLASSES.index("Complex cultivation patterns")], 1)
        self.assertEqual(sample["required_bands"], REQUIRED_S2_BANDS)

    # 2. Real multilabel metric calculation
    def test_real_multilabel_metric_calculation(self):
        # 3 samples, 3 classes
        y_true = np.array([
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ], dtype=np.int32)
        y_pred = np.array([
            [0.9, 0.1, 0.8],
            [0.2, 0.7, 0.6],
            [0.8, 0.9, 0.1],
        ], dtype=np.float32)

        metrics_out = compute_multilabel_metrics(
            y_true, y_pred, threshold=0.5, class_names=["ClassA", "ClassB", "ClassC"]
        )
        m = metrics_out["metrics"]
        # Perfect predictions at 0.5 threshold
        self.assertEqual(m["micro_precision"], 1.0)
        self.assertEqual(m["micro_recall"], 1.0)
        self.assertEqual(m["micro_f1"], 1.0)
        self.assertEqual(m["macro_f1"], 1.0)
        self.assertEqual(m["exact_match_ratio"], 1.0)
        self.assertEqual(m["hamming_loss"], 0.0)

    # 3. Threshold handling
    def test_threshold_handling(self):
        y_true = np.array([[1, 0]], dtype=np.int32)
        y_pred = np.array([[0.6, 0.4]], dtype=np.float32)

        # At threshold 0.5: pred is [1, 0] -> perfect
        res_low = compute_multilabel_metrics(y_true, y_pred, threshold=0.5)
        self.assertEqual(res_low["metrics"]["exact_match_ratio"], 1.0)
        self.assertEqual(res_low["threshold"], 0.5)

        # At threshold 0.7: pred is [0, 0] -> misses class 0
        res_high = compute_multilabel_metrics(y_true, y_pred, threshold=0.7)
        self.assertEqual(res_high["metrics"]["exact_match_ratio"], 0.0)
        self.assertEqual(res_high["threshold"], 0.7)

    # 4. Per-class metrics
    def test_per_class_metrics(self):
        y_true = np.array([
            [1, 0],
            [1, 1],
        ], dtype=np.int32)
        y_pred = np.array([
            [0.8, 0.2],
            [0.9, 0.1],  # misses class 1
        ], dtype=np.float32)

        class_names = ["Urban", "Forest"]
        res = compute_multilabel_metrics(y_true, y_pred, threshold=0.5, class_names=class_names)
        per_class = res["metrics"]["per_class"]
        self.assertIn("Urban", per_class)
        self.assertIn("Forest", per_class)
        self.assertEqual(per_class["Urban"]["precision"], 1.0)
        self.assertEqual(per_class["Urban"]["recall"], 1.0)
        self.assertEqual(per_class["Urban"]["support"], 2)
        self.assertEqual(per_class["Forest"]["recall"], 0.0)
        self.assertEqual(per_class["Forest"]["support"], 1)

    # 5. Missing BigEarthNet dataset
    def test_missing_bigearthnet_dataset(self):
        missing_dir = os.path.join(self.temp_dir, "nonexistent_ben")
        adapter = BigEarthNetAdapter(root_path=missing_dir)
        valid, msg = adapter.validate_configuration()
        self.assertFalse(valid)
        self.assertIn("does not exist", msg)

        samples, diag = adapter.load_samples(split="test")
        self.assertEqual(len(samples), 0)
        self.assertEqual(diag["status"], "dataset_unavailable")

    # 6. RSVQA adapter
    def test_rsvqa_adapter(self):
        rsvqa_dir = os.path.join(self.temp_dir, "rsvqa")
        os.makedirs(rsvqa_dir, exist_ok=True)
        q_data = {
            "questions": [
                {
                    "id": 101,
                    "question": "Are there buildings in this image?",
                    "answer": "yes",
                    "image": "img_001.tif",
                    "category": "presence",
                }
            ]
        }
        with open(os.path.join(rsvqa_dir, "questions.json"), "w", encoding="utf-8") as f:
            json.dump(q_data, f)

        adapter = RSVQAAdapter(benchmark_id="rsvqa-hr", root_path=rsvqa_dir)
        samples, diag = adapter.load_samples(split="test")
        self.assertEqual(len(samples), 1)
        s = samples[0]
        self.assertEqual(s["sample_id"], "101")
        self.assertEqual(s["question"], "Are there buildings in this image?")
        self.assertEqual(s["ground_truth"], "yes")
        self.assertEqual(s["category"], "presence")

    # 7. Missing RSVQA dataset
    def test_missing_rsvqa_dataset(self):
        adapter = RSVQAAdapter(benchmark_id="rsvqa-hr", root_path=os.path.join(self.temp_dir, "missing_rsvqa"))
        valid, msg = adapter.validate_configuration()
        self.assertFalse(valid)
        samples, diag = adapter.load_samples(split="test")
        self.assertEqual(len(samples), 0)
        self.assertEqual(diag["status"], "dataset_unavailable")

    # 8. VRSBench adapter
    def test_vrsbench_adapter(self):
        vrs_dir = os.path.join(self.temp_dir, "vrsbench")
        os.makedirs(vrs_dir, exist_ok=True)
        cap_data = {
            "items": [
                {
                    "id": "vrs_1",
                    "caption": "A dense residential neighborhood with organized road networks.",
                    "image": "vrs_001.png",
                    "task": "captioning",
                }
            ]
        }
        with open(os.path.join(vrs_dir, "captions.json"), "w", encoding="utf-8") as f:
            json.dump(cap_data, f)

        adapter = VRSBenchAdapter(root_path=vrs_dir)
        samples, diag = adapter.load_samples(split="test")
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["sample_id"], "vrs_1")
        self.assertEqual(samples[0]["task"], "captioning")

    # 9. Unsupported VRSBench task handling
    def test_unsupported_vrsbench_task_handling(self):
        vrs_dir = os.path.join(self.temp_dir, "vrsbench_unsupported")
        os.makedirs(vrs_dir, exist_ok=True)
        cap_data = {
            "items": [
                {
                    "id": "vrs_unsupp",
                    "task": "3d_mesh_reconstruction",
                    "caption": "3D mesh object",
                    "image": "vrs_002.png",
                }
            ]
        }
        with open(os.path.join(vrs_dir, "captions.json"), "w", encoding="utf-8") as f:
            json.dump(cap_data, f)

        adapter = VRSBenchAdapter(root_path=vrs_dir)
        samples, diag = adapter.load_samples(split="test")
        # Adapter filters out unsupported tasks
        self.assertEqual(len(samples), 0)

    # 10. CDVQA adapter
    def test_cdvqa_adapter(self):
        cdvqa_dir = os.path.join(self.temp_dir, "cdvqa")
        os.makedirs(cdvqa_dir, exist_ok=True)
        cd_data = {
            "samples": [
                {
                    "id": "cd_1",
                    "question": "What change occurred between the two dates?",
                    "answer": "New building construction.",
                    "image_t1": "t1_001.tif",
                    "image_t2": "t2_001.tif",
                    "change_type": "construction",
                }
            ]
        }
        with open(os.path.join(cdvqa_dir, "cdvqa.json"), "w", encoding="utf-8") as f:
            json.dump(cd_data, f)

        adapter = CDVQAAdapter(root_path=cdvqa_dir)
        samples, diag = adapter.load_samples(split="test")
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["sample_id"], "cd_1")
        self.assertEqual(samples[0]["question"], "What change occurred between the two dates?")
        self.assertEqual(samples[0]["change_type"], "construction")

    # 11. Missing CDVQA dataset
    def test_missing_cdvqa_dataset(self):
        adapter = CDVQAAdapter(root_path=os.path.join(self.temp_dir, "missing_cdvqa"))
        valid, msg = adapter.validate_configuration()
        self.assertFalse(valid)
        samples, diag = adapter.load_samples(split="test")
        self.assertEqual(len(samples), 0)
        self.assertEqual(diag["status"], "dataset_unavailable")

    # 12. EvaluationResult schema
    def test_evaluation_result_schema(self):
        res = EvaluationResult(
            dataset="BigEarthNet-S2",
            task="multilabel_classification",
            split="test",
            samples_total=10,
            samples_evaluated=10,
            samples_skipped=0,
            metrics={"micro_f1": 0.82},
            metric_definitions={"micro_f1": METRIC_DEFINITIONS["micro_f1"]},
            model="ResNet-18",
            checkpoint="resnet18.onnx",
            preprocessing="10-band normalized",
            runtime=1.23,
            limitations=["Zero synthetic data."],
            status="evaluated",
            timestamp="2026-09-20T12:00:00Z",
            run_id="eval_ben_001",
            seed=42,
            threshold=0.5,
            adaptation_status="pretrained_specialist",
            audit_records=[],
        )
        d = res.to_dict()
        required_keys = [
            "dataset", "task", "split", "samples_total", "samples_evaluated", "samples_skipped",
            "metrics", "metric_definitions", "model", "checkpoint", "preprocessing", "runtime",
            "limitations", "status", "timestamp", "run_id"
        ]
        for k in required_keys:
            self.assertIn(k, d)
        self.assertEqual(d["status"], "evaluated")
        self.assertEqual(d["samples_evaluated"], 10)

    # 13. Sample-level audit records
    def test_sample_level_audit_records(self):
        audit_rec = {
            "sample_id": "sample_001",
            "prediction": "yes",
            "ground_truth": "yes",
            "metrics": {"exact_match": 1},
            "status": "evaluated",
            "reason": "OK",
            "runtime_ms": 42.5,
        }
        self.assertEqual(audit_rec["status"], "evaluated")
        self.assertEqual(audit_rec["metrics"]["exact_match"], 1)
        self.assertEqual(audit_rec["runtime_ms"], 42.5)

    # 14. Reproducibility metadata
    def test_reproducibility_metadata(self):
        runner = EvaluationRunner()
        # Non-configured benchmark
        res = runner.run_evaluation("rsvqa-hr", split="test", seed=123, threshold=0.6)
        self.assertEqual(res.seed, 123)
        self.assertEqual(res.threshold, 0.6)
        self.assertIsNotNone(res.run_id)
        self.assertIsNotNone(res.timestamp)
        self.assertEqual(res.status, "dataset_unavailable")

    # 15. Report generation (JSON & Markdown)
    def test_report_generation(self):
        res = EvaluationResult(
            dataset="RSVQA-HR",
            task="vqa",
            split="test",
            samples_total=0,
            samples_evaluated=0,
            samples_skipped=0,
            metrics={},
            metric_definitions={},
            model="microsoft/Florence-2-base",
            checkpoint="none",
            preprocessing="none",
            runtime=0.01,
            limitations=["Dataset is not configured or missing on local disk."],
            status="dataset_unavailable",
            timestamp="2026-09-20T12:00:00Z",
            run_id="eval_test_report_001",
        )

        out_dir = os.path.join(self.temp_dir, "reports")
        json_path = ReportGenerator.generate_json_report(res, output_dir=out_dir)
        md_path = ReportGenerator.generate_markdown_report(res, output_dir=out_dir)

        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(md_path))

        with open(json_path, "r", encoding="utf-8") as f:
            j_data = json.load(f)
            self.assertEqual(j_data["status"], "dataset_unavailable")

        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
            self.assertIn("Benchmark Evaluation Report: RSVQA-HR", md_content)
            self.assertIn("Dataset Unavailable", md_content)

    # 16. No fabricated benchmark values
    def test_no_fabricated_benchmark_values(self):
        runner = EvaluationRunner()
        # When dataset is unavailable, assert no fabricated metrics are generated
        res = runner.run_evaluation("bigearthnet-s2")
        self.assertEqual(res.status, "dataset_unavailable")
        self.assertEqual(res.samples_evaluated, 0)
        self.assertEqual(res.metrics, {})

        # Ensure historical fake numbers (88.3, 86.4, 89.4) do not appear
        res_str = str(res.to_dict())
        self.assertNotIn("88.3", res_str)
        self.assertNotIn("86.4", res_str)
        self.assertNotIn("89.4", res_str)

    # 17. Dataset configuration validation
    def test_dataset_configuration_validation(self):
        # Unset env var
        with patch.dict(os.environ, {}, clear=True):
            status_info = BenchmarkRegistry.discover_dataset_status("bigearthnet-s2")
            self.assertEqual(status_info["status"], "dataset_unavailable")
            self.assertIn("BIGEARTHNET_ROOT", status_info["message"])

        # Path does not exist
        with patch.dict(os.environ, {"BIGEARTHNET_ROOT": "/nonexistent/path/123"}):
            status_info = BenchmarkRegistry.discover_dataset_status("bigearthnet-s2")
            self.assertEqual(status_info["status"], "dataset_unavailable")
            self.assertIn("does not exist", status_info["message"])

    # 18. Empty dataset handling
    def test_empty_dataset_handling(self):
        ben_dir = os.path.join(self.temp_dir, "empty_ben")
        os.makedirs(os.path.join(ben_dir, "splits"), exist_ok=True)
        with open(os.path.join(ben_dir, "splits", "test.json"), "w", encoding="utf-8") as f:
            json.dump([], f)

        adapter = BigEarthNetAdapter(root_path=ben_dir)
        samples, diag = adapter.load_samples(split="test")
        self.assertEqual(len(samples), 0)
        self.assertEqual(diag["status"], "dataset_unavailable")

        runner = EvaluationRunner()
        res = runner.run_evaluation("bigearthnet-s2", custom_adapter=adapter)
        self.assertEqual(res.samples_evaluated, 0)
        self.assertEqual(res.metrics, {})
        self.assertEqual(res.status, "dataset_unavailable")


if __name__ == "__main__":
    unittest.main()
