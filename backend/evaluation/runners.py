"""
SatQuery AI - Benchmark Evaluation Runners.
Executes reproducible, evidence-grounded evaluations against real remote sensing benchmarks
when configured, or strictly reports dataset unmount/unavailability without fabricating scores.

Distinguishes:
1. Dataset available + evaluated
2. Dataset available but partially evaluated / skipped
3. Dataset unavailable / not configured
4. Capability implemented but benchmark not yet measured
"""
import os
import time
import uuid
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from backend.evaluation.registry import BenchmarkRegistry, BenchmarkSpec
from backend.evaluation.dataset_adapters import (
    DatasetAdapter,
    BigEarthNetAdapter,
    RSVQAAdapter,
    VRSBenchAdapter,
    CDVQAAdapter,
)
from backend.evaluation.metrics import (
    compute_multilabel_metrics,
    compute_vqa_metrics,
    compute_captioning_metrics,
    METRIC_DEFINITIONS,
)
from backend.models.bigearthnet_loader import (
    bigearthnet_loader,
    BIGEARTHNET_19_CLASSES,
    REQUIRED_S2_BANDS,
)

logger = logging.getLogger("satquery.evaluation.runners")


@dataclass
class EvaluationResult:
    """
    Standardized, reproducible evaluation result container.
    """
    dataset: str
    task: str
    split: str
    samples_total: int
    samples_evaluated: int
    samples_skipped: int
    metrics: Dict[str, Any]
    metric_definitions: Dict[str, Any]
    model: str
    checkpoint: str
    preprocessing: str
    runtime: float
    limitations: List[str]
    status: str  # "evaluated", "partially_evaluated", "dataset_unavailable", "not_configured", "failed"
    timestamp: str
    run_id: str
    seed: Optional[int] = None
    threshold: Optional[float] = None
    adaptation_status: Optional[str] = None
    audit_records: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        return res


class EvaluationRunner:
    """
    Reproducible benchmark evaluation execution runner.
    """

    def __init__(self, registry: Optional[BenchmarkRegistry] = None):
        self.registry = registry or BenchmarkRegistry

    def run_evaluation(
        self,
        benchmark_id: str,
        split: str = "test",
        max_samples: Optional[int] = None,
        threshold: float = 0.5,
        seed: int = 42,
        save_audit: bool = True,
        custom_adapter: Optional[DatasetAdapter] = None,
    ) -> EvaluationResult:
        """
        Executes benchmark evaluation with deterministic seeding and sample auditing.
        Never fabricates metrics.
        """
        run_id = f"eval_{benchmark_id.replace('-', '_')}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        start_time = time.time()

        # Set deterministic seed
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass

        spec = self.registry.get_spec(benchmark_id)
        if not spec:
            return EvaluationResult(
                dataset=benchmark_id,
                task="unknown",
                split=split,
                samples_total=0,
                samples_evaluated=0,
                samples_skipped=0,
                metrics={},
                metric_definitions={},
                model="unknown",
                checkpoint="unknown",
                preprocessing="none",
                runtime=round(time.time() - start_time, 4),
                limitations=[f"Benchmark '{benchmark_id}' is not registered in the benchmark registry."],
                status="not_configured",
                timestamp=timestamp,
                run_id=run_id,
                seed=seed,
                threshold=threshold,
                audit_records=[],
            )

        # Get adapter
        adapter = custom_adapter or self._get_adapter(spec)
        valid, msg = adapter.validate_configuration()
        if not valid:
            return EvaluationResult(
                dataset=spec.name,
                task=spec.task,
                split=split,
                samples_total=0,
                samples_evaluated=0,
                samples_skipped=0,
                metrics={},
                metric_definitions={m: METRIC_DEFINITIONS.get(m, {}) for m in spec.prescribed_metrics},
                model=spec.model_used,
                checkpoint="none",
                preprocessing="none",
                runtime=round(time.time() - start_time, 4),
                limitations=[
                    f"Dataset is not configured or missing on local disk: {msg}",
                    "No benchmark score can be reported until dataset files are configured.",
                ],
                status="dataset_unavailable",
                timestamp=timestamp,
                run_id=run_id,
                seed=seed,
                threshold=threshold,
                adaptation_status=spec.adaptation_status,
                audit_records=[],
            )

        # Load samples
        samples, load_diag = adapter.load_samples(split=split, max_samples=max_samples)
        if not samples:
            return EvaluationResult(
                dataset=spec.name,
                task=spec.task,
                split=split,
                samples_total=0,
                samples_evaluated=0,
                samples_skipped=0,
                metrics={},
                metric_definitions={m: METRIC_DEFINITIONS.get(m, {}) for m in spec.prescribed_metrics},
                model=spec.model_used,
                checkpoint="none",
                preprocessing="none",
                runtime=round(time.time() - start_time, 4),
                limitations=[
                    load_diag.get("message", "No valid evaluation samples found."),
                    "Dataset configuration exists but contains 0 valid samples.",
                ],
                status="dataset_unavailable" if load_diag.get("status") == "dataset_unavailable" else "failed",
                timestamp=timestamp,
                run_id=run_id,
                seed=seed,
                threshold=threshold,
                adaptation_status=spec.adaptation_status,
                audit_records=[],
            )

        # Execute task-specific evaluation
        if spec.task == "multilabel_classification":
            return self._evaluate_bigearthnet(
                spec, samples, split, threshold, seed, run_id, timestamp, start_time, save_audit
            )
        elif spec.task == "vqa":
            return self._evaluate_rsvqa(
                spec, samples, split, threshold, seed, run_id, timestamp, start_time, save_audit
            )
        elif spec.task == "captioning":
            return self._evaluate_vrsbench(
                spec, samples, split, threshold, seed, run_id, timestamp, start_time, save_audit
            )
        elif spec.task == "change_vqa":
            return self._evaluate_cdvqa(
                spec, samples, split, threshold, seed, run_id, timestamp, start_time, save_audit
            )
        else:
            return EvaluationResult(
                dataset=spec.name,
                task=spec.task,
                split=split,
                samples_total=len(samples),
                samples_evaluated=0,
                samples_skipped=len(samples),
                metrics={},
                metric_definitions={},
                model=spec.model_used,
                checkpoint="none",
                preprocessing="none",
                runtime=round(time.time() - start_time, 4),
                limitations=[f"Task '{spec.task}' is currently unsupported by the evaluation runner."],
                status="failed",
                timestamp=timestamp,
                run_id=run_id,
                seed=seed,
                threshold=threshold,
                adaptation_status=spec.adaptation_status,
                audit_records=[],
            )

    def _get_adapter(self, spec: BenchmarkSpec) -> DatasetAdapter:
        if spec.benchmark_id == "bigearthnet-s2":
            return BigEarthNetAdapter()
        elif spec.benchmark_id in ["rsvqa-hr", "rsvqa-lr"]:
            return RSVQAAdapter(spec.benchmark_id)
        elif spec.benchmark_id == "vrsbench":
            return VRSBenchAdapter()
        elif spec.benchmark_id == "cdvqa":
            return CDVQAAdapter()
        else:
            return DatasetAdapter(spec.benchmark_id, spec.env_var)

    def _evaluate_bigearthnet(
        self,
        spec: BenchmarkSpec,
        samples: List[Dict[str, Any]],
        split: str,
        threshold: float,
        seed: int,
        run_id: str,
        timestamp: str,
        start_time: float,
        save_audit: bool,
    ) -> EvaluationResult:
        """
        Runs real BigEarthNet ResNet-18 10-band inference.
        """
        audit_records = []
        y_true_list = []
        y_pred_list = []
        skipped_count = 0

        # Attempt to load model
        is_loaded, status, msg = bigearthnet_loader.load_model()
        checkpoint_name = (
            os.path.basename(bigearthnet_loader.onnx_path or "none")
            if bigearthnet_loader.onnx_path
            else os.path.basename(bigearthnet_loader.checkpoint_path or "none")
        )

        if not is_loaded:
            return EvaluationResult(
                dataset=spec.name,
                task=spec.task,
                split=split,
                samples_total=len(samples),
                samples_evaluated=0,
                samples_skipped=len(samples),
                metrics={},
                metric_definitions={m: METRIC_DEFINITIONS.get(m, {}) for m in spec.prescribed_metrics},
                model=spec.model_used,
                checkpoint=checkpoint_name,
                preprocessing="10-band Sentinel-2 reflectance normalized to [0, 1] at 120x120",
                runtime=round(time.time() - start_time, 4),
                limitations=[
                    f"BigEarthNet ResNet-18 model weights not loaded: {msg}",
                    "Cannot compute neural predictions without model checkpoint.",
                ],
                status="failed",
                timestamp=timestamp,
                run_id=run_id,
                seed=seed,
                threshold=threshold,
                adaptation_status=spec.adaptation_status,
                audit_records=[],
            )

        for s in samples:
            s_id = s.get("sample_id", "unknown")
            gt_labels = s.get("labels")
            t_s = time.time()

            # Load raster data if sample has raw array or file path
            raster_arr = s.get("raster")
            if raster_arr is None and "raster_path" in s and os.path.exists(s["raster_path"]):
                try:
                    if s["raster_path"].endswith(".npy"):
                        raster_arr = np.load(s["raster_path"])
                    elif s["raster_path"].endswith((".tif", ".tiff")):
                        import tifffile
                        raster_arr = tifffile.imread(s["raster_path"])
                except Exception as e:
                    logger.warning("Failed loading raster for sample %s: %s", s_id, e)

            if raster_arr is None:
                skipped_count += 1
                if save_audit:
                    audit_records.append({
                        "sample_id": s_id,
                        "prediction": None,
                        "ground_truth": s.get("label_names", []),
                        "metrics": {},
                        "status": "skipped",
                        "reason": "Raster data file not found or unreadable.",
                        "runtime_ms": round((time.time() - t_s) * 1000, 2),
                    })
                continue

            try:
                probs = bigearthnet_loader.run_inference(raster_arr)
                runtime_ms = round((time.time() - t_s) * 1000, 2)

                y_true_list.append(gt_labels)
                y_pred_list.append(probs)

                # Binary prediction for sample audit
                pred_bin = (probs >= threshold).astype(np.int32)
                pred_classes = [
                    BIGEARTHNET_19_CLASSES[i] for i, val in enumerate(pred_bin) if val == 1
                ]
                sample_em = int(np.array_equal(gt_labels, pred_bin))

                if save_audit:
                    audit_records.append({
                        "sample_id": s_id,
                        "prediction": pred_classes,
                        "ground_truth": s.get("label_names", []),
                        "metrics": {"exact_match": sample_em},
                        "status": "evaluated",
                        "reason": "OK",
                        "runtime_ms": runtime_ms,
                    })
            except Exception as e:
                skipped_count += 1
                if save_audit:
                    audit_records.append({
                        "sample_id": s_id,
                        "prediction": None,
                        "ground_truth": s.get("label_names", []),
                        "metrics": {},
                        "status": "failed",
                        "reason": str(e),
                        "runtime_ms": round((time.time() - t_s) * 1000, 2),
                    })

        evaluated_count = len(y_pred_list)
        if evaluated_count == 0:
            status_str = "failed"
            metrics_dict = {}
        elif evaluated_count < len(samples):
            status_str = "partially_evaluated"
            m_res = compute_multilabel_metrics(
                np.array(y_true_list), np.array(y_pred_list), threshold=threshold, class_names=BIGEARTHNET_19_CLASSES
            )
            metrics_dict = m_res["metrics"]
        else:
            status_str = "evaluated"
            m_res = compute_multilabel_metrics(
                np.array(y_true_list), np.array(y_pred_list), threshold=threshold, class_names=BIGEARTHNET_19_CLASSES
            )
            metrics_dict = m_res["metrics"]

        return EvaluationResult(
            dataset=spec.name,
            task=spec.task,
            split=split,
            samples_total=len(samples),
            samples_evaluated=evaluated_count,
            samples_skipped=skipped_count,
            metrics=metrics_dict,
            metric_definitions={m: METRIC_DEFINITIONS.get(m, {}) for m in spec.prescribed_metrics},
            model=spec.model_used,
            checkpoint=checkpoint_name,
            preprocessing="10-band Sentinel-2 reflectance normalized to [0, 1] at 120x120 spatial resolution",
            runtime=round(time.time() - start_time, 4),
            limitations=[
                f"Evaluated with fixed decision threshold: {threshold}.",
                "Requires genuine 10-band Sentinel-2 spectral order: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12.",
                "Zero RGB padding or channel synthesis permitted.",
            ],
            status=status_str,
            timestamp=timestamp,
            run_id=run_id,
            seed=seed,
            threshold=threshold,
            adaptation_status=spec.adaptation_status,
            audit_records=audit_records if save_audit else [],
        )

    def _evaluate_rsvqa(
        self,
        spec: BenchmarkSpec,
        samples: List[Dict[str, Any]],
        split: str,
        threshold: float,
        seed: int,
        run_id: str,
        timestamp: str,
        start_time: float,
        save_audit: bool,
    ) -> EvaluationResult:
        """
        Runs real VQA evaluation using Florence-2 base pipeline.
        Explicitly records adaptation_status = "not_remote_sensing_finetuned".
        """
        from backend.app.models.florence2_inference import florence2_service

        audit_records = []
        eval_pairs = []
        skipped_count = 0

        for s in samples:
            s_id = s.get("sample_id", "unknown")
            q_text = s.get("question", "")
            gt_ans = s.get("ground_truth", "")
            img_path = s.get("image_path", "")
            cat = s.get("category", "general")
            t_s = time.time()

            if not os.path.exists(img_path):
                skipped_count += 1
                if save_audit:
                    audit_records.append({
                        "sample_id": s_id,
                        "prediction": None,
                        "ground_truth": gt_ans,
                        "metrics": {},
                        "status": "skipped",
                        "reason": f"Image file not found: '{img_path}'.",
                        "runtime_ms": round((time.time() - t_s) * 1000, 2),
                    })
                continue

            try:
                # Execute inference through Florence-2 VQA
                res = florence2_service.predict(img_path, task="vqa", prompt=q_text)
                pred_ans = res.get("text") or res.get("answer") or ""
                runtime_ms = round((time.time() - t_s) * 1000, 2)

                pair = {
                    "ground_truth": gt_ans,
                    "prediction": pred_ans,
                    "category": cat,
                }
                eval_pairs.append(pair)

                # Compute per-sample metrics
                from backend.evaluation.metrics import normalize_text_answer
                is_em = str(pred_ans).strip() == str(gt_ans).strip()
                is_norm = normalize_text_answer(str(pred_ans)) == normalize_text_answer(str(gt_ans))

                if save_audit:
                    audit_records.append({
                        "sample_id": s_id,
                        "prediction": pred_ans,
                        "ground_truth": gt_ans,
                        "metrics": {
                            "exact_match": int(is_em),
                            "normalized_exact_match": int(is_norm),
                        },
                        "status": "evaluated",
                        "reason": "OK",
                        "runtime_ms": runtime_ms,
                    })
            except Exception as e:
                skipped_count += 1
                if save_audit:
                    audit_records.append({
                        "sample_id": s_id,
                        "prediction": None,
                        "ground_truth": gt_ans,
                        "metrics": {},
                        "status": "failed",
                        "reason": str(e),
                        "runtime_ms": round((time.time() - t_s) * 1000, 2),
                    })

        evaluated_count = len(eval_pairs)
        if evaluated_count == 0:
            status_str = "failed"
            metrics_dict = {}
        elif evaluated_count < len(samples):
            status_str = "partially_evaluated"
            m_res = compute_vqa_metrics(eval_pairs)
            metrics_dict = m_res["metrics"]
        else:
            status_str = "evaluated"
            m_res = compute_vqa_metrics(eval_pairs)
            metrics_dict = m_res["metrics"]

        return EvaluationResult(
            dataset=spec.name,
            task=spec.task,
            split=split,
            samples_total=len(samples),
            samples_evaluated=evaluated_count,
            samples_skipped=skipped_count,
            metrics=metrics_dict,
            metric_definitions={m: METRIC_DEFINITIONS.get(m, {}) for m in spec.prescribed_metrics},
            model=spec.model_used,
            checkpoint="microsoft/Florence-2-base",
            preprocessing="Standard Florence-2 RGB processor (scaling, tokenization)",
            runtime=round(time.time() - start_time, 4),
            limitations=[
                "Model is base Florence-2 and NOT remote-sensing fine-tuned (zero-shot evaluation).",
                "Evaluates natural language answer matches via Exact Match and Normalized Exact Match.",
            ],
            status=status_str,
            timestamp=timestamp,
            run_id=run_id,
            seed=seed,
            threshold=threshold,
            adaptation_status="not_remote_sensing_finetuned",
            audit_records=audit_records if save_audit else [],
        )

    def _evaluate_vrsbench(
        self,
        spec: BenchmarkSpec,
        samples: List[Dict[str, Any]],
        split: str,
        threshold: float,
        seed: int,
        run_id: str,
        timestamp: str,
        start_time: float,
        save_audit: bool,
    ) -> EvaluationResult:
        """
        Runs real VRSBench scene description / captioning evaluation.
        Explicitly isolates unsupported tasks.
        """
        from backend.app.models.florence2_inference import florence2_service

        audit_records = []
        eval_pairs = []
        skipped_count = 0

        for s in samples:
            s_id = s.get("sample_id", "unknown")
            gt_cap = s.get("ground_truth", "")
            img_path = s.get("image_path", "")
            task = s.get("task", "captioning")
            t_s = time.time()

            # Unsupported tasks
            if task not in ["captioning", "scene_description"]:
                skipped_count += 1
                if save_audit:
                    audit_records.append({
                        "sample_id": s_id,
                        "prediction": None,
                        "ground_truth": gt_cap,
                        "metrics": {},
                        "status": "skipped",
                        "reason": "unsupported_task",
                        "runtime_ms": round((time.time() - t_s) * 1000, 2),
                    })
                continue

            if not os.path.exists(img_path):
                skipped_count += 1
                if save_audit:
                    audit_records.append({
                        "sample_id": s_id,
                        "prediction": None,
                        "ground_truth": gt_cap,
                        "metrics": {},
                        "status": "skipped",
                        "reason": f"Image file not found: '{img_path}'.",
                        "runtime_ms": round((time.time() - t_s) * 1000, 2),
                    })
                continue

            try:
                res = florence2_service.caption(img_path, detailed=True)
                pred_cap = res.get("caption") or res.get("text") or ""
                runtime_ms = round((time.time() - t_s) * 1000, 2)

                pair = {
                    "ground_truth": gt_cap,
                    "prediction": pred_cap,
                }
                eval_pairs.append(pair)

                from backend.evaluation.metrics import compute_rouge_l, compute_token_overlap
                r_metrics = compute_rouge_l(gt_cap, pred_cap)
                t_metrics = compute_token_overlap(gt_cap, pred_cap)

                if save_audit:
                    audit_records.append({
                        "sample_id": s_id,
                        "prediction": pred_cap,
                        "ground_truth": gt_cap,
                        "metrics": {
                            "rouge_l": r_metrics["f1"],
                            "token_f1": t_metrics["f1"],
                        },
                        "status": "evaluated",
                        "reason": "OK",
                        "runtime_ms": runtime_ms,
                    })
            except Exception as e:
                skipped_count += 1
                if save_audit:
                    audit_records.append({
                        "sample_id": s_id,
                        "prediction": None,
                        "ground_truth": gt_cap,
                        "metrics": {},
                        "status": "failed",
                        "reason": str(e),
                        "runtime_ms": round((time.time() - t_s) * 1000, 2),
                    })

        evaluated_count = len(eval_pairs)
        if evaluated_count == 0:
            status_str = "failed"
            metrics_dict = {}
        elif evaluated_count < len(samples):
            status_str = "partially_evaluated"
            m_res = compute_captioning_metrics(eval_pairs)
            metrics_dict = m_res["metrics"]
        else:
            status_str = "evaluated"
            m_res = compute_captioning_metrics(eval_pairs)
            metrics_dict = m_res["metrics"]

        return EvaluationResult(
            dataset=spec.name,
            task=spec.task,
            split=split,
            samples_total=len(samples),
            samples_evaluated=evaluated_count,
            samples_skipped=skipped_count,
            metrics=metrics_dict,
            metric_definitions={m: METRIC_DEFINITIONS.get(m, {}) for m in spec.prescribed_metrics},
            model=spec.model_used,
            checkpoint="microsoft/Florence-2-base",
            preprocessing="Detailed captioning prompt via Florence-2 processor",
            runtime=round(time.time() - start_time, 4),
            limitations=[
                "Supported subset is strictly scene description and captioning.",
                "Model is base Florence-2 and NOT remote-sensing fine-tuned.",
                "Metrics: ROUGE-L (LCS) and token overlap F1 without external NLP package dependencies.",
            ],
            status=status_str,
            timestamp=timestamp,
            run_id=run_id,
            seed=seed,
            threshold=threshold,
            adaptation_status="not_remote_sensing_finetuned",
            audit_records=audit_records if save_audit else [],
        )

    def _evaluate_cdvqa(
        self,
        spec: BenchmarkSpec,
        samples: List[Dict[str, Any]],
        split: str,
        threshold: float,
        seed: int,
        run_id: str,
        timestamp: str,
        start_time: float,
        save_audit: bool,
    ) -> EvaluationResult:
        """
        Runs real CDVQA bi-temporal visual question answering evaluation.
        """
        from backend.app.models.bitemporal_inference import BiTemporalChangeService

        change_service = BiTemporalChangeService()
        audit_records = []
        eval_pairs = []
        skipped_count = 0

        for s in samples:
            s_id = s.get("sample_id", "unknown")
            q_text = s.get("question", "")
            gt_ans = s.get("ground_truth", "")
            img_t1 = s.get("image_t1")
            img_t2 = s.get("image_t2")
            chg_type = s.get("change_type", "general")
            t_s = time.time()

            # Validate images exist
            t1_exists = (isinstance(img_t1, str) and os.path.exists(img_t1)) or isinstance(img_t1, np.ndarray)
            t2_exists = (isinstance(img_t2, str) and os.path.exists(img_t2)) or isinstance(img_t2, np.ndarray)

            if not (t1_exists and t2_exists):
                skipped_count += 1
                if save_audit:
                    audit_records.append({
                        "sample_id": s_id,
                        "prediction": None,
                        "ground_truth": gt_ans,
                        "metrics": {},
                        "status": "skipped",
                        "reason": "Bi-temporal image pair (T1 or T2) missing or unreadable.",
                        "runtime_ms": round((time.time() - t_s) * 1000, 2),
                    })
                continue

            try:
                res = change_service.predict(
                    query=q_text,
                    inputs={"image": img_t1, "image_b": img_t2},
                )
                pred_ans = res.get("text") or res.get("explanation") or res.get("summary") or ""
                runtime_ms = round((time.time() - t_s) * 1000, 2)

                pair = {
                    "ground_truth": gt_ans,
                    "prediction": pred_ans,
                    "category": chg_type,
                }
                eval_pairs.append(pair)

                from backend.evaluation.metrics import normalize_text_answer
                is_em = str(pred_ans).strip() == str(gt_ans).strip()
                is_norm = normalize_text_answer(str(pred_ans)) == normalize_text_answer(str(gt_ans))

                if save_audit:
                    audit_records.append({
                        "sample_id": s_id,
                        "prediction": pred_ans,
                        "ground_truth": gt_ans,
                        "metrics": {
                            "exact_match": int(is_em),
                            "normalized_exact_match": int(is_norm),
                        },
                        "status": "evaluated",
                        "reason": "OK",
                        "runtime_ms": runtime_ms,
                    })
            except Exception as e:
                skipped_count += 1
                if save_audit:
                    audit_records.append({
                        "sample_id": s_id,
                        "prediction": None,
                        "ground_truth": gt_ans,
                        "metrics": {},
                        "status": "failed",
                        "reason": str(e),
                        "runtime_ms": round((time.time() - t_s) * 1000, 2),
                    })

        evaluated_count = len(eval_pairs)
        if evaluated_count == 0:
            status_str = "failed"
            metrics_dict = {}
        elif evaluated_count < len(samples):
            status_str = "partially_evaluated"
            m_res = compute_vqa_metrics(eval_pairs)
            metrics_dict = m_res["metrics"]
        else:
            status_str = "evaluated"
            m_res = compute_vqa_metrics(eval_pairs)
            metrics_dict = m_res["metrics"]

        return EvaluationResult(
            dataset=spec.name,
            task=spec.task,
            split=split,
            samples_total=len(samples),
            samples_evaluated=evaluated_count,
            samples_skipped=skipped_count,
            metrics=metrics_dict,
            metric_definitions={m: METRIC_DEFINITIONS.get(m, {}) for m in spec.prescribed_metrics},
            model=spec.model_used,
            checkpoint="BiTemporalChangeService + Florence-2",
            preprocessing="Bi-temporal image alignment and differencing pipeline",
            runtime=round(time.time() - start_time, 4),
            limitations=[
                "Bi-temporal pipeline combines algorithmic differencing with zero-shot VLM.",
                "Change detection accuracy is only reported when ground-truth change masks/labels exist.",
            ],
            status=status_str,
            timestamp=timestamp,
            run_id=run_id,
            seed=seed,
            threshold=threshold,
            adaptation_status="algorithmic_differencing_with_vlm",
            audit_records=audit_records if save_audit else [],
        )
