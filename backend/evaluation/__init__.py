"""
SatQuery AI - Evaluation & Benchmarking Layer.
Provides reproducible evaluation pipelines, dataset adapters, metrics,
reporters, and benchmark registry for remote sensing benchmarks.
"""
from backend.evaluation.registry import (
    BenchmarkSpec,
    BenchmarkRegistry,
    BENCHMARK_SPECS,
)
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
from backend.evaluation.runners import (
    EvaluationResult,
    EvaluationRunner,
)
from backend.evaluation.reports import (
    ReportGenerator,
)

__all__ = [
    "BenchmarkSpec",
    "BenchmarkRegistry",
    "BENCHMARK_SPECS",
    "compute_multilabel_metrics",
    "compute_vqa_metrics",
    "compute_captioning_metrics",
    "compute_rouge_l",
    "compute_token_overlap",
    "normalize_text_answer",
    "METRIC_DEFINITIONS",
    "DatasetAdapter",
    "BigEarthNetAdapter",
    "RSVQAAdapter",
    "VRSBenchAdapter",
    "CDVQAAdapter",
    "EvaluationResult",
    "EvaluationRunner",
    "ReportGenerator",
]
