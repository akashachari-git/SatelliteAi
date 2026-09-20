"""
SatQuery AI - Remote Sensing Benchmark Registry.
Defines official benchmark specifications, dataset contracts, supported tasks,
prescribed metrics, and adaptation statuses for:
- BigEarthNet-S2 (Multilabel Land Cover Classification)
- RSVQA-HR / RSVQA-LR (Remote Sensing Visual Question Answering)
- VRSBench (Visual Remote Sensing Scene Description & Grounding)
- CDVQA (Change Detection Visual Question Answering)
"""
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional

from backend.models.bigearthnet_loader import BIGEARTHNET_19_CLASSES, REQUIRED_S2_BANDS


@dataclass
class BenchmarkSpec:
    """
    Formal specification and contract for a remote sensing benchmark.
    """
    benchmark_id: str
    name: str
    task: str
    env_var: str
    target_modalities: List[str]
    model_used: str
    adaptation_status: str
    prescribed_metrics: List[str]
    description: str
    supported_subset: str = "full"
    required_bands: Optional[List[str]] = None
    classes: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


BENCHMARK_SPECS: Dict[str, BenchmarkSpec] = {
    "bigearthnet-s2": BenchmarkSpec(
        benchmark_id="bigearthnet-s2",
        name="BigEarthNet-S2 (19-Class CORINE Land Cover)",
        task="multilabel_classification",
        env_var="BIGEARTHNET_ROOT",
        target_modalities=["Sentinel-2 Multispectral (10-Band)"],
        required_bands=REQUIRED_S2_BANDS,
        classes=BIGEARTHNET_19_CLASSES,
        model_used="BIFOLD-BigEarthNetv2-0/resnet18-s2-v0.2.0 (ResNet-18)",
        adaptation_status="pretrained_specialist",
        prescribed_metrics=[
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "macro_f1",
            "per_class_f1",
            "exact_match_ratio",
            "hamming_loss",
        ],
        description=(
            "Official BigEarthNet-S2 19-class multilabel land-cover classification benchmark. "
            "Evaluates genuine 10-band Sentinel-2 patches at 120x120 resolution against CORINE ground truth."
        ),
        supported_subset="10-band Sentinel-2 patches with CORINE Land Cover labels",
    ),
    "rsvqa-hr": BenchmarkSpec(
        benchmark_id="rsvqa-hr",
        name="RSVQA-HR (High Resolution Aerial VQA)",
        task="vqa",
        env_var="RSVQA_ROOT",
        target_modalities=["Optical RGB (High-Resolution Aerial)"],
        model_used="microsoft/Florence-2-base",
        adaptation_status="not_remote_sensing_finetuned",
        prescribed_metrics=[
            "exact_match",
            "normalized_exact_match",
            "category_stratified_accuracy",
        ],
        description=(
            "Remote Sensing Visual Question Answering on high-resolution aerial imagery (RSVQA-HR). "
            "Florence-2 base evaluates zero-shot remote sensing question answering across presence, "
            "count, comparison, and rural/urban questions."
        ),
        supported_subset="RSVQA-HR question-answer pairs with RGB imagery",
    ),
    "rsvqa-lr": BenchmarkSpec(
        benchmark_id="rsvqa-lr",
        name="RSVQA-LR (Low Resolution Sentinel-2 VQA)",
        task="vqa",
        env_var="RSVQA_ROOT",
        target_modalities=["Optical Multispectral (Sentinel-2)"],
        model_used="microsoft/Florence-2-base",
        adaptation_status="not_remote_sensing_finetuned",
        prescribed_metrics=[
            "exact_match",
            "normalized_exact_match",
            "category_stratified_accuracy",
        ],
        description=(
            "Remote Sensing Visual Question Answering on Sentinel-2 low-resolution observations (RSVQA-LR). "
            "Evaluates Florence-2 base on questions over low-resolution spaceborne tiles."
        ),
        supported_subset="RSVQA-LR question-answer pairs with Sentinel-2 RGB composite imagery",
    ),
    "vrsbench": BenchmarkSpec(
        benchmark_id="vrsbench",
        name="VRSBench (Visual Remote Sensing Benchmark)",
        task="captioning",
        env_var="VRSBENCH_ROOT",
        target_modalities=["Optical RGB"],
        model_used="microsoft/Florence-2-base",
        adaptation_status="not_remote_sensing_finetuned",
        prescribed_metrics=[
            "rouge_l",
            "token_f1",
            "token_precision",
            "token_recall",
        ],
        description=(
            "Visual Remote Sensing Benchmark for captioning and scene description. "
            "Evaluates descriptive fidelity against reference descriptions without external dependencies."
        ),
        supported_subset="Scene description / captioning subset compatible with current VLM pipeline",
    ),
    "cdvqa": BenchmarkSpec(
        benchmark_id="cdvqa",
        name="CDVQA (Change Detection Visual Question Answering)",
        task="change_vqa",
        env_var="CDVQA_ROOT",
        target_modalities=["Bi-Temporal Optical Pairs"],
        model_used="BiTemporalChangeService + Florence-2 VLM",
        adaptation_status="algorithmic_differencing_with_vlm",
        prescribed_metrics=[
            "exact_match",
            "normalized_exact_match",
            "change_type_accuracy",
        ],
        description=(
            "Change Detection Visual Question Answering across bi-temporal observation pairs (T1 and T2). "
            "Evaluates temporal reasoning and change description against ground-truth answers."
        ),
        supported_subset="Bi-temporal image pairs with natural language change questions and answers",
    ),
}


class BenchmarkRegistry:
    """
    Central registry for remote-sensing benchmark specifications and dataset discovery.
    """

    @classmethod
    def get_spec(cls, benchmark_id: str) -> Optional[BenchmarkSpec]:
        return BENCHMARK_SPECS.get(benchmark_id.lower())

    @classmethod
    def get_all_specs(cls) -> Dict[str, BenchmarkSpec]:
        return dict(BENCHMARK_SPECS)

    @classmethod
    def discover_dataset_status(cls, benchmark_id: str) -> Dict[str, Any]:
        """
        Discovers whether a benchmark dataset is locally configured and accessible.
        Does not download or fabricate data.
        """
        spec = cls.get_spec(benchmark_id)
        if not spec:
            return {
                "benchmark_id": benchmark_id,
                "status": "not_configured",
                "message": f"Benchmark '{benchmark_id}' is not registered.",
            }

        root_path = os.environ.get(spec.env_var)
        if not root_path:
            return {
                "benchmark_id": benchmark_id,
                "status": "dataset_unavailable",
                "env_var": spec.env_var,
                "message": f"Environment variable '{spec.env_var}' is not configured.",
            }

        if not os.path.exists(root_path):
            return {
                "benchmark_id": benchmark_id,
                "status": "dataset_unavailable",
                "configured_path": root_path,
                "message": f"Configured directory '{root_path}' does not exist on disk.",
            }

        if not os.path.isdir(root_path):
            return {
                "benchmark_id": benchmark_id,
                "status": "dataset_unavailable",
                "configured_path": root_path,
                "message": f"Configured path '{root_path}' is not a directory.",
            }

        return {
            "benchmark_id": benchmark_id,
            "status": "available",
            "configured_path": root_path,
            "message": f"Dataset directory verified at '{root_path}'.",
        }
