"""
SatQuery AI - Benchmark Dataset Adapters.
Encapsulates dataset loading, validation, splitting, and sample formatting for:
- BigEarthNet-S2 (10-Band Sentinel-2 Multilabel)
- RSVQA (High-Resolution & Low-Resolution VQA)
- VRSBench (Remote Sensing Scene Captioning / Description)
- CDVQA (Change Detection Visual Question Answering)

Strict principles:
- Configurable roots via environment variables (BIGEARTHNET_ROOT, RSVQA_ROOT, VRSBENCH_ROOT, CDVQA_ROOT).
- Validates local existence before loading.
- Never fabricates or synthesizes samples.
- Returns explicit 'dataset_unavailable' diagnostics when unconfigured or missing.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from backend.models.bigearthnet_loader import BIGEARTHNET_19_CLASSES, REQUIRED_S2_BANDS

logger = logging.getLogger("satquery.evaluation.adapters")


class DatasetAdapter:
    """
    Base class for remote-sensing benchmark dataset adapters.
    """
    def __init__(self, dataset_id: str, env_var: str, root_path: Optional[str] = None):
        self.dataset_id = dataset_id
        self.env_var = env_var
        self.root_path = root_path or os.environ.get(env_var)

    def validate_configuration(self) -> Tuple[bool, str]:
        """
        Validates whether the dataset root directory exists and contains necessary structure.
        """
        if not self.root_path:
            return False, f"Environment variable '{self.env_var}' is not configured."
        if not os.path.exists(self.root_path):
            return False, f"Configured directory '{self.root_path}' does not exist on disk."
        if not os.path.isdir(self.root_path):
            return False, f"Configured path '{self.root_path}' is not a directory."
        return True, f"Dataset root verified at '{self.root_path}'."

    def load_samples(
        self,
        split: str = "test",
        max_samples: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Loads structured evaluation samples.
        Returns (samples_list, diagnostics_dict).
        """
        raise NotImplementedError


class BigEarthNetAdapter(DatasetAdapter):
    """
    Adapter for BigEarthNet-S2 multilabel classification benchmark.
    Requires 10-band Sentinel-2 GeoTIFFs/NumPy arrays and 19-class CORINE labels.
    """
    def __init__(self, root_path: Optional[str] = None):
        super().__init__("bigearthnet-s2", "BIGEARTHNET_ROOT", root_path)

    def load_samples(
        self,
        split: str = "test",
        max_samples: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        valid, msg = self.validate_configuration()
        if not valid:
            return [], {
                "dataset": self.dataset_id,
                "status": "dataset_unavailable",
                "message": msg,
                "samples_loaded": 0,
            }

        # Check for split annotation file: splits/{split}.json or metadata.json
        split_file = os.path.join(self.root_path, "splits", f"{split}.json")
        if not os.path.exists(split_file):
            split_file = os.path.join(self.root_path, f"{split}.json")

        if not os.path.exists(split_file):
            return [], {
                "dataset": self.dataset_id,
                "status": "dataset_unavailable",
                "message": f"Split file '{split_file}' not found in BigEarthNet directory.",
                "samples_loaded": 0,
            }

        samples = []
        try:
            with open(split_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = data if isinstance(data, list) else data.get("samples", [])
            for item in items:
                if max_samples and len(samples) >= max_samples:
                    break

                patch_name = item.get("patch_name") or item.get("name") or item.get("id")
                labels_raw = item.get("labels") or item.get("classes") or []

                # Convert string labels or indices to 19-dim binary vector
                label_vector = np.zeros(len(BIGEARTHNET_19_CLASSES), dtype=np.int32)
                for lbl in labels_raw:
                    if isinstance(lbl, str) and lbl in BIGEARTHNET_19_CLASSES:
                        idx = BIGEARTHNET_19_CLASSES.index(lbl)
                        label_vector[idx] = 1
                    elif isinstance(lbl, int) and 0 <= lbl < len(BIGEARTHNET_19_CLASSES):
                        label_vector[lbl] = 1

                patch_dir = os.path.join(self.root_path, patch_name) if patch_name else ""

                samples.append({
                    "sample_id": patch_name or f"ben_{len(samples)}",
                    "patch_name": patch_name,
                    "patch_dir": patch_dir,
                    "labels": label_vector,
                    "label_names": [
                        BIGEARTHNET_19_CLASSES[i] for i, val in enumerate(label_vector) if val == 1
                    ],
                    "required_bands": REQUIRED_S2_BANDS,
                })

            status = "evaluated" if samples else "dataset_unavailable"
            return samples, {
                "dataset": self.dataset_id,
                "status": status,
                "message": f"Loaded {len(samples)} BigEarthNet-S2 samples for split '{split}'.",
                "samples_loaded": len(samples),
            }
        except Exception as e:
            return [], {
                "dataset": self.dataset_id,
                "status": "failed",
                "message": f"Failed reading BigEarthNet split: {e}",
                "samples_loaded": 0,
            }


class RSVQAAdapter(DatasetAdapter):
    """
    Adapter for RSVQA (RSVQA-HR / RSVQA-LR) visual question answering benchmark.
    Requires images and question-answer annotation JSON files.
    """
    def __init__(self, benchmark_id: str = "rsvqa-hr", root_path: Optional[str] = None):
        super().__init__(benchmark_id, "RSVQA_ROOT", root_path)

    def load_samples(
        self,
        split: str = "test",
        max_samples: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        valid, msg = self.validate_configuration()
        if not valid:
            return [], {
                "dataset": self.dataset_id,
                "status": "dataset_unavailable",
                "message": msg,
                "samples_loaded": 0,
            }

        # Check for questions file (e.g. questions.json, RSVQA_HR_questions.json)
        candidate_files = [
            os.path.join(self.root_path, f"{split}_questions.json"),
            os.path.join(self.root_path, "questions.json"),
            os.path.join(self.root_path, f"{self.dataset_id}_{split}.json"),
        ]
        q_file = next((f for f in candidate_files if os.path.exists(f)), None)

        if not q_file:
            return [], {
                "dataset": self.dataset_id,
                "status": "dataset_unavailable",
                "message": f"No valid questions annotation file found in '{self.root_path}'.",
                "samples_loaded": 0,
            }

        samples = []
        try:
            with open(q_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = data if isinstance(data, list) else data.get("questions", [])
            for item in items:
                if max_samples and len(samples) >= max_samples:
                    break

                q_id = item.get("id") or item.get("question_id") or f"rsvqa_{len(samples)}"
                q_text = item.get("question") or item.get("query")
                ans_text = item.get("answer") or item.get("ground_truth")
                img_name = item.get("image_name") or item.get("image") or item.get("img")
                cat = item.get("category") or item.get("type", "general")

                if not q_text or ans_text is None:
                    continue

                img_path = os.path.join(self.root_path, "images", img_name) if img_name else ""

                samples.append({
                    "sample_id": str(q_id),
                    "image_name": img_name,
                    "image_path": img_path,
                    "question": str(q_text).strip(),
                    "ground_truth": str(ans_text).strip(),
                    "category": str(cat).lower(),
                })

            status = "evaluated" if samples else "dataset_unavailable"
            return samples, {
                "dataset": self.dataset_id,
                "status": status,
                "message": f"Loaded {len(samples)} RSVQA samples from '{q_file}'.",
                "samples_loaded": len(samples),
            }
        except Exception as e:
            return [], {
                "dataset": self.dataset_id,
                "status": "failed",
                "message": f"Failed reading RSVQA questions: {e}",
                "samples_loaded": 0,
            }


class VRSBenchAdapter(DatasetAdapter):
    """
    Adapter for VRSBench scene description & captioning benchmark.
    Supports scene-description/captioning subset. Explicitly identifies unsupported tasks.
    """
    def __init__(self, root_path: Optional[str] = None):
        super().__init__("vrsbench", "VRSBENCH_ROOT", root_path)

    def load_samples(
        self,
        split: str = "test",
        max_samples: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        valid, msg = self.validate_configuration()
        if not valid:
            return [], {
                "dataset": self.dataset_id,
                "status": "dataset_unavailable",
                "message": msg,
                "samples_loaded": 0,
            }

        candidate_files = [
            os.path.join(self.root_path, f"vrsbench_{split}_captions.json"),
            os.path.join(self.root_path, "captions.json"),
            os.path.join(self.root_path, f"{split}.json"),
        ]
        cap_file = next((f for f in candidate_files if os.path.exists(f)), None)

        if not cap_file:
            return [], {
                "dataset": self.dataset_id,
                "status": "dataset_unavailable",
                "message": f"No valid VRSBench caption annotation file found in '{self.root_path}'.",
                "samples_loaded": 0,
            }

        samples = []
        try:
            with open(cap_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = data if isinstance(data, list) else data.get("items", [])
            for item in items:
                if max_samples and len(samples) >= max_samples:
                    break

                task = item.get("task", "captioning")
                # Filter to supported subset: scene description / captioning
                if task not in ["captioning", "scene_description"]:
                    continue

                s_id = item.get("id") or f"vrs_{len(samples)}"
                gt_cap = item.get("caption") or item.get("ground_truth") or item.get("description")
                img_name = item.get("image_name") or item.get("image")

                if not gt_cap:
                    continue

                img_path = os.path.join(self.root_path, "images", img_name) if img_name else ""

                samples.append({
                    "sample_id": str(s_id),
                    "image_name": img_name,
                    "image_path": img_path,
                    "task": task,
                    "ground_truth": str(gt_cap).strip(),
                })

            status = "evaluated" if samples else "dataset_unavailable"
            return samples, {
                "dataset": self.dataset_id,
                "status": status,
                "message": f"Loaded {len(samples)} VRSBench captioning samples.",
                "samples_loaded": len(samples),
            }
        except Exception as e:
            return [], {
                "dataset": self.dataset_id,
                "status": "failed",
                "message": f"Failed reading VRSBench captions: {e}",
                "samples_loaded": 0,
            }


class CDVQAAdapter(DatasetAdapter):
    """
    Adapter for CDVQA (Change Detection Visual Question Answering) benchmark.
    Requires bi-temporal image pairs (T1 and T2) and temporal change questions.
    """
    def __init__(self, root_path: Optional[str] = None):
        super().__init__("cdvqa", "CDVQA_ROOT", root_path)

    def load_samples(
        self,
        split: str = "test",
        max_samples: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        valid, msg = self.validate_configuration()
        if not valid:
            return [], {
                "dataset": self.dataset_id,
                "status": "dataset_unavailable",
                "message": msg,
                "samples_loaded": 0,
            }

        candidate_files = [
            os.path.join(self.root_path, f"cdvqa_{split}.json"),
            os.path.join(self.root_path, "cdvqa.json"),
            os.path.join(self.root_path, f"{split}.json"),
        ]
        cd_file = next((f for f in candidate_files if os.path.exists(f)), None)

        if not cd_file:
            return [], {
                "dataset": self.dataset_id,
                "status": "dataset_unavailable",
                "message": f"No valid CDVQA annotation file found in '{self.root_path}'.",
                "samples_loaded": 0,
            }

        samples = []
        try:
            with open(cd_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = data if isinstance(data, list) else data.get("samples", [])
            for item in items:
                if max_samples and len(samples) >= max_samples:
                    break

                s_id = item.get("id") or f"cdvqa_{len(samples)}"
                q_text = item.get("question")
                ans_text = item.get("answer") or item.get("ground_truth")
                img_t1 = item.get("image_t1") or item.get("before_image")
                img_t2 = item.get("image_t2") or item.get("after_image")
                chg_type = item.get("change_type", "general")

                if not q_text or ans_text is None:
                    continue

                samples.append({
                    "sample_id": str(s_id),
                    "image_t1": img_t1,
                    "image_t2": img_t2,
                    "question": str(q_text).strip(),
                    "ground_truth": str(ans_text).strip(),
                    "change_type": str(chg_type),
                })

            status = "evaluated" if samples else "dataset_unavailable"
            return samples, {
                "dataset": self.dataset_id,
                "status": status,
                "message": f"Loaded {len(samples)} CDVQA bi-temporal samples.",
                "samples_loaded": len(samples),
            }
        except Exception as e:
            return [], {
                "dataset": self.dataset_id,
                "status": "failed",
                "message": f"Failed reading CDVQA annotations: {e}",
                "samples_loaded": 0,
            }
