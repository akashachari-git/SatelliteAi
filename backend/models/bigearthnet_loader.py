"""
SatQuery AI - Dedicated BigEarthNet ResNet-18 Model Loader and Inference Engine.
Target Architecture: BIFOLD-BigEarthNetv2-0/resnet18-s2-v0.2.0
Utilizes the official ConfigILM architecture via:
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

Implements lazy loading on CPU, strict 10-band Sentinel-2 verification,
and 19-class multi-label sigmoid probability evaluation.
"""
import os
import sys
import logging
import threading
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

# Ensure backend directory is in sys.path for reben_publication resolution
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_CURRENT_DIR)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

logger = logging.getLogger("satquery.models.bigearthnet")

# 19-Class Corine Land Cover (CLC) Nomenclature (BigEarthNet v2.0 / reBEN standard)
BIGEARTHNET_19_CLASSES: List[str] = [
    "Continuous urban fabric",
    "Discontinuous urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grasslands and sclerophyllous vegetation",
    "Transitional woodland-shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
    "Bare rock and sparsely vegetated areas",
]

# Official Sentinel-2 10-band spectral contract required by BIFOLD-BigEarthNetv2-0/resnet18-s2-v0.2.0
REQUIRED_S2_BANDS: List[str] = [
    "B02",  # Blue (490 nm)
    "B03",  # Green (560 nm)
    "B04",  # Red (665 nm)
    "B05",  # Red Edge 1 (705 nm)
    "B06",  # Red Edge 2 (740 nm)
    "B07",  # Red Edge 3 (783 nm)
    "B08",  # NIR Broad (842 nm)
    "B8A",  # NIR Narrow (865 nm)
    "B11",  # SWIR 1 (1610 nm)
    "B12",  # SWIR 2 (2190 nm)
]

# Canonical model repository identifier
CANONICAL_MODEL_ID = "BIFOLD-BigEarthNetv2-0/resnet18-s2-v0.2.0"


class ModelStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    LOAD_ERROR = "LOAD_ERROR"


def normalize_band_name(name: str) -> str:
    """Normalizes band nomenclature (e.g. 'b2' -> 'B02', 'B8a' -> 'B8A')."""
    cleaned = str(name).strip().upper()
    mapping = {
        "B2": "B02",
        "B3": "B03",
        "B4": "B04",
        "B5": "B05",
        "B6": "B06",
        "B7": "B07",
        "B8": "B08",
        "B08A": "B8A",
    }
    return mapping.get(cleaned, cleaned)


def validate_s2_10band_input(
    image: Any,
    bands: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Strict safety validation for BigEarthNet Sentinel-2 10-band contract.
    Enforces:
    - Exactly 10 channels.
    - Verified Sentinel-2 band names matching REQUIRED_S2_BANDS in order.
    - Strict rejection of 3-band RGB imagery.
    - Strict rejection of unknown or unverified band sequences.
    - Prohibits synthesis, padding, channel duplication, or band guesswork.
    """
    # 1. Band list presence check
    if not bands:
        return False, "Band ordering is unknown or unverified; bypassing BigEarthNet model"

    # 2. Band count check
    if len(bands) == 3:
        return False, "3-band RGB imagery is incompatible with BigEarthNet 10-band ResNet-18 model; bypassing to deterministic fallback"

    if len(bands) != 10:
        return False, f"Incompatible band count: expected 10 Sentinel-2 bands, got {len(bands)}; bypassing model"

    # 3. Band nomenclature and ordering check
    normalized_bands = [normalize_band_name(b) for b in bands]
    if normalized_bands != REQUIRED_S2_BANDS:
        return (
            False,
            f"Unverified or misordered band sequence: expected {REQUIRED_S2_BANDS}, got {normalized_bands}; bypassing model"
        )

    # 4. Dimensionality inspection if array/tensor passed
    if image is not None:
        try:
            if hasattr(image, "shape"):
                shape = image.shape
                # Supported formats: (10, H, W), (B, 10, H, W), (H, W, 10)
                if len(shape) == 2:
                    return False, "Single-band 2D raster input cannot enter 10-band BigEarthNet model"
                elif len(shape) == 3:
                    if shape[0] == 3 or shape[-1] == 3:
                        return False, "3-channel RGB image cannot enter 10-band BigEarthNet model"
                    if shape[0] != 10 and shape[-1] != 10:
                        return False, f"Input tensor channels ({shape[0]} or {shape[-1]}) do not match required 10 bands"
                elif len(shape) == 4:
                    if shape[1] != 10 and shape[-1] != 10:
                        return False, f"Batch tensor channels do not match required 10 bands: shape {shape}"
        except Exception as e:
            return False, f"Failed to inspect tensor dimensions: {e}"

    return True, "Verified compatible 10-band Sentinel-2 input"


DEFAULT_CHECKPOINT_DIR = os.path.join(_BACKEND_DIR, "models", "checkpoints", "resnet18-s2-v0.2.0")
DEFAULT_ONNX_PATH = os.path.join(_BACKEND_DIR, "models", "bigearthnet_resnet18_10band.onnx")


class InferenceEngine(str, Enum):
    ONNX = "ONNX"
    PYTORCH = "PYTORCH"
    NONE = "NONE"


class BigEarthNetModelLoader:
    """
    Dedicated manager for BigEarthNet ResNet-18 model lifecycle.
    Features:
    - Primary Engine: ONNX Runtime on CPU (CPUExecutionProvider).
    - Fallback Engine: Official PyTorch BigEarthNetv2_0_ImageClassifier from checkpoint.
    - Lazy loading: Model is NOT instantiated upon server startup.
    - CPU inference: Strictly executes on CPU.
    - Strict 10-band Sentinel-2 validation and RGB bypass.
    - Explicit model status: AVAILABLE, UNAVAILABLE, LOAD_ERROR.
    - Explicit engine reporting: ONNX, PYTORCH, NONE.
    """

    def __init__(
        self,
        onnx_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
        env_onnx = os.environ.get("BIGEARTHNET_ONNX_PATH")
        env_ckpt = os.environ.get("BIGEARTHNET_CHECKPOINT_PATH")

        if onnx_path is not None:
            self._onnx_path = onnx_path
        elif checkpoint_path is not None:
            # Caller explicitly configured checkpoint_path only; do not silently mount default ONNX
            self._onnx_path = None
        elif env_onnx:
            self._onnx_path = env_onnx
        elif os.path.exists(DEFAULT_ONNX_PATH):
            self._onnx_path = DEFAULT_ONNX_PATH
        else:
            self._onnx_path = None

        if checkpoint_path is not None:
            self._checkpoint_path = checkpoint_path
        elif env_ckpt:
            self._checkpoint_path = env_ckpt
        elif os.path.exists(DEFAULT_CHECKPOINT_DIR):
            self._checkpoint_path = DEFAULT_CHECKPOINT_DIR
        else:
            self._checkpoint_path = None

        self._status: ModelStatus = ModelStatus.UNAVAILABLE
        self._engine: InferenceEngine = InferenceEngine.NONE
        self._onnx_session: Optional[Any] = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._model: Optional[Any] = None
        self._error_detail: Optional[str] = None
        self._load_attempted: bool = False
        self._lock = threading.Lock()

    @property
    def status(self) -> ModelStatus:
        """Returns the current operational status of the model."""
        return self._status

    @property
    def engine(self) -> InferenceEngine:
        """Returns the active inference engine (ONNX, PYTORCH, or NONE)."""
        return self._engine

    @property
    def is_available(self) -> bool:
        """Returns True only if model weights are loaded into memory and ready."""
        return self._status == ModelStatus.AVAILABLE and (
            self._onnx_session is not None or self._model is not None
        )

    @property
    def error_detail(self) -> Optional[str]:
        """Diagnostic description of loading failure or unmounted status."""
        return self._error_detail

    @property
    def onnx_path(self) -> Optional[str]:
        return self._onnx_path

    @property
    def checkpoint_path(self) -> Optional[str]:
        return self._checkpoint_path

    def set_onnx_path(self, path: str):
        """Updates the local ONNX path and resets state for re-evaluation."""
        self._onnx_path = path
        self._status = ModelStatus.UNAVAILABLE
        self._engine = InferenceEngine.NONE
        self._onnx_session = None
        self._error_detail = None
        self._load_attempted = False

    def set_checkpoint_path(self, path: str):
        """Updates the local PyTorch checkpoint path and resets state for re-evaluation."""
        self._checkpoint_path = path
        self._status = ModelStatus.UNAVAILABLE
        self._engine = InferenceEngine.NONE
        self._model = None
        self._error_detail = None
        self._load_attempted = False

    def load_model(self, force: bool = False) -> Tuple[bool, ModelStatus, str]:
        """
        Loads the BigEarthNet ResNet-18 model lazily in a thread-safe manner.
        Attempts ONNX Runtime first; falls back to PyTorch if ONNX is unavailable.
        Configures CPU execution provider with optimized thread counts.
        """
        if self._load_attempted and not force:
            return (
                self.is_available,
                self._status,
                self._error_detail or f"Model status unchanged ({self._engine.value})",
            )

        with self._lock:
            if self._load_attempted and not force:
                return (
                    self.is_available,
                    self._status,
                    self._error_detail or f"Model status unchanged ({self._engine.value})",
                )

            self._load_attempted = True

            # 1. Primary path: Attempt ONNX Runtime with CPUExecutionProvider
            target_onnx = self._onnx_path
            if target_onnx and os.path.exists(target_onnx):
                try:
                    import onnxruntime as ort

                    logger.info("Loading BigEarthNet ResNet-18 ONNX model from %s...", target_onnx)
                    sess_options = ort.SessionOptions()
                    intra_threads = int(os.environ.get("SATQUERY_ORT_INTRA_THREADS", min(4, os.cpu_count() or 1)))
                    inter_threads = int(os.environ.get("SATQUERY_ORT_INTER_THREADS", 1))
                    sess_options.intra_op_num_threads = intra_threads
                    sess_options.inter_op_num_threads = inter_threads
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                    session = ort.InferenceSession(
                        target_onnx,
                        sess_options=sess_options,
                        providers=["CPUExecutionProvider"],
                    )
                    self._onnx_session = session
                    self._input_name = session.get_inputs()[0].name
                    self._output_name = session.get_outputs()[0].name
                    self._engine = InferenceEngine.ONNX
                    self._status = ModelStatus.AVAILABLE
                    self._error_detail = None
                    logger.info(
                        "BigEarthNet ResNet-18 successfully loaded via ONNX Runtime (CPUExecutionProvider). "
                        "Input: '%s', Output: '%s' (intra_threads=%d, inter_threads=%d)",
                        self._input_name,
                        self._output_name,
                        intra_threads,
                        inter_threads,
                    )
                    return True, self._status, "ONNX model loaded successfully on CPU"
                except Exception as ort_err:
                    logger.warning(
                        "Failed to load ONNX model from '%s': %s. Attempting PyTorch fallback...",
                        target_onnx,
                        ort_err,
                    )
                    self._onnx_session = None

        # 2. Fallback path: Attempt verified PyTorch checkpoint
        target_ckpt = self._checkpoint_path
        if target_ckpt and os.path.exists(target_ckpt):
            try:
                from reben_publication.BigEarthNetv2_0_ImageClassifier import (
                    BigEarthNetv2_0_ImageClassifier,
                )

                logger.info("Loading BigEarthNet ResNet-18 PyTorch checkpoint from %s on CPU...", target_ckpt)
                model = BigEarthNetv2_0_ImageClassifier.from_pretrained(target_ckpt)
                if hasattr(model, "to"):
                    model.to("cpu")
                if hasattr(model, "eval"):
                    model.eval()

                self._model = model
                self._engine = InferenceEngine.PYTORCH
                self._status = ModelStatus.AVAILABLE
                self._error_detail = None
                logger.info("BigEarthNet ResNet-18 successfully loaded into CPU memory via PyTorch fallback.")
                return True, self._status, "PyTorch model loaded successfully on CPU as fallback"
            except (ImportError, ModuleNotFoundError) as err:
                self._status = ModelStatus.UNAVAILABLE
                self._engine = InferenceEngine.NONE
                self._error_detail = f"PyTorch fallback dependencies unavailable: {err}"
                logger.warning(self._error_detail)
                return False, self._status, self._error_detail
            except Exception as err:
                self._status = ModelStatus.LOAD_ERROR
                self._engine = InferenceEngine.NONE
                self._error_detail = f"Failed to load PyTorch fallback from '{target_ckpt}': {err}"
                logger.error(self._error_detail)
                return False, self._status, self._error_detail

        # Neither ONNX nor PyTorch checkpoint found
        self._status = ModelStatus.UNAVAILABLE
        self._engine = InferenceEngine.NONE
        self._error_detail = (
            f"Neither ONNX model (path='{target_onnx}') nor PyTorch checkpoint "
            f"(path='{target_ckpt}') is mounted locally. Model remains UNAVAILABLE."
        )
        logger.info(self._error_detail)
        return False, self._status, self._error_detail

    def run_inference(self, input_tensor: Any, return_logits: bool = False) -> np.ndarray:
        """
        Executes model forward pass strictly on CPU.
        Primary: ONNX Runtime (CPUExecutionProvider).
        Fallback: PyTorch BigEarthNetv2_0_ImageClassifier on CPU.

        Model contract:
        Input: (B, 10, H, W) tensor
        Output: 19 raw logits
        External sigmoid applied: 1 / (1 + exp(-logits)) -> 19 probabilities in [0.0, 1.0].
        Softmax is explicitly forbidden for multi-label classification.
        """
        if not self.is_available:
            raise RuntimeError(
                f"Cannot run BigEarthNet inference: model status is {self._status} ({self._error_detail})"
            )

        # Standardize input array to (B, 10, H, W) float32 numpy
        try:
            if hasattr(input_tensor, "cpu") and hasattr(input_tensor, "numpy"):
                arr = input_tensor.cpu().numpy().astype(np.float32)
            else:
                arr = np.asarray(input_tensor, dtype=np.float32)

            if arr.ndim == 3:
                if arr.shape[0] == 10:
                    arr = np.expand_dims(arr, axis=0)  # (1, 10, H, W)
                elif arr.shape[-1] == 10:
                    arr = np.transpose(arr, (2, 0, 1))  # (10, H, W)
                    arr = np.expand_dims(arr, axis=0)
                else:
                    raise ValueError(f"Input 3D array channels ({arr.shape}) do not match 10 bands")
            elif arr.ndim == 4:
                if arr.shape[1] == 10:
                    pass  # already (B, 10, H, W)
                elif arr.shape[-1] == 10:
                    arr = np.transpose(arr, (0, 3, 1, 2))
                else:
                    raise ValueError(f"Input 4D batch channels ({arr.shape}) do not match 10 bands")
            else:
                raise ValueError(f"Unsupported array dimensionality: {arr.ndim}D")
        except Exception as prep_err:
            raise ValueError(f"Failed to prepare input tensor for BigEarthNet: {prep_err}") from prep_err

        # Execute inference via active engine
        try:
            if self._engine == InferenceEngine.ONNX and self._onnx_session is not None:
                ort_inputs = {self._input_name: arr}
                logits = self._onnx_session.run([self._output_name], ort_inputs)[0]
                if return_logits:
                    return logits.squeeze()
                probs = 1.0 / (1.0 + np.exp(-logits))
                return probs.squeeze()

            elif self._engine == InferenceEngine.PYTORCH and self._model is not None:
                import torch

                with torch.no_grad():
                    t_input = torch.from_numpy(arr).to("cpu")
                    logits = self._model(t_input).cpu().numpy()
                    if return_logits:
                        return logits.squeeze()
                    probs = 1.0 / (1.0 + np.exp(-logits))
                    return probs.squeeze()
            else:
                raise RuntimeError(f"No active inference engine (engine={self._engine})")
        except Exception as err:
            raise RuntimeError(f"BigEarthNet neural inference failed ({self._engine.value}): {err}") from err

    def run_batch_inference(self, input_batch: Any, return_logits: bool = False) -> np.ndarray:
        """
        Executes batched forward pass for multiple 10-band Sentinel-2 patches simultaneously.
        Accepts:
          - (B, 10, H, W) numpy array or torch tensor
          - list of (10, H, W) numpy arrays
        Returns:
          - (B, 19) numpy array of probabilities (or logits if return_logits=True).
        Numerically equivalent to iterating run_inference single-patch.
        """
        if isinstance(input_batch, list):
            input_batch = np.stack(input_batch, axis=0)

        out = self.run_inference(input_batch, return_logits=return_logits)
        if out.ndim == 1:
            out = np.expand_dims(out, axis=0)
        return out


# Singleton instance for lazy model management
bigearthnet_loader = BigEarthNetModelLoader()

