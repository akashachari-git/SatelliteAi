"""
SatQuery AI - Florence-2 Vision-Language Model Specialist Service
Executes genuine image captioning, visual question answering (VQA),
and text-guided phrase grounding on local CPU runtime.

Key Principles:
1. Lazy loading: Model and processor are loaded only when requested.
2. Local checkpoint only: Loads from backend/models/checkpoints/florence2-base.
3. CPU & Float32: Optimized for local CPU execution without CUDA requirement.
4. Scientific honesty:
   - Ingests native RGB images or authentic B04-B03-B02 true-color composites from 10-band Sentinel-2 rasters.
   - Preserves distinction between multispectral rasters and optical RGB visualizations.
   - No fabricated confidence scores.
   - Coordinates remain strictly in image pixel space (no guessed geocoordinates).
   - No fake/template silent fallbacks.
"""
import os
import io
import time
import base64
import logging
import threading
import shutil
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

logger = logging.getLogger("satquery.models.florence2")

# Local canonical checkpoint path
CHECKPOINT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "models",
        "checkpoints",
        "florence2-base"
    )
)


def _sync_patched_modeling_file():
    """
    Ensures the compatibility-fixed modeling_florence2.py is present in the
    HuggingFace transformers_modules cache directory if it exists.
    """
    src_path = os.path.join(CHECKPOINT_DIR, "modeling_florence2.py")
    if not os.path.exists(src_path):
        return

    cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/florence2_hyphen_base")
    if os.path.exists(cache_dir):
        dst_path = os.path.join(cache_dir, "modeling_florence2.py")
        try:
            shutil.copy2(src_path, dst_path)
            logger.debug("Synchronized patched modeling_florence2.py to HF cache: %s", dst_path)
        except Exception as e:
            logger.warning("Could not sync modeling_florence2.py to HF cache: %s", e)


class Florence2InferenceService:
    """
    Thread-safe specialist service wrapping Microsoft Florence-2-base.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path or CHECKPOINT_DIR
        self._model = None
        self._processor = None
        self._is_loaded = False
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        """Returns True if the checkpoint exists and is loadable/loaded."""
        if self._is_loaded:
            return True
        weights_path = os.path.join(self.checkpoint_path, "pytorch_model.bin")
        config_path = os.path.join(self.checkpoint_path, "config.json")
        return os.path.exists(weights_path) and os.path.exists(config_path)

    def load_model(self) -> Tuple[bool, str]:
        """
        Loads the Florence-2 processor and model from the local checkpoint into CPU memory.
        Configures CPU thread count to avoid logical core oversubscription.
        """
        if self._is_loaded:
            return True, "Florence-2 model is already loaded."

        with self._lock:
            if self._is_loaded:
                return True, "Florence-2 model is already loaded."

            if not os.path.exists(self.checkpoint_path):
                msg = f"Florence-2 checkpoint directory not found: {self.checkpoint_path}"
                self._load_error = msg
                logger.error(msg)
                return False, msg

            weights_path = os.path.join(self.checkpoint_path, "pytorch_model.bin")
            if not os.path.exists(weights_path):
                msg = f"Florence-2 weights not found at: {weights_path}"
                self._load_error = msg
                logger.error(msg)
                return False, msg

            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoProcessor

                # Configure CPU thread count
                num_threads = int(os.environ.get("SATQUERY_TORCH_THREADS", min(4, os.cpu_count() or 1)))
                torch.set_num_threads(num_threads)
                logger.info("Configured PyTorch CPU thread count to %d", num_threads)

                # Ensure patched modeling file is active
                _sync_patched_modeling_file()

                t0 = time.time()
                logger.info("Loading Florence-2 processor from %s...", self.checkpoint_path)
                self._processor = AutoProcessor.from_pretrained(
                    self.checkpoint_path,
                    trust_remote_code=True,
                    local_files_only=True,
                )

                logger.info("Loading Florence-2 model on CPU (float32)...")
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.checkpoint_path,
                    trust_remote_code=True,
                    dtype=torch.float32,
                    local_files_only=True,
                )
                self._model.eval()

                self._is_loaded = True
                self._load_error = None
                duration = time.time() - t0
                logger.info("Florence-2-base model loaded successfully in %.2fs", duration)
                return True, f"Florence-2 model loaded in {duration:.2f}s."

            except Exception as e:
                self._load_error = str(e)
                logger.error("Failed to load Florence-2 model: %s", e, exc_info=True)
                return False, f"Failed to load Florence-2 model: {e}"

    # ------------------------------------------------------------------
    # Image Preparation Pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_image(
        image_source: Any
    ) -> Tuple[Optional[Image.Image], str, Dict[str, Any]]:
        """
        Converts any valid image input into a PIL RGB Image suitable for Florence-2.

        Distinguishes:
        - Native Optical RGB images
        - True-Color RGB composite derived from 10-band Sentinel-2 rasters (B04-B03-B02)
        - Grayscale / single-band rasters

        Returns:
            (pil_image, modality_description, metadata)
        """
        if image_source is None:
            return None, "No image source provided.", {}

        # 1. Unpack dictionary container
        if isinstance(image_source, dict):
            if "array" in image_source and image_source["array"] is not None:
                image_source = image_source["array"]
            elif "data" in image_source and image_source["data"] is not None:
                image_source = image_source["data"]
            elif "path" in image_source and image_source["path"] is not None:
                image_source = image_source["path"]
            elif "fileDataUri" in image_source and image_source["fileDataUri"] is not None:
                image_source = image_source["fileDataUri"]
            elif "filename" in image_source and os.path.exists(str(image_source["filename"])):
                image_source = image_source["filename"]

        # 2. PIL Image directly
        if isinstance(image_source, Image.Image):
            rgb_img = image_source.convert("RGB")
            return (
                rgb_img,
                "Optical RGB",
                {"width": rgb_img.width, "height": rgb_img.height, "is_multispectral_derived": False}
            )

        # 3. NumPy Array
        if isinstance(image_source, np.ndarray):
            arr = image_source
            # Check for 10-band Sentinel-2 raster: shape (10, H, W) or (H, W, 10)
            if arr.ndim == 3 and (arr.shape[0] == 10 or arr.shape[2] == 10):
                if arr.shape[2] == 10:
                    arr = np.transpose(arr, (2, 0, 1))  # (10, H, W)
                # Sentinel-2 bands in standard order:
                # 0: B02 (Blue), 1: B03 (Green), 2: B04 (Red)
                # True-color RGB: Red = B04 (idx 2), Green = B03 (idx 1), Blue = B02 (idx 0)
                red = arr[2].astype(np.float32)
                green = arr[1].astype(np.float32)
                blue = arr[0].astype(np.float32)
                rgb_stack = np.stack([red, green, blue], axis=-1)

                # Contrast stretch with percentile normalization
                p2, p98 = np.percentile(rgb_stack, (2, 98))
                if p98 > p2:
                    stretched = np.clip((rgb_stack - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
                else:
                    stretched = np.clip(rgb_stack / 3000.0 * 255.0, 0, 255).astype(np.uint8)

                pil_img = Image.fromarray(stretched, mode="RGB")
                return (
                    pil_img,
                    "Sentinel-2 Derived True-Color RGB Composite (B04-B03-B02)",
                    {
                        "width": pil_img.width,
                        "height": pil_img.height,
                        "is_multispectral_derived": True,
                        "note": (
                            "Prepared true-color RGB composite from authentic 10-band Sentinel-2 raster. "
                            "Raw multispectral channels are preserved separately for BigEarthNet."
                        )
                    }
                )

            # Standard 3-channel RGB array: shape (3, H, W) or (H, W, 3)
            elif arr.ndim == 3 and (arr.shape[0] == 3 or arr.shape[2] == 3):
                if arr.shape[0] == 3:
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.dtype != np.uint8:
                    p2, p98 = np.percentile(arr, (2, 98))
                    if p98 > p2:
                        arr = np.clip((arr - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
                    else:
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                pil_img = Image.fromarray(arr, mode="RGB")
                return (
                    pil_img,
                    "Optical RGB",
                    {"width": pil_img.width, "height": pil_img.height, "is_multispectral_derived": False}
                )

            # Single-band 2D array or (1, H, W)
            elif arr.ndim == 2 or (arr.ndim == 3 and arr.shape[0] == 1):
                if arr.ndim == 3:
                    arr = arr[0]
                p2, p98 = np.percentile(arr, (2, 98))
                if p98 > p2:
                    scaled = np.clip((arr - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
                else:
                    scaled = np.clip(arr, 0, 255).astype(np.uint8)
                pil_img = Image.fromarray(scaled).convert("RGB")
                return (
                    pil_img,
                    "Single-Band Optical (Rendered as Grayscale RGB)",
                    {"width": pil_img.width, "height": pil_img.height, "is_multispectral_derived": False}
                )

        # 4. File Path String
        if isinstance(image_source, str) and os.path.isfile(image_source):
            from backend.geospatial.raster_cache import raster_cache
            cached_arr = raster_cache.get_raster(image_source)
            if cached_arr is not None:
                return Florence2InferenceService.prepare_image(cached_arr)

            ext = os.path.splitext(image_source.lower())[1]
            if ext in [".tif", ".tiff", ".geotiff"]:
                try:
                    import tifffile
                    raw_array = tifffile.imread(image_source)
                    raster_cache.set_raster(image_source, raw_array)
                    return Florence2InferenceService.prepare_image(raw_array)
                except Exception as e:
                    logger.warning("tifffile failed for '%s': %s. Falling back to PIL.", image_source, e)
            try:
                with Image.open(image_source) as img:
                    rgb_img = img.convert("RGB")
                    raster_cache.set_raster(image_source, np.array(rgb_img))
                    return (
                        rgb_img,
                        "Optical RGB",
                        {"width": rgb_img.width, "height": rgb_img.height, "is_multispectral_derived": False}
                    )
            except Exception as e:
                return None, f"Failed to load image file '{image_source}': {e}", {}

        # 5. Base64 Data URI or raw base64 string
        if isinstance(image_source, str) and (image_source.startswith("data:") or len(image_source) > 100):
            try:
                data_str = image_source
                if "base64," in data_str:
                    data_str = data_str.split("base64,")[1]
                image_bytes = base64.b64decode(data_str)
                return Florence2InferenceService.prepare_image(image_bytes)
            except Exception as e:
                return None, f"Failed to decode base64 image data: {e}", {}

        # 6. Bytes data
        if isinstance(image_source, bytes):
            # Try reading multi-band TIFF via tifffile
            try:
                import tifffile
                raw_array = tifffile.imread(io.BytesIO(image_source))
                return Florence2InferenceService.prepare_image(raw_array)
            except Exception:
                pass
            try:
                with Image.open(io.BytesIO(image_source)) as img:
                    rgb_img = img.convert("RGB")
                    return (
                        rgb_img,
                        "Optical RGB",
                        {"width": rgb_img.width, "height": rgb_img.height, "is_multispectral_derived": False}
                    )
            except Exception as e:
                return None, f"Failed to decode image bytes: {e}", {}

        return None, f"Unsupported image source type: {type(image_source).__name__}", {}

    # ------------------------------------------------------------------
    # Specialized Task Execution
    # ------------------------------------------------------------------

    def _generate(
        self,
        image: Image.Image,
        task_prompt: str,
        text_input: Optional[str] = None
    ) -> Tuple[Dict[str, Any], float]:
        """
        Runs the Florence-2 forward pass and decodes output.
        Thread-safe execution under self._infer_lock to prevent CPU starvation.
        """
        import torch

        if not self._is_loaded:
            success, msg = self.load_model()
            if not success:
                raise RuntimeError(f"Cannot execute Florence-2: {msg}")

        full_prompt = task_prompt + text_input if text_input else task_prompt
        inputs = self._processor(text=full_prompt, images=image, return_tensors="pt")

        t0 = time.time()
        with self._infer_lock:
            with torch.no_grad():
                generated_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                    early_stopping=False,
                )
        duration = time.time() - t0

        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_result = self._processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        return parsed_result, duration

    def caption(self, image: Image.Image, detailed: bool = True) -> Dict[str, Any]:
        """
        Generates dense natural language scene descriptions.
        """
        prompt = "<MORE_DETAILED_CAPTION>" if detailed else "<DETAILED_CAPTION>"
        parsed, duration = self._generate(image, prompt)
        caption_text = parsed.get(prompt, "")
        if isinstance(caption_text, dict):
            caption_text = str(caption_text)

        return {
            "task": "image_captioning",
            "prompt": prompt,
            "caption": caption_text.strip(),
            "latency_s": round(duration, 2),
            "latency_ms": int(duration * 1000),
        }

    def vqa(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """
        Answers natural language visual questions about the image.
        """
        prompt = "<VQA>"
        parsed, duration = self._generate(image, prompt, question)
        raw_answer = parsed.get(prompt, "")
        if isinstance(raw_answer, dict):
            raw_answer = str(raw_answer)

        # Sanitize any leading vocabulary artifact tokens (e.g. "QA>", "<s>")
        clean_answer = raw_answer.strip()
        if clean_answer.startswith("QA>"):
            clean_answer = clean_answer[3:].strip()
        if clean_answer.startswith("<s>"):
            clean_answer = clean_answer[3:].strip()

        return {
            "task": "visual_question_answering",
            "question": question,
            "answer": clean_answer,
            "latency_s": round(duration, 2),
            "latency_ms": int(duration * 1000),
        }

    def ground(self, image: Image.Image, phrase: str) -> Dict[str, Any]:
        """
        Localizes text-guided phrases into spatial pixel bounding boxes.
        Uses <CAPTION_TO_PHRASE_GROUNDING>, falling back to <OPEN_VOCABULARY_DETECTION>.
        """
        prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        parsed, duration = self._generate(image, prompt, phrase)
        result_data = parsed.get(prompt, {})

        bboxes = result_data.get("bboxes", [])
        labels = result_data.get("labels", [])

        # Fallback to <OPEN_VOCABULARY_DETECTION> if no boxes were grounded
        if not bboxes:
            ovd_prompt = "<OPEN_VOCABULARY_DETECTION>"
            ovd_parsed, ovd_duration = self._generate(image, ovd_prompt, phrase)
            duration += ovd_duration
            ovd_data = ovd_parsed.get(ovd_prompt, {})
            bboxes = ovd_data.get("bboxes", [])
            labels = ovd_data.get("bboxes_labels", [])

        boxes = []
        for i, bbox in enumerate(bboxes):
            label_str = labels[i] if i < len(labels) else phrase
            x1, y1, x2, y2 = bbox
            boxes.append({
                "id": f"florence-box-{i + 1}",
                "label": str(label_str),
                "confidence": None,  # Florence-2 outputs spatial coords without fabricated score
                "x": round(float(x1), 2),
                "y": round(float(y1), 2),
                "width": round(float(x2 - x1), 2),
                "height": round(float(y2 - y1), 2),
                "description": f"Spatial region for '{label_str}' grounded in image pixel space."
            })

        return {
            "task": "phrase_grounding",
            "phrase": phrase,
            "boundingBoxes": boxes,
            "boxes_count": len(boxes),
            "image_dimensions": [image.width, image.height],
            "latency_s": round(duration, 2),
            "latency_ms": int(duration * 1000),
        }

    # ------------------------------------------------------------------
    # Unified Prediction Endpoint for SatQuery Agent
    # ------------------------------------------------------------------

    def predict(
        self,
        image_source: Any,
        query: str,
        task_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Primary entry point for SatQuery agent execution.
        Routes to captioning, VQA, or grounding based on task_type.
        """
        # Step 1: Ingest and validate image
        pil_img, modality_desc, img_meta = self.prepare_image(image_source)
        if pil_img is None:
            raise ValueError(f"Florence-2 input validation error: {modality_desc}")

        # Step 2: Ensure model is loaded
        if not self._is_loaded:
            success, msg = self.load_model()
            if not success:
                raise RuntimeError(f"Florence-2 model unavailable: {msg}")

        # Step 3: Route to specific task
        bounding_boxes = None
        overlay_type = "none"

        if task_type in ["scene-captioning", "image_captioning"]:
            res = self.caption(pil_img, detailed=True)
            answer = res["caption"]
            duration_ms = res["latency_ms"]
            task_label = "Image Captioning"
        elif task_type in ["text-guided-grounding", "phrase_grounding"]:
            # Extract target phrase from query (e.g. "locate the roads" -> "roads")
            phrase = query
            for prefix in ["where is", "where are", "locate", "highlight", "find", "show me", "ground"]:
                if query.lower().startswith(prefix):
                    phrase = query[len(prefix):].strip()
                    break
            phrase = phrase.strip("? .")
            if not phrase:
                phrase = query

            res = self.ground(pil_img, phrase)
            bounding_boxes = res["boundingBoxes"]
            duration_ms = res["latency_ms"]
            task_label = "Text-Guided Phrase Grounding"
            overlay_type = "grounding"
            box_count = len(bounding_boxes)
            if box_count > 0:
                answer = (
                    f"Florence-2 localized {box_count} spatial region(s) corresponding to '{phrase}' "
                    f"in image pixel space ({pil_img.width}×{pil_img.height} px)."
                )
            else:
                answer = (
                    f"Florence-2 scanned the image ({pil_img.width}×{pil_img.height} px) for '{phrase}', "
                    f"but detected no salient visual matches."
                )
        else:
            # Default: Visual Question Answering
            res = self.vqa(pil_img, query)
            answer = res["answer"]
            duration_ms = res["latency_ms"]
            task_label = "Visual Question Answering"

        # Step 4: Build scientifically honest evidence citations
        evidence = [
            f"VLM Task: {task_label} (executed via microsoft/Florence-2-base on CPU runtime).",
            f"Input Representation: {modality_desc} ({pil_img.width}×{pil_img.height} px).",
            f"Inference Latency: {duration_ms} ms.",
            "Coordinate Provenance: Bounding boxes (if present) are in image pixel space; no geographic CRS guessing.",
        ]
        if img_meta.get("is_multispectral_derived"):
            evidence.append(
                "Spectral Context: Visual interpretation operated on derived B04-B03-B02 true-color composite; "
                "full 10-band multispectral data is preserved for BigEarthNet specialist."
            )

        return {
            "model": "Florence-2",
            "model_name": "Florence-2 Vision-Language Specialist",
            "model_source": "microsoft/Florence-2-base",
            "task": task_type,
            "task_label": task_label,
            "answer": answer,
            "confidence": None,  # No fabricated confidence score
            "evidence": evidence,
            "boundingBoxes": bounding_boxes,
            "imageOverlayType": overlay_type,
            "execution_duration_ms": duration_ms,
            "input_modality": modality_desc,
            "image_dimensions": [pil_img.width, pil_img.height],
            "is_multispectral_derived": img_meta.get("is_multispectral_derived", False),
            "is_simulation": False,
        }


# Global singleton instance
florence2_service = Florence2InferenceService()
