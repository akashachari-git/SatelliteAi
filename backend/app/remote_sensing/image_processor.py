import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile
from ..config import PROCESSED_DIR, EVIDENCE_DIR

class ImageProcessor:
    """
    Robust remote-sensing image processor:
    - GeoTIFF/TIFF reading & normalization with LRU mtime-based array cache
    - 2-98% percentile radiometric stretching
    - Web-safe PNG generation and thumbnailing with preview reuse
    - Overlay generation (masks, heatmaps, bounding boxes)
    - False-color composite generation
    """

    _ARRAY_CACHE: Dict[str, Tuple[float, np.ndarray]] = {}
    _MAX_CACHE_ENTRIES = 32

    @classmethod
    def load_as_array(cls, file_path: str) -> np.ndarray:
        """Load image as float32 numpy array normalized to [0, 255] or [0, 1] with LRU mtime caching."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Raster file not found: {file_path}")

        mtime = path.stat().st_mtime
        cache_key = str(path.resolve())

        if cache_key in cls._ARRAY_CACHE:
            cached_mtime, cached_arr = cls._ARRAY_CACHE[cache_key]
            if cached_mtime == mtime:
                return cached_arr.copy()

        ext = path.suffix.lower()
        arr = None

        if ext in [".tif", ".tiff", ".geotiff"]:
            try:
                raw = tifffile.imread(str(path))
                if raw.ndim == 2:
                    arr = raw.astype(np.float32)
                elif raw.ndim == 3:
                    if raw.shape[0] < raw.shape[2]:
                        arr = np.transpose(raw, (1, 2, 0)).astype(np.float32)
                    else:
                        arr = raw.astype(np.float32)
            except Exception:
                pass

        if arr is None:
            with Image.open(str(path)) as img:
                rgb = img.convert("RGB")
                arr = np.array(rgb, dtype=np.float32)

        if len(cls._ARRAY_CACHE) >= cls._MAX_CACHE_ENTRIES:
            first_k = next(iter(cls._ARRAY_CACHE))
            del cls._ARRAY_CACHE[first_k]

        cls._ARRAY_CACHE[cache_key] = (mtime, arr)
        return arr.copy()

    @classmethod
    def stretch_to_uint8(cls, arr: np.ndarray, lower_pct: float = 2.0, upper_pct: float = 98.0) -> np.ndarray:
        """Radiometric percentile stretch (standard remote sensing visualization)"""
        arr = np.nan_to_num(arr, nan=0.0)
        if arr.ndim == 2:
            p_low, p_high = np.percentile(arr, (lower_pct, upper_pct))
            if p_high == p_low:
                p_high += 1e-5
            stretched = np.clip((arr - p_low) / (p_high - p_low) * 255.0, 0, 255)
            return stretched.astype(np.uint8)
        
        # Multi-band
        out = np.zeros(arr.shape[:2] + (min(arr.shape[2], 3),), dtype=np.uint8)
        for i in range(min(arr.shape[2], 3)):
            band = arr[:, :, i]
            p_low, p_high = np.percentile(band, (lower_pct, upper_pct))
            if p_high == p_low:
                p_high += 1e-5
            stretched = np.clip((band - p_low) / (p_high - p_low) * 255.0, 0, 255)
            out[:, :, i] = stretched.astype(np.uint8)
        return out

    @classmethod
    def generate_web_preview(cls, file_path: str, max_size: int = 1024) -> str:
        """
        Creates or retrieves a web-compatible PNG preview from GeoTIFF or other image.
        Reuses existing previews immediately if up to date.
        Returns the relative web path.
        """
        path = Path(file_path)
        out_filename = f"{path.stem}_preview.png"
        out_path = PROCESSED_DIR / out_filename

        # Re-use existing preview if it's already generated and up-to-date
        if out_path.exists():
            try:
                if out_path.stat().st_mtime >= path.stat().st_mtime and out_path.stat().st_size > 0:
                    return f"/storage/processed/{out_filename}"
            except Exception:
                pass

        # Fast path: standard web RGB image
        ext = path.suffix.lower()
        if ext in [".png", ".jpg", ".jpeg"]:
            try:
                with Image.open(str(path)) as img:
                    if img.mode == "RGB" and max(img.size) <= max_size:
                        img.save(str(out_path), "PNG", optimize=False)
                        return f"/storage/processed/{out_filename}"
                    elif max(img.size) > max_size:
                        resized = img.convert("RGB")
                        resized.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
                        resized.save(str(out_path), "PNG", optimize=False)
                        return f"/storage/processed/{out_filename}"
            except Exception:
                pass

        arr = cls.load_as_array(file_path)
        uint8_arr = cls.stretch_to_uint8(arr)

        if uint8_arr.ndim == 2:
            img = Image.fromarray(uint8_arr, mode="L").convert("RGB")
        else:
            img = Image.fromarray(uint8_arr, mode="RGB")

        # Fast Bilinear resize
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)

        img.save(str(out_path), "PNG", optimize=False)
        return f"/storage/processed/{out_filename}"

    @classmethod
    def create_mask_overlay(cls, base_file_path: str, mask: np.ndarray, color: Tuple[int, int, int] = (239, 68, 68), alpha: float = 0.45) -> str:
        """
        Renders a transparent colored mask overlay on top of the base image.
        Returns the relative web path.
        """
        base_arr = cls.load_as_array(base_file_path)
        uint8_base = cls.stretch_to_uint8(base_arr)
        if uint8_base.ndim == 2:
            base_img = Image.fromarray(uint8_base, mode="L").convert("RGBA")
        else:
            base_img = Image.fromarray(uint8_base, mode="RGB").convert("RGBA")

        # Create overlay
        mask_h, mask_w = mask.shape
        if (mask_w, mask_h) != base_img.size:
            # Resize mask to match
            mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
            mask_img = mask_img.resize(base_img.size, Image.Resampling.NEAREST)
            mask_resized = np.array(mask_img) > 128
        else:
            mask_resized = mask > 0

        overlay = np.zeros((base_img.size[1], base_img.size[0], 4), dtype=np.uint8)
        overlay[mask_resized, 0] = color[0]
        overlay[mask_resized, 1] = color[1]
        overlay[mask_resized, 2] = color[2]
        overlay[mask_resized, 3] = int(255 * alpha)

        overlay_img = Image.fromarray(overlay, mode="RGBA")
        combined = Image.alpha_composite(base_img, overlay_img).convert("RGB")

        if max(combined.size) > 1024:
            combined.thumbnail((1024, 1024), Image.Resampling.BILINEAR)

        out_name = f"evidence_mask_{np.random.randint(10000, 99999)}.png"
        out_path = EVIDENCE_DIR / out_name
        combined.save(str(out_path), "PNG")
        return f"/storage/evidence/{out_name}"

    @classmethod
    def create_bounding_box_overlay(cls, base_file_path: str, boxes: List[Dict[str, Any]], color: str = "#06b6d4") -> str:
        """
        Draws labeled bounding boxes on image.
        boxes = [{"box": [ymin, xmin, ymax, xmax], "label": "Water body", "score": 0.92}]
        Coordinates normalized [0, 1] or pixel coords.
        """
        base_arr = cls.load_as_array(base_file_path)
        uint8_base = cls.stretch_to_uint8(base_arr)
        if uint8_base.ndim == 2:
            img = Image.fromarray(uint8_base, mode="L").convert("RGB")
        else:
            img = Image.fromarray(uint8_base, mode="RGB")

        draw = ImageDraw.Draw(img)
        w, h = img.size

        for b in boxes:
            box = b["box"]
            # Detect if normalized
            if max(box) <= 1.0:
                y0, x0, y1, x1 = int(box[0]*h), int(box[1]*w), int(box[2]*h), int(box[3]*w)
            else:
                y0, x0, y1, x1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # Draw thick rectangle
            for offset in range(3):
                draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=color)

            # Draw tag label
            label = f"{b.get('label', 'Target')} ({int(b.get('score', 0.9) * 100)}%)"
            draw.rectangle([x0, max(0, y0 - 20), x0 + len(label) * 8 + 8, y0], fill=color)
            draw.text((x0 + 4, max(0, y0 - 18)), label, fill=(255, 255, 255))

        out_name = f"evidence_bbox_{np.random.randint(10000, 99999)}.png"
        out_path = EVIDENCE_DIR / out_name
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        img.save(str(out_path), "PNG")
        return f"/storage/evidence/{out_name}"

    @classmethod
    def create_difference_heatmap(cls, diff_arr: np.ndarray, base_file_path: Optional[str] = None) -> str:
        """
        Renders a scientific difference heatmap (cyan to yellow to red)
        for bi-temporal change magnitude.
        """
        norm_diff = (diff_arr - np.min(diff_arr)) / (np.max(diff_arr) - np.min(diff_arr) + 1e-6)
        
        # Color palette: low (dark slate) -> mid (amber/orange) -> high (vibrant red/cyan)
        h, w = diff_arr.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Simplified turbo/jet map
        rgba[:, :, 0] = np.clip(norm_diff * 2.0 * 255, 0, 255).astype(np.uint8) # Red
        rgba[:, :, 1] = np.clip((1.0 - np.abs(norm_diff - 0.5) * 2.0) * 220, 0, 255).astype(np.uint8) # Green
        rgba[:, :, 2] = np.clip((1.0 - norm_diff * 1.5) * 255, 0, 255).astype(np.uint8) # Blue
        rgba[:, :, 3] = 255

        heat_img = Image.fromarray(rgba, mode="RGBA")
        out_name = f"evidence_change_heat_{np.random.randint(10000, 99999)}.png"
        out_path = EVIDENCE_DIR / out_name
        heat_img.convert("RGB").save(str(out_path), "PNG")
        return f"/storage/evidence/{out_name}"

    @classmethod
    def create_optical_sar_composite(cls, optical_file: str, sar_file: str) -> str:
        """
        Creates a False-Color Optical + SAR fusion composite:
        Red: Optical Luminance / Red band
        Green: SAR Backscatter Amplitude (highlights metallic/dielectric structures)
        Blue: Optical Green/NIR band (highlights water absorption / vegetative texture)
        """
        opt_arr = cls.load_as_array(optical_file)
        sar_arr = cls.load_as_array(sar_file)

        opt_u8 = cls.stretch_to_uint8(opt_arr)
        sar_u8 = cls.stretch_to_uint8(sar_arr)

        if opt_u8.ndim == 3:
            r = opt_u8[:, :, 0]
            b = opt_u8[:, :, 1] if opt_u8.shape[2] > 1 else opt_u8[:, :, 0]
        else:
            r = opt_u8
            b = opt_u8

        if sar_u8.ndim == 3:
            g = sar_u8[:, :, 0]
        else:
            g = sar_u8

        # Resize if shapes mismatch
        if r.shape != g.shape:
            sar_pil = Image.fromarray(g).resize((r.shape[1], r.shape[0]), Image.Resampling.BILINEAR)
            g = np.array(sar_pil)

        fused = np.stack([r, g, b], axis=-1)
        fused_img = Image.fromarray(fused, mode="RGB")

        out_name = f"evidence_fusion_{np.random.randint(10000, 99999)}.png"
        out_path = EVIDENCE_DIR / out_name
        fused_img.save(str(out_path), "PNG")
        return f"/storage/evidence/{out_name}"
