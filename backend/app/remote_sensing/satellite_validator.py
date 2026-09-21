import numpy as np
import io
import base64
import json
import urllib.request
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from PIL import Image, ExifTags
from ..config import DEMO_DATA_DIR

REJECTION_MSG_CLEAR = (
    "Unsupported image. SatQuery AI accepts only satellite or remote-sensing imagery. "
    "Please upload a valid satellite image such as GeoTIFF/TIFF or compatible satellite imagery."
)

REJECTION_MSG_UNCERTAIN = (
    "Unable to verify this as satellite imagery. Please upload a clearer satellite/remote-sensing image."
)

VALID_CONFIRMATION_MSG = "Verified authentic remote-sensing / satellite imagery."


class SatelliteImageValidator:
    """
    Validates whether an input image is a legitimate satellite / aerial remote-sensing image,
    and strictly rejects everyday non-EO photos (ground-level photography, indoor rooms, selfies,
    portraits, animals, vehicles, food, documents, screenshots, and synthetic graphics),
    while reliably supporting Earth Observation PNG, JPG, and GeoTIFF rasters.
    """

    REJECTION_CLEAR = REJECTION_MSG_CLEAR
    REJECTION_UNCERTAIN = REJECTION_MSG_UNCERTAIN

    _VALIDATION_CACHE: Dict[str, Tuple[float, Tuple[bool, str, float]]] = {}
    _MAX_CACHE = 64

    @classmethod
    def validate_satellite_image(
        cls,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, float]:
        path = Path(file_path)
        meta = metadata or {}

        if path.exists():
            try:
                mtime = path.stat().st_mtime
                cache_key = str(path.resolve())
                if cache_key in cls._VALIDATION_CACHE:
                    cached_mtime, cached_res = cls._VALIDATION_CACHE[cache_key]
                    if cached_mtime == mtime:
                        return cached_res
            except Exception:
                mtime = 0.0
        else:
            mtime = 0.0

        res = cls._validate_image_uncached(path, meta)
        if path.exists() and mtime > 0:
            if len(cls._VALIDATION_CACHE) >= cls._MAX_CACHE:
                first_k = next(iter(cls._VALIDATION_CACHE))
                del cls._VALIDATION_CACHE[first_k]
            cls._VALIDATION_CACHE[str(path.resolve())] = (mtime, res)
        return res

    @classmethod
    def _validate_image_uncached(
        cls,
        path: Path,
        meta: Dict[str, Any]
    ) -> Tuple[bool, str, float]:
        file_path = str(path)

        # -------------------------------------------------------------
        # 1. Authentic Demo Data Whitelist
        # -------------------------------------------------------------
        try:
            if DEMO_DATA_DIR.exists() and path.resolve().parent == DEMO_DATA_DIR.resolve():
                return True, "Verified authentic curated Earth Observation scenario raster.", 0.99
        except Exception:
            pass

        # -------------------------------------------------------------
        # 2. Digital Camera / Smartphone EXIF Inspection
        # -------------------------------------------------------------
        # Everyday consumer devices (smartphones, DSLRs) embed camera EXIF metadata
        # that never exists on satellite or aerial remote-sensing sensors.
        try:
            with Image.open(file_path) as test_img:
                exif_data = test_img._getexif()
                if exif_data:
                    camera_signatures = [
                        "apple", "iphone", "samsung", "google", "pixel",
                        "canon", "nikon", "sony", "huawei", "xiaomi",
                        "oneplus", "oppo", "vivo", "motorola", "panasonic",
                        "fujifilm", "olympus"
                    ]
                    exif_str = ""
                    for tag_id, value in exif_data.items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id)).lower()
                        if tag_name in ["make", "model", "lensmodel", "software"]:
                            exif_str += f" {str(value).lower()}"
                    
                    if any(brand in exif_str for brand in camera_signatures):
                        return False, REJECTION_MSG_CLEAR, 0.02
        except Exception:
            pass

        # -------------------------------------------------------------
        # 3. GeoTIFF & Geospatial Metadata Verification
        # -------------------------------------------------------------
        # Legitimate GeoTIFFs contain coordinate reference systems, map projections, or tie points.
        if meta.get("is_geotiff"):
            geo_tags = meta.get("geo_tags", {})
            has_geo = any(k.lower() in str(geo_tags).lower() for k in [
                "model", "geokey", "tiepoint", "pixelscale", "crs", "projection"
            ])
            has_crs = "EPSG" in str(meta.get("crs", "")) or "326" in str(meta.get("crs", ""))
            has_bounds = meta.get("bounds") is not None
            has_multiband = meta.get("bands", 1) > 3

            if has_geo or has_crs or has_bounds or has_multiband:
                return True, "Verified authentic geospatial raster with embedded coordinate reference tags.", 0.99

        # -------------------------------------------------------------
        # 4. Multispectral & SAR Physical Characteristics
        # -------------------------------------------------------------
        bands = meta.get("bands", 1)
        # Multispectral rasters with > 3 bands (e.g. RedEdge, NIR, SWIR)
        if bands > 3:
            return True, "Verified authentic multispectral remote-sensing imagery.", 0.98

        # -------------------------------------------------------------
        # 5. Load and Prepare Image Array for Visual Analysis
        # -------------------------------------------------------------
        try:
            with Image.open(file_path) as img:
                # Basic dimensions check
                orig_w, orig_h = img.size
                if orig_w < 32 or orig_h < 32:
                    return False, REJECTION_MSG_CLEAR, 0.0

                # Alpha compositing
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    rgba = img.convert("RGBA")
                    bg = Image.new("RGB", rgba.size, (15, 23, 42))
                    bg.paste(rgba, mask=rgba.split()[3])
                    rgb_img = bg
                else:
                    rgb_img = img.convert("RGB")

                # Analysis resolution
                resized = rgb_img.resize((256, 256), Image.Resampling.BILINEAR)
                arr = np.array(resized, dtype=np.float32)
        except Exception:
            return False, REJECTION_MSG_CLEAR, 0.0

        h, w, c = arr.shape
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        overall_std = float(np.std(gray))
        overall_mean = float(np.mean(gray))

        # -------------------------------------------------------------
        # 6. Check for Completely Blank / Uniform Images
        # -------------------------------------------------------------
        if overall_std < 2.0:
            return False, REJECTION_MSG_CLEAR, 0.0

        # Calculate local spatial gradient & block distribution
        gx = np.abs(np.diff(gray, axis=1))[:-1, :]  # shape (255, 255)
        gy = np.abs(np.diff(gray, axis=0))[:, :-1]  # shape (255, 255)
        edge_energy = gx + gy  # shape (255, 255)

        # 4x4 spatial grid analysis
        grid_h, grid_w = edge_energy.shape[0] // 4, edge_energy.shape[1] // 4
        block_energies = []
        for i in range(4):
            for j in range(4):
                block = edge_energy[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                block_energies.append(float(np.mean(block)))

        block_energies = np.array(block_energies)
        center_indices = [5, 6, 9, 10]
        border_indices = [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]

        center_energy = float(np.mean(block_energies[center_indices]))
        border_energy = float(np.mean(block_energies[border_indices]))

        if border_energy > 0.1:
            central_concentration = center_energy / border_energy
        else:
            central_concentration = 1.0

        # Saliency entropy across the 16 blocks (1.0 = evenly distributed Earth surface)
        norm_blocks = block_energies / (np.sum(block_energies) + 1e-6)
        entropy = -float(np.sum(norm_blocks * np.log2(norm_blocks + 1e-8)))
        entropy_ratio = entropy / np.log2(16)

        # -------------------------------------------------------------
        # 6b. Check for Low Information / Blurry Images (Uncertainty)
        # -------------------------------------------------------------
        laplacian = (
            gray[1:-1, 2:] + gray[1:-1, :-2] +
            gray[2:, 1:-1] + gray[:-2, 1:-1] -
            4 * gray[1:-1, 1:-1]
        )
        lap_var = float(np.var(laplacian))

        # If image lacks clarity / is too blurry to verify as satellite imagery
        if lap_var < 10.0 and not meta.get("is_geotiff"):
            return False, REJECTION_MSG_UNCERTAIN, 0.45

        # -------------------------------------------------------------
        # 7. Human Subjects, Faces, & Selfies (Skin Tone & Portrait Layout)
        # -------------------------------------------------------------
        cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b

        # Human skin tone cluster in YCbCr
        skin_mask = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173) & (gray > 40) & (gray < 235)
        skin_ratio = float(np.sum(skin_mask) / (h * w))
        
        # Center crop (central 50% region where faces and selfies concentrate)
        center_skin = skin_mask[64:192, 64:192]
        center_skin_ratio = float(np.sum(center_skin) / (128 * 128))

        # Distinguish human portraits from natural soil/terracotta roofs:
        # Portraits have a central subject with low/smooth texture.
        # Satellite terrain is decentralized (entropy_ratio > 0.88, central_concentration < 1.6, lap_var > 80).
        is_centered_human = (central_concentration > 1.6 or entropy_ratio < 0.88 or lap_var < 80.0)
        if skin_ratio > 0.45 or ((skin_ratio > 0.18 or center_skin_ratio > 0.28) and is_centered_human):
            return False, REJECTION_MSG_CLEAR, 0.03

        # -------------------------------------------------------------
        # 8. Document, Receipt, UI Graphic, & Text Screenshot Check
        # -------------------------------------------------------------
        # Scanned papers and text graphics typically have flat white/black backgrounds
        white_pixels = (r > 240) & (g > 240) & (b > 240)
        black_pixels = (r < 18) & (g < 18) & (b < 18)
        white_ratio = float(np.mean(white_pixels))
        black_ratio = float(np.mean(black_pixels))

        if white_ratio > 0.55 or black_ratio > 0.70:
            return False, REJECTION_MSG_CLEAR, 0.04

        # Check for extreme bimodal text contrast
        hist, _ = np.histogram(gray, bins=32, range=(0, 256))
        bimodal_peaks = (hist[0] + hist[-1]) / float(np.sum(hist))
        if bimodal_peaks > 0.65 and white_ratio > 0.40:
            return False, REJECTION_MSG_CLEAR, 0.04

        # -------------------------------------------------------------
        # 9. Synthetic Clip Art & Artificial Neon Graphics
        # -------------------------------------------------------------
        hsv_img = resized.convert("HSV")
        hsv_arr = np.array(hsv_img, dtype=np.float32)
        sat = hsv_arr[:, :, 1]
        val = hsv_arr[:, :, 2]

        neon_mask = (sat > 230) & (val > 230)
        if float(np.mean(neon_mask)) > 0.28:
            return False, REJECTION_MSG_CLEAR, 0.05

        # -------------------------------------------------------------
        # 10. Terrestrial Ground-Level Horizon & Atmospheric Sky Perspective
        # -------------------------------------------------------------
        # Outdoor eye-level photos have a distinct horizon: the upper region is sky
        # (bright, uniform, low-variance blue or overcast) while the lower region is ground.
        # Nadir satellite imagery has no atmospheric sky or horizon.
        top_third_gray = gray[:85, :]
        bot_two_thirds_gray = gray[85:, :]
        top_std = float(np.std(top_third_gray))
        top_mean = float(np.mean(top_third_gray))
        bot_mean = float(np.mean(bot_two_thirds_gray))

        top_r = r[:85, :]
        top_g = g[:85, :]
        top_b = b[:85, :]

        # Clear blue sky
        sky_blue = (top_b > top_r + 20) & (top_b > top_g + 8) & (top_b > 110)
        sky_blue_ratio = float(np.sum(sky_blue) / (85 * 256))

        # Overcast or bright white/gray sky
        overcast_sky = (top_third_gray > 185) & (np.abs(top_r - top_g) < 20) & (np.abs(top_g - top_b) < 20)
        overcast_ratio = float(np.sum(overcast_sky) / (85 * 256))

        # Check for sky horizon transition
        if (sky_blue_ratio > 0.35 or overcast_ratio > 0.45) and top_std < 28.0 and (top_mean - bot_mean) > 25:
            return False, REJECTION_MSG_CLEAR, 0.05

        # -------------------------------------------------------------
        # 11. Central Foreground Object / Close-Up Saliency Check
        #     (Cars, Animals/Pets, Food Plates, Indoor Furniture, Handheld Objects)
        # -------------------------------------------------------------
        # Remote sensing scenes have decentralized spatial structure across the frame.
        # Everyday photos of animals, cars, food, and indoor objects have a dominant
        # salient object in the center, with a blurred or plain periphery (depth of field bokeh).
        if central_concentration > 3.8 and center_energy > 8.0:
            return False, REJECTION_MSG_CLEAR, 0.06

        # Food plate check: warm high-saturation food colors centrally isolated
        # on plate/table background
        warm_food_mask = (r > g + 25) & (g > b) & (r > 120) & (sat > 80)
        center_food = warm_food_mask[64:192, 64:192]
        border_food = np.concatenate([
            warm_food_mask[:64, :].ravel(),
            warm_food_mask[192:, :].ravel(),
            warm_food_mask[64:192, :64].ravel(),
            warm_food_mask[64:192, 192:].ravel()
        ])
        center_food_ratio = float(np.mean(center_food))
        border_food_ratio = float(np.mean(border_food))
        if center_food_ratio > 0.30 and border_food_ratio < 0.08:
            return False, REJECTION_MSG_CLEAR, 0.05

        # -------------------------------------------------------------
        # 12. Multimodal AI Verification (Gemini Vision) if active
        # -------------------------------------------------------------
        gemini_result = cls._verify_with_gemini(file_path)
        if gemini_result is not None:
            is_eo, gemini_status, gemini_conf = gemini_result
            if not is_eo:
                if gemini_status == "UNCERTAIN":
                    return False, REJECTION_MSG_UNCERTAIN, gemini_conf
                return False, REJECTION_MSG_CLEAR, gemini_conf
            return True, VALID_CONFIRMATION_MSG, gemini_conf

        # -------------------------------------------------------------
        # 13. Algorithmic Remote Sensing Characteristics Evaluation
        # -------------------------------------------------------------
        # Positive indicators for satellite / aerial Earth Observation:
        # - Top-down orthographic perspective (uniform spatial scale)
        # - Decentralized spatial frequency & natural terrain texture
        # - Plausible land-cover / remote sensing spectral signatures
        
        # Saliency entropy: how evenly distributed is texture across the 16 blocks?
        norm_blocks = block_energies / (np.sum(block_energies) + 1e-6)
        entropy = -float(np.sum(norm_blocks * np.log2(norm_blocks + 1e-8)))
        max_entropy = np.log2(16)  # 4.0
        entropy_ratio = entropy / max_entropy  # close to 1.0 = highly decentralized

        # If spatial information is extremely concentrated in a single spot (entropy ratio < 0.62)
        # and not a GeoTIFF, it's very likely an isolated object
        if entropy_ratio < 0.62 and not meta.get("is_geotiff"):
            return False, REJECTION_MSG_CLEAR, 0.15

        # Ambiguous / uncertain range check:
        # If overall standard deviation is marginal or edge energy is very low
        if overall_std < 10.0 and lap_var < 20.0 and not meta.get("is_geotiff"):
            return False, REJECTION_MSG_UNCERTAIN, 0.48

        # Confidence calculation
        if meta.get("is_geotiff"):
            confidence = 0.96
        elif entropy_ratio > 0.85 and lap_var > 35.0:
            confidence = 0.92
        else:
            confidence = 0.86

        return True, VALID_CONFIRMATION_MSG, confidence

    @classmethod
    def _verify_with_gemini(cls, file_path: str) -> Optional[Tuple[bool, str, float]]:
        """
        Uses Google Gemini Multimodal Vision to verify whether the image is authentic
        Earth Observation / satellite / aerial remote sensing imagery.
        Returns None if Gemini is unavailable, quota is exhausted, or not configured.
        """
        try:
            from ..tools.gemini_service import GeminiVisionService
            api_key = GeminiVisionService._get_api_key()
            if not api_key:
                return None

            image_b64 = GeminiVisionService._prepare_image_b64(file_path)
            if not image_b64:
                return None

            prompt = (
                "You are a strict Earth Observation and Satellite Remote Sensing Validator for SatQuery AI.\n"
                "Determine whether the uploaded image is an authentic satellite, orbital, or high-altitude aerial remote sensing image of Earth terrain, geography, land cover, or water bodies.\n\n"
                "STRICTLY REJECT non-satellite images such as:\n"
                "- Selfies, human faces, portraits, people\n"
                "- Cars, vehicles, street-level views\n"
                "- Animals, pets, organisms\n"
                "- Food, plates, dining tables\n"
                "- Scanned documents, receipts, text, screenshots, diagrams\n"
                "- Indoor photos, rooms, furniture, ceilings\n"
                "- Regular ground-level photos with sky, horizon, or buildings viewed from the side\n\n"
                "Output JSON ONLY:\n"
                "{\n"
                '  "is_satellite": boolean,\n'
                '  "confidence": number between 0.0 and 1.0,\n'
                '  "status": "VALID" | "CLEARLY_NOT_SATELLITE" | "UNCERTAIN",\n'
                '  "reason": "short explanation (max 10 words)"\n'
                "}"
            )

            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": image_b64
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 150
                }
            }

            models_to_try = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-flash-latest"]
            for model_name in models_to_try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                try:
                    with urllib.request.urlopen(req, timeout=6) as response:
                        res_data = json.loads(response.read().decode("utf-8"))
                    candidates = res_data.get("candidates", [])
                    if not candidates:
                        continue
                    cand_parts = candidates[0].get("content", {}).get("parts", [])
                    text_parts = [p.get("text", "") for p in cand_parts if "text" in p and not p.get("thought", False)]
                    raw_text = " ".join(text_parts).strip()
                    clean_json = raw_text.replace("```json", "").replace("```", "").strip()
                    data = json.loads(clean_json)
                    is_sat = bool(data.get("is_satellite", False))
                    status = str(data.get("status", "VALID" if is_sat else "CLEARLY_NOT_SATELLITE")).upper()
                    conf = float(data.get("confidence", 0.9 if is_sat else 0.1))
                    return is_sat, status, conf
                except Exception:
                    continue
        except Exception:
            pass

        return None
