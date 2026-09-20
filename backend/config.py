"""
SatQuery AI - Remote Sensing Intelligence Backend
Configuration and Hardware Setup
"""
import os
import platform

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Hardware Runtime Detection
def detect_accelerator() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda: {torch.cuda.get_device_name(0)}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps: Apple Silicon Metal"
    except ImportError:
        pass
    return f"cpu: {platform.processor() or 'Standard Architecture'}"

DEVICE = detect_accelerator()

# Remote Sensing Dataset Standards
SUPPORTED_FORMATS = [".tif", ".tiff", ".geotiff", ".png", ".jpg", ".jpeg"]
MODALITIES = ["Optical RGB", "Multispectral (12-Band)", "SAR (Sentinel-1 C-Band)", "Thermal Infrared"]

# API Metadata
API_TITLE = "SatQuery AI Remote Sensing Backend"
API_VERSION = "2.0.0"
API_DESCRIPTION = (
    "Production FastAPI backend for SatQuery AI: An Interactive Vision-Language Assistant "
    "for Multimodal Remote Sensing Image Analysis through Text Queries."
)
