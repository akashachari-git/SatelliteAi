import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env if present
for env_candidate in [BASE_DIR / ".env", BASE_DIR.parent / ".env"]:
    if env_candidate.exists():
        try:
            with open(env_candidate, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip("'\""))
        except Exception as e:
            print(f"Notice: Failed loading {env_candidate}: {e}")

STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
PROCESSED_DIR = STORAGE_DIR / "processed"
EVIDENCE_DIR = STORAGE_DIR / "evidence"
REPORTS_DIR = STORAGE_DIR / "reports"
DEMO_DATA_DIR = STORAGE_DIR / "demo_data"

# Create required directories
for d in [STORAGE_DIR, UPLOADS_DIR, PROCESSED_DIR, EVIDENCE_DIR, REPORTS_DIR, DEMO_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE_MB = 100
ALLOWED_EXTENSIONS = {".tif", ".tiff", ".geotiff", ".png", ".jpg", ".jpeg"}

# Google OAuth & Authentication settings
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
SESSION_COOKIE_NAME = "satquery_session"
SESSION_EXPIRY_HOURS = 24 * 7  # 7 days

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "*"
]

