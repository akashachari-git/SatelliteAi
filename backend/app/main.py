"""
SatQuery AI - Uvicorn Application Entrypoint
Exposes the FastAPI application instance for `uvicorn app.main:app`.
"""
import os
import sys

# Ensure backend root is on sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Also ensure parent of backend is on sys.path if running from repository root
repo_dir = os.path.dirname(backend_dir)
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

try:
    from backend.main import app
except ImportError:
    from main import app

__all__ = ["app"]
