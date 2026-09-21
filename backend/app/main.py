import os
import shutil
import uuid
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Ensure backend root is on sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
repo_dir = os.path.dirname(backend_dir)
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from .config import UPLOADS_DIR, STORAGE_DIR, CORS_ORIGINS, DEMO_DATA_DIR, REPORTS_DIR, GOOGLE_CLIENT_ID, SESSION_COOKIE_NAME
from .auth.auth_service import AuthService
from .remote_sensing.metadata_extractor import RemoteSensingMetadataExtractor
from .remote_sensing.co_registration import CoRegistrationChecker
from .remote_sensing.satellite_validator import SatelliteImageValidator
from .remote_sensing.image_processor import ImageProcessor
from .agents.agent_controller import AgentController
from .tools.tool_registry import tool_registry
from .models.adaptation import BigEarthNetAdapter, get_bigearthnet_adapter
from .reports.report_service import ReportService
from .evaluation.benchmark_service import BenchmarkEvaluationService
from .demo.demo_generator import DemoDataGenerator

# Ensure demo datasets on startup
try:
    demo_scenarios = DemoDataGenerator.ensure_demo_files()
except Exception:
    demo_scenarios = {}

app = FastAPI(
    title="SatQuery AI API",
    description="Agentic Vision-Language Assistant for Multimodal Remote Sensing Analysis",
    version="1.0.0"
)

# CORS configuration
_env_origins = os.environ.get("ALLOWED_ORIGINS")
if _env_origins:
    _allowed = [o.strip() for o in _env_origins.split(",") if o.strip()]
else:
    _allowed = CORS_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files to serve generated previews, overlays, and reports if storage exists
if STORAGE_DIR.exists():
    app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")

# In-memory storage for analysis results cache
ANALYSIS_CACHE: Dict[str, Dict[str, Any]] = {}
controller = AgentController()

# -------------------------------------------------------------
# Request Schemas
# -------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    query: str
    image_paths: List[str]
    user_metadata: Optional[List[Dict[str, Any]]] = None
    gemini_api_key: Optional[str] = None

class ValidateRequest(BaseModel):
    image_paths: List[str]
    pair_mode: Optional[str] = "AUTO"

class ReportRequest(BaseModel):
    analysis_data: Dict[str, Any]

class GoogleLoginRequest(BaseModel):
    id_token: Optional[str] = None
    credential: Optional[str] = None

# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------

@app.get("/api/auth/config")
def get_auth_config():
    """Returns public Google OAuth Client ID and configuration."""
    return {
        "client_id": GOOGLE_CLIENT_ID,
        "auth_enabled": True
    }

@app.post("/api/auth/google")
def login_with_google(req: GoogleLoginRequest, response: Response):
    token = req.id_token or req.credential
    if not token or not token.strip():
        raise HTTPException(status_code=400, detail="Google sign-in failed. Missing Google token.")

    try:
        user_profile = AuthService.verify_google_id_token(token)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Google sign-in failed. Please try again.")

    session_token = AuthService.create_session(user_profile)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        httponly=True,
        max_age=3600 * 24 * 7,
        samesite="lax"
    )
    return {
        "success": True,
        "session_token": session_token,
        "user": user_profile
    }

@app.get("/api/auth/me")
def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_session_id: Optional[str] = Header(None)
):
    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_token and authorization and authorization.startswith("Bearer "):
        session_token = authorization.replace("Bearer ", "").strip()
    if not session_token and x_session_id:
        session_token = x_session_id.strip()

    user = AuthService.get_session_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthenticated session. Please sign in with Google.")

    return {
        "authenticated": True,
        "user": user
    }

@app.post("/api/auth/logout")
def logout_user(
    request: Request,
    response: Response,
    authorization: Optional[str] = Header(None),
    x_session_id: Optional[str] = Header(None)
):
    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_token and authorization and authorization.startswith("Bearer "):
        session_token = authorization.replace("Bearer ", "").strip()
    if not session_token and x_session_id:
        session_token = x_session_id.strip()

    if session_token:
        AuthService.destroy_session(session_token)

    response.delete_cookie(key=SESSION_COOKIE_NAME)
    return {
        "success": True,
        "message": "Logged out successfully"
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "SatQuery AI Remote Sensing Platform",
        "device": "CPU (Optimized)",
        "active_tools": len(tool_registry.list_tools()),
        "adaptation_status": "BigEarthNet-19 Spectral Adaptation Active"
    }

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    ext = Path(file.filename or "image.png").suffix.lower()
    if ext not in [".tif", ".tiff", ".geotiff", ".png", ".jpg", ".jpeg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Supported: GeoTIFF, TIFF, PNG, JPG.")

    unique_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = UPLOADS_DIR / unique_filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    metadata = RemoteSensingMetadataExtractor.extract_metadata(str(file_path))
    is_satellite, reason, conf = SatelliteImageValidator.validate_satellite_image(str(file_path), metadata)
    if not is_satellite:
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass
        raise HTTPException(
            status_code=400,
            detail=reason
        )

    preview_url = ImageProcessor.generate_web_preview(str(file_path))
    metadata["preview_url"] = preview_url
    metadata["server_path"] = str(file_path)
    metadata["satellite_verification"] = {
        "is_satellite": True,
        "confidence": conf,
        "verification_summary": reason
    }

    return {
        "success": True,
        "filename": file.filename,
        "server_path": str(file_path),
        "preview_url": preview_url,
        "metadata": metadata
    }

@app.post("/api/validate")
def validate_images(req: ValidateRequest):
    if len(req.image_paths) == 0:
        raise HTTPException(status_code=400, detail="No images provided.")

    for idx, p in enumerate(req.image_paths):
        is_sat, reason, _ = SatelliteImageValidator.validate_satellite_image(p)
        if not is_sat:
            return {
                "is_compatible": False,
                "co_registration_score": 0,
                "error": reason,
                "checks": [{"name": f"Satellite Validation (Image {idx+1})", "status": "FAIL", "details": reason}],
                "metadata": {}
            }

    metas = [RemoteSensingMetadataExtractor.extract_metadata(p) for p in req.image_paths]
    if len(req.image_paths) == 1:
        return {
            "is_compatible": True,
            "co_registration_score": 100,
            "checks": [{"name": "Single Image Integrity", "status": "PASS", "details": "Spatial and radiometric profiles valid"}],
            "metadata": metas[0]
        }

    compat = CoRegistrationChecker.evaluate_pair(metas[0], metas[1], pair_mode=req.pair_mode or "AUTO")
    return {
        "metadata_a": metas[0],
        "metadata_b": metas[1],
        "compatibility": compat
    }

@app.post("/api/analyze")
def analyze_query(
    req: AnalyzeRequest,
    x_gemini_api_key: Optional[str] = Header(None)
):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(req.image_paths) == 0:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    resolved_paths = []
    for p in req.image_paths:
        path_obj = Path(p)
        if path_obj.is_absolute() and path_obj.exists():
            resolved_paths.append(str(path_obj))
        elif (UPLOADS_DIR / path_obj.name).exists():
            resolved_paths.append(str(UPLOADS_DIR / path_obj.name))
        elif (STORAGE_DIR / p).exists():
            resolved_paths.append(str((STORAGE_DIR / p).resolve()))
        elif (DEMO_DATA_DIR / p).exists():
            resolved_paths.append(str((DEMO_DATA_DIR / p).resolve()))
        elif path_obj.exists():
            resolved_paths.append(str(path_obj.resolve()))
        else:
            resolved_paths.append(p)

    key = req.gemini_api_key or x_gemini_api_key
    result = controller.process_analysis_request(
        query=req.query,
        image_paths=resolved_paths,
        user_metadata=req.user_metadata,
        gemini_api_key=key
    )

    analysis_id = str(uuid.uuid4())
    ANALYSIS_CACHE[analysis_id] = result
    result["analysis_id"] = analysis_id

    return result

@app.post("/api/gemini/validate")
def validate_gemini_key(body: Dict[str, str] = Body(...)):
    import urllib.request
    key = body.get("api_key", "").strip()
    if not key:
        return {"valid": False, "error": "API key cannot be empty"}
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash?key={key}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=8) as res:
            if res.status == 200:
                return {"valid": True, "model": "gemini-3.6-flash"}
        return {"valid": False, "error": "Invalid response from Gemini API"}
    except Exception as e:
        try:
            url2 = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
            req2 = urllib.request.Request(url2)
            with urllib.request.urlopen(req2, timeout=8) as res2:
                if res2.status == 200:
                    return {"valid": True, "model": "gemini-flash"}
        except Exception:
            pass
        return {"valid": False, "error": str(e)}

@app.get("/api/analysis/{analysis_id}")
def get_analysis(analysis_id: str):
    if analysis_id not in ANALYSIS_CACHE:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return ANALYSIS_CACHE[analysis_id]

@app.get("/api/analysis/{analysis_id}/trace")
def get_analysis_trace(analysis_id: str):
    if analysis_id not in ANALYSIS_CACHE:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return {
        "analysis_id": analysis_id,
        "trace": ANALYSIS_CACHE[analysis_id].get("execution_trace", [])
    }

@app.post("/api/report/generate-pdf")
def generate_pdf(req: ReportRequest):
    pdf_url = ReportService.generate_pdf_report(req.analysis_data)
    return {"pdf_url": pdf_url}

@app.post("/api/report/generate-json")
def generate_json(req: ReportRequest):
    json_url = ReportService.generate_json_export(req.analysis_data)
    return {"json_url": json_url}

@app.get("/api/tools")
def list_tools():
    return {"tools": tool_registry.list_tools()}

@app.get("/api/models")
def list_models():
    adapter = get_bigearthnet_adapter()
    return {
        "adaptation": adapter.get_training_config(),
        "classes": adapter.classes,
        "sensors_supported": ["Sentinel-2 MSI (10m/20m)", "Sentinel-1 C-SAR (VV/VH)", "Landsat 8/9", "PlanetScope"],
        "nomenclature": "CORINE 2018 (19-Class Aggregation for Remote Sensing)"
    }

@app.get("/api/demo-scenarios")
def get_demo_scenarios():
    scenarios = DemoDataGenerator.ensure_demo_files()
    return {"scenarios": scenarios}

@app.post("/api/demo-scenarios/load/{scenario_id}")
def load_demo_scenario(scenario_id: str):
    scenarios = DemoDataGenerator.ensure_demo_files()
    if scenario_id not in scenarios:
        raise HTTPException(status_code=404, detail="Demo scenario not found.")

    sc = scenarios[scenario_id]
    loaded_images = []
    for img_info in sc["images"]:
        file_path = DEMO_DATA_DIR / img_info["filename"]
        metadata = RemoteSensingMetadataExtractor.extract_metadata(str(file_path))
        preview_url = ImageProcessor.generate_web_preview(str(file_path))
        metadata["preview_url"] = preview_url
        metadata["server_path"] = str(file_path)
        loaded_images.append({
            "filename": img_info["filename"],
            "server_path": str(file_path),
            "preview_url": preview_url,
            "modality": img_info["modality"],
            "description": img_info["description"],
            "metadata": metadata
        })

    return {
        "scenario": sc,
        "loaded_images": loaded_images,
        "default_query": sc["default_query"],
        "suggested_queries": sc["suggested_queries"],
        "task": sc["task"]
    }

@app.get("/api/benchmarks")
def list_benchmarks():
    return {"benchmarks": BenchmarkEvaluationService.list_benchmarks()}

@app.post("/api/benchmarks/run/{benchmark_key}")
def run_benchmark(benchmark_key: str):
    return BenchmarkEvaluationService.run_benchmark_eval(benchmark_key)

# Mount production router for additional endpoints
try:
    from backend.main import app as prod_app
    app.include_router(prod_app.router)
except Exception:
    pass

__all__ = ["app"]
