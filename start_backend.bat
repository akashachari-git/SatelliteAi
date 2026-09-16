@echo off
setlocal
cd /d "%~dp0backend"
echo ========================================================
echo Starting SatQuery AI FastAPI Backend on port 8000...
echo ========================================================
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
