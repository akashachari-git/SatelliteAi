@echo off
setlocal
cd /d "%~dp0"

echo ======================================================================
echo                  SATQUERY AI — GEOSPATIAL INTELLIGENCE
echo       Multimodal Remote Sensing Vision-Language Assistant
echo ======================================================================
echo.
echo [1/2] Launching Backend (FastAPI on http://127.0.0.1:8000)...
start "SatQuery AI Backend" cmd /k "cd /d "%~dp0backend" && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

echo [2/2] Launching Frontend (React on http://localhost:5173)...
start "SatQuery AI Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev -- --host"

echo.
echo Opening browser to http://localhost:5173...
timeout /t 3 >nul
start http://localhost:5173
echo.
echo ======================================================================
echo SatQuery AI is running! Keep the terminal windows open.
echo ======================================================================
pause
