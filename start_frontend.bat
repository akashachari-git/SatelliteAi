@echo off
setlocal
cd /d "%~dp0frontend"
echo ========================================================
echo Starting SatQuery AI React Frontend on port 5173...
echo ========================================================
npm run dev -- --host
pause
