@echo off
setlocal
echo ╔══════════════════════════════════════════════════════╗
echo ║         NYDRA — UNIFIED STARTUP ENGINE              ║
echo ║    Standardized Protocol: HTTP (Port 8000/3000)     ║
echo ╚══════════════════════════════════════════════════════╝

echo.
echo [1/2] Starting Nydra Backend Engine (API + WebSockets)...
start /b .\.venv\Scripts\python.exe -m uvicorn api:app --port 8000 --log-level info

echo [2/2] Starting Nydra Frontend UI (Next.js)...
cd frontend
npm run dev

pause
