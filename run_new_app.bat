@echo off
TITLE NYDRA - Production Launcher
echo Starting Nydra Intelligence...
echo.

:: 1. Start Backend in background
echo [1/2] Launching Backend Engine...
start /b python -m uvicorn api:app --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem

:: 2. Start Frontend and open browser
echo [2/2] Launching Frontend UI...
cd frontend
npm run dev

:: Note: Browser opens automatically via Next.js
pause
