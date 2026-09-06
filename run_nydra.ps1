
# NYDRA — One-Command Startup Script
# This script launches both the Python API and the React Frontend simultaneously.

$host.UI.RawUI.WindowTitle = "NYDRA — Autonomous Data Inspection"

Write-Host "`n🩺 Starting Nydra Intelligence..." -ForegroundColor Cyan
Write-Host "-------------------------------------------"

# 1. Start the Python API in the background
Write-Host "▶ Starting Backend Engine (Port 8000)..." -ForegroundColor Yellow
$BackendProcess = Start-Process .\.venv\Scripts\python.exe -ArgumentList "-m uvicorn api:app --port 8000" -NoNewWindow -PassThru

# 2. Start the React Frontend
Write-Host "▶ Starting Frontend UI (Port 3000)..." -ForegroundColor Yellow
Write-Host "▶ Opening Browser at http://localhost:3000" -ForegroundColor Green
Write-Host "-------------------------------------------"
Write-Host "Press CTRL+C to stop both servers.`n"

Set-Location frontend
npm run dev

# Cleanup: When the user stops the script, kill the backend too
Stop-Process -Id $BackendProcess.Id -Force
