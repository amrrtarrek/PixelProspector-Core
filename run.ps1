# PixelProspector V4.1 AI Pipeline - Startup Script

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  PixelProspector V4.1 AI Pipeline - Startup Script" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

Write-Host "`n[1/3] Database already initialized via pipeline." -ForegroundColor Yellow

Write-Host "`n[2/3] Starting FastAPI Orchestrator (New Window)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 05_fastapi_agent; python -m uvicorn core.agent_router:app --port 8000"

Write-Host "`n[3/3] Starting React Dashboard (New Window)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 06_react_dashboard; npm run dev"

Write-Host "`n=======================================================" -ForegroundColor Green
Write-Host "ALL SYSTEMS STARTING UP!" -ForegroundColor Green
Write-Host "1. Wait for Server terminal to show 'Uvicorn running on http://127.0.0.1:8000'."
Write-Host "2. The React Dashboard will be available at http://localhost:5173"
Write-Host "=======================================================" -ForegroundColor Green
