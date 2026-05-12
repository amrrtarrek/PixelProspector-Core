# PixelProspector V4.1 AI Pipeline - Startup Script

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  PixelProspector V4.1 AI Pipeline - Startup Script" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

Write-Host "`n[1/3] Database already initialized via pipeline." -ForegroundColor Yellow

Write-Host "`n[2/3] Starting FastAPI Orchestrator (New Window)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 05_fastapi_agent; python -m uvicorn core.agent_router:app --port 8000"

Write-Host "`n[3/3] Starting Streamlit Dashboard (New Window)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m streamlit run 01_data_ingestion/dashboard.py"

Write-Host "`n=======================================================" -ForegroundColor Green
Write-Host "ALL SYSTEMS STARTING UP!" -ForegroundColor Green
Write-Host "1. Wait for Server terminal to show 'Uvicorn running on http://127.0.0.1:8000'."
Write-Host "2. The Dashboard will open automatically in your browser."
Write-Host "=======================================================" -ForegroundColor Green
