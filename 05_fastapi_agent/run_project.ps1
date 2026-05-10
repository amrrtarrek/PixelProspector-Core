# PixelProspector V4.1 AI Pipeline - PowerShell Startup Script

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  PixelProspector V4.1 AI Pipeline - Startup Script" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

Write-Host "`n[1/2] Initializing Database (SQLite Fallback)..." -ForegroundColor Yellow
Set-Location 01_data_ingestion
python db.py --reset
Set-Location ..

Write-Host "`n[2/3] Starting FastAPI Orchestrator (New Window)..." -ForegroundColor Yellow
Start-Process cmd.exe -ArgumentList "/k", "python app.py"

Write-Host "`n[3/3] Starting Streamlit Dashboard (New Window)..." -ForegroundColor Yellow
Start-Process cmd.exe -ArgumentList "/k", "streamlit run 05_dashboard_action/app.py"

Write-Host "`n=======================================================" -ForegroundColor Green
Write-Host "ALL SYSTEMS STARTING UP!" -ForegroundColor Green
Write-Host "1. Wait for Server terminal to show 'Uvicorn running'."
Write-Host "2. The Dashboard will open automatically in your browser."
Write-Host "=======================================================" -ForegroundColor Green

Read-Host "`nPress Enter to exit this window..."
