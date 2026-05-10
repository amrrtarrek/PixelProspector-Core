@echo off
setlocal
echo =======================================================
echo   PixelProspector V4.1 AI Pipeline - Startup Script
echo =======================================================

echo [1/2] Initializing Database (SQLite Fallback)...
cd 01_data_ingestion
python db.py --reset
cd ..

echo.
echo [2/3] Starting FastAPI Orchestrator...
start "PixelProspector-Server" cmd /k "python app.py"

echo.
echo [3/3] Starting Streamlit Dashboard...
start "PixelProspector-Dashboard" cmd /k "streamlit run 05_dashboard_action/app.py"

echo.
echo =======================================================
echo ALL SYSTEMS STARTING UP!
echo 1. Wait for Server terminal to show 'Uvicorn running'.
echo 2. The Dashboard will open automatically in your browser.
echo =======================================================
pause
