@echo off
title DeepFake Detector — Starting...
color 0B
echo.
echo  =====================================================
echo   DeepFake Detector - AI Media Forensics Platform
echo  =====================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python is not installed or not in PATH.
    echo  Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo  [1/4] Python found.

:: Create virtual environment if not exists
if not exist "venv\" (
    echo  [2/4] Creating virtual environment...
    python -m venv venv
) else (
    echo  [2/4] Virtual environment already exists.
)

:: Activate venv
call venv\Scripts\activate.bat

:: Install requirements
echo  [3/4] Installing requirements (this may take a moment)...
pip install -r backend\requirements.txt --quiet

echo  [4/4] Starting server at http://localhost:5000 ...
echo.
echo  Open your browser to:  http://localhost:5000
echo  Press Ctrl+C to stop the server.
echo.

:: Open browser after 2 seconds
start /b cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:5000"

:: Start Flask
cd backend
python app.py

pause
