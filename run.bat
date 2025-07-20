@echo off
REM =====================================================
REM PCB Auto-Inspection System Launcher
REM =====================================================

title PCB Inspection System

echo.
echo ====================================================
echo   PCB AUTO-INSPECTION SYSTEM
echo ====================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Navigate to project directory
cd /d "%~dp0"

echo [INFO] Starting PCB Inspection System...
echo [INFO] Project directory: %CD%
echo.

REM Kill any existing Python processes to free camera
echo [INFO] Releasing camera resources...
taskkill /f /im python.exe >nul 2>&1

REM Wait a moment for cleanup
timeout /t 2 /nobreak >nul

REM Run the main application
echo [INFO] Launching PCB Inspection System...
echo.
echo ====================================================
echo   System is starting... Please wait...
echo ====================================================
echo.

python main.py

REM Handle exit codes
if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] PCB Inspection System closed successfully
) else (
    echo.
    echo [ERROR] PCB Inspection System encountered an error
    echo Error code: %errorlevel%
)

echo.
echo Press any key to exit...
pause >nul