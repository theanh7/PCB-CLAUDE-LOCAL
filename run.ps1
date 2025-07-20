# =====================================================
# PCB Auto-Inspection System PowerShell Launcher
# =====================================================

# Set window title
$Host.UI.RawUI.WindowTitle = "PCB Inspection System"

# Clear screen
Clear-Host

Write-Host ""
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   PCB AUTO-INSPECTION SYSTEM" -ForegroundColor White
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>$null
    Write-Host "[INFO] Python detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python and try again" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Navigate to script directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "[INFO] Project directory: $PWD" -ForegroundColor Blue
Write-Host ""

# Kill any existing Python processes to free camera
Write-Host "[INFO] Releasing camera resources..." -ForegroundColor Yellow
try {
    Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
} catch {
    # Ignore errors
}

# Run the main application
Write-Host "[INFO] Launching PCB Inspection System..." -ForegroundColor Green
Write-Host ""
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   System is starting... Please wait..." -ForegroundColor White
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

try {
    python main.py
    $exitCode = $LASTEXITCODE
    
    Write-Host ""
    if ($exitCode -eq 0) {
        Write-Host "[SUCCESS] PCB Inspection System closed successfully" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] PCB Inspection System encountered an error" -ForegroundColor Red
        Write-Host "Error code: $exitCode" -ForegroundColor Yellow
    }
} catch {
    Write-Host ""
    Write-Host "[ERROR] Failed to start PCB Inspection System" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to exit"