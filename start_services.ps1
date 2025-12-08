# start_services.ps1
# PowerShell script to start all ModelX Anomaly Detection services

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "🚀 Starting ModelX Anomaly Detection Pipeline" -ForegroundColor Cyan
Write-Host "==================================================="

# 1. Check for dependencies
Write-Host "[1/3] Checking dependencies..." -ForegroundColor Yellow
$ModelPath = "models\anomaly-detection\models_cache\lid.176.bin"
if (-not (Test-Path $ModelPath)) {
    Write-Host "⚠️  WARNING: FastText model not found in $ModelPath" -ForegroundColor Red
    Write-Host "   Please download it from: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin" -ForegroundColor Gray
}

# 2. Start Vectorization API (New Window)
Write-Host "[2/3] Starting Vectorization Agent API..." -ForegroundColor Yellow
Write-Host "   - Port: 8001" -ForegroundColor Gray

# Start python process in a new window
$ApiProcess = Start-Process python -ArgumentList "src\api\vectorization_api.py" -PassThru -WindowStyle Minimized
Write-Host "   ✅ API started with PID: $($ApiProcess.Id)" -ForegroundColor Green

# Wait for API to be ready
Start-Sleep -Seconds 5

# 3. Start Airflow (Astro)
Write-Host "[3/3] Starting Apache Airflow (Astro)..." -ForegroundColor Yellow
Write-Host "   - Dashboard: http://localhost:8080" -ForegroundColor Gray

Set-Location "models\anomaly-detection"

# Check if astro is installed
if (Get-Command "astro" -ErrorAction SilentlyContinue) {
    astro dev start
} else {
    Write-Host "❌ Error: 'astro' command not found. Please install Astronomer CLI." -ForegroundColor Red
    Stop-Process -Id $ApiProcess.Id -Force
}

# Note: The script will pause here while Astro is running. 
# If Astro returns immediately (daemon mode), the script might end.
# If you want to keep the script running to manage the API process:

Write-Host "Press any key to stop services..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Cleanup
Write-Host "🛑 Stopping services..." -ForegroundColor Yellow
Stop-Process -Id $ApiProcess.Id -Force -ErrorAction SilentlyContinue
Write-Host "✅ Services stopped." -ForegroundColor Green
