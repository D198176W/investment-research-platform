Write-Host "================================"
Write-Host "Smart Investment Research Platform"
Write-Host "================================"
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

if (-not (Test-Path ".env")) {
    Write-Host "[Info] Copying .env.example to .env..."
    Copy-Item ".env.example" ".env"
    Write-Host "[Warning] Please edit .env and add your API key!"
    Write-Host ""
}

Write-Host "[1/2] Starting FastAPI Backend..."
$backendJob = Start-Job -ScriptBlock {
    param($dir)
    Set-Location "$dir\backend\gateway"
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
} -ArgumentList $scriptDir

Start-Sleep -Seconds 3

Write-Host "[2/2] Starting Streamlit Frontend..."
$frontendJob = Start-Job -ScriptBlock {
    param($dir)
    Set-Location "$dir\frontend"
    streamlit run streamlit_app.py --server.port 8501
} -ArgumentList $scriptDir

Write-Host ""
Write-Host "================================"
Write-Host "Services Started Successfully!"
Write-Host "================================"
Write-Host "Backend API: http://localhost:8000"
Write-Host "API Docs:    http://localhost:8000/docs"
Write-Host "Frontend:    http://localhost:8501"
Write-Host "================================"
Write-Host ""
Write-Host "Press any key to exit (services will continue running)..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")