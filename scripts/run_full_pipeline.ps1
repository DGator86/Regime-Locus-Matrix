$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

Step "Verifying project files exist"
$required = @(
    "pyproject.toml",
    "scripts/build_rolling_backtest_dataset.py",
    "scripts/build_features.py",
    "scripts/run_forecast_pipeline.py",
    "scripts/run_backtest.py"
)
foreach ($f in $required) {
    if (-not (Test-Path $f)) {
        throw "Missing required file: $f. Run this from the repo root."
    }
}

Step "Creating virtual environment (.venv) if needed"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.12 -m venv .venv
}

$python = ".\.venv\Scripts\python.exe"
$pip = ".\.venv\Scripts\pip.exe"

Step "Upgrading pip and installing dependencies"
& $python -m pip install --upgrade pip
& $pip install -e ".[dev]"

Step "Building demo rolling backtest dataset"
& $python scripts/build_rolling_backtest_dataset.py --demo

Step "Building features (--no-vix)"
& $python scripts/build_features.py --no-vix

Step "Running forecast pipeline (--no-vix)"
& $python scripts/run_forecast_pipeline.py --no-vix

Step "Running backtest (--no-vix)"
& $python scripts/run_backtest.py --no-vix

Write-Host ""
Write-Host "Pipeline complete." -ForegroundColor Green
