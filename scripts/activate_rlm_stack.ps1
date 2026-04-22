#Requires -Version 5.1
<#
  Activate the RLM stack (repo on PATH, venv, .env) and optionally start the **unified** process:
  master loop + IBKR paper equity + Telegram bot:
      python  scripts\run_master.py  --telegram-bot
  (Do not also run rlm_telegram_bot.py - that would duplicate the bot.)
  From repo root:
      .\scripts\activate_rlm_stack.ps1
      .\scripts\activate_rlm_stack.ps1 -Run
#>
param(
  [string]$VenvPython = "",
  [switch]$Run
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not (Test-Path (Join-Path $Repo "pyproject.toml"))) {
  Write-Error "Run from RLM clone; expected pyproject.toml under $Repo"
}
Set-Location $Repo
$Py = if ($VenvPython) { $VenvPython } else { "py" }
# Prefer project venv if present (common: .venv or rlm-venv on VPS)
$Candidates = @(
  (Join-Path $Repo ".venv\Scripts\python.exe"),
  (Join-Path $Repo "..\rlm-venv\Scripts\python.exe"),
  "C:\opt\rlm-venv\Scripts\python.exe"
)
foreach ($c in $Candidates) {
  if (Test-Path $c) { $Py = $c; break }
}
Write-Host "Repo: $Repo"
Write-Host "Python: $Py"
if (-not (Test-Path (Join-Path $Repo ".env"))) {
  Write-Warning "No .env in repo root - copy from .env.example and add API keys, TELEGRAM_BOT_TOKEN, IBKR_* as needed."
}
$env:PYTHONUNBUFFERED = "1"
$RunScript = (Join-Path $Repo "scripts\run_master.py")
if ($Run) {
  $Base = [IO.Path]::GetFileName($Py)
  if ($Py -eq "py" -or $Base -ieq "py.exe") {
    $ArgList = @("-3", $RunScript, "--telegram-bot")
  } else {
    $ArgList = @($RunScript, "--telegram-bot")
  }
  Write-Host "Starting: $Py $($ArgList -join ' ')" -ForegroundColor Green
  Start-Process -FilePath $Py -ArgumentList $ArgList -WorkingDirectory $Repo -WindowStyle Minimized
  Write-Host "Started in background. Optional Streamlit: py -3 scripts\run_control_center.py --public --port 8501"
} else {
  Write-Host "Dry run. To start background master + Telegram: .\scripts\activate_rlm_stack.ps1 -Run" -ForegroundColor Cyan
}
