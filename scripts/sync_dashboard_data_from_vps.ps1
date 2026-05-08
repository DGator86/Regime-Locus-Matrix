#!/usr/bin/env pwsh
# Pull live processed artefacts + challenge state from Hostinger VPS into this repo clone.
# Usage (from repo root): .\scripts\sync_dashboard_data_from_vps.ps1
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$LocalProcessed = Join-Path $RepoRoot 'data' 'processed'
$LocalChallenge = Join-Path $RepoRoot 'data' 'challenge'

$RemoteHost = if ($env:VPS_HOST) { $env:VPS_HOST } else { '2.24.28.77' }
$RemoteUser = if ($env:VPS_USER) { $env:VPS_USER } else { 'root' }
$RemoteRepo = if ($env:VPS_REPO) { $env:VPS_REPO } else { '/opt/Regime-Locus-Matrix' }
$Target = "${RemoteUser}@${RemoteHost}:${RemoteRepo}"

Write-Host "[sync] $Target -> repo data/" -ForegroundColor Cyan

New-Item -ItemType Directory -Force -Path $LocalProcessed | Out-Null
New-Item -ItemType Directory -Force -Path $LocalChallenge | Out-Null

$processedFiles = @(
  'universe_trade_plans.json',
  'trade_log.csv',
  'equity_trade_log.csv',
  'equity_positions_state.json',
  'forecast_features_SPY.csv',
  'trade_monitor_state.json'
)

foreach ($f in $processedFiles) {
  $src = "${Target}/data/processed/${f}"
  $dst = Join-Path $LocalProcessed $f
  scp -o BatchMode=yes -o ConnectTimeout=25 $src $dst
  Write-Host "  synced processed/$f" -ForegroundColor Green
}

$challengeFiles = @('state.json', 'trade_log.csv')
foreach ($f in $challengeFiles) {
  $src = "${Target}/data/challenge/${f}"
  $dst = Join-Path $LocalChallenge $f
  scp -o BatchMode=yes -o ConnectTimeout=25 $src $dst
  Write-Host "  synced challenge/$f" -ForegroundColor Green
}

Write-Host "[sync] done. Restart dashboard dev server / hard-refresh browser." -ForegroundColor Cyan
