# Sync repo to VPS: git push (unless -SkipPush), then ssh pull + restart rlm-telegram.
# From repo root:  .\scripts\deploy_vps.ps1
# Overrides: $env:VPS_HOST, $env:VPS_USER, $env:VPS_REPO

param(
    [switch] $SkipPush,
    [switch] $SkipRestart,
    [switch] $StashOnVpsBeforePull
)

$ErrorActionPreference = "Stop"
$VpsHost = if ($env:VPS_HOST) { $env:VPS_HOST } else { "2.24.28.77" }
$VpsUser = if ($env:VPS_USER) { $env:VPS_USER } else { "root" }
$VpsRepo = if ($env:VPS_REPO) { $env:VPS_REPO } else { "/opt/Regime-Locus-Matrix" }

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

if (-not $SkipPush) {
    $dirty = git status --porcelain
    if ($dirty) {
        Write-Host "Commit or stash changes before deploy (or use -SkipPush)." -ForegroundColor Yellow
        exit 1
    }
    git push origin main
}

$remote = "cd $VpsRepo && "
if ($StashOnVpsBeforePull) {
    $remote += "git stash push -u -m deploy_vps_autostash 2>/dev/null || true; "
}
$remote += "git pull --ff-only origin main"
if ($StashOnVpsBeforePull) {
    $remote += " && (git stash pop || true)"
}
if (-not $SkipRestart) {
    $remote += " && systemctl restart rlm-telegram.service && systemctl is-active rlm-telegram.service"
}

ssh -o BatchMode=yes "$VpsUser@$VpsHost" $remote
Write-Host "Deploy finished." -ForegroundColor Green
