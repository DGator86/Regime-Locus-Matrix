# Sync repo to VPS: git push (unless -SkipPush), then ssh pull + restart systemd unit(s).
# From repo root:  .\scripts\deploy_vps.ps1
# Overrides: $env:VPS_HOST, $env:VPS_USER, $env:VPS_REPO, $env:VPS_SYSTEMD_UNITS
#
# -SystemdUnits: comma-separated base names or full unit names (e.g. regime-locus-master,rlm-master-telegram).
#   Only units that are active are restarted. Default list covers master+telegram and standalone telegram.
#   Example: .\scripts\deploy_vps.ps1 -SystemdUnits "regime-locus-master,rlm-control-center"

param(
    [switch] $SkipPush,
    [switch] $SkipRestart,
    [switch] $StashOnVpsBeforePull,
    [string] $SystemdUnits = ""
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
    $unitsRaw = $SystemdUnits
    if ([string]::IsNullOrWhiteSpace($unitsRaw)) { $unitsRaw = $env:VPS_SYSTEMD_UNITS }
    if ([string]::IsNullOrWhiteSpace($unitsRaw)) {
        $unitsRaw = "regime-locus-master,regime-locus-crew,rlm-master-telegram,rlm-telegram,rlm-telegram-bot,rlm-systems-control-telegram,rlm-host-watchdog"
    }
    $unitNames = @(
        $unitsRaw.Split(",", [StringSplitOptions]::RemoveEmptyEntries) |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ }
    )
    $withService = @()
    foreach ($u in $unitNames) {
        if ($u.EndsWith(".service")) { $withService += $u }
        else { $withService += "$u.service" }
    }
    $bashList = ($withService | ForEach-Object { "`"$_`"" }) -join " "
    $remote += " && restarted=0; for u in $bashList; do if systemctl is-active --quiet `"`$u`"; then systemctl restart `"`$u`" && restarted=1; fi; done; if [ `"`$restarted`" -eq 0 ]; then echo 'deploy_vps: no active systemd unit from the deploy list was running (check systemctl status)'; fi"
}

ssh -o BatchMode=yes "$VpsUser@$VpsHost" $remote
Write-Host "Deploy finished." -ForegroundColor Green
