# Sync repo to VPS: git push (unless -SkipPush), then ssh pull + restart systemd unit(s).
# From repo root:  .\scripts\deploy_vps.ps1
# Overrides: $env:VPS_HOST, $env:VPS_USER, $env:VPS_REPO, $env:VPS_SYSTEMD_UNITS, $env:VPS_ENSURE_UNITS
#
# -SystemdUnits: comma-separated base names or full unit names (e.g. regime-locus-master,rlm-master-telegram).
#   Units in -EnsureUnits (default: regime-locus-crew) are enabled and started or restarted so Hermes
#   stays up even if it was stopped. Other units are only restarted when already active.
#   Example: .\scripts\deploy_vps.ps1 -SystemdUnits "regime-locus-master,rlm-control-center"

param(
    [switch] $SkipPush,
    [switch] $SkipRestart,
    [switch] $SkipEnsure,
    [switch] $StashOnVpsBeforePull, # Stash before pull so host-only data/processed/*.json survives if git drops tracked copies
    [string] $SystemdUnits = "",
    [string] $EnsureUnits = ""
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
# Match deploy_to_vps.sh: refresh editable install (PEP 668 safe via /opt/rlm-venv on Ubuntu VPS)
$remote += " && PY=/opt/rlm-venv/bin/python; if [ -x `"`$PY`" ]; then `"`$PY`" -m pip install -e . -q; elif [ -x .venv/bin/python ]; then .venv/bin/python -m pip install -e . -q; fi"
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
    $ensureRaw = $EnsureUnits
    if (-not $SkipEnsure -and [string]::IsNullOrWhiteSpace($ensureRaw)) { $ensureRaw = $env:VPS_ENSURE_UNITS }
    if (-not $SkipEnsure -and [string]::IsNullOrWhiteSpace($ensureRaw)) { $ensureRaw = "regime-locus-crew" }
    $ensureWithService = @()
    if (-not $SkipEnsure -and -not [string]::IsNullOrWhiteSpace($ensureRaw)) {
        foreach ($u in $ensureRaw.Split(",", [StringSplitOptions]::RemoveEmptyEntries)) {
            $t = $u.Trim()
            if (-not $t) { continue }
            if ($t.EndsWith(".service")) { $ensureWithService += $t }
            else { $ensureWithService += "$t.service" }
        }
    }
    $ensureSet = [System.Collections.Generic.HashSet[string]]::new()
    foreach ($x in $ensureWithService) {
        [void]$ensureSet.Add($x)
    }
    $restartOnly = @($withService | Where-Object { -not $ensureSet.Contains($_) })
    $bashEnsure = ($ensureWithService | ForEach-Object { "`"$_`"" }) -join " "
    $bashRestart = ($restartOnly | ForEach-Object { "`"$_`"" }) -join " "
    $remote += " && restarted=0"
    if (-not [string]::IsNullOrWhiteSpace($bashEnsure)) {
        $remote += " ; for u in $bashEnsure; do systemctl enable `"`$u`" 2>/dev/null || true; if systemctl is-active --quiet `"`$u`"; then systemctl restart `"`$u`" && restarted=1; else systemctl start `"`$u`" && restarted=1; fi; done"
    }
    if (-not [string]::IsNullOrWhiteSpace($bashRestart)) {
        $remote += " ; for u in $bashRestart; do if systemctl is-active --quiet `"`$u`"; then systemctl restart `"`$u`" && restarted=1; fi; done"
    }
    $remote += " ; if [ `"`$restarted`" -eq 0 ]; then echo 'deploy_vps: no systemd unit from the deploy list was restarted or started (check systemctl status)'; fi"
}

ssh -o BatchMode=yes "$VpsUser@$VpsHost" $remote
Write-Host "Deploy finished." -ForegroundColor Green
