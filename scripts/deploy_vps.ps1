# Sync repo to VPS: git push (unless -SkipPush), then ssh pull + restart systemd unit(s).
# From repo root:  .\scripts\deploy_vps.ps1
# After pull, seeds data/processed/live_*.json from configs/*.seed.json only when those files are
# missing (host-tuned JSON stays gitignored). Optional -StashOnVpsBeforePull for pre-pull local edits
# to tracked files; configs/*.seed.json are restored to HEAD after stash pop.
# Overrides: $env:VPS_HOST, $env:VPS_USER, $env:VPS_REPO, $env:VPS_SYSTEMD_UNITS, $env:VPS_ENSURE_UNITS
#
# -SystemdUnits: comma-separated base names or full unit names (e.g. regime-locus-master,rlm-master-telegram).
#   Units in -EnsureUnits are enabled and started or restarted so always-on services stay up even if they
#   were stopped. Default ensure excludes market-hours trading units; those are started by market timers.
#   Other units are only restarted when already active.
#   Example: .\scripts\deploy_vps.ps1 -SystemdUnits "regime-locus-master,rlm-control-center"

param(
    [switch] $SkipPush,
    [switch] $SkipRestart,
    [switch] $SkipEnsure,
    [switch] $StashOnVpsBeforePull,
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
# Host runtime JSON is gitignored; after a pull that drops tracked copies, seed from configs/ if missing.
$remote += " && mkdir -p data/processed"
$remote += " && (test -f data/processed/live_regime_model.json || (test -f configs/live_regime_model.seed.json && cp configs/live_regime_model.seed.json data/processed/live_regime_model.json))"
$remote += " && (test -f data/processed/live_nightly_hyperparams.json || (test -f configs/live_nightly_hyperparams.seed.json && cp configs/live_nightly_hyperparams.seed.json data/processed/live_nightly_hyperparams.json))"
if ($StashOnVpsBeforePull) {
    $remote += " && (git stash pop || true)"
}
$remote += " && (git restore configs/live_regime_model.seed.json configs/live_nightly_hyperparams.seed.json 2>/dev/null || true)"
# Match deploy_to_vps.sh: refresh editable install (PEP 668 safe via /opt/rlm-venv on Ubuntu VPS)
$remote += " && PY=/opt/rlm-venv/bin/python; if [ -x `"`$PY`" ]; then `"`$PY`" -m pip install -e . -q; elif [ -x .venv/bin/python ]; then .venv/bin/python -m pip install -e . -q; fi"
if (-not $SkipRestart) {
    $unitsRaw = $SystemdUnits
    if ([string]::IsNullOrWhiteSpace($unitsRaw)) { $unitsRaw = $env:VPS_SYSTEMD_UNITS }
    if ([string]::IsNullOrWhiteSpace($unitsRaw)) {
        $unitsRaw = "rlm-master-trader,regime-locus-master,rlm-challenge-loop,rlm-systems-control-telegram,regime-locus-crew,rlm-host-watchdog,rlm-master-telegram,rlm-telegram,rlm-telegram-bot"
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
    if (-not $SkipEnsure -and [string]::IsNullOrWhiteSpace($ensureRaw)) {
        $ensureRaw = "rlm-systems-control-telegram,rlm-host-watchdog,regime-locus-crew"
    }
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
$remote += " ; if [ -f scripts/rlm_enable_startup_services.sh ]; then bash scripts/rlm_enable_startup_services.sh; fi"

ssh -o BatchMode=yes "$VpsUser@$VpsHost" $remote
Write-Host "Deploy finished." -ForegroundColor Green
