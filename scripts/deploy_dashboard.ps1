# Sync repo to VPS and deploy the Next.js dashboard
# From repo root:  .\scripts\deploy_dashboard.ps1

param(
    [switch] $SkipPush,
    [string] $VpsHost = "2.24.28.77",
    [string] $VpsUser = "root",
    [string] $VpsRepo = "/opt/Regime-Locus-Matrix"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

if (-not $SkipPush) {
    Write-Host "Pushing changes to origin..." -ForegroundColor Cyan
    git push origin main
}

Write-Host "Deploying to VPS..." -ForegroundColor Cyan

$remote = @"
cd $VpsRepo && \
git pull origin main && \
# Install Node.js if missing
if ! command -v node &> /dev/null; then
    echo 'Installing Node.js via NVM...'
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
    export NVM_DIR="`$HOME/.nvm"
    [ -s "`$NVM_DIR/nvm.sh" ] && \. "`$NVM_DIR/nvm.sh"
    nvm install 20
    nvm use 20
fi

# Ensure nvm is loaded for the rest of the script
export NVM_DIR="`$HOME/.nvm"
[ -s "`$NVM_DIR/nvm.sh" ] && \. "`$NVM_DIR/nvm.sh"

# Build dashboard
cd dashboard && \
npm install && \
npm run build && \

# Restart/Start using PM2 (install if missing)
if ! command -v pm2 &> /dev/null; then
    npm install -g pm2
fi

pm2 delete rlm-dashboard 2>/dev/null || true
PORT=3000 RLM_DATA_DIR=$VpsRepo/data/processed pm2 start npm --name "rlm-dashboard" -- run start
pm2 save
pm2 startup systemd -u root --hp /root 2>/dev/null || true
"@

ssh -o BatchMode=yes "$VpsUser@$VpsHost" "$remote"
Write-Host "Dashboard deployed successfully at http://$VpsHost:3000" -ForegroundColor Green
