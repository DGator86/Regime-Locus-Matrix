#!/usr/bin/env bash
# deploy_to_vps.sh -- push current branch to GitHub then pull + restart on VPS
#
# Usage (from repo root):
#   bash scripts/deploy_to_vps.sh                      # push current branch, deploy to VPS
#   bash scripts/deploy_to_vps.sh --skip-push          # VPS pull only (branch already pushed)
#   bash scripts/deploy_to_vps.sh --skip-restart       # pull only, no systemd restart
#   bash scripts/deploy_to_vps.sh --health-check       # run rlm_health_check.py on VPS after deploy
#
# Environment overrides:
#   VPS_HOST   (default: 2.24.28.77)
#   VPS_USER   (default: root)
#   VPS_REPO   (default: /opt/Regime-Locus-Matrix)
#   VPS_BRANCH (default: main)

set -euo pipefail

VPS_HOST="${VPS_HOST:-2.24.28.77}"
VPS_USER="${VPS_USER:-root}"
VPS_REPO="${VPS_REPO:-/opt/Regime-Locus-Matrix}"
VPS_BRANCH="${VPS_BRANCH:-main}"

SKIP_PUSH=0
SKIP_RESTART=0
RUN_HEALTH_CHECK=0

for arg in "$@"; do
    case "$arg" in
        --skip-push)     SKIP_PUSH=1 ;;
        --skip-restart)  SKIP_RESTART=1 ;;
        --health-check)  RUN_HEALTH_CHECK=1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── 1. Push to GitHub ────────────────────────────────────────────────────────
if [[ "$SKIP_PUSH" -eq 0 ]]; then
    CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "[deploy] Uncommitted changes detected — commit or stash first."
        exit 1
    fi
    echo "[deploy] Pushing $CURRENT_BRANCH to origin..."
    git push -u origin "$CURRENT_BRANCH"
fi

# ── 2. SSH into VPS ──────────────────────────────────────────────────────────
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 ${VPS_USER}@${VPS_HOST}"

echo "[deploy] Connecting to ${VPS_USER}@${VPS_HOST}..."

VPS_CMD="
set -euo pipefail
cd ${VPS_REPO}

echo '[vps] Current branch:' \$(git rev-parse --abbrev-ref HEAD)
echo '[vps] Fetching origin...'
git fetch origin ${VPS_BRANCH}

echo '[vps] Pulling ${VPS_BRANCH}...'
git checkout ${VPS_BRANCH}
git pull --ff-only origin ${VPS_BRANCH}

echo '[vps] Latest commit:' \$(git log --oneline -1)

# Resolve Python
if [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
elif command -v python3 &>/dev/null; then
    PY=python3
else
    echo '[vps] ERROR: no python3 found' && exit 1
fi

# Install/update package in case pyproject.toml changed
echo '[vps] pip install -e . (quiet)...'
\$PY -m pip install -e '.' -q
"

# ── 3. Restart systemd units ─────────────────────────────────────────────────
if [[ "$SKIP_RESTART" -eq 0 ]]; then
    VPS_CMD+="
UNITS=(regime-locus-master regime-locus-crew rlm-control-center rlm-telegram rlm-master-telegram rlm-telegram-bot)
for unit in \"\${UNITS[@]}\"; do
    if systemctl is-active --quiet \"\${unit}.service\" 2>/dev/null; then
        echo \"[vps] Restarting \${unit}.service...\"
        systemctl restart \"\${unit}.service\" && echo \"[vps]   OK\" || echo \"[vps]   FAILED (not fatal)\"
    fi
done
echo '[vps] Active RLM services:'
systemctl list-units 'regime-locus-*' 'rlm-*' --no-pager --no-legend 2>/dev/null | head -20 || true
"
fi

# ── 4. Health check on VPS ───────────────────────────────────────────────────
if [[ "$RUN_HEALTH_CHECK" -eq 1 ]]; then
    VPS_CMD+="
echo '[vps] Running rlm_health_check.py...'
cd ${VPS_REPO}
\$PY scripts/rlm_health_check.py --force
"
fi

$SSH bash -c "$VPS_CMD"

echo "[deploy] Done. VPS is on ${VPS_BRANCH} at ${VPS_HOST}:${VPS_REPO}"
