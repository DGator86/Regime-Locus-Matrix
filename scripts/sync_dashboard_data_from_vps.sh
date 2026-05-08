#!/usr/bin/env bash
# Pull live processed artefacts + challenge state from Hostinger VPS into this repo clone.
# Bash equivalent of scripts/sync_dashboard_data_from_vps.ps1 (Linux / CI).
# Usage (from repo root): bash scripts/sync_dashboard_data_from_vps.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_PROCESSED="${REPO_ROOT}/data/processed"
LOCAL_CHALLENGE="${REPO_ROOT}/data/challenge"

REMOTE_HOST="${VPS_HOST:-2.24.28.77}"
REMOTE_USER="${VPS_USER:-root}"
REMOTE_REPO="${VPS_REPO:-/opt/Regime-Locus-Matrix}"
TARGET="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_REPO}"

echo "[sync] ${TARGET} -> repo data/"

mkdir -p "${LOCAL_PROCESSED}" "${LOCAL_CHALLENGE}"

_processed=(
  universe_trade_plans.json
  trade_log.csv
  equity_trade_log.csv
  equity_positions_state.json
  forecast_features_SPY.csv
  trade_monitor_state.json
)
for f in "${_processed[@]}"; do
  scp -o BatchMode=yes -o ConnectTimeout=25 \
    "${TARGET}/data/processed/${f}" "${LOCAL_PROCESSED}/${f}"
  echo "  synced processed/${f}"
done

_challenge=(state.json trade_log.csv)
for f in "${_challenge[@]}"; do
  if scp -o BatchMode=yes -o ConnectTimeout=25 \
    "${TARGET}/data/challenge/${f}" "${LOCAL_CHALLENGE}/${f}" 2>/dev/null; then
    echo "  synced challenge/${f}"
  else
    echo "  [warn] skip challenge/${f} (missing on host or scp failed)"
  fi
done

echo "[sync] done. Restart dashboard dev server / hard-refresh browser."
