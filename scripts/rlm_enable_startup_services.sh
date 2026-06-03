#!/usr/bin/env bash
# Enable and start the full three-track RLM stack (run on VPS after deploy).
set -euo pipefail

ROOT="${RLM_ROOT:-/opt/Regime-Locus-Matrix}"
PY="${RLM_PYTHON:-/opt/rlm-venv/bin/python}"

cd "${ROOT}"

if [[ -x "${PY}" && -f "${ROOT}/scripts/migrate_vps_three_tracks.py" ]]; then
  echo "[startup] applying three-track .env profile"
  "${PY}" "${ROOT}/scripts/migrate_vps_three_tracks.py" || true
fi

_enable() {
  local u="$1"
  systemctl enable "${u}" 2>/dev/null || true
}

_start() {
  local u="$1"
  systemctl start "${u}" 2>/dev/null || true
}

echo "[startup] enabling timers and core units"
for u in \
  rlm-market-open.timer \
  rlm-market-close.timer \
  rlm-forecast.timer \
  rlm-nightly-opt.timer \
  rlm-weekly-calibrate.timer \
  rlm-startup-decision-health.service \
  rlm-host-watchdog.service \
  rlm-systems-control-telegram.service \
  rlm-challenge-loop.service \
  regime-locus-crew.service; do
  _enable "${u}"
done

# Prefer three-track master unit when present.
if systemctl list-unit-files rlm-master-trader.service --no-legend 2>/dev/null | grep -q rlm-master-trader; then
  _enable rlm-master-trader.service
else
  _enable regime-locus-master.service
fi

systemctl daemon-reload

echo "[startup] starting always-on services"
_start rlm-host-watchdog.service
_start rlm-systems-control-telegram.service
_start regime-locus-crew.service
_start rlm-challenge-loop.service

if systemctl list-unit-files rlm-master-trader.service --no-legend 2>/dev/null | grep -q rlm-master-trader; then
  _start rlm-master-trader.service
elif systemctl list-unit-files regime-locus-master.service --no-legend 2>/dev/null | grep -q regime-locus-master; then
  _start regime-locus-master.service
fi

_start rlm-market-open.timer
_start rlm-market-close.timer

if [[ -x "${PY}" && -f "${ROOT}/scripts/verify_kronos_gpu.py" ]]; then
  echo "[startup] Kronos GPU probe"
  "${PY}" "${ROOT}/scripts/verify_kronos_gpu.py" || echo "[startup] WARN: Kronos GPU probe failed (check RLM_KRONOS_REMOTE_URL)"
fi

echo "[startup] active units:"
systemctl is-active rlm-master-trader.service regime-locus-master.service rlm-challenge-loop.service \
  rlm-systems-control-telegram.service rlm-host-watchdog.service regime-locus-crew.service 2>/dev/null || true
