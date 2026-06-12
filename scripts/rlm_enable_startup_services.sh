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

_disable() {
  local u="$1"
  systemctl disable "${u}" 2>/dev/null || true
}

_market_services_window_open() {
  case "${RLM_FORCE_MARKET_SERVICE_WINDOW:-}" in
    1|true|TRUE|yes|YES|open|OPEN)
      echo "forced_open"
      return 0
      ;;
    0|false|FALSE|no|NO|closed|CLOSED)
      echo "forced_closed"
      return 1
      ;;
  esac

  local pybin="${PY}"
  if [[ ! -x "${pybin}" ]]; then
    pybin="python3"
  fi

  "${pybin}" - <<'PY'
from datetime import datetime, time
from zoneinfo import ZoneInfo

now = datetime.now(ZoneInfo("America/New_York"))
t = now.time().replace(second=0, microsecond=0)
if now.weekday() < 5 and time(9, 0) <= t < time(16, 30):
    print("market_service_window_open")
    raise SystemExit(0)
print("market_service_window_closed")
raise SystemExit(1)
PY
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
  regime-locus-crew.service; do
  _enable "${u}"
done

systemctl daemon-reload

echo "[startup] disabling direct boot for market-hours trading units"
for u in \
  rlm-master-trader.service \
  regime-locus-master.service \
  rlm-master-telegram.service \
  rlm-challenge-loop.service; do
  _disable "${u}"
done

echo "[startup] starting always-on services"
_start rlm-host-watchdog.service
_start rlm-systems-control-telegram.service
_start regime-locus-crew.service

if window_label="$(_market_services_window_open)"; then
  echo "[startup] market service window open (${window_label}); starting trading units"
  _start rlm-challenge-loop.service
  # Prefer three-track master unit when present.
  if systemctl list-unit-files rlm-master-trader.service --no-legend 2>/dev/null | grep -q rlm-master-trader; then
    _start rlm-master-trader.service
  elif systemctl list-unit-files regime-locus-master.service --no-legend 2>/dev/null | grep -q regime-locus-master; then
    _start regime-locus-master.service
  fi
else
  echo "[startup] market service window closed (${window_label:-unknown}); trading units stay stopped"
fi

_start rlm-market-open.timer
_start rlm-market-close.timer

if [[ -x "${PY}" && -f "${ROOT}/scripts/verify_kronos_gpu.py" ]]; then
  echo "[startup] Kronos GPU probe"
  (
    cd "${ROOT}"
    export RLM_ROOT="${ROOT}"
    "${PY}" "${ROOT}/scripts/verify_kronos_gpu.py"
  ) || echo "[startup] WARN: Kronos GPU probe failed (check RLM_KRONOS_REMOTE_URL)"
fi

echo "[startup] active units:"
systemctl is-active rlm-master-trader.service regime-locus-master.service rlm-challenge-loop.service \
  rlm-systems-control-telegram.service rlm-host-watchdog.service regime-locus-crew.service 2>/dev/null || true
