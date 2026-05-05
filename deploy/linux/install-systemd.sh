#!/usr/bin/env bash
# Install regime-locus-master.service on a Linux VPS (systemd).
#
# Usage (as root on the VPS, from repo root or deploy/linux):
#   export INSTALL_ROOT=/root/Regime-Locus-Matrix   # or /opt/regime-locus-matrix
#   sudo bash deploy/linux/install-systemd.sh
#
# Before enabling:
#   cd "$INSTALL_ROOT" && python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e ".[ibkr]"
#   cp your-secrets.env .env
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-$ROOT_DEFAULT}"

if [[ ! -f "${INSTALL_ROOT}/scripts/run_master.py" ]]; then
  echo "error: INSTALL_ROOT=${INSTALL_ROOT} does not look like the repo (missing scripts/run_master.py)" >&2
  exit 1
fi
if [[ ! -x "${INSTALL_ROOT}/.venv/bin/python" ]]; then
  echo "error: venv not found at ${INSTALL_ROOT}/.venv/bin/python — create it and pip install -e . first" >&2
  exit 1
fi

_install_unit() {
  local src="$1" dst="$2"
  local tmp
  tmp="$(mktemp)"
  sed "s|@INSTALL_ROOT@|${INSTALL_ROOT}|g" "${src}" >"${tmp}"
  install -m 0644 "${tmp}" "${dst}"
  rm -f "${tmp}"
  echo "Installed ${dst}"
}

_install_script() {
  local src="$1" dst="$2"
  local tmp
  tmp="$(mktemp)"
  sed "s|@INSTALL_ROOT@|${INSTALL_ROOT}|g" "${src}" >"${tmp}"
  install -m 0755 "${tmp}" "${dst}"
  rm -f "${tmp}"
  echo "Installed ${dst}"
}

_install_unit "${SCRIPT_DIR}/regime-locus-master.service" \
              "/etc/systemd/system/regime-locus-master.service"

_install_unit "${SCRIPT_DIR}/rlm-forecast.service" \
              "/etc/systemd/system/rlm-forecast.service"
_install_unit "${SCRIPT_DIR}/rlm-forecast.timer" \
              "/etc/systemd/system/rlm-forecast.timer"

_install_script "${SCRIPT_DIR}/rlm-market-hours-start.sh" \
                "/usr/local/bin/rlm-market-hours-start.sh"
_install_script "${SCRIPT_DIR}/rlm-market-hours-stop.sh" \
                "/usr/local/bin/rlm-market-hours-stop.sh"
_install_unit "${SCRIPT_DIR}/rlm-market-open.service" \
              "/etc/systemd/system/rlm-market-open.service"
_install_unit "${SCRIPT_DIR}/rlm-market-open.timer" \
              "/etc/systemd/system/rlm-market-open.timer"
_install_unit "${SCRIPT_DIR}/rlm-market-close.service" \
              "/etc/systemd/system/rlm-market-close.service"
_install_unit "${SCRIPT_DIR}/rlm-market-close.timer" \
              "/etc/systemd/system/rlm-market-close.timer"
_install_unit "${SCRIPT_DIR}/rlm-challenge-loop.service" \
              "/etc/systemd/system/rlm-challenge-loop.service"

_install_unit "${SCRIPT_DIR}/rlm-host-watchdog.service" \
              "/etc/systemd/system/rlm-host-watchdog.service"

_install_unit "${SCRIPT_DIR}/rlm-nightly-opt.service" \
              "/etc/systemd/system/rlm-nightly-opt.service"
_install_unit "${SCRIPT_DIR}/rlm-nightly-opt.timer" \
              "/etc/systemd/system/rlm-nightly-opt.timer"

_install_unit "${SCRIPT_DIR}/rlm-weekly-calibrate.service" \
              "/etc/systemd/system/rlm-weekly-calibrate.service"
_install_unit "${SCRIPT_DIR}/rlm-weekly-calibrate.timer" \
              "/etc/systemd/system/rlm-weekly-calibrate.timer"

systemctl daemon-reload
systemctl enable regime-locus-master.service
systemctl enable rlm-forecast.timer
systemctl enable rlm-market-open.timer
systemctl enable rlm-market-close.timer
# Challenge loop: started at NY open / stopped at NY close via rlm-market-hours-*.sh (avoid 24/7 unless you enable it).
systemctl enable rlm-host-watchdog.service
systemctl enable rlm-nightly-opt.timer
systemctl enable rlm-weekly-calibrate.timer

echo ""
echo "  Optional: copy deploy/systemd/ib-gateway.service.example with IB_GATEWAY_DIR sed, enable ib-gateway before master."
echo ""
echo "  Start master:    systemctl start regime-locus-master"
echo "  Start timers:    systemctl start rlm-market-open.timer rlm-market-close.timer rlm-nightly-opt.timer rlm-weekly-calibrate.timer"
echo "  Follow log:      journalctl -u regime-locus-master -f"
echo "  Market open log: journalctl -u rlm-market-open.service -f"
echo "  Market close log:journalctl -u rlm-market-close.service -f"
echo "  Nightly opt log: tail -f /var/log/rlm-nightly-opt.log"
echo "  Calibrate log:   tail -f /var/log/rlm-weekly-calibrate.log"
echo "  Host watchdog:   journalctl -u rlm-host-watchdog -f"
