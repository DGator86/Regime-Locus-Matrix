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

UNIT_SRC="${SCRIPT_DIR}/regime-locus-master.service"
UNIT_DST="/etc/systemd/system/regime-locus-master.service"

tmp="$(mktemp)"
sed "s|@INSTALL_ROOT@|${INSTALL_ROOT}|g" "${UNIT_SRC}" >"${tmp}"
install -m 0644 "${tmp}" "${UNIT_DST}"
rm -f "${tmp}"

systemctl daemon-reload
systemctl enable regime-locus-master.service
echo "Installed ${UNIT_DST}"
echo "  Optional: copy deploy/systemd/ib-gateway.service.example with IB_GATEWAY_DIR sed, enable ib-gateway before master."
echo "  Optional: deploy/systemd/rlm-preopen*.example + rlm-postclose*.example for session brief timers."
echo "  Start now:  systemctl start regime-locus-master"
echo "  Follow log: journalctl -u regime-locus-master -f"
echo "  Status:     systemctl status regime-locus-master"
