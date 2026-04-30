#!/usr/bin/env bash
# Run ON the VPS (or pipe via SSH). Installs Hermes into the repo .venv and restarts crew.
set -eu
REPO="${VPS_REPO:-/opt/Regime-Locus-Matrix}"
cd "$REPO"
if [[ ! -f pyproject.toml ]]; then
  echo "error: no pyproject.toml in $REPO" >&2
  exit 1
fi
if [[ ! -x .venv/bin/pip ]]; then
  echo "error: missing $REPO/.venv — run: python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ." >&2
  exit 1
fi
echo "==> git pull (ff-only)"
git pull --ff-only origin main || true
echo "==> pip install Hermes extra into .venv"
.venv/bin/pip install -U pip -q
.venv/bin/pip install -e ".[hermes]"
echo "==> smoke: rlm toolset registered"
PYTHONPATH="$REPO/src:$REPO" .venv/bin/python -c "
import run_agent
import rlm_hermes_tools.register_rlm_tools  # noqa: F401
from tools.registry import registry
print('toolset rlm:', registry.get_tool_names_for_toolset('rlm'))
"
if systemctl is-active --quiet regime-locus-crew.service 2>/dev/null; then
  systemctl restart regime-locus-crew.service
  echo "==> regime-locus-crew restarted"
  systemctl --no-pager -l status regime-locus-crew.service | head -20
else
  echo "==> regime-locus-crew is not active (skip restart). Enable with: systemctl enable --now regime-locus-crew"
fi
