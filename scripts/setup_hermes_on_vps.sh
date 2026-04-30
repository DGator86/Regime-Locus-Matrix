#!/usr/bin/env bash
# Run ON the VPS (or pipe via SSH). Installs Hermes into the Python venv that actually runs regime-locus-crew.
set -eu
REPO="${VPS_REPO:-/opt/Regime-Locus-Matrix}"
cd "$REPO"
if [[ ! -f pyproject.toml ]]; then
  echo "error: no pyproject.toml in $REPO" >&2
  exit 1
fi
# Many VPS units use a shared venv (e.g. ExecStart=/opt/rlm-venv/bin/python); repo .venv alone is not enough.
if [[ -n "${RLM_CREW_VENV:-}" ]]; then
  VENV_ROOT="${RLM_CREW_VENV}"
elif [[ -x /opt/rlm-venv/bin/pip ]]; then
  VENV_ROOT="/opt/rlm-venv"
elif [[ -x "$REPO/.venv/bin/pip" ]]; then
  VENV_ROOT="$REPO/.venv"
else
  echo "error: no pip found — set RLM_CREW_VENV to your venv root (directory containing bin/pip), or create $REPO/.venv" >&2
  exit 1
fi
PIP="${VENV_ROOT}/bin/pip"
PY="${VENV_ROOT}/bin/python"
echo "==> using venv: $VENV_ROOT"
echo "==> git pull (ff-only)"
git pull --ff-only origin main || true
echo "==> pip install Hermes extra"
"$PIP" install -U pip -q
"$PIP" install -e ".[hermes]"
echo "==> smoke: rlm toolset registered"
PYTHONPATH="$REPO/src:$REPO" "$PY" -c "
import run_agent
import rlm_hermes_tools.register_rlm_tools  # noqa: F401
from tools.registry import registry
print('toolset rlm:', registry.get_tool_names_for_toolset('rlm'))
"
if [[ -f scripts/smoke_hermes_imports.py ]]; then
  echo "==> smoke: scripts/smoke_hermes_imports.py"
  "$PY" scripts/smoke_hermes_imports.py
fi
if systemctl is-active --quiet regime-locus-crew.service 2>/dev/null; then
  systemctl restart regime-locus-crew.service
  echo "==> regime-locus-crew restarted"
  systemctl --no-pager -l status regime-locus-crew.service | head -20
else
  echo "==> regime-locus-crew is not active (skip restart). Enable with: systemctl enable --now regime-locus-crew"
fi
